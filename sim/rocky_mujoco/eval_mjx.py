"""Score a train_mjx.py policy on the GPU (many parallel rollouts, deterministic policy, no DR).

    python eval_mjx.py runs/mjx_300m                 # flat: velocity tracking + falls per command
    python eval_mjx.py runs/mjx_terrain --terrain     # + success per terrain type and level
    python eval_mjx.py runs/mjx_300m --terrain        # a flat-trained policy on the terrain course (baseline)

Reads runs/<name>/config.json and params.pkl (or params_latest.pkl). Writes runs/<name>/eval.json.

All rollouts of one evaluation run as a single GPU batch (small batches are dominated by Warp launch
overhead). Flat: 256 envs per command, 10 s; tracking error = |mean measured velocity - command| over the last 7.5 s.
Terrain: 128 envs per (type, level), spawned on the tile pad with a random heading, commanded 0.10 m/s
straight ahead for 15 s; success = no fall and >= 70 % of the commanded distance covered (the same rule
that moves the training curriculum up a level).
"""
import argparse
import functools
import json
import os
import pickle
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import rocky_mjx_env as R

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT_CMDS = [("forward +y 8 cm/s", (0.0, 0.08, 0.0)), ("back -y 8 cm/s", (0.0, -0.08, 0.0)),
             ("side +x 8 cm/s", (0.08, 0.0, 0.0)), ("side -x 8 cm/s", (-0.08, 0.0, 0.0)),
             ("fast +y 12 cm/s", (0.0, 0.12, 0.0)), ("turn 0.4 rad/s", (0.0, 0.0, 0.4)), ("stand", (0.0, 0.0, 0.0))]


def load(run, terrain=None):
    cfgj = json.load(open(os.path.join(run, "config.json")))
    args = cfgj["args"]
    terrain = terrain or args["terrain"]
    cfg = R.default_config()
    cfg.impl = args["impl"]
    cfg.est_obs = args.get("est_obs", False)
    cfg.naconmax = 3072 * (64 if terrain == "curriculum" else 10)   # contacts for all eval envs (one batch)
    cfg.njmax = 4 * (64 if terrain == "curriculum" else 10) + 16
    env = R.RockyJoystick(terrain=terrain, config=cfg)
    p = os.path.join(run, "params.pkl")
    if not os.path.exists(p):
        p = os.path.join(run, "params_latest.pkl")
    params = pickle.load(open(p, "rb"))
    obs_shape = jax.tree.map(lambda x: x.shape, env.reset(jax.random.PRNGKey(0)).obs)
    # a flat-trained policy on the terrain env: the policy input is identical, only the critic's
    # privileged input is longer (terrain scan) -> trim it to the size the saved normaliser expects
    n_priv = params[0].mean["privileged_state"].shape[-1]
    obs_shape["privileged_state"] = (n_priv,)
    nets = ppo_networks.make_ppo_networks(
        obs_shape, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(256, 128, 128), value_hidden_layer_sizes=(256, 256, 128),
        policy_obs_key="state", value_obs_key="privileged_state")
    pol = ppo_networks.make_inference_fn(nets)(params, deterministic=True)

    def policy(obs, k):
        return pol(dict(obs, privileged_state=obs["privileged_state"][..., :n_priv]), k)
    return env, policy, args, p


def rollout(env, policy, cmds, tiles=None, steps=500, seed=0):
    """One batch: env i gets command cmds[i] (and, if given, spawns on tile tiles[i] = (level, type)).
    Small batches waste the GPU (Warp launch overhead per step), so callers batch everything at once.
    Returns per-step body velocity (n, T, 3), yaw rate (n, T), alive (n, T), final displacement, alive at end."""
    cmds = jp.array(cmds, jp.float32)
    n = cmds.shape[0]
    on_tile = tiles is not None
    tiles = jp.array(tiles if on_tile else np.zeros((n, 2)), jp.int32)
    fn = _rollout_fn(env, policy, n, steps, on_tile)
    return fn(jax.random.PRNGKey(seed), cmds, tiles)


_CACHE = {}


def _rollout_fn(env, policy, n, steps, on_tile):
    key = (id(env), n, steps, on_tile)
    if key in _CACHE:
        return _CACHE[key]

    def one(key, cmd, tile):
        k1, k2 = jax.random.split(key)
        st = env.reset(k1)
        if on_tile:
            data, spawn = env._spawn(st.data, tile[0], tile[1], k2)
            st = st.replace(data=data)
            st.info.update(spawn=spawn, ep_step=jp.int32(0), respawn=jp.array(False))
        start = st.data.qpos[0:2]

        def body(carry, k):
            st, alive = carry
            st.info["command"] = cmd
            a, _ = policy(st.obs, k)
            st = env.step(st, a)
            alive = alive & (st.done < 0.5)
            Rm = st.data.xmat[1]
            return (st, alive), (Rm.T @ st.data.qvel[0:3], st.data.qvel[5], alive)

        (st, alive), (v, wz, al) = jax.lax.scan(body, (st, jp.array(True)), jax.random.split(k2, steps))
        return v, wz, al, jp.linalg.norm(st.data.qpos[0:2] - start), alive

    def batch(key, cmds, tiles):
        return jax.vmap(one)(jax.random.split(key, n), cmds, tiles)
    _CACHE[key] = jax.jit(batch)
    return _CACHE[key]


def eval_flat(env, policy, n=256, seconds=10.0):
    steps = int(seconds / env.dt)
    cmds = np.repeat([c for _, c in FLAT_CMDS], n, axis=0)
    v, wz, al, _, alive = (np.asarray(x) for x in rollout(env, policy, cmds, steps=steps))
    out = {}
    print(f"{'command':20s} {'measured vx, vy, wz':>26s}  {'|v err|':>8s} {'|wz err|':>8s}  falls")
    k0 = steps // 4
    for i, (name, c) in enumerate(FLAT_CMDS):
        s = slice(i * n, (i + 1) * n)
        m = al[s, k0:]
        vv = v[s, k0:, :2][m].mean(0)
        ww = float(wz[s, k0:][m].mean())
        ev = float(np.hypot(vv[0] - c[0], vv[1] - c[1]))
        ew = abs(ww - c[2])
        falls = float(1 - alive[s].mean())
        out[name] = {"cmd": c, "v": vv.tolist(), "wz": ww, "v_err": ev, "wz_err": ew, "fall_rate": falls}
        print(f"{name:20s} ({vv[0]:+.3f}, {vv[1]:+.3f}, {ww:+.2f})  {ev:8.3f} {ew:8.2f}  {falls * 100:4.1f} %")
    return out


def eval_terrain(env, policy, levels=(0, 3, 6, 9), n=128, seconds=15.0, speed=0.10):
    steps = int(seconds / env.dt)
    T = env._terrain
    cells = [(lv, ti) for ti in range(len(T.types)) for lv in levels]
    tiles = np.repeat(cells, n, axis=0)
    cmds = np.tile([0.0, speed, 0.0], (len(tiles), 1))
    _, _, _, dist, alive = (np.asarray(x) for x in rollout(env, policy, cmds, tiles, steps=steps))
    out = {}
    print(f"{'terrain':11s} " + " ".join(f"{'L' + str(l):>12s}" for l in levels) + "   (success %, falls %)")
    for ti, kind in enumerate(T.types):
        row = {}
        for lv in levels:
            k = cells.index((lv, ti))
            d, al = dist[k * n:(k + 1) * n], alive[k * n:(k + 1) * n]
            row[lv] = {"success": float(np.mean(al & (d >= 0.7 * speed * seconds))),
                       "fall_rate": float(1 - al.mean()), "mean_dist": float(d.mean())}
        out[kind] = row
        print(f"{kind:11s} " + " ".join(f"{row[l]['success'] * 100:5.0f} {row[l]['fall_rate'] * 100:4.0f}f " for l in levels))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--terrain", action="store_true", help="also score each terrain type and level")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    env, policy, args, path = load(run, "curriculum" if a.terrain else None)
    print(f"{os.path.basename(run)}: {path}  terrain={args['terrain']} est_obs={args.get('est_obs')}")
    res = {"params": os.path.basename(path)}
    if not a.terrain:
        res["flat"] = eval_flat(env, policy)
    if a.terrain:
        res["terrain"] = eval_terrain(env, policy)
    out = os.path.join(run, "eval_terrain.json" if a.terrain else "eval.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
