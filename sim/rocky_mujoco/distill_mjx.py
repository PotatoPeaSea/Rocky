"""Teacher-student distillation (DAgger) on the GPU: a student policy that sees the simulated OAK-D
height map (scan="camera": noisy, only cells the camera has seen, + a known-mask) learns to output the
actions of a teacher trained with exact terrain heights (train_mjx.py --scan true).

    python distill_mjx.py --teacher runs/mjx_teacher --name mjx_student --num-steps 60000000

Each iteration: 2048 envs run 20 steps. The student drives (during warm-up some envs are driven by the
teacher, fraction beta falling 1 -> 0), the teacher labels every visited state with its deterministic
action, and the student is fitted to those labels (MSE on the squashed action) for a few epochs.
Physics domain randomisation and the curriculum are on, as in training.

Output (runs/<name>/, same layout as train_mjx.py, so eval_mjx.py, watch_training.py and the video
recorder all work on it): params.pkl / params_latest.pkl / checkpoints/ (brax PPO format: normaliser,
policy, value), progress.csv (eval_reward column = student rollout reward per 1000 steps), config.json,
videos/step_*.mp4.
"""
import argparse
import csv
import functools
import json
import os
import pickle
import sys
import time

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src import wrapper

import rocky_mjx_env as R
from train_mjx import TERRAIN_VIDEO_CMDS_CAM, VideoRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = dict(policy_hidden_layer_sizes=(256, 128, 128), value_hidden_layer_sizes=(256, 256, 128))


def make_nets(obs_shape):
    return ppo_networks.make_ppo_networks(obs_shape, 15, preprocess_observations_fn=running_statistics.normalize,
                                          policy_obs_key="state", value_obs_key="privileged_state", **SIZES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="run folder of a --scan true policy")
    ap.add_argument("--name", default="mjx_student")
    ap.add_argument("--num-steps", type=int, default=60_000_000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.15, help="fraction of iterations with teacher-driven envs")
    ap.add_argument("--num-evals", type=int, default=10, help="checkpoints + videos (plus step 0)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = os.path.join(HERE, "runs", a.name)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    targs = json.load(open(os.path.join(HERE, a.teacher, "config.json")))["args"]
    assert targs.get("scan") == "true", "the teacher must be a --scan true run"
    cfg = R.default_config()
    cfg.impl, cfg.est_obs, cfg.scan = "warp", targs.get("est_obs", False), "camera"
    tcfg = json.load(open(os.path.join(HERE, a.teacher, "config.json"))).get("env_config", {})
    cfg.gait.enabled = "gait" in tcfg and tcfg["gait"]["enabled"]      # same gait clock as the teacher
    if cfg.gait.enabled:
        cfg.gait.update({k: tuple(v) if isinstance(v, list) else v for k, v in tcfg["gait"].items()})
    cfg.naconmax, cfg.njmax = 64 * a.num_envs, 160
    env = R.RockyJoystick(terrain="curriculum", config=cfg)
    ecfg = R.default_config(); ecfg.update(cfg.to_dict()); ecfg.naconmax = 64 * 128
    eval_env = R.RockyJoystick(terrain="curriculum", config=ecfg)
    json.dump({"args": {"terrain": "curriculum", "impl": "warp", "est_obs": cfg.est_obs, "scan": "camera",
                        "num_timesteps": a.num_steps, "teacher": a.teacher, "distill": vars(a)},
               "env_config": cfg.to_dict()}, open(os.path.join(out, "config.json"), "w"), indent=1)

    key = jax.random.PRNGKey(a.seed)
    key, kr, kd = jax.random.split(key, 3)
    rand = functools.partial(R.domain_randomize, rng=jax.random.split(kd, a.num_envs))
    wenv = wrapper.wrap_for_brax_training(env, episode_length=cfg.episode_length, action_repeat=1,
                                          randomization_fn=rand)
    state = jax.jit(wenv.reset)(jax.random.split(kr, a.num_envs))
    obs_shape = {k: v.shape[1:] for k, v in state.obs.items()}

    # teacher (frozen)
    tp = os.path.join(HERE, a.teacher, "params.pkl")
    if not os.path.exists(tp):
        tp = os.path.join(HERE, a.teacher, "params_latest.pkl")
    print("teacher:", tp, flush=True)
    tparams = pickle.load(open(tp, "rb"))
    tnets = make_nets({"state": obs_shape["teacher_state"], "privileged_state": obs_shape["privileged_state"]})
    tdist = tnets.parametric_action_distribution

    def teacher_act(obs):
        o = {"state": obs["teacher_state"], "privileged_state": obs["privileged_state"]}
        return tdist.mode(tnets.policy_network.apply(tparams[0], tparams[1], o))

    # student
    sshape = {"state": obs_shape["state"], "privileged_state": obs_shape["privileged_state"]}
    snets = make_nets(sshape)
    sdist = snets.parametric_action_distribution
    key, kp, kv = jax.random.split(key, 3)
    norm = running_statistics.init_state({k: specs.Array(v, jp.float32) for k, v in sshape.items()})
    pparams = snets.policy_network.init(kp)
    vparams = snets.value_network.init(kv)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = opt.init(pparams)

    def sobs(obs):
        return {"state": obs["state"], "privileged_state": obs["privileged_state"]}

    def student_act(norm, pparams, obs):
        return sdist.mode(snets.policy_network.apply(norm, pparams, sobs(obs)))

    per_iter = a.num_envs * a.unroll
    iters = max(1, a.num_steps // per_iter)
    warm = int(a.warmup_frac * iters)

    @jax.jit
    def collect(state, norm, pparams, beta, key):
        def body(carry, k):
            st = carry
            k1, k2 = jax.random.split(k)
            at = teacher_act(st.obs)
            as_ = student_act(norm, pparams, st.obs)
            use_t = jax.random.uniform(k1, (a.num_envs, 1)) < beta
            act = jp.where(use_t, at, as_)
            nxt = wenv.step(st, act)
            return nxt, (sobs(st.obs), at, nxt.reward, nxt.done)
        state, (o, at, r, d) = jax.lax.scan(body, state, jax.random.split(key, a.unroll))
        return state, o, at, r.mean(), d.mean()

    @jax.jit
    def fit(norm, pparams, opt_state, o, at, key):
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), o)
        at = at.reshape(-1, 15)
        norm = running_statistics.update(norm, flat)
        n = at.shape[0]
        mb = n // a.minibatches

        def loss_fn(p, ob, y):
            return jp.mean((sdist.mode(snets.policy_network.apply(norm, p, ob)) - y) ** 2)

        def epoch(carry, k):
            p, s = carry
            perm = jax.random.permutation(k, n)

            def mstep(carry, i):
                p, s = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                ob = jax.tree.map(lambda x: x[idx], flat)
                l, g = jax.value_and_grad(loss_fn)(p, ob, at[idx])
                u, s = opt.update(g, s, p)
                return (optax.apply_updates(p, u), s), l
            (p, s), ls = jax.lax.scan(mstep, (p, s), jp.arange(a.minibatches))
            return (p, s), ls.mean()
        (pparams, opt_state), ls = jax.lax.scan(epoch, (pparams, opt_state), jax.random.split(key, a.epochs))
        return norm, pparams, opt_state, ls[-1]

    holder = {"make_policy": lambda p, deterministic=True: (
        lambda obs, k: ppo_networks.make_inference_fn(snets)(p, deterministic)(sobs(obs), k))}
    recorder = VideoRecorder(eval_env, holder, os.path.join(out, "videos"), TERRAIN_VIDEO_CMDS_CAM)
    every = max(1, iters // a.num_evals)
    rows, t0 = [], time.time()

    def checkpoint(it, last_loss, last_rew):
        step = it * per_iter
        params = (norm, pparams, vparams)
        finite = all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jax.tree_util.tree_leaves(params)
                     if np.asarray(x).dtype.kind == "f")
        with open(os.path.join(out, "checkpoints", f"params_step_{step // 1000:07d}k.pkl"), "wb") as f:
            pickle.dump(params, f)
        if finite:
            with open(os.path.join(out, "params_latest.pkl"), "wb") as f:
                pickle.dump(params, f)
        else:
            print(f"WARNING: non-finite parameters at step {step}", flush=True)
        r = {"step": step, "time_s": round(time.time() - t0, 1), "eval_reward": float(last_rew) * cfg.episode_length,
             "imitation_mse": float(last_loss)}
        rows.append(r)
        print(r, flush=True)
        with open(os.path.join(out, "progress.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        try:
            recorder((norm, pparams, vparams), step, a.num_steps)
        except Exception as e:
            print("video skipped:", e, flush=True)

    loss, rew = float("nan"), 0.0
    for it in range(iters + 1):
        if it % every == 0 or it == iters:
            checkpoint(it, loss, rew)
        if it == iters:
            break
        beta = max(0.0, 1.0 - it / max(warm, 1)) if warm else 0.0
        key, kc, kf = jax.random.split(key, 3)
        state, o, at, rew, dn = collect(state, norm, pparams, beta, kc)
        norm, pparams, opt_state, loss = fit(norm, pparams, opt_state, o, at, kf)
        if it % 25 == 0:
            print(f"iter {it}/{iters} beta {beta:.2f} imitation mse {float(loss):.4f} reward/step {float(rew):.4f} "
                  f"done {float(dn) * 100:.2f} %  ({(it + 1) * per_iter / (time.time() - t0):.0f} steps/s)", flush=True)
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump((norm, pparams, vparams), f)
    print("saved", out)


if __name__ == "__main__":
    main()
