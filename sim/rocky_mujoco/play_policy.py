"""Watch a trained PPO policy (train_ppo.py) and steer it.

    python play_policy.py runs/ppo_rocky/latest.zip
    python play_policy.py runs/ppo_rocky/latest.zip --cmd 0 0.08 0     # fixed (vx, vy, wz) command
    python play_policy.py runs/ppo_rocky/latest.zip --record media/policy.mp4 --seconds 10
    python play_policy.py runs/ppo_rocky/latest.zip --eval     # headless score on 6 fixed commands
    python play_policy.py zero --eval                            # same score for "just stand" (baseline)

vecnormalize.pkl next to the .zip is loaded automatically (observation normalisation).
"""
import argparse
import os
import pickle
import time

import numpy as np
from stable_baselines3 import PPO

import rocky_env

ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--cmd", type=float, nargs=3, default=None, help="vx vy wz (default: random, resampled)")
ap.add_argument("--terrain", default="flat")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--dr", action="store_true", help="keep domain randomisation on")
ap.add_argument("--record", help="write an MP4 instead of opening a window")
ap.add_argument("--residual", default="stand", choices=["stand", "gait"], help="must match training")
ap.add_argument("--eval", action="store_true", help="headless: fixed commands, print tracking error and return")
a = ap.parse_args()

model = None if a.policy == "zero" else PPO.load(a.policy, device="cpu")
norm = None
npath = os.path.join(os.path.dirname(os.path.abspath(a.policy)), "vecnormalize.pkl")
if model is not None and os.path.exists(npath):
    with open(npath, "rb") as f:
        norm = pickle.load(f)          # VecNormalize object: use its running obs mean/var


def normalize(o):
    if norm is None:
        return o
    return np.clip((o - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon), -norm.clip_obs, norm.clip_obs)


def act_fn(obs):
    if model is None:
        return np.zeros(15)              # "zero" = reference only (stand, or the open-loop gait)
    return model.predict(normalize(obs), deterministic=True)[0]


if a.eval:
    cmds = [(0, 0.08, 0), (0, -0.08, 0), (0.08, 0, 0), (-0.08, 0, 0), (0, 0, 0.4), (0, 0, 0)]
    env = rocky_env.RockyEnv(terrain=a.terrain, domain_rand=a.dr, episode_seconds=10.0, residual_around=a.residual)
    tot = []
    for c in cmds:
        obs, _ = env.reset(seed=1, options={"command": c})
        ret, ev, ew, fell = 0.0, [], [], False
        for k in range(500):
            env.command = np.asarray(c, float)
            obs, r, term, trunc, info = env.step(act_fn(obs))
            ret += r
            if k >= 100:
                ev.append(np.linalg.norm(info["v_body"][:2] - c[:2]))
                ew.append(abs(info["w_body"][2] - c[2]))
            if term:
                fell = True
                break
        tot.append(ret)
        print(f"cmd {c}: return {ret:6.2f}  |v err| {np.mean(ev):.3f} m/s  |wz err| {np.mean(ew):.3f} rad/s  fell={fell}")
    print(f"mean return {np.mean(tot):.2f}")
    raise SystemExit

env = rocky_env.RockyEnv(terrain=a.terrain, domain_rand=a.dr, episode_seconds=a.seconds, residual_around=a.residual,
                         render_mode="rgb_array" if a.record else "human")
opts = {"command": a.cmd} if a.cmd else None
obs, info = env.reset(seed=0, options=opts)
frames, t0, ret = [], time.time(), 0.0
for k in range(int(a.seconds / env.control_dt)):
    act = act_fn(obs)
    if a.cmd:
        env.command = np.asarray(a.cmd, float)
    obs, r, term, trunc, info = env.step(act)
    ret += r
    if a.record and k % 2 == 0:
        frames.append(env.render())
    elif not a.record:
        time.sleep(max(0.0, env.control_dt * (k + 1) - (time.time() - t0)))
    if k % 50 == 0:
        print(f"t={k*env.control_dt:5.1f}s cmd={np.round(info['command'], 3)} v=({info['v_body'][0]:+.3f},"
              f" {info['v_body'][1]:+.3f}) wz={info['w_body'][2]:+.3f}")
    if term or trunc:
        print("episode end", "(fell)" if term else "", "return", round(ret, 2))
        obs, info = env.reset(options=opts)
        ret = 0.0
if frames:
    import imageio
    imageio.mimsave(a.record, frames, fps=25, quality=6, macro_block_size=8)
    print("wrote", a.record)
env.close()
