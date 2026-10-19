"""GPU-parallel PPO for Rocky with MJX + Brax (MuJoCo Playground style). Linux / WSL2 + NVIDIA GPU.

    # in WSL2 (Ubuntu) with an NVIDIA driver installed on Windows:
    pip install "jax[cuda12]==0.9.2" mujoco mujoco-mjx playground brax mediapy
    (tested with jax 0.9.2 / brax 0.14.2 / playground 0.2.0 / mujoco 3.16: jax 0.12 breaks brax 0.14.2)
    python train_mjx.py --num-timesteps 100000000 --num-envs 4096
    python train_mjx.py --impl warp ...                # MuJoCo Warp backend (newer GPUs, faster)
    python train_mjx.py --smoke                        # tiny CPU run to check everything compiles

Writes runs/mjx_rocky/params.pkl (+ progress.csv) and an MP4 rollout of the final policy.
"""
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")      # headless rendering of the rollout video

import argparse
import csv
import functools
import pickle
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground._src import wrapper

import rocky_mjx_env as R

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-timesteps", type=int, default=100_000_000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--name", default="mjx_rocky")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run (CPU ok) to test the pipeline")
    a = ap.parse_args()
    if a.smoke:
        a.num_timesteps, a.num_envs = 20_000, 32

    out = os.path.join(HERE, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    cfg = R.default_config()
    cfg.impl = a.impl
    env = R.RockyJoystick(terrain=a.terrain, config=cfg)
    eval_env = R.RockyJoystick(terrain=a.terrain, config=cfg)

    net = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=(256, 128, 128), value_hidden_layer_sizes=(256, 256, 128),
                            policy_obs_key="state", value_obs_key="privileged_state")
    rows, t0 = [], time.time()

    def progress(step, metrics):
        r = {"step": step, "time_s": round(time.time() - t0, 1),
             "eval_reward": float(metrics.get("eval/episode_reward", np.nan)),
             "eval_len": float(metrics.get("eval/avg_episode_length", np.nan))}
        rows.append(r)
        print(r, flush=True)

    small = a.smoke
    make_policy, params, _ = ppo.train(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=None if a.no_dr else R.domain_randomize,
        num_timesteps=a.num_timesteps, num_envs=a.num_envs, episode_length=cfg.episode_length,
        num_evals=2 if small else 10, num_eval_envs=8 if small else 128,
        unroll_length=20, batch_size=16 if small else 256, num_minibatches=2 if small else 32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
        normalize_observations=True, reward_scaling=1.0, max_grad_norm=1.0, clipping_epsilon=0.2,
        network_factory=net, progress_fn=progress, seed=0,
    )
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(out, "progress.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)

    # short deterministic rollout -> MP4
    try:
        import mediapy
        inference = jax.jit(make_policy(params, deterministic=True))
        reset, step = jax.jit(eval_env.reset), jax.jit(eval_env.step)
        state = reset(jax.random.PRNGKey(1))
        state.info["command"] = jp.array([0.0, 0.08, 0.0])
        traj, rng = [state], jax.random.PRNGKey(2)
        for _ in range(250 if not small else 25):
            rng, k = jax.random.split(rng)
            act, _ = inference(state.obs, k)
            state = step(state, act)
            traj.append(state)
        frames = eval_env.render(traj[::2], height=360, width=640)
        mediapy.write_video(os.path.join(out, "rollout.mp4"), frames, fps=25)
    except Exception as e:  # rendering needs EGL/OSMesa; training results are saved regardless
        print("rollout video skipped:", e)


if __name__ == "__main__":
    main()
