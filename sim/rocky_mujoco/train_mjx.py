"""GPU-parallel PPO for Rocky with MJX + Brax (MuJoCo Playground style). Linux / WSL2 + NVIDIA GPU.

    # in WSL2 (Ubuntu) with an NVIDIA driver installed on Windows:
    pip install "jax[cuda12]==0.9.2" mujoco mujoco-mjx playground brax mediapy
    (tested with jax 0.9.2 / brax 0.14.2 / playground 0.2.0 / mujoco 3.16: jax 0.12 breaks brax 0.14.2)
    python train_mjx.py --num-timesteps 100000000 --num-envs 4096
    python train_mjx.py --impl warp ...                # MuJoCo Warp backend (newer GPUs, faster)
    python train_mjx.py --smoke                        # tiny CPU run to check everything compiles

Writes runs/<name>/params.pkl, params_latest.pkl (every eval), progress.csv (every eval), config.json,
and runs/<name>/videos/step_XXXXXXXk.mp4: a 9 s clip (forward, sideways, turn) of the policy at step 0,
at every eval and at the end, so you can watch it improve (--num-evals sets how many).
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

# Every progress video runs the same command script, so clips from different checkpoints compare directly.
VIDEO_CMDS = [(3.0, (0.0, 0.08, 0.0), "forward 8 cm/s (toward leg 1)"),
              (3.0, (0.08, 0.0, 0.0), "sideways 8 cm/s"),
              (3.0, (0.0, 0.0, 0.4), "turn 0.4 rad/s")]


class VideoRecorder:
    """Rolls out the current deterministic policy on the eval env and writes an MP4 with a caption.
    reset / step / policy are jitted once; params are an argument, so later checkpoints don't recompile."""

    def __init__(self, env, make_policy_fn_holder, folder):
        self.env, self.folder = env, folder
        self.reset, self.step = jax.jit(env.reset), jax.jit(env.step)
        self.holder = make_policy_fn_holder
        self.policy = jax.jit(lambda p, obs, k: self.holder["make_policy"](p, deterministic=True)(obs, k)[0])
        os.makedirs(folder, exist_ok=True)

    def __call__(self, params, step, total):
        import mediapy
        from PIL import Image, ImageDraw
        t0 = time.time()
        state = self.reset(jax.random.PRNGKey(1))
        traj, labels, rng = [], [], jax.random.PRNGKey(2)
        for dur, cmd, label in VIDEO_CMDS:
            for _ in range(int(dur / self.env.dt)):
                state.info["command"] = jp.array(cmd)
                rng, k = jax.random.split(rng)
                state = self.step(state, self.policy(params, state.obs, k))
                traj.append(state)
                labels.append(label)
        frames = self.env.render(traj[::2], height=360, width=640, camera="track")
        out = []
        for img, label in zip(frames, labels[::2]):
            im = Image.fromarray(np.asarray(img))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 640, 26], fill=(0, 0, 0))
            d.text((8, 7), f"step {step / 1e6:6.1f} M / {total / 1e6:.0f} M   command: {label}", fill=(255, 255, 255))
            out.append(np.asarray(im))
        path = os.path.join(self.folder, f"step_{step // 1000:07d}k.mp4")
        mediapy.write_video(path, out, fps=25)
        print(f"video {path} ({time.time() - t0:.0f} s)", flush=True)
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-timesteps", type=int, default=100_000_000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--impl", default="jax", choices=["jax", "warp"])
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--name", default="mjx_rocky")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--est-obs", action="store_true",
                    help="policy also sees the (noisy) Kalman-filter body velocity + height, like rocky_env estimator=ekf")
    ap.add_argument("--smoke", action="store_true", help="tiny run (CPU ok) to test the pipeline")
    ap.add_argument("--num-evals", type=int, default=10, help="evals = checkpoints = progress videos (plus step 0)")
    ap.add_argument("--no-video", action="store_true", help="skip the progress videos")
    a = ap.parse_args()
    if a.smoke:
        a.num_timesteps, a.num_envs = 20_000, 32

    out = os.path.join(HERE, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    cfg = R.default_config()
    cfg.impl = a.impl
    cfg.est_obs = a.est_obs
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
        with open(os.path.join(out, "progress.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    holder = {}
    recorder = None if a.no_video else VideoRecorder(eval_env, holder, os.path.join(out, "videos"))

    def checkpoint(step, make_policy, params):        # brax: once before training (step 0) and after every eval
        with open(os.path.join(out, "params_latest.pkl"), "wb") as f:
            pickle.dump(params, f)
        if recorder is not None:
            holder["make_policy"] = make_policy
            try:
                recorder(params, step, a.num_timesteps)
            except Exception as e:      # rendering needs EGL/OSMesa; never kill a long run over a video
                print("video skipped:", e, flush=True)

    import json
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "env_config": cfg.to_dict()}, f, indent=1)

    small = a.smoke
    make_policy, params, _ = ppo.train(
        environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=None if a.no_dr else R.domain_randomize,
        num_timesteps=a.num_timesteps, num_envs=a.num_envs, episode_length=cfg.episode_length,
        num_evals=2 if small else a.num_evals, num_eval_envs=8 if small else 128,
        unroll_length=20, batch_size=16 if small else 256, num_minibatches=2 if small else 32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
        normalize_observations=True, reward_scaling=1.0, max_grad_norm=1.0, clipping_epsilon=0.2,
        network_factory=net, progress_fn=progress, policy_params_fn=checkpoint, seed=0,
    )
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    print("saved", out)

    if recorder is not None:
        holder["make_policy"] = make_policy
        recorder(params, a.num_timesteps, a.num_timesteps)


if __name__ == "__main__":
    main()
