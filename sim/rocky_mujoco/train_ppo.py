"""Train a velocity-tracking walking policy with Stable-Baselines3 PPO (CPU, Windows-friendly).

    python train_ppo.py                                  # 10 M steps, n_envs = CPU count
    python train_ppo.py --steps 200000 --n-envs 4        # quick sanity run
    python train_ppo.py --terrain rough --resume runs/ppo_rocky/latest.zip
    python train_ppo.py --residual gait --name ppo_gait    # RL corrections on top of the wave gait
    python train_ppo.py --estimator ekf --name ppo_ekf     # policy also sees the EKF velocity / height

Outputs (runs/<name>/): checkpoints, latest.zip, vecnormalize.pkl, progress.csv, TensorBoard logs
(tensorboard --logdir runs). View a policy with play_policy.py.

Rough budget on a desktop CPU: ~1.5 k env steps/s per core. A usable flat-ground gait usually needs
20-50 M steps - for that, use the GPU path (train_mjx.py) or leave this running overnight.
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

HERE = os.path.dirname(os.path.abspath(__file__))


def make_env(rank, seed, terrain, domain_rand, residual, estimator="none"):
    def _f():
        import rocky_env
        env = rocky_env.RockyEnv(terrain=terrain, domain_rand=domain_rand, seed=seed + rank, residual_around=residual,
                                 estimator=estimator)
        return env
    return _f


class LogRewardTerms(BaseCallback):
    """Logs the mean raw (unscaled) value of every reward term to progress.csv / TensorBoard."""

    def __init__(self):
        super().__init__()
        self.acc, self.n = {}, 0

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.get("terms", {}).items():
                self.acc[k] = self.acc.get(k, 0.0) + v
            self.n += 1
        return True

    def _on_rollout_end(self):
        if self.n:
            for k, v in self.acc.items():
                self.logger.record(f"terms/{k}", v / self.n)
            self.acc, self.n = {}, 0


class SaveLatest(BaseCallback):
    def __init__(self, path, every):
        super().__init__()
        self.path, self.every = path, every

    def _on_step(self):
        if self.n_calls % self.every == 0:
            self.model.save(os.path.join(self.path, "latest"))
            self.training_env.save(os.path.join(self.path, "vecnormalize.pkl"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--terrain", default="flat", choices=["flat", "rough"])
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomisation")
    ap.add_argument("--residual", default="stand", choices=["stand", "gait"],
                    help="actions are offsets from the stand pose (pure RL) or from the open-loop wave gait")
    ap.add_argument("--estimator", default="none", choices=["none", "ekf", "ekf_cam"],
                    help="add the Kalman-filter velocity / height estimate to the observation")
    ap.add_argument("--name", default="ppo_rocky")
    ap.add_argument("--resume", help="path to a .zip to continue from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    out = os.path.join(HERE, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    fns = [make_env(i, a.seed, a.terrain, not a.no_dr, a.residual, a.estimator) for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    norm_path = os.path.join(out, "vecnormalize.pkl")
    if a.resume and os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, venv)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    n_steps = max(64, 4096 // a.n_envs)               # ~4 k samples per update
    if a.resume:
        model = PPO.load(a.resume, env=venv, device=a.device)
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=3e-4, n_steps=n_steps, batch_size=min(1024, n_steps * a.n_envs),
            n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            vf_coef=0.5, max_grad_norm=1.0, target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]), log_std_init=-1.6),
            seed=a.seed, device=a.device, verbose=1,
        )
    fmts = ["stdout", "csv"]
    try:
        import tensorboard  # noqa: F401
        fmts.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(out, fmts))
    cbs = [CheckpointCallback(save_freq=max(1, 500_000 // a.n_envs), save_path=out, name_prefix="ckpt",
                              save_vecnormalize=True),
           SaveLatest(out, max(1, 50_000 // a.n_envs)), LogRewardTerms()]
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cbs, reset_num_timesteps=not a.resume)
    model.save(os.path.join(out, "latest"))
    venv.save(norm_path)
    print(f"done: {a.steps} steps in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":       # required on Windows (SubprocVecEnv spawns processes)
    main()
