"""Side-by-side video: several policies climbing out of the same stair pit, seen from the side.

    python compare_stairs.py runs/mjx_terrain2 runs/mjx_teacher runs/mjx_student \
        --labels blind teacher student --level 3 --out media/training/compare_stairs_L3.mp4

Each policy runs 8 rollouts in one batch on the same stairs_dn tile ("stairs, climbing out"), spawned
at the pit's centre with 8 evenly spaced headings and commanded 10 cm/s camera-first (body -y) for 15 s.
The video shows rollout 0 (heading 0, walking towards world -y) of every policy, side by side, from a
fixed camera beside the stairs. The caption gives how many of the 8 got >= 1.05 m out (the eval's
success rule) and how high rollout 0 climbed.
"""
import argparse
import math
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import rocky_terrain
from eval_mjx import load

HERE = os.path.dirname(os.path.abspath(__file__))
N = 8


def run_policy(run, level, seconds):
    env, policy, _, _ = load(os.path.join(HERE, run), "curriculum", n_envs=N)
    steps = int(seconds / env.dt)
    tt = env._terrain.types.index("stairs_dn")

    def one(key, yaw):
        k1, k2, k3 = jax.random.split(key, 3)
        st = env.reset(k1)
        data, spawn = env._spawn(st.data, jp.int32(level), jp.int32(tt), k2)
        qpos = data.qpos.at[0:2].set(env._centers[level, tt, 0:2])
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        data = data.replace(qpos=qpos)
        st.info.update(level=jp.int32(level), ttype=jp.int32(tt), spawn=qpos[0:2], ep_step=jp.int32(0),
                       respawn=jp.array(False))
        if "cam_p" in st.info:
            st.info.update(env._fresh_map(data, k3))
        st = st.replace(data=data)

        def body(carry, k):
            st, alive = carry
            st.info["command"] = jp.array([0.0, -0.10, 0.0])
            a, _ = policy(st.obs, k)
            st = env.step(st, a)
            alive = alive & (st.done < 0.5)
            return (st, alive), st.data.qpos

        (st, alive), qs = jax.lax.scan(body, (st, jp.array(True)), jax.random.split(k3, steps))
        return qs, alive

    yaws = jp.arange(N) * (2 * math.pi / N)
    qs, alive = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(0), N), yaws)
    qs, alive = np.asarray(qs), np.asarray(alive)
    dist = np.linalg.norm(qs[:, -1, 0:2] - qs[:, 0, 0:2], axis=1)
    ok = int(np.sum(alive & (dist >= 1.05)))
    rise = float(qs[0, -1, 2] - qs[0, 0, 2])
    return env, qs[0], ok, rise, float(dist[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(HERE, "media", "training", "compare_stairs.mp4"))
    ap.add_argument("--rerender", action="store_true", help="reuse the rollouts saved next to --out (<out>.npz)")
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(r) for r in a.runs]
    cache = os.path.splitext(a.out)[0] + ".npz"

    if a.rerender:
        z = np.load(cache)
        results = [(str(lab), z[f"q{i}"], int(z["ok"][i]), float(z["rise"][i]), float(z["dist"][i]))
                   for i, lab in enumerate(z["labels"])]
        dt = float(z["dt"])
    else:
        results = []
        for run, lab in zip(a.runs, labels):
            env, q, ok, rise, dist = run_policy(run, a.level, a.seconds)
            print(f"{lab:10s} {ok}/{N} out of the pit; rollout 0: {dist:.2f} m, climbed {rise * 100:.1f} cm", flush=True)
            results.append((lab, q, ok, rise, dist))
        dt = env.dt
        np.savez(cache, labels=np.array([r[0] for r in results]), ok=[r[2] for r in results],
                 rise=[r[3] for r in results], dist=[r[4] for r in results], dt=dt,
                 **{f"q{i}": r[1] for i, r in enumerate(results)})

    T = rocky_terrain.generate()
    tt = T.types.index("stairs_dn")
    m = rocky_terrain.build_render_model(T, [("stairs_dn", a.level)])
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    c = T.centers[a.level, tt]
    cam.lookat[:] = [c[0], c[1] - 0.75, c[2] + 0.06]          # middle of the path out (world -y)
    cam.distance, cam.azimuth, cam.elevation = 1.9, 0.0, -8.0     # nearly side-on: risers show in profile
    opt = mujoco.MjvOption()
    from PIL import Image, ImageDraw
    import mediapy
    riser = 0.01 + (rocky_terrain.MAX_STEP - 0.01) * a.level / (T.levels - 1)    # rocky_terrain._tile
    frames = []
    for k in range(0, len(results[0][1]), 2):
        row = []
        for lab, q, ok, rise, dist in results:
            d.qpos[:] = q[min(k, len(q) - 1)]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam, scene_option=opt)
            im = Image.fromarray(r.render())
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 480, 40], fill=(0, 0, 0))
            dr.text((8, 5), f"{lab}: {ok}/{N} headings got out", fill=(255, 255, 255))
            dr.text((8, 22), f"this one: {dist:.2f} m, climbed {rise * 100:.1f} cm", fill=(200, 200, 200))
            row.append(np.asarray(im))
        frame = np.concatenate(row, 1)
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text((8, 342), f"climbing out of a {riser * 100:.1f} cm stair pit (level {a.level}), "
                                          f"10 cm/s camera-first   t = {k * dt:4.1f} s", fill=(255, 255, 255))
        frames.append(np.asarray(im))
    r.close()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mediapy.write_video(a.out, frames, fps=25)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
