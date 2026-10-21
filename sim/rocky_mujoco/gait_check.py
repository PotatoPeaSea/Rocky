"""Gait quality of a GPU policy: numbers, a foot-contact diagram and a close-up video.

    python gait_check.py runs/mjx_gait1                 # flat ground
    python gait_check.py runs/mjx_teacher2 --tile stairs_dn 3

Runs the deterministic policy (no domain randomisation) for 10 s per command, one env per command, all
in one GPU batch, then works out from the joint and foot positions:
  speed      achieved ground speed vs the command
  stride     distance per step of one leg; step rate per leg (Hz) and how evenly the legs share the work
  lift       median and 90th-percentile height of a swinging foot above the ground under it
  jitter     joint direction reversals per joint per second (a smooth 1.5 Hz gait has ~3)
  slip       mean horizontal speed of feet that are on the ground
  tilt       roll / pitch standard deviation
Writes <run>/gait/gait.json, gait_contacts.png (which foot is down when) and gait.mp4 (side close-up
that follows the robot, all commands one after another).
"""
import argparse
import json
import math
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np

from eval_mjx import load

HERE = os.path.dirname(os.path.abspath(__file__))
CMDS = [("forward 10 cm/s", (0.0, 0.10, 0.0)), ("forward 20 cm/s", (0.0, 0.20, 0.0)),
        ("camera-first 15 cm/s", (0.0, -0.15, 0.0)), ("sideways 15 cm/s", (0.15, 0.0, 0.0)),
        ("turn 0.6 rad/s", (0.0, 0.0, 0.6)), ("stand", (0.0, 0.0, 0.0))]
AIR = 0.004          # a foot more than 4 mm above the ground counts as in the air


def rollouts(env, policy, cmds, steps, tile):
    n = len(cmds)

    def one(key, cmd):
        k1, k2 = jax.random.split(key)
        st = env.reset(k1)
        yaw = 0.0
        if tile is not None:
            data, _ = env._spawn(st.data, jp.int32(tile[1]), jp.int32(tile[0]), k2)
            st.info.update(level=jp.int32(tile[1]), ttype=jp.int32(tile[0]), spawn=data.qpos[0:2],
                           ep_step=jp.int32(0), respawn=jp.array(False))
            st = st.replace(data=data)
        qpos = st.data.qpos.at[3:7].set(jp.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
        st = st.replace(data=st.data.replace(qpos=qpos))
        if "cam_p" in st.info:
            st.info.update(env._fresh_map(st.data, k2))

        def body(st, k):
            st.info["command"] = cmd
            a, _ = policy(st.obs, k)
            st = env.step(st, a)
            return st, (st.data.qpos, st.done)

        _, (qs, dn) = jax.lax.scan(body, st, jax.random.split(k2, steps))
        return qs, dn

    f = jax.jit(jax.vmap(one))
    qs, dn = f(jax.random.split(jax.random.PRNGKey(0), n), jp.array(cmds, jp.float32))
    return np.asarray(qs), np.asarray(dn)


def analyse(m, q, dt, ground, cmd):
    import mujoco
    d = mujoco.MjData(m)
    feet = [m.geom(f"leg{n}_foot").id for n in range(1, 6)]
    r = m.geom_size[feet[0]][0]
    fp, rp, yaw = [], [], []
    for k in range(len(q)):
        d.qpos[:] = q[k]
        mujoco.mj_kinematics(m, d)
        fp.append(d.geom_xpos[feet].copy())
        R = d.xmat[1].reshape(3, 3)
        rp.append([math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(-math.asin(np.clip(R[2, 0], -1, 1)))])
        yaw.append(math.atan2(R[1, 0], R[0, 0]))
    fp, rp = np.array(fp), np.array(rp)
    clr = fp[..., 2] - r - ground(fp[..., 0], fp[..., 1])
    air = clr > AIR
    k0 = len(q) // 5                                           # skip the first 2 s (start-up)
    T = (len(q) - k0) * dt
    steps = (air[k0 + 1:] & ~air[k0:-1]).sum(0)
    v = np.diff(q[k0:, 0:2], axis=0) / dt
    speed = float(np.linalg.norm(v.mean(0)))
    fv = np.linalg.norm(np.diff(fp[k0:, :, 0:2], axis=0), axis=2) / dt
    stance = ~air[k0 + 1:] & ~air[k0:-1]                     # planted this frame and the last (as the reward)
    dq = np.diff(q[k0:, 7:], axis=0)
    rev = ((np.sign(dq[1:]) != np.sign(dq[:-1])) & (np.abs(dq[1:]) > 1e-3)).sum(0) / T
    rate = steps / T
    sw = clr[k0:][air[k0:]]
    out = {
        "cmd": list(cmd), "speed": speed, "cmd_speed": float(np.hypot(cmd[0], cmd[1])),
        "yaw_rate": float((np.unwrap(yaw)[-1] - np.unwrap(yaw)[k0]) / T),
        "step_rate_hz": rate.round(2).tolist(),
        "stride_cm": 100 * speed / rate.mean() if rate.mean() > 0.05 else 0.0,
        "evenness": float(rate.min() / max(rate.max(), 1e-6)),
        "lift_median_cm": 100 * float(np.median(sw)) if sw.size else 0.0,
        "lift_p90_cm": 100 * float(np.percentile(sw, 90)) if sw.size else 0.0,
        "feet_in_air": float(air[k0:].sum(1).mean()),
        "jitter_rev_per_s": float(rev.mean()),
        "slip_cm_s": 100 * float(fv[stance].mean()) if stance.any() else 0.0,
        "tilt_std_deg": rp[k0:].std(0).round(2).tolist(),
    }
    return out, air


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tile", nargs=2, metavar=("TYPE", "LEVEL"), help="run on a terrain tile instead of flat ground")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--video-seconds", type=float, default=4.0, help="per command in gait.mp4")
    a = ap.parse_args()
    import mujoco
    import rocky_terrain

    run = os.path.join(HERE, a.run) if not os.path.isabs(a.run) else a.run
    terrain = "curriculum" if a.tile else "flat"
    env, policy, _, _ = load(run, terrain, n_envs=len(CMDS))
    tile = None
    if a.tile:
        tile = (env._terrain.types.index(a.tile[0]), int(a.tile[1]))
    steps = int(a.seconds / env.dt)
    qs, dn = rollouts(env, policy, [c for _, c in CMDS], steps, tile)

    if a.tile:
        T = env._terrain
        m = rocky_terrain.build_render_model(T, [(a.tile[0], int(a.tile[1]))])
        ground = lambda x, y: T.height_at(x, y)
    else:
        m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
        ground = lambda x, y: np.zeros_like(x)
    res, airs = {}, []
    print(f"{'command':22s} {'speed':>11s} {'stride':>7s} {'step Hz (legs 1-5)':>26s} {'even':>5s} "
          f"{'lift med/p90':>13s} {'jitter':>7s} {'slip':>6s} {'fell':>5s}")
    for (name, c), q, d in zip(CMDS, qs, dn):
        r, air = analyse(m, q, env.dt, ground, c)
        r["fell"] = bool(d.any())
        res[name] = r
        airs.append(air)
        print(f"{name:22s} {r['speed'] * 100:4.1f}/{r['cmd_speed'] * 100:4.1f} cm/s {r['stride_cm']:5.1f}cm "
              f"{str(r['step_rate_hz']):>26s} {r['evenness']:5.2f} {r['lift_median_cm']:5.1f}/{r['lift_p90_cm']:4.1f} cm "
              f"{r['jitter_rev_per_s']:5.1f}/s {r['slip_cm_s']:4.1f}cm/s {'yes' if r['fell'] else 'no':>5s}")
    outdir = os.path.join(run, "gait" + (f"_{a.tile[0]}{a.tile[1]}" if a.tile else ""))
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, "gait.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(CMDS), 1, figsize=(10, 1.1 * len(CMDS) + 0.6), sharex=True)
    t = np.arange(steps) * env.dt
    w = (t >= 4.0) & (t < 7.0)                                 # 3 s window after start-up
    for ax, (name, _), air in zip(axs, CMDS, airs):
        ax.imshow((~air[w].T).astype(float), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  extent=[t[w][0], t[w][-1], 5.5, 0.5])
        ax.set_yticks(range(1, 6))
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axs[-1].set_xlabel("time (s); black = foot on the ground, rows = legs 1-5")
    fig.suptitle(f"{os.path.basename(run)}: foot contacts" + (f" on {a.tile[0]} level {a.tile[1]}" if a.tile else ""),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "gait_contacts.png"), dpi=110)

    from PIL import Image, ImageDraw
    import mediapy
    rend = mujoco.Renderer(m, 360, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 0.9, -14.0
    d = mujoco.MjData(m)
    frames, nv = [], int(a.video_seconds / env.dt)
    for (name, c), q in zip(CMDS, qs):
        r = res[name]
        heading = math.atan2(c[1], c[0]) if np.hypot(c[0], c[1]) > 0 else math.pi / 2
        cam.azimuth = math.degrees(heading) + 90.0             # side-on to the walking direction
        start = steps - nv
        for k in range(start, steps, 2):
            d.qpos[:] = q[k]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [q[k, 0], q[k, 1], ground(np.array([q[k, 0]]), np.array([q[k, 1]]))[0] + 0.06]
            rend.update_scene(d, camera=cam)
            im = Image.fromarray(rend.render())
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 640, 40], fill=(0, 0, 0))
            dr.text((8, 5), f"{os.path.basename(run)}   {name}   t = {k * env.dt:4.1f} s", fill=(255, 255, 255))
            dr.text((8, 22), f"speed {r['speed'] * 100:.1f} cm/s  stride {r['stride_cm']:.1f} cm  "
                             f"lift {r['lift_median_cm']:.1f} cm  jitter {r['jitter_rev_per_s']:.0f}/s  "
                             f"evenness {r['evenness']:.2f}", fill=(200, 200, 200))
            frames.append(np.asarray(im))
    rend.close()
    mediapy.write_video(os.path.join(outdir, "gait.mp4"), frames, fps=25)
    print("wrote", outdir)


if __name__ == "__main__":
    main()
