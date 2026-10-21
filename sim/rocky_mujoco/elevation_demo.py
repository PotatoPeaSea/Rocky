"""Check the robot-side height map (rocky_vision.ElevationMap) against the true terrain, on the CPU.

The robot is moved kinematically (stand pose, small body bob and sway) straight out of a stair pit
and across rocks and blocks on the terrain-curriculum tiles, camera side first (body -y), at 10 cm/s.
Every 0.1 s the simulated OAK-D renders a noisy depth frame and the map fuses it using a pose with a
slowly drifting error (the estimator's). The 13 x 13 scan the policy would get is compared with the
true one (the --scan true teacher's input).

    python elevation_demo.py               # prints the table, writes media/elevation_map.png
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import rocky_terrain
from rocky_sensors import DepthCamera
from rocky_vision import ElevationMap

HERE = os.path.dirname(os.path.abspath(__file__))
N, SPACING = 13, 0.06                      # rocky_mjx_env scan_grid
TILES = [("stairs_dn", 3), ("stairs_dn", 6), ("stairs_dn", 9), ("rough", 9), ("blocks", 6)]


def quat_yaw(yaw, roll=0.0, pitch=0.0):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--drift", type=float, default=0.005,
                    help="horizontal pose error random walk, m per sqrt(s) (estimate_demo.py: ~2 cm in 20 s with VO)")
    ap.add_argument("--z-err", type=float, default=0.0015, help="height error, m (white: leg odometry pins it)")
    ap.add_argument("--png", default=os.path.join(HERE, "media", "elevation_map.png"))
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()

    T = rocky_terrain.generate()
    m = rocky_terrain.build_render_model(T, TILES)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    base_z = float(d.qpos[2])
    cam = DepthCamera(m, rng=np.random.default_rng(1))
    mujoco.mj_forward(m, d)
    R_bc, t_bc = cam.extrinsics(d)
    g = (np.arange(N) - (N - 1) / 2) * SPACING
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    rng = np.random.default_rng(0)

    rows, last = [], None
    for kind, lv in TILES:
        emap = ElevationMap(cam.intrinsics(), R_bc, t_bc)
        c = T.centers[lv, T.types.index(kind)]
        yaw = rng.uniform(-math.pi, math.pi)
        fwd = np.array([math.sin(yaw), -math.cos(yaw)])              # body -y in the world: camera side
        err_p = np.zeros(3)
        errs, cover, cover_front = [], [], []
        for k in range(int(a.seconds / 0.1)):
            t = k * 0.1
            xy = c[:2] + fwd * 0.10 * t
            gz = float(np.max(T.height_at(xy[0] + np.array([-.08, 0, .08, 0, 0]),
                                          xy[1] + np.array([0, -.08, 0, .08, 0]))))
            d.qpos[0:2] = xy
            d.qpos[2] = gz + base_z + 0.004 * math.sin(2 * math.pi * 1.5 * t)
            d.qpos[3:7] = quat_yaw(yaw + 0.03 * math.sin(0.7 * t), 0.02 * math.sin(3.1 * t), 0.02 * math.cos(2.3 * t))
            mujoco.mj_forward(m, d)
            depth, _ = cam.render(d)
            err_p[:2] += rng.normal(0, a.drift * math.sqrt(0.1), 2)
            err_p[2] = rng.normal(0, a.z_err)
            p_est = d.qpos[0:3] + err_p
            R_wb = d.xmat[1].reshape(3, 3)
            emap.insert(depth, p_est, R_wb)
            meas, known = emap.scan(p_est, R_wb, grid)
            yaw_now = math.atan2(R_wb[1, 0], R_wb[0, 0])
            cs, sn = math.cos(yaw_now), math.sin(yaw_now)
            pts = d.qpos[0:2] + grid @ np.array([[cs, sn], [-sn, cs]])
            true = np.clip(d.qpos[2] - T.height_at(pts[:, 0], pts[:, 1]), -0.5, 0.5)
            kb = known > 0
            front = grid[:, 1] < 0                                 # camera half of the grid (body -y)
            if kb.any():
                errs.append(np.abs(meas - true)[kb])
            cover.append(kb.mean())
            cover_front.append(kb[front].mean())
            last = (true.reshape(N, N), np.where(kb, meas, np.nan).reshape(N, N), kind, lv)
        e = np.concatenate(errs)
        rows.append((kind, lv, np.mean(cover[10:]), np.mean(cover_front[10:]), 1000 * e.mean(),
                     1000 * np.percentile(e, 95), 100 * np.mean(e > 0.02), 1000 * np.linalg.norm(err_p[:2])))
        if kind == "stairs_dn" and lv == 6:
            shown = last
    cam.close()

    print(f"{'tile':12s} {'lvl':>3s} {'cover':>6s} {'front':>6s} {'mean mm':>8s} {'p95 mm':>7s} {'>2cm %':>7s} {'xy err mm':>9s}")
    for r in rows:
        print(f"{r[0]:12s} {r[1]:3d} {r[2]:6.2f} {r[3]:6.2f} {r[4]:8.1f} {r[5]:7.1f} {r[6]:7.1f} {r[7]:9.1f}")

    if a.no_png:
        return
    true, meas, kind, lv = shown
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    vmin, vmax = np.nanmin(true), np.nanmax(true)
    for a_, im, title in [(ax[0], true, "true scan (teacher input)"), (ax[1], meas, "camera map (student input)"),
                          (ax[2], meas - true, "map - true")]:
        h = a_.imshow(im * 100, origin="lower", cmap="RdBu_r" if a_ is ax[2] else "viridis",
                      vmin=-3 if a_ is ax[2] else vmin * 100, vmax=3 if a_ is ax[2] else vmax * 100,
                      extent=[g[0] - .03, g[-1] + .03, g[0] - .03, g[-1] + .03])
        a_.set_title(title, fontsize=9)
        a_.set_xlabel("body x (m)")
        plt.colorbar(h, ax=a_, label="cm" if a_ is ax[2] else "base height above ground (cm)", fraction=0.046)
    ax[0].set_ylabel("body y (m); camera looks to -y")
    fig.suptitle(f"{kind} level {lv}, after {a.seconds:.0f} s walking camera-first (grey = not seen yet)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.png, dpi=110)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
