"""State-estimation benchmark: walk the open-loop gait with realistic sensors and compare three
estimators fed from the same run against ground truth.

    python estimate_demo.py                      # 30 s on flat ground, writes media/estimator_flat.png
    python estimate_demo.py --terrain rough
    python estimate_demo.py --seconds 10 --no-plot

  imu        IMU strapdown only (dead reckoning)
  imu+legs   + servo encoders: leg odometry and stance-foot height
  full       + OAK-D Lite: ground plane (tilt, height) and visual odometry (velocity)

Sensors: BNO085 gyro / accel at 250 Hz with noise and a random constant bias; 15 encoders at 50 Hz,
12-bit quantised with +-1.5 deg calibration offsets; depth camera at 25 Hz (160 x 120, stereo noise,
holes), delivered `--cam-latency` s late.
"""
import argparse
import math
import os
import time
from collections import deque

import mujoco
import numpy as np

import rocky_kinematics as K
import rocky_sensors as S
from gait_demo import WaveGait
from rocky_estimator import Estimator
from rocky_vision import GroundPlane, VisualOdometry

HERE = os.path.dirname(os.path.abspath(__file__))
ENC = 2 * math.pi / 4096


def command(t):
    """A schedule that exercises every axis: forward, sideways, turn, diagonal, stop."""
    seq = [(4, (0.0, 0.0, 0.0)), (6, (0.0, -0.08, 0.0)), (6, (0.08, 0.0, 0.0)), (5, (0.0, 0.0, 0.4)),
           (6, (0.06, -0.06, 0.2)), (3, (0.0, 0.0, 0.0))]
    for dur, c in seq:
        if t < dur:
            return np.array(c)
        t -= dur
    return np.zeros(3)


def rpy(R):
    return np.array([math.atan2(R[2, 1], R[2, 2]), -math.asin(np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0])])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(a):
    rng = np.random.default_rng(a.seed)
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene_rough.xml" if a.terrain == "rough" else "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    yaw0 = rng.uniform(-math.pi, math.pi)
    d.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
    mujoco.mj_forward(m, d)
    base = m.body("base").id
    dt_ctrl, dt_imu = 0.02, 0.004
    nsub, imu_div = int(round(dt_ctrl / m.opt.timestep)), int(round(dt_imu / m.opt.timestep))
    cam_div = int(round(1.0 / a.cam_hz / dt_ctrl))

    imu = S.ImuModel(m, rng)
    calib = rng.uniform(-1, 1, 15) * math.radians(1.5)
    cam = S.DepthCamera(m, 160, 120, rng=rng)
    R_bc, t_bc = cam.extrinsics(d)
    gp = GroundPlane(cam.intrinsics(), R_bc, t_bc)
    vo = VisualOdometry(cam.intrinsics(), R_bc, t_bc)
    gait = WaveGait("wave", 1.0)

    names = ["imu", "imu+legs", "full"]
    ests = {n: Estimator(r_cam=t_bc) for n in names}
    accs = []
    for _ in range(25):                                   # 0.1 s standing: average the accelerometer
        mujoco.mj_step(m, d, imu_div)
        accs.append(imu.sample(d)[1])
    acc0 = np.mean(accs, axis=0)
    for e in ests.values():
        e.init_from_accel(acc0, yaw=yaw0, p=(d.qpos[0], d.qpos[1], 0.09))   # start heading known

    cam_queue = deque()
    log = {k: [] for k in ["t", "v", "rpy", "z", "pos"] + [f"{n}_{q}" for n in names for q in ("v", "rpy", "z", "pos")]}
    stats = {"vo_frames": 0, "vo_ok": 0, "gp_ok": 0, "cam_ms": [], "est_ms": []}
    t_wall = time.time()
    for k in range(int(a.seconds / dt_ctrl)):
        t = d.time
        c = command(t)
        d.ctrl[:] = gait.joints(t, c) - calib if t > 0.5 else K.stand_pose() - calib
        for s in range(nsub):
            mujoco.mj_step(m, d)
            if (s + 1) % imu_div == 0:
                g, ac = imu.sample(d)
                t0 = time.perf_counter()
                for e in ests.values():
                    e.predict(g, ac, dt_imu)
                stats["est_ms"].append((time.perf_counter() - t0) * 1e3 / 3)
        q_meas = np.round((d.qpos[7:] + calib) / ENC) * ENC        # what the servos report
        for n in ("imu+legs", "full"):
            ests[n].update_legs(q_meas, dt_ctrl)
        if k % cam_div == 0:
            t0 = time.perf_counter()
            depth, gray = cam.render(d)
            plane = gp.fit(depth)
            v = vo.update(d.time, gray, depth)
            stats["cam_ms"].append((time.perf_counter() - t0) * 1e3)
            cam_queue.append((d.time + a.cam_latency, plane, v))
            stats["vo_frames"] += 1
        while cam_queue and cam_queue[0][0] <= d.time + 1e-9:
            _, plane, v = cam_queue.popleft()
            stats["gp_ok"] += ests["full"].update_ground(plane)
            stats["vo_ok"] += ests["full"].update_vo(v)

        R = d.xmat[base].reshape(3, 3)
        log["t"].append(d.time)
        log["v"].append(R.T @ d.qvel[0:3])
        log["rpy"].append(rpy(R))
        log["z"].append(d.qpos[2])
        log["pos"].append(d.qpos[0:3].copy())
        for n, e in ests.items():
            log[f"{n}_v"].append(e.v_body())
            log[f"{n}_rpy"].append(e.rpy())
            log[f"{n}_z"].append(e.height())
            log[f"{n}_pos"].append(e.p.copy())
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        if tilt > 45:
            print("robot fell at t=%.1f s" % d.time)
            break
    wall = time.time() - t_wall
    L = {k: np.array(v) for k, v in log.items()}

    print(f"\n{a.terrain} ground, {L['t'][-1]:.0f} s sim in {wall:.0f} s wall | camera {a.cam_hz:.0f} Hz, "
          f"latency {a.cam_latency*1e3:.0f} ms | VO accepted {stats['vo_ok']}/{stats['vo_frames']}, "
          f"ground plane {stats['gp_ok']}/{stats['vo_frames']}")
    print(f"per-step cost: EKF predict {np.mean(stats['est_ms']):.3f} ms, camera render + vision {np.mean(stats['cam_ms']):.1f} ms")
    sel = L["t"] > 1.0
    hdr = f"{'estimator':10s} {'v_xy RMSE':>10s} {'v_z RMSE':>9s} {'roll/pitch':>11s} {'height':>9s} {'xy drift':>9s} {'yaw drift':>10s}"
    print(hdr + "\n" + "-" * len(hdr))
    res = {}
    for n in names:
        ev = L[f"{n}_v"][sel] - L["v"][sel]
        er = L[f"{n}_rpy"][sel] - L["rpy"][sel]
        ez = L[f"{n}_z"][sel] - L["z"][sel]
        dxy = np.linalg.norm(L[f"{n}_pos"][-1, :2] - L["pos"][-1, :2])
        dyaw = abs(wrap(L[f"{n}_rpy"][-1, 2] - L["rpy"][-1, 2]))
        r = {"v_xy": math.sqrt(np.mean(np.sum(ev[:, :2] ** 2, 1))), "v_z": math.sqrt(np.mean(ev[:, 2] ** 2)),
             "tilt_deg": math.degrees(math.sqrt(np.mean(np.sum(er[:, :2] ** 2, 1)))),
             "z_mm": 1e3 * math.sqrt(np.mean(ez ** 2)), "xy_drift": dxy, "yaw_deg": math.degrees(dyaw)}
        res[n] = r
        def fmt(x, w, p):
            return f"{x:{w}.{p}f}" if abs(x) < 1e4 else f"{x:{w}.1e}"
        print(f"{n:10s} {fmt(r['v_xy'], 7, 3)} m/s {fmt(r['v_z'], 5, 3)} m/s {fmt(r['tilt_deg'], 7, 2)} deg "
              f"{fmt(r['z_mm'], 6, 1)} mm {fmt(r['xy_drift'], 7, 2)} m {fmt(r['yaw_deg'], 7, 1)} deg")
    dist = np.sum(np.linalg.norm(np.diff(L["pos"][:, :2], axis=0), axis=1))
    print(f"(distance walked {dist:.2f} m; truth speed RMS {math.sqrt(np.mean(np.sum(L['v'][sel, :2]**2, 1))):.3f} m/s)")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        t = L["t"]
        cols = {"imu": "#c0392b", "imu+legs": "#2874a6", "full": "#1e8449"}
        for i, lab in enumerate(["body vx (m/s)", "body vy (m/s)"]):
            ax[i].plot(t, L["v"][:, i], "k", lw=2, label="truth")
            for n in names[1:]:
                ax[i].plot(t, L[f"{n}_v"][:, i], color=cols[n], lw=1, label=n)
            ax[i].set_ylabel(lab)
            ax[i].set_ylim(-0.2, 0.2)
        ax[2].plot(t, np.degrees(L["rpy"][:, 0]), "k", lw=2, label="truth roll")
        ax[2].plot(t, np.degrees(L["rpy"][:, 1]), "k--", lw=2, label="truth pitch")
        for n in names[1:]:
            ax[2].plot(t, np.degrees(L[f"{n}_rpy"][:, 0]), color=cols[n], lw=1)
            ax[2].plot(t, np.degrees(L[f"{n}_rpy"][:, 1]), "--", color=cols[n], lw=1)
        ax[2].set_ylabel("roll / pitch (deg)")
        ax[3].plot(t, L["z"] * 1e3, "k", lw=2, label="truth")
        for n in names[1:]:
            ax[3].plot(t, L[f"{n}_z"] * 1e3, color=cols[n], lw=1, label=n)
        ax[3].set_ylabel("height (mm)")
        ax[3].set_xlabel("time (s)")
        ax[0].legend(loc="upper right", ncol=3)
        ax[0].set_title(f"Rocky state estimation, {a.terrain} ground (IMU-only diverges: off scale)")
        fig.tight_layout()
        out = os.path.join(HERE, "media", f"estimator_{a.terrain}.png")
        fig.savefig(out, dpi=110)
        print("wrote", out)
    cam.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="flat", choices=["flat", "rough"])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--cam-hz", type=float, default=25.0)
    ap.add_argument("--cam-latency", type=float, default=0.04, help="s from capture to measurement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    run(ap.parse_args())
