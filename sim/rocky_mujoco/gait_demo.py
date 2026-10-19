"""Classical baseline: open-loop wave / ripple gait from leg IK, position targets at 50 Hz.

    python gait_demo.py                         # interactive viewer, walk forward (+y = leg 1 direction)
    python gait_demo.py --vx 0.0 --vy 0.06      # body-frame velocity command (m/s)
    python gait_demo.py --wz 0.4                # turn on the spot (rad/s)
    python gait_demo.py --gait ripple           # 2 legs swing at a time (faster)
    python gait_demo.py --headless --seconds 10 # no window, prints speed / tilt
    python gait_demo.py --headless --record media/gait.mp4

The gait is what you would run on the ESP32/Pi: every 20 ms compute 5 foot targets, run leg IK
(rocky_kinematics.leg_ik) and send 15 servo positions. Nothing here reads the IMU - closing
the loop (body levelling, step-height adaptation) is the obvious next step.
"""
import argparse
import math
import os
import time

import numpy as np

import rocky_kinematics as K

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_DT = 0.02          # 50 Hz, as the ESP32 bus loop

# swing order 0, 3, 1, 4, 2  -> consecutive swings are never neighbours
PHASE_OFFSET = np.array([0.0, 2 / 5, 4 / 5, 1 / 5, 3 / 5])
GAITS = {"wave": 0.8, "ripple": 0.6}   # duty factor (fraction of the cycle a foot is on the ground)


class WaveGait:
    def __init__(self, gait="wave", period=1.0, step_height=0.035, radius=K.STAND_R, foot_z=K.STAND_FOOT_Z):
        self.duty = GAITS[gait]
        self.period = period
        self.h = step_height
        self.p0 = K.stand_feet(radius, foot_z)

    def feet(self, t, cmd):
        """Body-frame foot targets at time t for command (vx, vy, wz)."""
        vx, vy, wz = cmd
        ts = self.duty * self.period
        out = np.empty((5, 3))
        for i in range(5):
            p0 = self.p0[i]
            # displacement of the ground under this foot during one stance phase
            d = np.array([vx - wz * p0[1], vy + wz * p0[0], 0.0]) * ts
            ph = (t / self.period + PHASE_OFFSET[i]) % 1.0
            if ph < self.duty:                       # stance: foot slides back under the body
                s = ph / self.duty
                out[i] = p0 + d * (0.5 - s)
            else:                                    # swing: lift and bring forward
                s = (ph - self.duty) / (1 - self.duty)
                sm = 0.5 - 0.5 * math.cos(math.pi * s)   # smooth start / stop
                out[i] = p0 + d * (-0.5 + sm)
                out[i, 2] += self.h * math.sin(math.pi * s)
        return out

    def joints(self, t, cmd):
        q = K.ik_all(self.feet(t, cmd))
        return np.clip(q, K.LIMITS[:, 0], K.LIMITS[:, 1])


def run(args):
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, args.scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    nsub = int(round(CONTROL_DT / m.opt.timestep))
    gait = WaveGait(args.gait, args.period, args.step_height)
    cmd = (args.vx, args.vy, args.wz)
    q_stand = K.stand_pose()

    def ctrl_at(t):
        ramp = min(1.0, max(0.0, (t - 0.5) / 1.0))          # stand still 0.5 s, ramp in over 1 s
        c = tuple(ramp * x for x in cmd)
        return gait.joints(t, c) if t > 0.5 else q_stand

    log = {"t": [], "pos": [], "tilt": [], "yaw": []}

    def record_state():
        log["t"].append(d.time)
        log["pos"].append(d.qpos[:3].copy())
        up = d.xmat[1].reshape(3, 3)[:, 2]
        log["tilt"].append(math.degrees(math.acos(np.clip(up[2], -1, 1))))
        xm = d.xmat[1].reshape(3, 3)
        log["yaw"].append(math.atan2(xm[1, 0], xm[0, 0]))

    frames = []
    renderer = None
    if args.record:
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(m, 360, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = m.body("base").id
        cam.distance, cam.azimuth, cam.elevation = 0.85, 160, -22

    if args.headless:
        k = 0
        while d.time < args.seconds:
            d.ctrl[:] = ctrl_at(d.time)
            mujoco.mj_step(m, d, nsub)
            record_state()
            if renderer is not None and k % 2 == 0:           # 25 fps video
                renderer.update_scene(d, cam)
                frames.append(renderer.render())
            k += 1
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = m.body("base").id
            v.cam.distance = 0.9
            while v.is_running() and d.time < args.seconds:
                t0 = time.time()
                d.ctrl[:] = ctrl_at(d.time)
                mujoco.mj_step(m, d, nsub)
                record_state()
                v.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))

    pos, t = np.array(log["pos"]), np.array(log["t"])
    sel = t > 2.0
    if sel.sum() > 10:
        dp = pos[sel][-1] - pos[sel][0]
        dt = t[sel][-1] - t[sel][0]
        yaw = np.unwrap(np.array(log["yaw"])[sel])
        print(f"average velocity after t=2 s: vx={dp[0]/dt:+.4f} vy={dp[1]/dt:+.4f} m/s, "
              f"wz={(yaw[-1]-yaw[0])/dt:+.3f} rad/s (cmd {args.vx:+.3f}, {args.vy:+.3f}, {args.wz:+.3f})")
    print(f"max tilt {max(log['tilt']):.2f} deg, base z min/max {pos[:,2].min():.4f}/{pos[:,2].max():.4f} m, "
          f"upright={max(log['tilt']) < 30}")
    if frames:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        imageio.mimsave(args.record, frames, fps=25, quality=6, macro_block_size=8)
        print("wrote", args.record)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene.xml")
    ap.add_argument("--gait", choices=list(GAITS), default="wave")
    ap.add_argument("--vx", type=float, default=0.0, help="m/s, body x")
    ap.add_argument("--vy", type=float, default=0.06, help="m/s, body y (leg 1 points +y)")
    ap.add_argument("--wz", type=float, default=0.0, help="rad/s yaw rate")
    ap.add_argument("--period", type=float, default=1.0, help="gait cycle s")
    ap.add_argument("--step-height", type=float, default=0.035)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", help="write an MP4 (implies --headless)")
    a = ap.parse_args()
    if a.record:
        a.headless = True
    run(a)
