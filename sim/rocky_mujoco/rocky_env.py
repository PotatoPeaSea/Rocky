"""Gymnasium environment: Rocky pentapod velocity-command locomotion.

Interface mirrors the real robot:
  * action  (15,) in [-1, 1]: residual servo position targets around the stand pose,
            target = q_stand + action_scale * action, clipped to the joint limits, sent at 50 Hz
            (the ESP32 bus loop). Actuator / servo ID order: leg n -> IDs 3n-2 (hip), 3n-1 (femur), 3n (knee).
  * observation (54,), everything the robot itself can measure:
        [0:3]   gyro (BNO085, body frame, rad/s)
        [3:6]   gravity direction in the body frame (from the IMU quaternion)
        [6:21]  joint positions - stand pose (rad), quantised like the HX-30HM 12-bit encoder
        [21:36] joint velocities (rad/s, finite difference of the reported positions)
        [36:51] last action
        [51:54] command (vx, vy, wz) in the body frame (m/s, m/s, rad/s)
        with estimator="ekf" / "ekf_cam" (rocky_estimator.py), 4 more:
        [54:57] estimated body velocity (m/s, body frame)   [57] estimated height - stand height (m)
  * reward: standard legged-locomotion terms (legged_gym / MuJoCo Playground style), see REWARD_SCALES.
  * termination: dome/chassis touches the ground, or body tilt > 60 deg, or body below 3 cm.

Domain randomisation (all on by default, `domain_rand=False` turns it off):
  friction, payload mass + CoM shift, servo kp, battery voltage (stall torque / speed), joint
  damping & friction, calibration offsets, actuation latency, observation noise, random pushes.
"""
import math
import os
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import rocky_kinematics as K

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER_RES = 2 * math.pi / 4096           # HX-30HM: 12-bit magnetic encoder over 360 deg

REWARD_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "base_height": -50.0,
    "torques": -0.002,
    "dof_vel": -0.001,
    "action_rate": -0.01,
    "feet_air_time": 1.0,
    "collision": -1.0,
    "dof_pos_limits": -5.0,
    "termination": -50.0,
    "alive": 0.2,
}

DEFAULT_DR = {
    "friction": (0.5, 1.25),          # x nominal foot / floor friction
    "added_mass": (-0.15, 0.35),      # kg on the base (PCB, wiring, payload)
    "com_shift": 0.015,               # m, +- base CoM offset in x and y
    "kp_scale": (0.75, 1.25),         # servo P gain
    "voltage": (9.9 / 11.1, 12.6 / 11.1),  # 3S pack: scales stall torque, speed and gain
    "damping_scale": (0.8, 1.2),
    "frictionloss_scale": (0.5, 2.0),
    "calib_offset": math.radians(1.5),  # +- per-servo zero offset (horn spline / calibration)
    "latency_s": (0.0, 0.012),        # bus write + servo response latency
    "push_interval_s": 4.0,           # random velocity kicks
    "push_vel": 0.15,                 # m/s
    "obs_noise": {"gyro": 0.05, "gravity": 0.02, "qpos": 0.0, "qvel": 0.3},
}


class RockyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, terrain="flat", render_mode=None, episode_seconds=20.0, action_scale=0.35,
                 domain_rand=True, dr_params=None, command_ranges=None, reward_scales=None,
                 control_dt=0.02, base_height_target=None, seed=None, residual_around="stand",
                 estimator="none", camera_hz=25.0, camera_latency=0.04):
        """residual_around: "stand" (default) - actions are offsets from the fixed stand pose;
        "gait" - actions are offsets from the open-loop wave gait of gait_demo.py driven by the
        current command (RL only has to learn corrections; observation gets 2 extra phase inputs).
        estimator: "none" (default); "ekf" - the policy also sees the Kalman-filter estimate of body
        velocity and height from a noisy IMU (250 Hz) + the servo encoders; "ekf_cam" - plus the
        OAK-D Lite depth camera (rendered, ~17 ms per frame: slow, for evaluation more than training)."""
        scene = {"flat": "scene.xml", "rough": "scene_rough.xml"}.get(terrain, terrain)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(HERE, scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.control_dt = control_dt
        self.n_sub = int(round(control_dt / m.opt.timestep))
        self.max_steps = int(episode_seconds / control_dt)
        self.action_scale = action_scale
        self.domain_rand = domain_rand
        self.dr = dict(DEFAULT_DR, **(dr_params or {}))
        self.cmd_ranges = command_ranges or {"vx": (-0.12, 0.12), "vy": (-0.12, 0.12), "wz": (-0.5, 0.5)}
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        self.render_mode = render_mode

        self.key_stand = m.key("stand").id
        self.q_stand = m.key_qpos[self.key_stand][7:].copy()
        self.base_z0 = m.key_qpos[self.key_stand][2]
        self.base_height_target = base_height_target or self.base_z0 - 0.001
        self.lo, self.hi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
        self.base_id = m.body("base").id
        self.floor_id = m.geom("floor").id
        self.foot_geoms = np.array([m.geom(f"leg{n}_foot").id for n in range(1, 6)])
        self.body_geoms = {m.geom("c_chassis").id, m.geom("c_dome").id}
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "c_camera") >= 0:
            self.body_geoms.add(m.geom("c_camera").id)
        self.leg_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and g not in self.body_geoms
                          and g not in set(self.foot_geoms) and g != self.floor_id}
        s = lambda name: m.sensor_adr[m.sensor(name).id]
        self.adr_quat, self.adr_gyro, self.adr_acc = s("imu_quat"), s("imu_gyro"), s("imu_acc")
        self.adr_linvel = s("base_linvel")

        # nominal values for domain randomisation
        self.nom = {
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_ipos": m.body_ipos.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "actuator_forcerange": m.actuator_forcerange.copy(),
            "dof_damping": m.dof_damping.copy(),
            "dof_frictionloss": m.dof_frictionloss.copy(),
        }

        self.action_space = spaces.Box(-1.0, 1.0, (15,), np.float32)
        self.residual_around = residual_around
        if residual_around == "gait":
            from gait_demo import WaveGait
            self.gait = WaveGait("wave", period=1.0)
        assert estimator in ("none", "ekf", "ekf_cam"), estimator
        self.estimator = estimator
        self.est = self.imu = self.cam = None
        self.imu_div = int(round(0.004 / m.opt.timestep))           # BNO085 raw reports at 250 Hz
        self.cam_div = max(1, int(round(1.0 / camera_hz / control_dt)))
        self.camera_latency = camera_latency
        n_obs = 54 + (2 if residual_around == "gait" else 0) + (4 if estimator != "none" else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self._renderer = None
        self._viewer = None
        self._init_seed = seed

    # ------------------------------------------------------------------ helpers
    def _apply_domain_rand(self):
        m, r, dr = self.model, self.np_random, self.dr
        for k, v in self.nom.items():
            getattr(m, k)[:] = v
        self.calib = np.zeros(15)
        self.latency_sub = 0
        if not self.domain_rand:
            return
        f = r.uniform(*dr["friction"])
        m.geom_friction[self.foot_geoms, 0] = self.nom["geom_friction"][self.foot_geoms, 0] * f
        m.geom_friction[self.floor_id, 0] = self.nom["geom_friction"][self.floor_id, 0] * f
        m.body_mass[self.base_id] = self.nom["body_mass"][self.base_id] + r.uniform(*dr["added_mass"])
        m.body_ipos[self.base_id, :2] = self.nom["body_ipos"][self.base_id, :2] + r.uniform(-1, 1, 2) * dr["com_shift"]
        volt = r.uniform(*dr["voltage"])
        kp = self.nom["actuator_gainprm"][:, 0] * r.uniform(*dr["kp_scale"], 15) * volt
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_forcerange[:] = self.nom["actuator_forcerange"] * volt
        m.dof_damping[6:] = self.nom["dof_damping"][6:] * r.uniform(*dr["damping_scale"], 15)
        m.dof_frictionloss[6:] = self.nom["dof_frictionloss"][6:] * r.uniform(*dr["frictionloss_scale"], 15)
        self.calib = r.uniform(-1, 1, 15) * dr["calib_offset"]
        self.latency_sub = int(round(r.uniform(*dr["latency_s"]) / m.opt.timestep))

    def _sample_command(self):
        r = self.np_random
        if r.random() < 0.1:
            return np.zeros(3)
        c = np.array([r.uniform(*self.cmd_ranges["vx"]), r.uniform(*self.cmd_ranges["vy"]), r.uniform(*self.cmd_ranges["wz"])])
        if r.random() < 0.25:
            c[:2] = 0.0                      # turn in place
        return c

    def _base_rot(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def _measured_q(self):
        q = self.data.qpos[7:] + self.calib
        q = np.round(q / ENCODER_RES) * ENCODER_RES
        return q

    def _get_obs(self):
        d, r = self.data, self.np_random
        quat = d.sensordata[self.adr_quat:self.adr_quat + 4]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        grav = R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[self.adr_gyro:self.adr_gyro + 3].copy()
        q = self._measured_q()
        dq = (q - self._last_q_meas) / self.control_dt
        self._last_q_meas = q
        if self.domain_rand:
            n = self.dr["obs_noise"]
            gyro += r.normal(0, n["gyro"], 3)
            grav += r.normal(0, n["gravity"], 3)
            dq += r.normal(0, n["qvel"], 15)
        obs = np.concatenate([gyro, grav, q - self.q_stand, dq, self.last_action, self.command])
        if self.residual_around == "gait":
            ph = 2 * math.pi * (self.step_count * self.control_dt / self.gait.period)
            obs = np.concatenate([obs, [math.sin(ph), math.cos(ph)]])
        if self.est is not None:
            obs = np.concatenate([obs, self.est.v_body(), [self.est.height() - self.base_z0]])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ state estimator
    def _reset_estimator(self):
        import rocky_sensors as S
        from rocky_estimator import Estimator, quat_exp, quat_mul
        m, d = self.model, self.data
        if self.imu is None:
            self.imu = S.ImuModel(m, self.np_random)
            t_bc = np.zeros(3)
            if self.estimator == "ekf_cam":
                from rocky_vision import GroundPlane, VisualOdometry
                self.cam = S.DepthCamera(m, 160, 120, rng=self.np_random)
                R_bc, t_bc = self.cam.extrinsics(d)
                self.gp = GroundPlane(self.cam.intrinsics(), R_bc, t_bc)
                self.vo = VisualOdometry(self.cam.intrinsics(), R_bc, t_bc)
            self.est = Estimator(r_cam=t_bc)
        self.imu.rng = self.np_random
        self.imu.reset()
        # boot: the robot starts still; attitude from the BNO085 to ~1 deg, heading taken as known
        self.est.reset(p=(d.qpos[0], d.qpos[1], self.base_z0))
        err = self.np_random.normal(0, math.radians(1.0), 3) * (1.0 if self.domain_rand else 0.0)
        self.est.q = quat_mul(d.qpos[3:7].copy(), quat_exp(err))
        if self.cam is not None:
            self.cam.rng = self.np_random
            self.vo.prev = None
            self.cam_queue = deque()

    def _estimator_camera(self):
        d = self.data
        if self.step_count % self.cam_div == 0:
            depth, gray = self.cam.render(d)
            self.cam_queue.append((d.time + self.camera_latency, self.gp.fit(depth), self.vo.update(d.time, gray, depth)))
        while self.cam_queue and self.cam_queue[0][0] <= d.time + 1e-9:
            _, plane, vo = self.cam_queue.popleft()
            self.est.update_ground(plane)
            self.est.update_vo(vo)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._init_seed)
        self._init_seed = None
        m, d = self.model, self.data
        self._apply_domain_rand()
        mujoco.mj_resetDataKeyframe(m, d, self.key_stand)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[7:] += self.np_random.uniform(-0.05, 0.05, 15)
        d.ctrl[:] = self.q_stand
        mujoco.mj_forward(m, d)
        if self.estimator != "none":
            self._reset_estimator()
        self.command = self._sample_command() if not (options and "command" in options) else np.asarray(options["command"], float)
        self.last_action = np.zeros(15)
        self.last_last_action = np.zeros(15)
        self.pending = deque([self.q_stand.copy()] * (self.latency_sub + 1), maxlen=self.latency_sub + 1)
        self.feet_air_time = np.zeros(5)
        self.last_contact = np.ones(5, bool)
        self.step_count = 0
        self._last_q_meas = self._measured_q()
        self._last_dq = d.qvel[6:].copy()
        self.next_push = self.dr["push_interval_s"] * self.np_random.uniform(0.5, 1.5)
        return self._get_obs(), {"command": self.command.copy()}

    def _contacts(self):
        d = self.data
        foot = np.zeros(5, bool)
        body_hit = False
        leg_hits = 0
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            other = g2 if g1 == self.floor_id else g1 if g2 == self.floor_id else None
            if other is None:
                if g1 in self.leg_geoms and g2 in self.leg_geoms:
                    leg_hits += 1                 # leg-leg self collision
                continue
            if other in self.body_geoms:
                body_hit = True
            elif other in self.leg_geoms:
                leg_hits += 1
            else:
                idx = np.nonzero(self.foot_geoms == other)[0]
                if len(idx):
                    foot[idx[0]] = True
        return foot, body_hit, leg_hits

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.residual_around == "gait":
            ref = self.gait.joints(self.step_count * self.control_dt, self.command)
        else:
            ref = self.q_stand
        target = np.clip(ref + self.action_scale * action, self.lo, self.hi)
        # latency: the target takes effect `latency_sub` physics steps late (calibration offset
        # moves where the servo thinks zero is)
        tgt = target - self.calib
        if self.latency_sub == 0 and self.est is None:
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d, self.n_sub)
        else:
            for i in range(self.n_sub):
                if self.latency_sub:
                    self.pending.append(tgt)
                    d.ctrl[:] = self.pending[0]
                else:
                    d.ctrl[:] = tgt
                mujoco.mj_step(m, d)
                if self.est is not None and (i + 1) % self.imu_div == 0:
                    g, acc = self.imu.sample(d)
                    self.est.predict(g, acc, self.imu_div * m.opt.timestep)
        self.step_count += 1
        if self.est is not None:
            self.est.update_legs(self._measured_q(), self.control_dt)
            if self.cam is not None:
                self._estimator_camera()

        if self.domain_rand and d.time > self.next_push:
            d.qvel[:2] += self.np_random.uniform(-1, 1, 2) * self.dr["push_vel"]
            self.next_push = d.time + self.dr["push_interval_s"] * self.np_random.uniform(0.5, 1.5)
        if self.step_count % int(10.0 / self.control_dt) == 0:
            self.command = self._sample_command()

        R = self._base_rot()
        v_body = R.T @ d.sensordata[self.adr_linvel:self.adr_linvel + 3]
        w_body = d.qvel[3:6].copy()     # free-joint angular velocity is already in the body frame
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        foot, body_hit, leg_hits = self._contacts()
        qd = d.qvel[6:]
        q = d.qpos[7:]

        # ---- reward terms ----
        dt = self.control_dt
        first_contact = foot & ~self.last_contact
        self.feet_air_time += dt
        # legged_gym-style air time, clipped below so contact chatter during exploration costs little
        air_rew = np.sum(np.clip(self.feet_air_time - 0.2, -0.05, 0.3) * first_contact) * (np.linalg.norm(self.command) > 0.02)
        self.feet_air_time[foot] = 0.0
        self.last_contact = foot
        lin_err = np.sum((self.command[:2] - v_body[:2]) ** 2)
        ang_err = (self.command[2] - w_body[2]) ** 2
        out_lo = np.clip(self.lo + 0.05 - q, 0, None)
        out_hi = np.clip(q - (self.hi - 0.05), 0, None)
        terms = {
            "tracking_lin_vel": math.exp(-lin_err / 0.01),
            "tracking_ang_vel": math.exp(-ang_err / 0.1),
            "lin_vel_z": v_body[2] ** 2,
            "ang_vel_xy": float(np.sum(w_body[:2] ** 2)),
            "orientation": float(np.sum(grav[:2] ** 2)),
            "base_height": (d.qpos[2] - self.base_height_target) ** 2,
            "torques": float(np.sum(d.actuator_force ** 2)),
            "dof_vel": float(np.sum(qd ** 2)),
            "action_rate": float(np.sum((action - self.last_action) ** 2)),
            "feet_air_time": float(air_rew),
            "collision": float(leg_hits),
            "dof_pos_limits": float(np.sum(out_lo + out_hi)),
            "alive": 1.0,
        }
        tilt_cos = -grav[2]
        terminated = bool(body_hit or tilt_cos < 0.5 or d.qpos[2] < 0.03 or not np.all(np.isfinite(d.qpos)))
        terms["termination"] = float(terminated)
        reward = sum(self.scales[k] * v for k, v in terms.items()) * dt
        truncated = self.step_count >= self.max_steps
        self.last_last_action = self.last_action
        self.last_action = action
        obs = self._get_obs()
        info = {"terms": terms, "command": self.command.copy(), "v_body": v_body, "w_body": w_body,
                "foot_contact": foot, "imu_acc": d.sensordata[self.adr_acc:self.adr_acc + 3].copy()}
        if self.est is not None:
            info["est_v_body"] = self.est.v_body()
            info["est_height"] = self.est.height()
            info["true_height"] = float(d.qpos[2])
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 360, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._cam.trackbodyid = self.base_id
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.85, 150, -22
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._viewer.cam.trackbodyid = self.base_id
                self._viewer.cam.distance = 0.9
            self._viewer.sync()

    def close(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="Rocky-v0", entry_point=RockyEnv)
