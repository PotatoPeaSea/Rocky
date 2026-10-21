"""MuJoCo Playground / MJX version of the Rocky joystick task (GPU-parallel training).

Same robot (rocky.xml), same 50 Hz position-target interface, same observation layout and reward
terms as rocky_env.py, written in JAX so thousands of copies run in parallel on an NVIDIA GPU
(Linux or WSL2). Train with train_mjx.py.

Differences from rocky_env.py (kept deliberately simple for MJX speed):
  * contacts: only the 5 foot spheres collide with the floor (body hulls / leg boxes are off);
    falls are detected from tilt and base height instead of a dome contact.
  * no encoder quantisation / latency buffer (add them once the basic policy trains).
  * est_obs=True: the policy also gets [body velocity (3), height - stand height (1)] like
    RockyEnv(estimator="ekf"). Instead of running the Kalman filter per env, it is the true value
    plus the error measured for rocky_estimator.py in estimate_demo.py (white noise + a constant
    per-episode bias), so the policy learns to cope with estimator error.
  * terrain="curriculum" (Warp only): rocky_terrain.py heightfield of 4 m tiles (rocks, slopes, stairs,
    blocks x 10 difficulty levels). Every env keeps a level; after each episode it moves up if the robot
    covered >= 70 % of the commanded distance without falling, down if < 35 % or it fell, and respawns on a
    tile of that level. Body and leg collisions are on (capsules / cylinders, robot vs terrain); plate,
    dome or camera on the ground ends the episode, shin / link / belly hits are penalised. The critic
    also sees a 7 x 7 grid of terrain heights around the robot.
  * scan (terrain="curriculum" only): what the POLICY sees of the terrain, a 13 x 13 grid at 6 cm
    (+-0.36 m) around the base, aligned with the body yaw, value = base height above that ground point.
      "none"   blind (step 2)
      "true"   exact heights: the teacher (step 3)
      "camera" what an OAK-D Lite height map would give: a cell has a value only if it was inside the
               camera's view (73 x 58 deg, 0.2 - 3 m) at some point in the last 4 s (camera poses every
               0.1 s), plus noise growing with viewing distance and a per-episode offset (pose drift);
               a mask says which cells are known. The student (step 4); obs also carries
               "teacher_state" (the teacher's input) for distillation.
    With a scan the robot walks camera-first: commands point within +-45 deg of body -y (the camera).
"""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

HERE = os.path.dirname(os.path.abspath(__file__))


def jnp_deg(d):
    return d * jp.pi / 180.0


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,             # MJX: 4 ms, 5 substeps (the CPU env uses 2 ms)
        episode_length=1000,
        action_scale=0.35,
        impl="jax",               # "warp" = MuJoCo Warp (faster on recent NVIDIA GPUs)
        iterations=10,            # constraint solver: MuJoCo's default 100 made terrain steps crawl
        ls_iterations=20,
        naconmax=16 * 1024,
        njmax=64,
        noise=config_dict.create(gyro=0.05, gravity=0.02, qpos=0.003, qvel=0.3),
        est_obs=False,
        # estimate_demo.py, IMU + servos: v_xy RMSE 0.021 m/s, v_z 0.019 m/s, height 1.4 mm (incl. a
        # ~1.5 mm calibration bias)
        est_noise=config_dict.create(vel=0.018, vel_bias=0.008, height=0.0012, height_bias=0.0015),
        reward=config_dict.create(
            tracking_sigma=0.01, tracking_ang_sigma=0.1,
            # second, sharp velocity term: the coarse one above pays 78 % for being 5 cm/s off, so on its
            # own the policy settled for half the commanded speed; the sharp one alone (0.0025) gave no
            # gradient from standing and the policy marched in place (gait2)
            tracking_sigma_fine=0.001,
            scales=config_dict.create(
                tracking_lin_vel=2.0, tracking_ang_vel=1.0, lin_vel_z=-2.0, ang_vel_xy=-0.05,
                orientation=-5.0, base_height=-50.0, torques=-0.002, action_rate=-0.02,
                feet_air_time=0.0, dof_pos_limits=-5.0, termination=-50.0, alive=0.2,
                # gait shaping (gait.enabled): follow the swing schedule, lift swing feet to the target
                # height, no sliding stance feet, no high-frequency jitter (2nd difference of the action)
                tracking_lin_vel_fine=2.0,
                gait_contact=1.5, feet_clearance=-500.0, foot_slip=-10.0, action_acc=-0.02, dof_acc=-2.5e-6,
            ),
        ),
        command=config_dict.create(vx=0.25, vy=0.25, wz=0.8),
        # gait clock: legs swing in the order 1, 3, 5, 2, 4 (two non-adjacent legs at a time, three on the
        # ground), each for `swing` of the cycle; the policy sees the phase. Off for runs made before it.
        gait=config_dict.create(enabled=True, freq=1.5, swing=0.4, offsets=(0.0, 0.6, 0.2, 0.8, 0.4),
                                clearance=0.035, terrain_clearance=0.05),
        # terrain="curriculum" only
        curriculum=config_dict.create(init_max_level=2, up_frac=0.7, down_frac=0.35, min_expected=0.4),
        terrain_scales=config_dict.create(base_height=0.0, orientation=-2.0, collision=-1.0, belly=-1.0),
        scan="none",
        scan_grid=config_dict.create(n=13, spacing=0.06),
        # simulated camera height map: noise 4 mm + 4 mm per metre of viewing distance, per-episode offset
        # 1 cm in height and 1 cm in xy (EKF drift over a few metres), memory 4 s
        cam_map=config_dict.create(noise=0.004, noise_per_m=0.004, bias=0.01, xy_bias=0.01, hist=40, every=5,
                                   min_range=0.2, max_range=3.0, hfov_deg=73.0, vfov_deg=58.0),
        cam_command=config_dict.create(speed_min=0.05, speed_max=0.25, half_angle_deg=45.0, wz=0.5),
    )


class RockyJoystick(mjx_env.MjxEnv):
    def __init__(self, terrain: str = "flat", config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        self._curriculum = terrain == "curriculum"
        if self._curriculum:
            import rocky_terrain
            assert self._config.impl == "warp", "terrain curriculum needs --impl warp (heightfield contacts)"
            self._terrain = rocky_terrain.generate()
            self._xml_path = os.path.join(HERE, "scene.xml")
            m = rocky_terrain.build_model(self._terrain)
        else:
            scene = "scene.xml" if terrain == "flat" else "scene_rough.xml"
            self._xml_path = os.path.join(HERE, scene)
            m = mujoco.MjModel.from_xml_path(self._xml_path)
        m.opt.timestep = self._config.sim_dt
        m.opt.iterations = self._config.iterations
        m.opt.ls_iterations = self._config.ls_iterations
        feet = [m.geom(f"leg{n}_foot").id for n in range(1, 6)]
        floor = m.geom("floor").id
        if not self._curriculum:
            for g in range(m.ngeom):
                if g not in feet and g != floor:
                    m.geom_contype[g] = 0
                    m.geom_conaffinity[g] = 0
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
        if self._config.impl == "warp":
            # Warp's overflow warnings are GPU printfs, some once per env per step: thousands of lines a step
            # stalled terrain training. Iteration caps are intended here (iterations / ls_iterations above).
            o = self._mjx_model.opt
            self._mjx_model = self._mjx_model.replace(opt=o.replace(_impl=o._impl.replace(warn_overflow=0)))
        key = m.key("stand").id
        self._init_q = jp.array(m.key_qpos[key])
        self._q_stand = jp.array(m.key_qpos[key][7:])
        self._lo = jp.array(m.jnt_range[1:, 0])
        self._hi = jp.array(m.jnt_range[1:, 1])
        self._base_z = float(m.key_qpos[key][2])
        self._foot_geoms = np.array(feet)
        self._foot_r = float(m.geom_size[feet[0]][0])
        self._floor = floor
        self._imu = m.site("imu").id
        self._gyro_adr = m.sensor_adr[m.sensor("imu_gyro").id]
        if self._curriculum:
            T = self._terrain
            sadr = lambda n: int(m.sensor_adr[m.sensor(n).id])
            self._fatal_adr = np.array([sadr(f"{n}_hit") for n in ("t_plate", "t_dome", "t_camera")])
            self._belly_adr = sadr("t_tray_hit")
            self._leg_hit_adr = np.array([sadr(m.sensor(i).name) for i in range(m.nsensor)
                                          if m.sensor(i).name.startswith("leg") and m.sensor(i).name.endswith("_hit")])
            self._centers = jp.array(T.centers)
            self._hmap = jp.array(T.heights)
            self._hmap_info = (T.res, T.size_x, T.size_y, T.nx, T.ny)
            g = (np.arange(7) - 3) * 0.1
            self._scan = jp.array(np.stack(np.meshgrid(g, g), -1).reshape(-1, 2))
        self._scan_mode = self._config.scan if self._curriculum else "none"
        if self._scan_mode != "none":
            sg = self._config.scan_grid
            g = (np.arange(sg.n) - (sg.n - 1) / 2) * sg.spacing
            self._pscan = jp.array(np.stack(np.meshgrid(g, g), -1).reshape(-1, 2))     # body-yaw frame xy
            # camera mount in the base frame (OpenCV axes: x right, y down, z forward), from the CPU model
            d0 = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d0, key)
            mujoco.mj_forward(m, d0)
            cid = m.camera("oakd").id
            Rb = d0.xmat[1].reshape(3, 3)
            self._cam_p = jp.array(Rb.T @ (d0.cam_xpos[cid] - d0.xpos[1]))
            self._cam_R = jp.array(Rb.T @ d0.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]))

    # ---------------- properties required by MjxEnv
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 15

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ---------------- helpers
    def _sample_cmd(self, rng):
        if getattr(self, "_scan_mode", "none") != "none":
            return self._sample_cam_cmd(rng)
        c = self._config.command
        k1, k2, k3 = jax.random.split(rng, 3)
        cmd = jax.random.uniform(k1, (3,), minval=-1.0, maxval=1.0) * jp.array([c.vx, c.vy, c.wz])
        zero = jax.random.uniform(k2) < 0.1
        turn_only = jax.random.uniform(k3) < 0.25
        cmd = jp.where(turn_only, cmd * jp.array([0.0, 0.0, 1.0]), cmd)
        return jp.where(zero, jp.zeros(3), cmd)

    def _sample_cam_cmd(self, rng):
        """Camera-first commands: heading within +-half_angle of body -y, plus some turning."""
        c = self._config.cam_command
        k = jax.random.split(rng, 5)
        th = jnp_deg(c.half_angle_deg) * jax.random.uniform(k[0], minval=-1.0, maxval=1.0)
        sp = jax.random.uniform(k[1], minval=c.speed_min, maxval=c.speed_max)
        wz = jax.random.uniform(k[2], minval=-c.wz, maxval=c.wz)
        cmd = jp.array([sp * jp.sin(th), -sp * jp.cos(th), wz])
        cmd = jp.where(jax.random.uniform(k[3]) < 0.1, cmd * jp.array([0.0, 0.0, 1.0]), cmd)
        return jp.where(jax.random.uniform(k[4]) < 0.1, jp.zeros(3), cmd)

    def _scan_world(self, data, xy_offset=None):
        """World xy of the policy scan points (body-yaw aligned grid around the base)."""
        R = data.xmat[1]
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        c, s_ = jp.cos(yaw), jp.sin(yaw)
        pts = data.qpos[0:2] + self._pscan @ jp.array([[c, s_], [-s_, c]])
        return pts if xy_offset is None else pts + xy_offset

    def _cam_pose(self, data):
        R = data.xmat[1]
        return data.qpos[0:3] + R @ self._cam_p, R @ self._cam_R

    def _scan_obs(self, data, info, rng):
        """(teacher scan, camera-map scan, known-mask); scans are base height above the ground there."""
        pts = self._scan_world(data)
        true = jp.clip(data.qpos[2] - self._ground(pts), -0.5, 0.5)
        if self._scan_mode != "camera":
            return true, None, None
        cm = self._config.cam_map
        pts_b = pts + info["map_bias"][0:2]                      # map built with a drifted pose estimate
        P = jp.concatenate([pts_b, self._ground(pts_b)[:, None]], axis=1)          # (N, 3) world points
        Xc = jp.einsum("kji,knj->kni", info["cam_R"], P[None] - info["cam_p"][:, None])   # (K, N, 3) cv frame
        z = Xc[..., 2]
        th, tv = jp.tan(jnp_deg(cm.hfov_deg) / 2), jp.tan(jnp_deg(cm.vfov_deg) / 2)
        vis = (z > cm.min_range) & (z < cm.max_range) & (jp.abs(Xc[..., 0]) < th * z) & (jp.abs(Xc[..., 1]) < tv * z)
        seen = jp.any(vis, axis=0)
        dist = jp.min(jp.where(vis, z, 10.0), axis=0)
        noise = (cm.noise + cm.noise_per_m * dist) * jax.random.normal(rng, true.shape)
        meas = jp.clip(data.qpos[2] - P[:, 2] + info["map_bias"][2] + noise, -0.5, 0.5)
        return true, jp.where(seen, meas, 0.0), seen.astype(jp.float32)

    def _cam_hist_update(self, data, info, force=False):
        p, R = self._cam_pose(data)
        push = force | (info["ep_step"] % self._config.cam_map.every == 0)
        hp = jp.where(push, jp.concatenate([p[None], info["cam_p"][:-1]], 0), info["cam_p"])
        hR = jp.where(push, jp.concatenate([R[None], info["cam_R"][:-1]], 0), info["cam_R"])
        return hp, hR

    def _ground(self, xy):
        """Terrain height (m) at world xy (..., 2), nearest heightfield cell."""
        res, sx, sy, nx, ny = self._hmap_info
        j = jp.clip(jp.round((xy[..., 0] + sx / 2) / res).astype(jp.int32), 0, nx - 1)
        i = jp.clip(jp.round((xy[..., 1] + sy / 2) / res).astype(jp.int32), 0, ny - 1)
        return self._hmap[i, j]

    def _height(self, data):
        """Base height above the local ground (flat: above z = 0)."""
        if self._curriculum:
            return data.qpos[2] - self._ground(data.qpos[0:2])
        return data.qpos[2]

    def _spawn(self, data, level, ttype, rng):
        """Teleport the robot (still at its stand pose) to the centre pad of tile (level, ttype)."""
        k1, k2 = jax.random.split(rng)
        c = self._centers[level, ttype]
        yaw = jax.random.uniform(k1, minval=-jp.pi, maxval=jp.pi)
        off = jax.random.uniform(k2, (2,), minval=-0.1, maxval=0.1)
        qpos = data.qpos.at[0:2].set(c[0:2] + off).at[2].set(c[2] + self._base_z + 0.005)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        return data.replace(qpos=qpos, qvel=jp.zeros_like(data.qvel)), qpos[0:2]

    def _foot_contact(self, data):
        return jp.stack([mjx_env.get_sensor_data(self.mj_model, data, f"leg{n}_floor_found")[0] > 0
                         for n in range(1, 6)])

    def _obs(self, data, info, rng):
        n = self._config.noise
        R = data.xmat[1]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        gyro = data.sensordata[self._gyro_adr:self._gyro_adr + 3]
        q = data.qpos[7:] - self._q_stand
        dq = data.qvel[6:]
        k = jax.random.split(rng, 4)
        gyro = gyro + n.gyro * jax.random.normal(k[0], (3,))
        grav = grav + n.gravity * jax.random.normal(k[1], (3,))
        q = q + n.qpos * jax.random.normal(k[2], (15,))
        dq = dq + n.qvel * jax.random.normal(k[3], (15,))
        state = jp.concatenate([gyro, grav, q, dq, info["last_act"], info["command"]])
        if self._config.gait.enabled:
            ph = 2 * jp.pi * info["phase"]
            state = jp.concatenate([state, jp.array([jp.sin(ph), jp.cos(ph)])])
        v_world = data.qvel[0:3]
        if self._config.est_obs:
            e = self._config.est_noise
            ke = jax.random.split(k[3], 2)
            v_est = R.T @ v_world + info["est_bias"][:3] + e.vel * jax.random.normal(ke[0], (3,))
            h_est = self._height(data) - self._base_z + info["est_bias"][3] + e.height * jax.random.normal(ke[1])
            state = jp.concatenate([state, v_est, jp.array([h_est])])
        privileged = jp.concatenate([state, R.T @ v_world, data.qvel[3:6], jp.array([self._height(data)])])
        if self._curriculum:          # critic only: terrain around the robot, yaw-aligned, relative to the base
            yaw = jp.arctan2(R[1, 0], R[0, 0])
            c, s_ = jp.cos(yaw), jp.sin(yaw)
            pts = data.qpos[0:2] + self._scan @ jp.array([[c, s_], [-s_, c]])
            privileged = jp.concatenate([privileged, jp.clip(data.qpos[2] - self._ground(pts), -0.5, 0.5)])
        out = {}
        if self._scan_mode != "none":
            true, meas, mask = self._scan_obs(data, info, jax.random.fold_in(rng, 7))
            privileged = jp.concatenate([privileged, true])
            if self._scan_mode == "true":
                state = jp.concatenate([state, true])
            else:
                out["teacher_state"] = jp.concatenate([state, true])
                state = jp.concatenate([state, meas, mask])
        # an exploded env must never reach the shared observation normaliser: one NaN there poisons every
        # policy input and then the weights (mjx_terrain died this way at 202 M steps)
        clean = lambda x: jp.clip(jp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
        out.update({"state": clean(state), "privileged_state": clean(privileged)})
        return {k: clean(v) for k, v in out.items()}

    # ---------------- API
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
        qpos = self._init_q
        yaw = jax.random.uniform(k1, minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        qpos = qpos.at[7:].add(jax.random.uniform(k2, (15,), minval=-0.05, maxval=0.05))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mj_model.nv), ctrl=self._q_stand,
                                 impl=self.mjx_model.impl.value, naconmax=self._config.naconmax,
                                 njmax=self._config.njmax)
        data = mjx.forward(self.mjx_model, data)
        info = {"rng": rng, "command": self._sample_cmd(k3), "last_act": jp.zeros(15),
                "feet_air_time": jp.zeros(5), "last_contact": jp.ones(5, dtype=bool), "step": 0,
                "phase": jax.random.uniform(k2), "last_act2": jp.zeros(15), "last_dq": jp.zeros(15),
                "last_feet": data.geom_xpos[self._foot_geoms][:, 0:2],
                "est_bias": jax.random.normal(k5, (4,)) * jp.array([self._config.est_noise.vel_bias] * 3
                                                                   + [self._config.est_noise.height_bias])}
        if self._curriculum:
            rng, kl, kt, ks = jax.random.split(rng, 4)
            level = jax.random.randint(kl, (), 0, self._config.curriculum.init_max_level + 1)
            ttype = jax.random.randint(kt, (), 0, len(self._terrain.types))
            data, spawn = self._spawn(data, level, ttype, ks)
            data = mjx.forward(self.mjx_model, data)
            info.update(rng=rng, level=level, ttype=ttype, spawn=spawn, ep_step=0, respawn=jp.array(False))
            if self._scan_mode == "camera":
                info.update(self._fresh_map(data, ks))
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._scales()}
        if self._curriculum:
            metrics["terrain/level"] = jp.zeros(())
        obs = self._obs(data, info, k4)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _fresh_map(self, data, rng):
        """New episode: camera history = the current pose only (nothing seen yet), new pose-drift offset."""
        cm = self._config.cam_map
        p, R = self._cam_pose(data)
        b = jax.random.normal(rng, (3,)) * jp.array([cm.xy_bias, cm.xy_bias, cm.bias])
        return {"cam_p": jp.tile(p[None], (cm.hist, 1)), "cam_R": jp.tile(R[None], (cm.hist, 1, 1)), "map_bias": b}

    def _scales(self):
        sc = dict(self._config.reward.scales)
        if not self._config.gait.enabled:
            for k in ("gait_contact", "feet_clearance", "foot_slip", "action_acc", "dof_acc"):
                sc.pop(k, None)
        if self._curriculum:
            sc.update(dict(self._config.terrain_scales))
        return sc

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        rng, k_cmd, k_obs, k_sp = jax.random.split(state.info["rng"], 4)
        data = state.data
        info = dict(state.info)
        if self._curriculum:
            # the auto-reset wrapper put this env back at its first spawn: move it to its current level
            moved, spawn = self._spawn(data, info["level"], info["ttype"], k_sp)
            r = info["respawn"]                    # only qpos / qvel: Warp's contact buffers can't be tree-merged
            data = data.replace(qpos=jp.where(r, moved.qpos, data.qpos), qvel=jp.where(r, moved.qvel, data.qvel))
            info["spawn"] = jp.where(info["respawn"], spawn, info["spawn"])
            info["ep_step"] = jp.where(info["respawn"], 0, info["ep_step"])
            info["command"] = jp.where(info["respawn"], self._sample_cmd(k_cmd), info["command"])
            if self._scan_mode == "camera":
                fresh = self._fresh_map(data, k_sp)
                for kk, v in fresh.items():
                    info[kk] = jp.where(info["respawn"], v, info[kk])
            info["respawn"] = jp.array(False)
        target = jp.clip(self._q_stand + cfg.action_scale * action, self._lo, self._hi)
        data = mjx_env.step(self.mjx_model, data, target, self.n_substeps)

        R = data.xmat[1]
        v = R.T @ data.qvel[0:3]
        w = data.qvel[3:6]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        contact = self._foot_contact(data)
        cmd = info["command"]
        first = contact & ~info["last_contact"]
        air = info["feet_air_time"] + self.dt
        moving = jp.linalg.norm(cmd) > 0.02
        q = data.qpos[7:]
        out = jp.clip(self._lo + 0.05 - q, 0.0) + jp.clip(q - (self._hi - 0.05), 0.0)
        height = self._height(data)
        blown = ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel)) | jp.any(jp.abs(data.qvel) > 100.0)
        done = (-grav[2] < 0.5) | (height < 0.03) | blown
        if self._curriculum:
            done = done | jp.any(data.sensordata[self._fatal_adr] > 0)
        terms = {
            "tracking_lin_vel": jp.exp(-jp.sum((cmd[:2] - v[:2]) ** 2) / cfg.reward.tracking_sigma),
            "tracking_ang_vel": jp.exp(-((cmd[2] - w[2]) ** 2) / cfg.reward.tracking_ang_sigma),
            "tracking_lin_vel_fine": jp.exp(-jp.sum((cmd[:2] - v[:2]) ** 2) / cfg.reward.tracking_sigma_fine),
            "lin_vel_z": v[2] ** 2,
            "ang_vel_xy": jp.sum(w[:2] ** 2),
            "orientation": jp.sum(grav[:2] ** 2),
            "base_height": (height - (self._base_z - 0.001)) ** 2,
            "torques": jp.sum(data.actuator_force ** 2),
            "action_rate": jp.sum((action - info["last_act"]) ** 2),
            "feet_air_time": jp.sum(jp.clip(air - 0.2, -0.05, 0.3) * first) * moving,
            "dof_pos_limits": jp.sum(out),
            "termination": done.astype(jp.float32),
            "alive": jp.ones(()),
        }
        if cfg.gait.enabled:
            g = cfg.gait
            ph = jp.mod(info["phase"] + jp.array(g.offsets), 1.0)            # per-leg phase
            swing = (ph < g.swing) & moving                                  # scheduled swing legs
            feet = data.geom_xpos[self._foot_geoms]
            ground = self._ground(feet[:, 0:2]) if self._curriculum else jp.zeros(5)
            clr = feet[:, 2] - self._foot_r - ground
            h = g.terrain_clearance if self._curriculum else g.clearance
            h_des = jp.where(swing, h * jp.sin(jp.pi * jp.clip(ph / g.swing, 0.0, 1.0)), 0.0)   # stance: on the ground
            fv = jp.minimum(jp.sum(((feet[:, 0:2] - info["last_feet"]) / self.dt) ** 2, axis=1), 1.0)
            terms["gait_contact"] = jp.mean((contact == ~swing).astype(jp.float32))
            terms["feet_clearance"] = jp.sum(jp.minimum(clr - h_des, 0.1) ** 2)
            # feet planted this step and the last: a foot touching down still moves at swing speed (gait3
            # counted that as slip and learned to take tiny, slow steps)
            terms["foot_slip"] = jp.sum((contact & info["last_contact"]) * fv)
            terms["action_acc"] = jp.sum((action - 2 * info["last_act"] + info["last_act2"]) ** 2)
            terms["dof_acc"] = jp.sum(((data.qvel[6:] - info["last_dq"]) / self.dt) ** 2)
            info["phase"] = jp.mod(info["phase"] + g.freq * self.dt, 1.0)
            info["last_feet"] = feet[:, 0:2]
            info["last_dq"] = data.qvel[6:]
            info["last_act2"] = info["last_act"]
        if self._curriculum:
            terms["collision"] = jp.sum(data.sensordata[self._leg_hit_adr] > 0).astype(jp.float32)
            terms["belly"] = (data.sensordata[self._belly_adr] > 0).astype(jp.float32)
        scales = self._scales()
        terms = {k: jp.where(blown, 0.0, jp.nan_to_num(v)) for k, v in terms.items()}
        terms["termination"] = done.astype(jp.float32)
        reward = sum(scales[k] * terms[k] for k in terms) * self.dt

        info["feet_air_time"] = air * ~contact
        info["last_contact"] = contact
        info["last_act"] = action
        info["step"] = info["step"] + 1
        resample = (info["step"] % 500) == 0
        info["command"] = jp.where(resample, self._sample_cmd(k_cmd), cmd)
        info["rng"] = rng
        if self._curriculum:
            info["ep_step"] = info["ep_step"] + 1
            end = done | (info["ep_step"] >= cfg.episode_length)
            cc = cfg.curriculum
            dist = jp.linalg.norm(data.qpos[0:2] - info["spawn"])
            expected = jp.linalg.norm(cmd[0:2]) * info["ep_step"] * self.dt
            judged = expected > cc.min_expected
            up = judged & ~done & (dist >= cc.up_frac * expected)
            down = done | (judged & (dist < cc.down_frac * expected))
            nl = info["level"] + up.astype(jp.int32) - down.astype(jp.int32)
            top = len(self._terrain.centers) - 1
            nl = jp.where(nl > top, jax.random.randint(k_sp, (), 0, top + 1), jp.clip(nl, 0, top))
            info["level"] = jp.where(end, nl, info["level"])
            info["ttype"] = jp.where(end, jax.random.randint(k_obs, (), 0, len(self._terrain.types)), info["ttype"])
            info["respawn"] = end
            if self._scan_mode == "camera":
                info["cam_p"], info["cam_R"] = self._cam_hist_update(data, info)
        obs = self._obs(data, info, k_obs)
        metrics = dict(state.metrics)          # keep keys added by wrappers
        metrics.update({f"reward/{k}": terms[k] for k in terms})
        if self._curriculum:
            metrics["terrain/level"] = info["level"].astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32),
                             metrics=metrics, info=info)


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Per-environment physics randomisation (brax `randomization_fn`)."""
    nu = 15

    @jax.vmap
    def rand(rng):
        k = jax.random.split(rng, 6)
        fr = jax.random.uniform(k[0], minval=0.5, maxval=1.25)
        geom_friction = model.geom_friction.at[:, 0].multiply(fr)
        body_mass = model.body_mass.at[1].add(jax.random.uniform(k[1], minval=-0.15, maxval=0.35))
        body_ipos = model.body_ipos.at[1, :2].add(jax.random.uniform(k[2], (2,), minval=-0.015, maxval=0.015))
        volt = jax.random.uniform(k[3], minval=9.9 / 11.1, maxval=12.6 / 11.1)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(k[4], (nu,), minval=0.75, maxval=1.25) * volt
        gain = model.actuator_gainprm.at[:, 0].set(kp)
        bias = model.actuator_biasprm.at[:, 1].set(-kp)
        frange = model.actuator_forcerange * volt
        damp = model.dof_damping.at[6:].multiply(jax.random.uniform(k[5], (nu,), minval=0.8, maxval=1.2))
        return geom_friction, body_mass, body_ipos, gain, bias, frange, damp

    gf, bm, bi, ga, bp, fr, dd = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    fields = {"geom_friction": gf, "body_mass": bm, "body_ipos": bi, "actuator_gainprm": ga,
              "actuator_biasprm": bp, "actuator_forcerange": fr, "dof_damping": dd}
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    model = model.tree_replace(fields)
    return model, in_axes
