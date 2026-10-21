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
    also sees a 7 x 7 grid of terrain heights around the robot; the policy stays blind.
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
            scales=config_dict.create(
                tracking_lin_vel=2.0, tracking_ang_vel=1.0, lin_vel_z=-2.0, ang_vel_xy=-0.05,
                orientation=-5.0, base_height=-50.0, torques=-0.002, action_rate=-0.01,
                feet_air_time=1.0, dof_pos_limits=-5.0, termination=-50.0, alive=0.2,
            ),
        ),
        command=config_dict.create(vx=0.12, vy=0.12, wz=0.5),
        # terrain="curriculum" only
        curriculum=config_dict.create(init_max_level=2, up_frac=0.7, down_frac=0.35, min_expected=0.4),
        terrain_scales=config_dict.create(base_height=0.0, orientation=-2.0, collision=-1.0, belly=-1.0),
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
        c = self._config.command
        k1, k2, k3 = jax.random.split(rng, 3)
        cmd = jax.random.uniform(k1, (3,), minval=-1.0, maxval=1.0) * jp.array([c.vx, c.vy, c.wz])
        zero = jax.random.uniform(k2) < 0.1
        turn_only = jax.random.uniform(k3) < 0.25
        cmd = jp.where(turn_only, cmd * jp.array([0.0, 0.0, 1.0]), cmd)
        return jp.where(zero, jp.zeros(3), cmd)

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
        return {"state": state, "privileged_state": privileged}

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
                "est_bias": jax.random.normal(k5, (4,)) * jp.array([self._config.est_noise.vel_bias] * 3
                                                                   + [self._config.est_noise.height_bias])}
        if self._curriculum:
            rng, kl, kt, ks = jax.random.split(rng, 4)
            level = jax.random.randint(kl, (), 0, self._config.curriculum.init_max_level + 1)
            ttype = jax.random.randint(kt, (), 0, len(self._terrain.types))
            data, spawn = self._spawn(data, level, ttype, ks)
            data = mjx.forward(self.mjx_model, data)
            info.update(rng=rng, level=level, ttype=ttype, spawn=spawn, ep_step=0, respawn=jp.array(False))
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._scales()}
        if self._curriculum:
            metrics["terrain/level"] = jp.zeros(())
        obs = self._obs(data, info, k4)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _scales(self):
        sc = dict(self._config.reward.scales)
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
        done = (-grav[2] < 0.5) | (height < 0.03) | jp.any(jp.isnan(data.qpos))
        if self._curriculum:
            done = done | jp.any(data.sensordata[self._fatal_adr] > 0)
        terms = {
            "tracking_lin_vel": jp.exp(-jp.sum((cmd[:2] - v[:2]) ** 2) / cfg.reward.tracking_sigma),
            "tracking_ang_vel": jp.exp(-((cmd[2] - w[2]) ** 2) / cfg.reward.tracking_ang_sigma),
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
        if self._curriculum:
            terms["collision"] = jp.sum(data.sensordata[self._leg_hit_adr] > 0).astype(jp.float32)
            terms["belly"] = (data.sensordata[self._belly_adr] > 0).astype(jp.float32)
        scales = self._scales()
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
