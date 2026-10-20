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
    )


class RockyJoystick(mjx_env.MjxEnv):
    def __init__(self, terrain: str = "flat", config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)
        scene = "scene.xml" if terrain == "flat" else "scene_rough.xml"
        self._xml_path = os.path.join(HERE, scene)
        m = mujoco.MjModel.from_xml_path(self._xml_path)
        m.opt.timestep = self._config.sim_dt
        feet = [m.geom(f"leg{n}_foot").id for n in range(1, 6)]
        floor = m.geom("floor").id
        for g in range(m.ngeom):
            if g not in feet and g != floor:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        self._mj_model = m
        self._mjx_model = mjx.put_model(m, impl=self._config.impl)
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
            h_est = data.qpos[2] - self._base_z + info["est_bias"][3] + e.height * jax.random.normal(ke[1])
            state = jp.concatenate([state, v_est, jp.array([h_est])])
        privileged = jp.concatenate([state, R.T @ v_world, data.qvel[3:6], jp.array([data.qpos[2]])])
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
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales.keys()}
        obs = self._obs(data, info, k4)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        rng, k_cmd, k_obs = jax.random.split(state.info["rng"], 3)
        target = jp.clip(self._q_stand + cfg.action_scale * action, self._lo, self._hi)
        data = mjx_env.step(self.mjx_model, state.data, target, self.n_substeps)

        R = data.xmat[1]
        v = R.T @ data.qvel[0:3]
        w = data.qvel[3:6]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        contact = self._foot_contact(data)
        info = dict(state.info)
        cmd = info["command"]
        first = contact & ~info["last_contact"]
        air = info["feet_air_time"] + self.dt
        moving = jp.linalg.norm(cmd) > 0.02
        q = data.qpos[7:]
        out = jp.clip(self._lo + 0.05 - q, 0.0) + jp.clip(q - (self._hi - 0.05), 0.0)
        done = (-grav[2] < 0.5) | (data.qpos[2] < 0.03) | jp.any(jp.isnan(data.qpos))
        terms = {
            "tracking_lin_vel": jp.exp(-jp.sum((cmd[:2] - v[:2]) ** 2) / cfg.reward.tracking_sigma),
            "tracking_ang_vel": jp.exp(-((cmd[2] - w[2]) ** 2) / cfg.reward.tracking_ang_sigma),
            "lin_vel_z": v[2] ** 2,
            "ang_vel_xy": jp.sum(w[:2] ** 2),
            "orientation": jp.sum(grav[:2] ** 2),
            "base_height": (data.qpos[2] - (self._base_z - 0.001)) ** 2,
            "torques": jp.sum(data.actuator_force ** 2),
            "action_rate": jp.sum((action - info["last_act"]) ** 2),
            "feet_air_time": jp.sum(jp.clip(air - 0.2, -0.05, 0.3) * first) * moving,
            "dof_pos_limits": jp.sum(out),
            "termination": done.astype(jp.float32),
            "alive": jp.ones(()),
        }
        reward = sum(cfg.reward.scales[k] * terms[k] for k in terms) * self.dt

        info["feet_air_time"] = air * ~contact
        info["last_contact"] = contact
        info["last_act"] = action
        info["step"] = info["step"] + 1
        resample = (info["step"] % 500) == 0
        info["command"] = jp.where(resample, self._sample_cmd(k_cmd), cmd)
        info["rng"] = rng
        obs = self._obs(data, info, k_obs)
        metrics = dict(state.metrics)          # keep keys added by wrappers
        metrics.update({f"reward/{k}": terms[k] for k in terms})
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
