"""Simulated sensors that MuJoCo does not model on its own: IMU noise / bias, and the OAK-D Lite
depth camera (rendered, then corrupted like a stereo depth map).

Sim side only. The robot-side code (rocky_estimator.py, rocky_vision.py) never imports this.

    cam = DepthCamera(model)                  # 160 x 120, 73 x 58 deg, stereo noise on
    depth, gray = cam.render(data)            # metres (0 = no depth), uint8 image
    imu = ImuModel(rng); gyro, acc = imu.sample(data, dt)
"""
import math

import mujoco
import numpy as np

# Luxonis OAK-D Lite (DM9095) stereo pair: OV7251 640 x 480 global shutter, 75 mm baseline, 73 x 58 deg
OAKD = {
    "baseline": 0.075,
    "native_width": 640,          # stereo matching resolution: sets the disparity step in metres
    "hfov_deg": 73.0,
    "min_depth": 0.20,            # extended-disparity mode; standard mode is ~0.35 m
    "max_depth": 8.0,
    "disp_sigma_px": 0.15,        # stereo matching noise (sub-pixel), gives ~2 % at 3 m like the datasheet
    "subpixel": 1.0 / 8.0,        # 3-bit sub-pixel disparity
    "dropout": 0.02,              # random invalid pixels
    "edge_jump": 0.15,            # relative depth jump between neighbour pixels that becomes a hole (occlusion)
}


class DepthCamera:
    """Renders the `oakd` camera of rocky.xml: depth (metres along the optical axis) + grey image.

    Image convention is OpenCV: row 0 at the top, x right, y down, z forward. intrinsics() gives
    fx, fy, cx, cy for the render resolution."""

    def __init__(self, model, width=160, height=120, camera="oakd", noise=True, rng=None, params=None):
        self.model = model
        self.w, self.h = width, height
        self.cam_id = model.camera(camera).id
        self.p = dict(OAKD, **(params or {}))
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.renderer = mujoco.Renderer(model, height, width)
        fovy = math.radians(model.cam_fovy[self.cam_id])
        self.fy = (height / 2) / math.tan(fovy / 2)
        self.fx = self.fy
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2
        # focal length (px) at the stereo matching resolution -> disparity in px for a given depth
        self.f_native = (self.p["native_width"] / 2) / math.tan(math.radians(self.p["hfov_deg"]) / 2)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]     # what a real camera sees: no collision / site groups

    def intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def extrinsics(self, data):
        """Camera pose in the robot body frame: (R_body_cv, t_body) with R mapping OpenCV camera axes
        (x right, y down, z forward) to body axes. Constant (rigid mount); computed from the model."""
        base = self.model.body("base").id
        Rb = data.xmat[base].reshape(3, 3)
        Rc = data.cam_xmat[self.cam_id].reshape(3, 3)          # MuJoCo camera: x right, y up, -z forward
        R_cv = Rc @ np.diag([1.0, -1.0, -1.0])
        return Rb.T @ R_cv, Rb.T @ (data.cam_xpos[self.cam_id] - data.xpos[base])

    def render(self, data):
        r = self.renderer
        r.update_scene(data, camera=self.cam_id, scene_option=self.opt)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.cam_id, scene_option=self.opt)
        z = r.render().astype(np.float64)
        r.disable_depth_rendering()
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
        return self._stereo(z), gray

    def _stereo(self, z):
        p = self.p
        valid = (z > p["min_depth"]) & (z < p["max_depth"])
        if not self.noise:
            return np.where(valid, z, 0.0)
        fb = self.f_native * p["baseline"]
        disp = fb / np.maximum(z, 1e-6)
        disp = disp + self.rng.normal(0.0, p["disp_sigma_px"], z.shape)
        disp = np.round(disp / p["subpixel"]) * p["subpixel"]
        zn = fb / np.maximum(disp, p["subpixel"])
        # holes: random dropouts and both sides of depth discontinuities (left-right check failures)
        valid &= self.rng.random(z.shape) > p["dropout"]
        jump = np.zeros_like(valid)
        zs = np.maximum(z, 1e-3)
        dzx = np.abs(np.diff(z, axis=1)) > p["edge_jump"] * np.minimum(zs[:, 1:], zs[:, :-1])
        dzy = np.abs(np.diff(z, axis=0)) > p["edge_jump"] * np.minimum(zs[1:, :], zs[:-1, :])
        jump[:, 1:] |= dzx; jump[:, :-1] |= dzx
        jump[1:, :] |= dzy; jump[:-1, :] |= dzy
        valid &= ~jump
        valid &= (zn > p["min_depth"]) & (zn < p["max_depth"])
        return np.where(valid, zn, 0.0)

    def close(self):
        self.renderer.close()


# BNO085 raw reports (calibrated gyroscope / accelerometer), sampled at the IMU rate
IMU_NOISE = {
    "gyro_sigma": 0.004,          # rad/s per sample at 250 Hz (~0.015 deg/s/sqrt(Hz))
    "acc_sigma": 0.04,            # m/s^2 per sample
    "gyro_bias": 0.01,            # rad/s, +- per axis, constant per episode (after BNO085 calibration)
    "acc_bias": 0.08,             # m/s^2
}


class ImuModel:
    """Adds white noise and a constant random bias to MuJoCo's ideal gyro / accelerometer at site imu."""

    def __init__(self, model, rng, params=None, enabled=True):
        self.p = dict(IMU_NOISE, **(params or {}))
        self.rng = rng
        self.enabled = enabled
        s = lambda n: model.sensor_adr[model.sensor(n).id]
        self.adr_gyro, self.adr_acc = s("imu_gyro"), s("imu_acc")
        self.reset()

    def reset(self):
        p, r = self.p, self.rng
        on = 1.0 if self.enabled else 0.0
        self.bg = r.uniform(-1, 1, 3) * p["gyro_bias"] * on
        self.ba = r.uniform(-1, 1, 3) * p["acc_bias"] * on

    def sample(self, data):
        g = data.sensordata[self.adr_gyro:self.adr_gyro + 3].copy()
        a = data.sensordata[self.adr_acc:self.adr_acc + 3].copy()
        if self.enabled:
            g += self.bg + self.rng.normal(0, self.p["gyro_sigma"], 3)
            a += self.ba + self.rng.normal(0, self.p["acc_sigma"], 3)
        return g, a
