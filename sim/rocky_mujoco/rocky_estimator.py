"""Rocky state estimator: error-state Kalman filter (ESKF) fusing the BNO085 IMU, the 15 servo
encoders (leg kinematics) and the OAK-D Lite depth camera. Robot side: numpy only, so it runs
unchanged on the Pi 5.

State (nominal):  p (world, m), v (world, m/s), q (body -> world), b_g (gyro bias), b_a (accel bias)
Error state (15): dp, dv, dtheta (body-frame rotation error), db_g, db_a

  predict(gyro, acc, dt)            IMU strapdown at the IMU rate (250 Hz in the sim)
  update_legs(q_meas, dt)           servo encoders: stance-foot velocity (leg odometry) and stance-foot height
  update_ground(plane)              camera ground plane: roll / pitch and height above the ground
  update_vo(vo)                     camera visual odometry: body velocity and rotation rate (-> gyro bias, so
                                    yaw drift is bounded by the gyro-bias estimate instead of the raw bias)

World z is up and the ground under the robot is taken as z = 0 (flat ground). On rough ground p_z is
"height above the local ground" (the stance-foot and ground-plane heights both measure that).
Absolute x, y and yaw are unobservable from these sensors (no GPS / magnetometer in the loop): their
covariance grows and they drift, as on the real robot.

Usage (50 Hz control loop, IMU at 250 Hz):
    est = Estimator(); est.init_from_accel(acc0)
    every IMU sample:  est.predict(gyro, acc, 0.004)
    every 20 ms:       est.update_legs(q_servo, 0.02)
    every camera frame: est.update_ground(gp.fit(depth)); est.update_vo(vo.update(t, gray, depth))
    est.v_body(), est.height(), est.gravity_body(), est.rpy()
"""
import math
from collections import deque

import numpy as np

import rocky_kinematics as K

G = np.array([0.0, 0.0, -9.81])
IMU_POS = np.array([-0.0483, 0.04145, 0.0605])      # site "imu" in the body frame (v5 BNO085 spot)

# noise parameters (1 sigma). Process: continuous-time densities; measurements: per update.
DEFAULT_NOISE = {
    "gyro": 0.004 * math.sqrt(0.004),    # rad/s/sqrt(Hz)   (0.004 rad/s per 250 Hz sample)
    "acc": 0.04 * math.sqrt(0.004),      # m/s^2/sqrt(Hz)
    "acc_extra": 0.3,                    # m/s^2: unmodelled vibration / foot impacts (dominates)
    "gyro_bias_rw": 1e-4,                # rad/s/sqrt(s)
    "acc_bias_rw": 1e-3,                 # m/s^2/sqrt(s)
    "leg_vel": 0.03,                     # m/s per axis, averaged stance-foot velocity
    "leg_height": 0.006,                 # m, stance-foot height above the ground
    "ground_normal": 0.01,               # unit-vector components (~0.6 deg)
    "ground_height": 0.008,              # m
    "vo_vel": 0.02,                      # m/s per axis
    "vo_rate": 0.02,                     # rad/s per axis (VO rotation over one frame interval)
}
INIT_STD = {"p": [0.01, 0.01, 0.01], "v": [0.05] * 3, "th": [0.02, 0.02, 0.01], "bg": [0.02] * 3, "ba": [0.15] * 3}
CHI2_GATE = {1: 10.8, 2: 13.8, 3: 16.3}             # 99.9 %


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_exp(th):
    a = np.linalg.norm(th)
    if a < 1e-12:
        return np.array([1.0, *(0.5 * th)])
    return np.array([math.cos(a / 2), *(math.sin(a / 2) * th / a)])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class ContactDetector:
    """Stance detection from servo feedback only (no foot switches).

    kinematic (default): a foot is in stance when its height in the gravity-aligned body frame is
      within `band` of the lowest foot and it is not rising fast. Needs only encoders + attitude.
    load (optional): |femur torque| above `load_thresh` N.m - use if the servos report load."""

    def __init__(self, band=0.010, vz_max=0.05, load_thresh=0.15):
        self.band, self.vz_max, self.load_thresh = band, vz_max, load_thresh

    def __call__(self, feet_g, feet_g_vel, femur_load=None):
        h = feet_g[:, 2]
        c = (h < h.min() + self.band) & (np.abs(feet_g_vel[:, 2]) < self.vz_max)   # low and not lifting / landing
        if femur_load is not None:
            c &= np.abs(femur_load) > self.load_thresh
        return c


class Estimator:
    def __init__(self, noise=None, r_cam=None, history_s=0.3):
        self.n = dict(DEFAULT_NOISE, **(noise or {}))
        self.r_cam = np.zeros(3) if r_cam is None else np.asarray(r_cam, float)   # camera origin, body frame
        self.contact = ContactDetector()
        self.hist = deque(maxlen=max(4, int(history_s / 0.004)))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, p=(0.0, 0.0, 0.09), yaw=0.0):
        self.p = np.array(p, float)
        self.v = np.zeros(3)
        self.q = yaw_quat(yaw)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        s = INIT_STD
        self.P = np.diag(np.concatenate([s["p"], s["v"], s["th"], s["bg"], s["ba"]]) ** 2)
        self.w = np.zeros(3)                     # bias-corrected angular rate (body)
        self.t = 0.0
        self._q_prev = None
        self._feet_prev = None
        self.stance = np.ones(5, bool)
        self.hist.clear()
        self.stats = {"rejected": 0, "updates": 0}

    def init_from_accel(self, acc, yaw=0.0, p=(0.0, 0.0, 0.09)):
        """Level the filter from a static accelerometer reading (roll / pitch); yaw is given."""
        self.reset(p, yaw)
        if not 5.0 < np.linalg.norm(acc) < 15.0:
            raise ValueError(f"accelerometer reading {acc} is not ~1 g: robot moving or IMU not sampled yet")
        f = np.asarray(acc, float) / np.linalg.norm(acc)          # body-frame "up" at rest
        z = np.array([0.0, 0.0, 1.0])
        ax = np.cross(f, z)
        s, c = np.linalg.norm(ax), float(np.dot(f, z))
        th = ax / s * math.atan2(s, c) if s > 1e-9 else np.zeros(3)
        # world_R_body = Rz(yaw) * R_tilt where R_tilt maps body up f to z
        self.q = quat_mul(yaw_quat(yaw), quat_exp(th))

    @property
    def R(self):
        return quat_to_R(self.q)

    def v_body(self):
        return self.R.T @ self.v

    def height(self):
        return float(self.p[2])

    def gravity_body(self):
        return self.R.T @ np.array([0.0, 0.0, -1.0])

    def rpy(self):
        R = self.R
        return np.array([math.atan2(R[2, 1], R[2, 2]), -math.asin(np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0])])

    def std(self):
        return np.sqrt(np.diag(self.P))

    # ------------------------------------------------------------------ IMU
    def predict(self, gyro, acc, dt):
        w = np.asarray(gyro) - self.bg
        f = np.asarray(acc) - self.ba - np.cross(w, np.cross(w, IMU_POS))   # IMU lever arm (centripetal)
        R = self.R
        a_w = R @ f + G
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = quat_mul(self.q, quat_exp(w * dt))
        self.q /= np.linalg.norm(self.q)
        self.w = w
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(f) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 6:9] = np.eye(3) - skew(w) * dt
        F[6:9, 9:12] = -np.eye(3) * dt
        n = self.n
        qa = n["acc"] ** 2 + n["acc_extra"] ** 2 * dt
        Q = np.diag(np.concatenate([np.zeros(3), [qa * dt] * 3, [n["gyro"] ** 2 * dt] * 3,
                                    [n["gyro_bias_rw"] ** 2 * dt] * 3, [n["acc_bias_rw"] ** 2 * dt] * 3]))
        self.P = F @ self.P @ F.T + Q
        self.t += dt
        self.hist.append((self.t, self.v.copy(), self.q.copy(), self.w.copy()))

    # ------------------------------------------------------------------ update core
    def _update(self, r, H, Rm, gate=True):
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        d2 = float(r @ Si @ r)
        if gate and d2 > CHI2_GATE.get(len(r), 3 * len(r) + 10):
            self.stats["rejected"] += 1
            return False
        Kg = self.P @ H.T @ Si
        dx = Kg @ r
        IKH = np.eye(15) - Kg @ H
        self.P = IKH @ self.P @ IKH.T + Kg @ Rm @ Kg.T
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_mul(self.q, quat_exp(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        self.stats["updates"] += 1
        return True

    # ------------------------------------------------------------------ servo encoders (leg kinematics)
    def update_legs(self, q_meas, dt, femur_load=None):
        """q_meas: 15 servo angles (rad, rocky_kinematics order). Stance feet are stationary on the
        ground: their velocity gives the body velocity, their height gives body height (and tilt)."""
        feet = K.fk_all(q_meas)                                  # (5, 3) body frame
        if self._feet_prev is None:
            self._feet_prev = feet
            return None
        dfeet = (feet - self._feet_prev) / dt
        self._feet_prev = feet
        R = self.R
        feet_g, vel_g = feet @ R.T, dfeet @ R.T                   # gravity-aligned (world-oriented) body frame
        st = self.contact(feet_g, vel_g, femur_load)
        self.stance = st
        if st.sum() < 3:
            return None
        # leg odometry: v_body = -(w x p_f + dp_f/dt) per stance foot; feet that disagree with the median
        # (slipping, or just landing / lifting) are dropped before averaging
        vi = -(np.cross(self.w, feet) + dfeet)
        med = np.median(vi[st], axis=0)
        st &= np.linalg.norm(vi - med, axis=1) < 0.04
        self.stance = st
        if st.sum() < 3:
            return None
        vb = vi[st].mean(axis=0)
        h = R.T @ self.v
        H = np.zeros((3, 15))
        H[:, 3:6] = R.T
        H[:, 6:9] = skew(R.T @ self.v)
        Rm = np.eye(3) * self.n["leg_vel"] ** 2 * 3.0 / st.sum()
        ok_v = self._update(vb - h, H, Rm)
        # stance-foot heights: p_z + (R p_f)_z = FOOT_R (sphere centre one radius above the ground)
        R = self.R
        idx = np.nonzero(st)[0]
        Hh = np.zeros((len(idx), 15))
        rh = np.zeros(len(idx))
        for k, i in enumerate(idx):
            Hh[k, 2] = 1.0
            Hh[k, 6:9] = -(R @ skew(feet[i]))[2]
            rh[k] = K.FOOT_R - (self.p[2] + (R @ feet[i])[2])
        ok_h = self._update(rh, Hh, np.eye(len(idx)) * self.n["leg_height"] ** 2, gate=False)
        return {"stance": st, "v_leg": vb, "ok_v": ok_v, "ok_h": ok_h}

    # ------------------------------------------------------------------ camera
    def _past(self, t):
        """Filter state (v, q, w) at time t from the history (the camera is ~one frame behind)."""
        if not self.hist:
            return self.v, self.q, self.w
        best = min(self.hist, key=lambda h: abs(h[0] - t))
        return best[1], best[2], best[3]

    def update_ground(self, plane):
        """Ground plane from the depth map: normal (body frame, up) = R^T z, height = p_z."""
        if plane is None or plane["inliers"] < 0.5:
            return False
        R = self.R
        n_pred = R.T @ np.array([0.0, 0.0, 1.0])
        H = np.zeros((3, 15))
        H[0:2, 6:9] = skew(n_pred)[0:2]
        H[2, 2] = 1.0
        r = np.array([plane["normal"][0] - n_pred[0], plane["normal"][1] - n_pred[1], plane["height"] - self.p[2]])
        Rm = np.diag([self.n["ground_normal"] ** 2] * 2 + [self.n["ground_height"] ** 2])
        return self._update(r, H, Rm)

    def update_vo(self, vo, t_now=None):
        """Camera-origin velocity (body axes) = R^T v + w x r_cam, evaluated at the middle of the VO
        interval (delayed measurement: residual from the past state, applied to the current one)."""
        if vo is None:
            return False
        t_mid = vo["t_mid"] if t_now is None else t_now - (self.t - vo["t_mid"])
        v_p, q_p, w_p = self._past(t_mid)
        Rp = quat_to_R(q_p)
        h = Rp.T @ v_p + np.cross(w_p, self.r_cam)
        R = self.R
        H = np.zeros((3, 15))
        H[:, 3:6] = R.T
        H[:, 6:9] = skew(R.T @ self.v)
        H[:, 9:12] = skew(self.r_cam)
        ok_v = self._update(vo["v_cam_body"] - h, H, np.eye(3) * self.n["vo_vel"] ** 2)
        # rotation rate: VO sees the true rate, the filter's rate is gyro - b_g  -> observes b_g (incl. yaw axis)
        Hw = np.zeros((3, 15))
        Hw[:, 9:12] = -np.eye(3)
        self._update(vo["w_body"] - w_p, Hw, np.eye(3) * self.n["vo_rate"] ** 2)
        return ok_v
