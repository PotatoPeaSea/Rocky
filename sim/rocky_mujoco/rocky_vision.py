"""Depth-camera measurements for the state estimator (robot side: numpy + OpenCV, runs on the Pi 5).

Input per frame: a depth map (metres along the optical axis, 0 = invalid) and the grey image of the
same camera, plus the fixed camera pose in the body frame (R_bc maps OpenCV camera axes, x right /
y down / z forward, to body axes; t_bc is the optical centre in the body frame).

  GroundPlane.fit(depth)          -> ground normal in the body frame + body height above it
                                     (roll / pitch relative to the ground, and height)
  VisualOdometry.update(...)      -> camera velocity over the last frame interval, in body axes
                                     (KLT features on the grey image, lifted to 3-D with depth,
                                     rigid fit with RANSAC)

On the real robot the OAK-D Lite can do the feature tracking on-device (DepthAI FeatureTracker node)
and send aligned depth; the maths below stays the same.
"""
import math

import numpy as np

try:
    import cv2
except ImportError:                       # ground plane works without OpenCV; VO needs it
    cv2 = None


def backproject(depth, intr, step=1, mask=None):
    """Depth map -> (N, 3) points in the OpenCV camera frame and their (row, col) pixel indices."""
    fx, fy, cx, cy = intr
    z = depth[::step, ::step]
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    ok = z > 0
    if mask is not None:
        ok &= mask[::step, ::step]
    z, u, v = z[ok], u[ok], v[ok]
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)


def fit_plane(pts):
    """Least-squares plane through pts: (unit normal, centroid)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return vt[2], c


class GroundPlane:
    """RANSAC plane fit on the depth points below the body. Returns the ground normal (pointing up,
    body frame), the height of the body origin above the plane, and the inlier fraction."""

    def __init__(self, intr, R_bc, t_bc, step=4, max_range=1.5, below=-0.03, thresh=0.01, iters=60, seed=0):
        self.intr, self.R, self.t = intr, np.asarray(R_bc), np.asarray(t_bc)
        self.step, self.max_range, self.below, self.thresh, self.iters = step, max_range, below, thresh, iters
        self.rng = np.random.default_rng(seed)

    def fit(self, depth):
        p = backproject(np.where(depth < self.max_range, depth, 0.0), self.intr, self.step)
        if len(p) < 30:
            return None
        pb = p @ self.R.T + self.t                         # body frame
        pb = pb[pb[:, 2] < self.below]                     # ground is below the body; drops walls / legs
        if len(pb) < 30:
            return None
        best, best_n = None, 0
        for _ in range(self.iters):
            s = pb[self.rng.choice(len(pb), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            inl = np.abs((pb - s[0]) @ n) < self.thresh
            k = int(inl.sum())
            if k > best_n:
                best, best_n = inl, k
        if best is None or best_n < 30:
            return None
        n, c = fit_plane(pb[best])
        if n[2] < 0:
            n = -n                                         # normal points up (towards the body)
        h = -float(np.dot(n, c))                           # signed distance of the body origin above the plane
        return {"normal": n, "height": h, "inliers": best_n / len(pb), "n_points": best_n}


def kabsch(A, B):
    """Rigid R, t minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (a / (2 * math.sin(a)))


class VisualOdometry:
    """Frame-to-frame RGB-D odometry. update() returns the camera origin's mean velocity and the
    camera angular velocity over the interval since the previous frame, in BODY axes, or None."""

    def __init__(self, intr, R_bc, t_bc=None, max_corners=200, min_tracks=12, inlier_m=0.005, iters=80, max_range=1.2,
                 self_radius=0.28, quality=0.003, seed=0):
        if cv2 is None:
            raise ImportError("VisualOdometry needs opencv-python (pip install opencv-python-headless)")
        self.intr, self.R = intr, np.asarray(R_bc)
        self.t = np.zeros(3) if t_bc is None else np.asarray(t_bc)
        self.self_radius, self.quality = self_radius, quality      # legs move with the camera: mask them out
        self.max_corners, self.min_tracks, self.inlier_m, self.iters, self.max_range = max_corners, min_tracks, inlier_m, iters, max_range
        self.rng = np.random.default_rng(seed)
        self.prev = None
        self.last = {}

    def _lift(self, depth, pts):
        """Depth at sub-pixel feature positions by bilinear interpolation of INVERSE depth (exact on a
        plane, where 1/z is affine in the pixel coordinates; nearest-pixel depth on ground seen at a
        grazing angle is off by tens of mm at 1.5 m). Samples next to a hole are dropped."""
        fx, fy, cx, cy = self.intr
        u, v = pts[:, 0], pts[:, 1]
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (u0 >= 0) & (v0 >= 0) & (u0 < depth.shape[1] - 1) & (v0 < depth.shape[0] - 1)
        u0c, v0c = np.where(ok, u0, 0), np.where(ok, v0, 0)
        a, b = u - u0c, v - v0c
        z00, z01 = depth[v0c, u0c], depth[v0c, u0c + 1]
        z10, z11 = depth[v0c + 1, u0c], depth[v0c + 1, u0c + 1]
        ok &= (z00 > 0) & (z01 > 0) & (z10 > 0) & (z11 > 0)
        inv = lambda x: 1.0 / np.maximum(x, 1e-6)
        z = 1.0 / ((1 - b) * ((1 - a) * inv(z00) + a * inv(z01)) + b * ((1 - a) * inv(z10) + a * inv(z11)))
        ok &= z < self.max_range
        X = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
        pb = X @ self.R.T + self.t
        ok &= np.hypot(pb[:, 0], pb[:, 1]) > self.self_radius     # not on the robot's own legs / dome
        return X, ok

    def update(self, t, gray, depth):
        prev, self.prev = self.prev, (t, gray, depth)
        if prev is None:
            return None
        t0, g0, d0 = prev
        dt = t - t0
        if dt <= 0:
            return None
        p0 = cv2.goodFeaturesToTrack(g0, self.max_corners, self.quality, 4, mask=(d0 > 0).astype(np.uint8))
        if p0 is None or len(p0) < self.min_tracks:
            self.last = {"tracks": 0}
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, gray, p0, None, winSize=(15, 15), maxLevel=2)
        st = st.ravel().astype(bool)
        p0, p1 = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
        X0, ok0 = self._lift(d0, p0)
        X1, ok1 = self._lift(depth, p1)
        ok = ok0 & ok1
        X0, X1 = X0[ok], X1[ok]
        n = len(X0)
        if n < self.min_tracks:
            self.last = {"tracks": n}
            return None
        best, best_k = None, 0
        for _ in range(self.iters):
            s = self.rng.choice(n, 3, replace=False)
            R, tt = kabsch(X0[s], X1[s])
            inl = np.linalg.norm(X0 @ R.T + tt - X1, axis=1) < self.inlier_m
            k = int(inl.sum())
            if k > best_k:
                best, best_k = inl, k
        if best_k < self.min_tracks:
            self.last = {"tracks": n, "inliers": best_k}
            return None
        R, tt = kabsch(X0[best], X1[best])          # static points: X1 = R X0 + t
        R_motion = R.T                               # camera pose at t1 in the t0 camera frame
        t_motion = -R.T @ tt
        v_c = t_motion / dt
        w_c = rotvec(R_motion) / dt
        resid = np.linalg.norm(X0[best] @ R.T + tt - X1[best], axis=1)
        self.last = {"tracks": n, "inliers": best_k, "resid_m": float(resid.mean())}
        return {"v_cam_body": self.R @ v_c, "w_body": self.R @ w_c, "dt": dt, "t_mid": t0 + dt / 2,
                "inliers": best_k, "tracks": n}
