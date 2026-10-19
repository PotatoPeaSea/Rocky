"""Rocky pentapod - per-leg forward / inverse kinematics that match rocky.xml.

All numbers come from fusion/pentapod_build_v5.py (mm there, metres here).

Frames
------
Body frame = the Fusion model frame: origin on the axis, midway between the two chassis plates,
z up. Leg n (n = 1..5) points along LEG_ANG[n-1] = 90 + 72*(n-1) degrees, so leg 1 points +y.

Leg frame: the body frame rotated by the leg angle about z, so x points radially out along the leg.

Joint conventions (rocky.xml and everything in this folder):
  hip   (yaw)  : axis +z of the leg frame,   +  = counter-clockwise seen from above
  femur (pitch): axis -y of the coxa frame,  +  = femur UP   (knee rises)
  knee  (pitch): axis +y of the femur frame, +  = tibia INWARD (foot swings toward the body)
  q = 0 is the CAD pose: coxa radial, femur horizontal, tibia hanging straight down.

Servo IDs: leg n uses IDs 3n-2 (hip), 3n-1 (femur), 3n (knee) - the same order as the
15 joints / actuators in rocky.xml, so actuator index i <-> servo ID i+1.
"""
import math
import numpy as np

# ---- geometry (from pentapod_build_v5.py, metres) ----
R_HIP = 0.100                    # hip yaw axis radius from body centre
HIP_Z = 0.0                      # hip axis height (plate mid-plane)
COXA_X, COXA_Z = 0.0447, 0.01275  # femur pitch axis relative to the hip axis (PFX - R_HIP, PFZ)
L_FEMUR = 0.085                  # femur pitch axis -> knee pitch axis (L_FEM)
TIB_X, TIB_Z = 0.008, -0.128     # knee axis -> foot sphere centre (TDX, -(L_TIB + 2))
FOOT_R = 0.010                   # foot sphere radius
LEG_ANG_DEG = [90.0 + 72.0 * i for i in range(5)]

L_TIB = math.hypot(TIB_X, TIB_Z)
GAMMA0 = math.atan2(TIB_Z, TIB_X)  # absolute angle of the tibia vector at q = 0 (about -86.4 deg)

# ---- joint limits (rad) : v5 design notes ----
HIP_LIM = (-math.radians(75), math.radians(75))      # hip yaw +-75
FEMUR_LIM = (-math.radians(45), math.radians(85))    # down 45 ; up limited by the dome (hit at ~93 deg, yaw 0)
KNEE_LIM = (-math.radians(60), math.radians(75))     # inward 75 ; outward 60 (assumed, not from notes)
LIMITS = np.array([HIP_LIM, FEMUR_LIM, KNEE_LIM] * 5)

JOINT_NAMES = [f"leg{n}_{j}" for n in range(1, 6) for j in ("hip", "femur", "knee")]
SERVO_IDS = list(range(1, 16))

# ---- default standing pose: foot centre at radius STAND_R from the body axis, STAND_H below it ----
STAND_R = 0.220                  # m, horizontal distance body axis -> foot centre
STAND_FOOT_Z = -0.081            # m, foot centre height in the body frame (body z ~ 0.091 m)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leg_fk(q, leg_frame=True):
    """Foot-sphere centre for one leg, q = (hip, femur, knee) rad. Leg frame by default."""
    q1, q2, q3 = q
    knee_s = COXA_X + L_FEMUR * math.cos(q2)
    knee_z = COXA_Z + L_FEMUR * math.sin(q2)
    phi = q2 - q3                                  # absolute tibia rotation in the leg plane
    c, s = math.cos(phi), math.sin(phi)
    fs = knee_s + c * TIB_X - s * TIB_Z
    fz = knee_z + s * TIB_X + c * TIB_Z
    p = np.array([R_HIP + math.cos(q1) * fs, math.sin(q1) * fs, HIP_Z + fz])
    return p


def leg_ik(p, knee_up=True):
    """Leg-frame foot-centre target -> (hip, femur, knee) rad. Raises ValueError if unreachable."""
    x, y, z = p
    dx = x - R_HIP
    q1 = math.atan2(y, dx)
    s = math.hypot(dx, y) - COXA_X
    zz = z - HIP_Z - COXA_Z
    D2 = s * s + zz * zz
    c = (D2 - L_FEMUR ** 2 - L_TIB ** 2) / (2 * L_FEMUR * L_TIB)
    if c > 1.0 or c < -1.0:
        raise ValueError(f"target out of reach (cos={c:.3f})")
    psi = -math.acos(c) if knee_up else math.acos(c)      # tibia angle relative to the femur
    q2 = math.atan2(zz, s) - math.atan2(L_TIB * math.sin(psi), L_FEMUR + L_TIB * math.cos(psi))
    q3 = GAMMA0 - psi
    return np.array([q1, q2, q3])


def body_to_leg(p_body, leg):
    """Body-frame point -> leg-frame point (leg = 0..4)."""
    return rotz(math.radians(LEG_ANG_DEG[leg])).T @ np.asarray(p_body)


def leg_to_body(p_leg, leg):
    return rotz(math.radians(LEG_ANG_DEG[leg])) @ np.asarray(p_leg)


def fk_all(q15):
    """15 joint angles -> (5, 3) foot centres in the body frame."""
    q15 = np.asarray(q15).reshape(5, 3)
    return np.stack([leg_to_body(leg_fk(q15[i]), i) for i in range(5)])


def ik_all(feet_body):
    """(5, 3) body-frame foot centres -> 15 joint angles."""
    return np.concatenate([leg_ik(body_to_leg(feet_body[i], i)) for i in range(5)])


def stand_feet(radius=STAND_R, foot_z=STAND_FOOT_Z):
    return np.stack([[radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), foot_z]
                     for a in LEG_ANG_DEG])


def stand_pose(radius=STAND_R, foot_z=STAND_FOOT_Z):
    return ik_all(stand_feet(radius, foot_z))


def within_limits(q15, margin=0.0):
    q15 = np.asarray(q15)
    return bool(np.all(q15 >= LIMITS[:, 0] + margin) and np.all(q15 <= LIMITS[:, 1] - margin))


if __name__ == "__main__":
    q = stand_pose()
    print("stand pose (deg):", np.round(np.degrees(q.reshape(5, 3)), 2).tolist())
    print("round-trip error (m):", np.abs(fk_all(q) - stand_feet()).max())
