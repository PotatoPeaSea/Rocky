"""Generate rocky.xml (and, optionally, meshes/) for the Rocky pentapod.

    python build_model.py                         # rewrite rocky.xml from meshes/massprops.json
    python build_model.py --fusion-export DIR     # also re-process raw STLs exported from Fusion
    python build_model.py --camera-export DIR     # re-process only the v7 camera + mount STLs

The raw STLs (DIR) were exported read-only from the "Pentapod Walker v5" Fusion document
(world frame, mm, leg 1 instance). This script moves each part into its MuJoCo link frame,
decimates the heavy vendor meshes, computes mass properties and writes meshes/*.stl (metres)
plus meshes/massprops.json. Edit MASS_ASSUMPTIONS below and re-run to change the mass model.
"""
import argparse
import json
import math
import os

import numpy as np

import rocky_kinematics as K

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "meshes")

# ----------------------------------------------------------------------------------------------
# Mass model (grams). Printed parts: CAD volume x density x fill factor.
# ----------------------------------------------------------------------------------------------
PLA = 1.24          # g/cc (PLA; PETG is 1.27)
TPU = 1.21
FILL = 0.80         # effective solid fraction of printed structural parts (4 walls + ~30 % infill)
FILL_DOME = 0.95    # the 2.4 mm dome shell is essentially all perimeters
SERVO_G = 52.0      # Hiwonder HX-30HM, spec mass
MASS_ASSUMPTIONS = {
    # name: (grams or None -> from volume, density g/cc, note)
    "servo": SERVO_G,
    "fasteners_per_link": 4.0,      # M2/M3 screws, horn screws, heat-sets per printed link
    "battery": 188.0,               # 3S 2200 mAh 25-30C LiPo (Turnigy-class), incl. leads
    "pi5": 46.0,                    # Raspberry Pi 5 board
    "pi5_cooler": 21.0,             # official Active Cooler
    "control_pcb": 65.0,            # new v6 control PCB (spec'd 50-80 g), placed at the old hub
    "wiring_misc": 60.0,            # servo leads, XT60s, fuse, loop key, zip ties
    "base_fasteners": 25.0,
    "oakd_lite": 61.0,              # Luxonis OAK-D Lite, datasheet mass
}
SERVO_TORQUE = 30.0 * 9.80665 / 100.0   # 30 kg.cm @ 11.1 V -> 2.94 N.m (used as stall torque)
SERVO_SPEED = math.radians(60) / 0.19   # 0.19 s / 60 deg @ 11.1 V -> 5.51 rad/s (no load)
SERVO_KV = SERVO_TORQUE / SERVO_SPEED   # back-EMF slope of the DC motor: 0.533 N.m.s/rad
SERVO_KP = 30.0                         # N.m/rad : saturates (2.94 N.m) at ~5.6 deg of error
SERVO_ARMATURE = 0.004                  # kg.m^2 reflected rotor inertia (~5e-8 kg.m^2 x ~285^2)
SERVO_FRICTION = 0.03                   # N.m gear-train Coulomb friction

# v7 depth camera (fusion/pentapod_camera_v7.py): optical centre (RGB lens = aligned depth frame) in the
# body frame, mm, and its pitch. The camera looks along (0, -cos p, -sin p): the gap between legs 3 and 4.
CAM_LENS_MM = (0.0, -38.87, 133.03)
CAM_PITCH_DEG = 15.0
CAM_FOVY_DEG = 58.0                     # OV7251 stereo pair: 73 x 58 deg (HFOV x VFOV), 640 x 480
V6_APEX_CAP_CM3 = None                  # computed in process_camera from the v6 dome dimensions

# link frames in the LEG frame (mm): coxa = hip axis, femur = femur axis, tibia = knee axis
PFX, PFZ = 144.7, 12.75
PKX = PFX + 85.0
LINK_ORIGIN = {"coxa": (100.0, 0.0, 0.0), "femur": (PFX, 0.0, PFZ), "tibia": (PKX, 0.0, PFZ)}


# ----------------------------------------------------------------------------------------------
def read_stl(path):
    d = open(path, "rb").read()
    n = int(np.frombuffer(d[80:84], "<u4")[0])
    if d[:5] == b"solid" and len(d) != 84 + 50 * n:
        import trimesh
        m = trimesh.load(path)
        return np.asarray(m.vertices)[np.asarray(m.faces)]
    a = np.frombuffer(d[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].astype(float)          # (n, 3, 3)


def write_stl(path, tris):
    tris = np.asarray(tris, dtype="<f4")
    n = len(tris)
    rec = np.zeros(n, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nn = np.cross(e1, e2)
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
    rec["n"], rec["v"] = nn, tris
    with open(path, "wb") as f:
        f.write(b"rocky_mujoco".ljust(80, b"\0"))
        f.write(np.uint32(n).tobytes())
        f.write(rec.tobytes())


def mass_props(tris):
    """Volume, centroid and inertia (about the centroid, unit density) of a closed triangle soup."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v6 = np.einsum("ij,ij->i", a, np.cross(b, c))           # 6 x signed tetra volumes
    vol = v6.sum() / 6.0
    com = (v6[:, None] * (a + b + c)).sum(0) / (24.0 * vol)
    # second moments  int x_i x_j dV  over the tetrahedra (origin, a, b, c)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (v6 * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                             + a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j] + a[:, j] * c[:, i]
                             + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 120.0
    S_c = S - vol * np.outer(com, com)
    I = np.trace(S_c) * np.eye(3) - S_c
    return vol, com, I


def box_inertia(m, size):
    x, y, z = size
    return m / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def combine(parts):
    """parts: list of (mass, com(3), I_about_com(3x3)) -> (mass, com, I_about_com)."""
    M = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1]) for p in parts) / M
    I = np.zeros((3, 3))
    for m, pc, Ic in parts:
        d = np.asarray(pc) - c
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, c, I


def world_to_leg(t):     # Fusion export has leg 1 (at +90 deg): rotate by -90 deg about z
    return np.stack([t[..., 1], -t[..., 0], t[..., 2]], axis=-1)


def rot_z_tris(t, deg):
    R = K.rotz(math.radians(deg))
    return t @ R.T


def decimate(tris, target):
    if len(tris) <= target:
        return tris
    try:
        import fast_simplification
        import trimesh
        m = trimesh.Trimesh(vertices=tris.reshape(-1, 3), faces=np.arange(len(tris) * 3).reshape(-1, 3), process=True)
        v, f = fast_simplification.simplify(np.asarray(m.vertices), np.asarray(m.faces), 1.0 - target / len(m.faces))
        return v[f]
    except Exception as e:   # no decimator available: keep the full mesh
        print("  (decimation skipped:", e, ")")
        return tris


def process_fusion(src):
    """Raw Fusion STLs (mm, world) -> meshes/*.stl (m, link frames) + massprops.json."""
    os.makedirs(MESH_DIR, exist_ok=True)
    raw = {os.path.splitext(f)[0]: read_stl(os.path.join(src, f)) for f in os.listdir(src) if f.endswith(".stl")}
    info = json.load(open(os.path.join(src, "export_info.json")))
    fusion_vol = {k: sum(b.get("vol_cm3", 0) for b in v["bodies"]) for k, v in info.items() if isinstance(v, dict)}
    out_meshes, report, props = {}, {}, {}

    def emit(name, tris_mm, target=None):
        t = decimate(tris_mm, target) if target else tris_mm
        write_stl(os.path.join(MESH_DIR, name + ".stl"), t * 1e-3)
        out_meshes[name] = len(t)

    def printed(key, tris_mm, density, fill):
        vol, com, I = mass_props(tris_mm * 1e-1)       # cm
        m_g = vol * density * fill
        report[key] = {"vol_cm3_mesh": round(vol, 2), "vol_cm3_fusion": round(fusion_vol.get(key, float("nan")), 2),
                       "mass_g": round(m_g, 1), "density": density, "fill": fill}
        return (m_g * 1e-3, com * 1e-2, I * (m_g * 1e-3 / vol) * 1e-4)

    def solid(key, tris_mm, m_g):
        vol, com, I = mass_props(tris_mm * 1e-1)
        report[key] = {"mass_g": m_g, "vol_cm3_mesh": round(vol, 2), "note": "uniform density over CAD volume"}
        return (m_g * 1e-3, com * 1e-2, I * (m_g * 1e-3 / vol) * 1e-4)

    # ---------------- legs (leg frame -> link frames) ----------------
    leg = {k: world_to_leg(raw[k]) for k in ("leg_coxa", "leg_femur", "leg_tibia", "leg_foot",
                                              "servo_hip", "servo_femur", "servo_knee")}
    fast = MASS_ASSUMPTIONS["fasteners_per_link"]
    servo_parts = {}
    for s in ("servo_hip", "servo_femur", "servo_knee"):
        servo_parts[s] = solid(s, leg[s], SERVO_G)
    links = {
        "coxa": (["leg_coxa"], ["servo_femur"]),
        "femur": (["leg_femur"], ["servo_knee"]),
        "tibia": (["leg_tibia", "leg_foot"], []),
    }
    for link, (prints, servos) in links.items():
        o = np.array(LINK_ORIGIN[link])
        parts = []
        for p in prints:
            dens, fill = (TPU, 0.9) if p == "leg_foot" else (PLA, FILL)
            parts.append(printed(p, leg[p] - o, dens, fill))
            emit(p, leg[p] - o)
        for s in servos:
            m, c, I = servo_parts[s]
            parts.append((m, c - o * 1e-3, I))
            emit(s, leg[s] - o, 1200)
        # fasteners: point mass at the printed-part centroid
        parts.append((fast * 1e-3, parts[0][1], np.eye(3) * 1e-9))
        props[link] = combine(parts)

    # ---------------- base (body frame) ----------------
    base_parts = []
    base_parts.append(printed("base_chassis", raw["base_chassis"], PLA, FILL))
    emit("base_chassis", raw["base_chassis"], 6000)
    base_parts.append(printed("base_dome", raw["base_dome"], PLA, FILL_DOME))
    emit("base_dome", raw["base_dome"], 5000)
    mr = raw["base_magnets_rods"]
    m_mr = sum(b["mass_kg"] for b in info["base_magnets_rods"]["bodies"]) * 1e3   # Fusion steel ~ NdFeB
    base_parts.append(solid("base_magnets_rods", mr, round(m_mr, 1)))
    hip_all = np.concatenate([rot_z_tris(leg["servo_hip"], a) for a in K.LEG_ANG_DEG])
    m, c, I = servo_parts["servo_hip"]
    for a in K.LEG_ANG_DEG:
        R = K.rotz(math.radians(a))
        base_parts.append((m, R @ c, R @ I @ R.T))
    emit("base_hip_servos", hip_all, 6000)
    base_parts.append(solid("battery", raw["battery"], MASS_ASSUMPTIONS["battery"]))
    emit("battery", raw["battery"])
    pi = np.concatenate([raw["pi5"], raw["pi5_cooler"]])
    lo, hi = pi.reshape(-1, 3).min(0), pi.reshape(-1, 3).max(0)
    m_pi = MASS_ASSUMPTIONS["pi5"] + MASS_ASSUMPTIONS["pi5_cooler"]
    base_parts.append((m_pi * 1e-3, (lo + hi) / 2e3, box_inertia(m_pi * 1e-3, (hi - lo) * 1e-3)))
    report["pi5+cooler"] = {"mass_g": m_pi, "note": "box over the CAD bounding box"}
    emit("pi5", raw["pi5"], 4000)
    emit("pi5_cooler", raw["pi5_cooler"], 3000)
    # control PCB (v6): 75 x 75 x 12 mm box centred on the top plate where the WAGO hub was
    pcb_c, pcb_s = np.array([0.0, 0.0, 0.0285]), np.array([0.075, 0.075, 0.012])
    base_parts.append((MASS_ASSUMPTIONS["control_pcb"] * 1e-3, pcb_c, box_inertia(MASS_ASSUMPTIONS["control_pcb"] * 1e-3, pcb_s)))
    base_parts.append((MASS_ASSUMPTIONS["wiring_misc"] * 1e-3, np.zeros(3), box_inertia(MASS_ASSUMPTIONS["wiring_misc"] * 1e-3, (0.18, 0.18, 0.04))))
    base_parts.append((MASS_ASSUMPTIONS["base_fasteners"] * 1e-3, np.array([0, 0, 0.01]), box_inertia(MASS_ASSUMPTIONS["base_fasteners"] * 1e-3, (0.2, 0.2, 0.06))))
    props["base"] = combine(base_parts)

    js = {k: {"mass": v[0], "pos": list(v[1]), "inertia": np.asarray(v[2]).tolist()} for k, v in props.items()}
    js["_report"] = report
    js["_meshes"] = out_meshes
    js["_assumptions"] = dict(MASS_ASSUMPTIONS, PLA=PLA, TPU=TPU, FILL=FILL, FILL_DOME=FILL_DOME)
    json.dump(js, open(os.path.join(MESH_DIR, "massprops.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


def v6_apex_cap_volume():
    """Volume (cm3) of the v6 dome apex cap: the 2.4 mm spherical shell inside r < 40 mm."""
    rr = math.sqrt(175.0 ** 2 - 115.0 ** 2)
    ro = (rr ** 2 + 70.0 ** 2) / (2 * 70.0)
    ri = ro - 2.4
    r = np.linspace(0, 40.0, 4001)
    dz = np.sqrt(ro ** 2 - r ** 2) - np.sqrt(ri ** 2 - r ** 2)
    return float(np.trapezoid(dz * 2 * math.pi * r, r)) * 1e-3


def process_camera(src):
    """v7 camera STLs (mm, body frame) -> meshes/camera_*.stl + massprops.json["camera"].

    camera_oakd_lite.stl : the two housing shells of the vendor STEP (front + back cover)
    camera_mount_cap.stl : "Dome Apex Cap (camera mount)", replaces the v6 apex cap
    Camera body frame = body axes, origin at CAM_LENS_MM. Its mass = camera (61 g) + the
    material the mount adds over the v6 cap (the cap itself is already in the dome mass)."""
    js = json.load(open(os.path.join(MESH_DIR, "massprops.json")))
    o = np.array(CAM_LENS_MM)
    cam = read_stl(os.path.join(src, "camera_oakd_lite.stl"))
    cap = read_stl(os.path.join(src, "camera_mount_cap.stl"))
    vol_c, com_c, I_c = mass_props(cam * 1e-1)
    vol_m, com_m, I_m = mass_props(cap * 1e-1)
    m_cam = MASS_ASSUMPTIONS["oakd_lite"]
    v_old = v6_apex_cap_volume()
    m_mount = (vol_m - v_old) * PLA * FILL
    parts = [(m_cam * 1e-3, com_c * 1e-2 - o * 1e-3, I_c * (m_cam * 1e-3 / vol_c) * 1e-4),
             (m_mount * 1e-3, com_m * 1e-2 - o * 1e-3, I_m * (m_mount * 1e-3 / vol_m) * 1e-4)]
    M, c, I = combine(parts)
    write_stl(os.path.join(MESH_DIR, "camera_oakd_lite.stl"), decimate(cam, 3000) * 1e-3 - o * 1e-3)
    write_stl(os.path.join(MESH_DIR, "camera_mount_cap.stl"), cap * 1e-3)
    js["camera"] = {"mass": M, "pos": list(c), "inertia": np.asarray(I).tolist()}
    js["_report"]["camera"] = {"oakd_lite_g": m_cam, "mount_cm3": round(vol_m, 2), "v6_cap_cm3": round(v_old, 2),
                               "mount_added_g": round(m_mount, 1), "note": "camera: uniform over the housing CAD volume"}
    js["_meshes"]["camera_oakd_lite"] = len(decimate(cam, 3000))
    js["_meshes"]["camera_mount_cap"] = len(cap)
    js["_assumptions"]["oakd_lite"] = m_cam
    json.dump(js, open(os.path.join(MESH_DIR, "massprops.json"), "w"), indent=1)
    print(json.dumps(js["_report"]["camera"], indent=1))


# ----------------------------------------------------------------------------------------------
def f(v, n=6):
    return " ".join(f"{x:.{n}g}" if abs(x) > 1e-12 else "0" for x in np.ravel(v))


def inertial(p):
    I = np.asarray(p["inertia"])
    full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    return f'<inertial pos="{f(p["pos"])}" mass="{p["mass"]:.5f}" fullinertia="{f(full, 5)}"/>'


def quat_z(deg):
    a = math.radians(deg) / 2
    return f"{math.cos(a):.8f} 0 0 {math.sin(a):.8f}"


def build_xml():
    mp = json.load(open(os.path.join(MESH_DIR, "massprops.json")))
    q0 = K.stand_pose()
    feet = K.stand_feet()
    base_z = -feet[0, 2] + K.FOOT_R
    lim = K.LIMITS
    L = []
    a = L.append
    a('<mujoco model="rocky_pentapod">')
    a('  <!-- GENERATED by build_model.py - edit that script, not this file. Units: m, kg, s, rad. -->')
    a('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
    a('  <option timestep="0.002" integrator="implicitfast" cone="pyramidal" impratio="1">')
    a('    <flag eulerdamp="disable"/>')
    a('  </option>')
    a('  <size memory="20M"/>')
    a('  <default>')
    a('    <default class="rocky">')
    a(f'      <joint type="hinge" armature="{SERVO_ARMATURE}" damping="{SERVO_KV:.4f}" frictionloss="{SERVO_FRICTION}" limited="true"/>')
    a('      <!-- HX-30HM: voltage-mode P servo. force = kp*(target - q) clipped to the stall torque;')
    a('           the back-EMF term (-kv * qdot) is the joint damping above, so the torque-speed line')
    a('           tau_max(w) = tau_stall (1 - w / w_noload) comes out of the physics. -->')
    a(f'      <general dyntype="none" gaintype="fixed" biastype="affine" gainprm="{SERVO_KP}" biasprm="0 -{SERVO_KP} 0"')
    a(f'               forcerange="-{SERVO_TORQUE:.4f} {SERVO_TORQUE:.4f}" ctrllimited="true"/>')
    a('      <geom contype="0" conaffinity="0" group="2" density="0" type="mesh"/>')
    a('      <site group="4"/>')
    a('      <default class="visual_body"><geom rgba="0.86 0.80 0.70 1"/></default>')
    a('      <default class="visual_leg"><geom rgba="0.72 0.62 0.50 1"/></default>')
    a('      <default class="visual_servo"><geom rgba="0.12 0.12 0.13 1"/></default>')
    a('      <default class="visual_foot"><geom rgba="0.20 0.20 0.20 1"/></default>')
    a('      <default class="visual_elec"><geom rgba="0.15 0.45 0.25 1"/></default>')
    a('      <default class="visual_batt"><geom rgba="0.15 0.25 0.65 1"/></default>')
    a('      <default class="collision">')
    a('        <geom group="3" contype="1" conaffinity="1" condim="3" friction="0.6 0.005 0.0001" rgba="0.8 0.3 0.3 0.4" density="0" solref="0.005 1"/>')
    a('        <default class="foot">')
    a('          <geom type="sphere" size="0.010" friction="1.0 0.01 0.001" priority="1" solref="0.006 1" solimp="0.9 0.95 0.001"/>')
    a('        </default>')
    a('      </default>')
    a('    </default>')
    a('  </default>')
    a('  <asset>')
    for m in sorted(mp["_meshes"]):
        a(f'    <mesh name="{m}" file="{m}.stl"/>')
    a('  </asset>')
    a('  <worldbody>')
    a(f'    <body name="base" pos="0 0 {base_z + 0.002:.4f}" childclass="rocky">')
    a('      <freejoint name="root"/>')
    a('      ' + inertial(mp["base"]))
    a('      <geom name="v_chassis" mesh="base_chassis" class="visual_body"/>')
    a('      <geom name="v_dome" mesh="base_dome" class="visual_body" group="1"/>')
    a('      <geom name="v_hip_servos" mesh="base_hip_servos" class="visual_servo"/>')
    a('      <geom name="v_battery" mesh="battery" class="visual_batt"/>')
    a('      <geom name="v_pi5" mesh="pi5" class="visual_elec"/>')
    a('      <geom name="v_pi5_cooler" mesh="pi5_cooler" class="visual_servo"/>')
    a('      <geom name="v_control_pcb" type="box" pos="0 0 0.0285" size="0.0375 0.0375 0.006" class="visual_elec"/>')
    a('      <!-- collision: convex hulls of the chassis (plates, tray, deck) and of the dome cap -->')
    a('      <geom name="c_chassis" mesh="base_chassis" type="mesh" class="collision"/>')
    a('      <geom name="c_dome" mesh="base_dome" type="mesh" class="collision"/>')
    a('      <site name="imu" pos="-0.0483 0.04145 0.0605" size="0.006" rgba="1 0.5 0 1" group="4"/>')
    a('      <site name="base_center" pos="0 0 0" size="0.005"/>')
    if "camera" in mp:
        cp, cs = math.cos(math.radians(CAM_PITCH_DEG)), math.sin(math.radians(CAM_PITCH_DEG))
        a('      <!-- v7: Luxonis OAK-D Lite on the camera-mount apex cap (fusion/pentapod_camera_v7.py) -->')
        a('      <geom name="v_camera_mount" mesh="camera_mount_cap" class="visual_body" group="1"/>')
        a(f'      <body name="camera" pos="{f(np.array(CAM_LENS_MM) * 1e-3)}">')
        a('        ' + inertial(mp["camera"]))
        a('        <geom name="v_camera" mesh="camera_oakd_lite" class="visual_servo"/>')
        # housing: 91 x 28 x 17.5 mm, lens 19 mm above the bottom face, on the cover glass
        hc = np.array([0.0, 0.0, 0.0]) + 0.00875 * np.array([0, cp, cs]) + (0.014 - 0.019) * np.array([0, -cs, cp])
        a(f'        <geom name="c_camera" type="box" pos="{f(hc)}" xyaxes="1 0 0 0 {cp:.5f} {cs:.5f}" '
          f'size="0.0455 0.00875 0.014" class="collision"/>')
        a('        <!-- MuJoCo cameras look along -z (y up): z = (0, cos p, sin p), image right = body -x.')
        a('             2 mm in front of the cover glass so the housing never enters the view. -->')
        a(f'        <camera name="oakd" pos="0 {-0.002 * cp:.5f} {-0.002 * cs:.5f}" xyaxes="-1 0 0 0 {-cs:.5f} {cp:.5f}" fovy="{CAM_FOVY_DEG}"/>')
        a('        <site name="camera" size="0.004" rgba="0 0.6 1 1"/>')
        a('      </body>')
    for n in range(1, 6):
        ang = K.LEG_ANG_DEG[n - 1]
        p = K.rotz(math.radians(ang)) @ np.array([K.R_HIP, 0, K.HIP_Z])
        a(f'      <!-- ===== leg {n}  (servo IDs {3*n-2}, {3*n-1}, {3*n}) at {ang:.0f} deg ===== -->')
        a(f'      <body name="leg{n}_coxa" pos="{f(p)}" quat="{quat_z(ang)}">')
        a('        ' + inertial(mp["coxa"]))
        a(f'        <joint name="leg{n}_hip" axis="0 0 1" range="{lim[0,0]:.5f} {lim[0,1]:.5f}"/>')
        a(f'        <geom name="leg{n}_v_coxa" mesh="leg_coxa" class="visual_leg"/>')
        a(f'        <geom name="leg{n}_v_femur_servo" mesh="servo_femur" class="visual_servo"/>')
        a(f'        <geom name="leg{n}_c_coxa" type="box" pos="0.0250 0 0" size="0.0390 0.0180 0.0275" class="collision"/>')
        a(f'        <body name="leg{n}_femur" pos="{K.COXA_X} 0 {K.COXA_Z}">')
        a('          ' + inertial(mp["femur"]))
        a(f'          <joint name="leg{n}_femur" axis="0 -1 0" range="{lim[1,0]:.5f} {lim[1,1]:.5f}"/>')
        a(f'          <geom name="leg{n}_v_femur" mesh="leg_femur" class="visual_leg"/>')
        a(f'          <geom name="leg{n}_v_knee_servo" mesh="servo_knee" class="visual_servo"/>')
        a(f'          <geom name="leg{n}_c_femur" type="box" pos="0.0450 0 0" size="0.0560 0.0214 0.0169" class="collision"/>')
        a(f'          <body name="leg{n}_tibia" pos="{K.L_FEMUR} 0 0">')
        a('            ' + inertial(mp["tibia"]))
        a(f'            <joint name="leg{n}_knee" axis="0 1 0" range="{lim[2,0]:.5f} {lim[2,1]:.5f}"/>')
        a(f'            <geom name="leg{n}_v_tibia" mesh="leg_tibia" class="visual_leg"/>')
        a(f'            <geom name="leg{n}_v_foot" mesh="leg_foot" class="visual_foot"/>')
        a(f'            <geom name="leg{n}_c_tibia" type="box" pos="0.004 0 -0.020" size="0.018 0.0268 0.036" class="collision"/>')
        a(f'            <geom name="leg{n}_c_shank" type="capsule" fromto="{K.TIB_X} 0 -0.058 {K.TIB_X} 0 -0.112" size="0.009" class="collision"/>')
        a(f'            <geom name="leg{n}_foot" pos="{K.TIB_X} 0 {K.TIB_Z}" class="foot"/>')
        a(f'            <site name="leg{n}_foot" pos="{K.TIB_X} 0 {K.TIB_Z}" size="0.0115" type="sphere" rgba="0 1 0 0.3"/>')
        a('          </body>')
        a('        </body>')
        a('      </body>')
    a('    </body>')
    a('  </worldbody>')
    a('  <contact>')
    a('    <!-- adjacent links inside one leg touch at the servo horns: no contact pairs needed -->')
    for n in range(1, 6):
        a(f'    <exclude body1="base" body2="leg{n}_femur"/>')
    a('  </contact>')
    a('  <actuator>')
    for n in range(1, 6):
        for j, (jn, (lo_, hi_)) in enumerate(zip(("hip", "femur", "knee"), lim[3 * (n - 1):3 * n])):
            sid = 3 * (n - 1) + j + 1
            a(f'    <general name="s{sid:02d}_leg{n}_{jn}" joint="leg{n}_{jn}" class="rocky" ctrlrange="{lo_:.5f} {hi_:.5f}"/>')
    a('  </actuator>')
    a('  <sensor>')
    a('    <!-- BNO085-like IMU on the logic deck (v5 location) -->')
    a('    <framequat name="imu_quat" objtype="site" objname="imu"/>')
    a('    <gyro name="imu_gyro" site="imu"/>')
    a('    <accelerometer name="imu_acc" site="imu"/>')
    a('    <framelinvel name="base_linvel" objtype="site" objname="base_center"/>')
    a('    <framepos name="base_pos" objtype="site" objname="base_center"/>')
    for n in range(1, 6):
        for jn in ("hip", "femur", "knee"):
            a(f'    <jointpos name="q_leg{n}_{jn}" joint="leg{n}_{jn}"/>')
    for n in range(1, 6):
        for jn in ("hip", "femur", "knee"):
            a(f'    <jointvel name="dq_leg{n}_{jn}" joint="leg{n}_{jn}"/>')
    for n in range(1, 6):
        for jn in ("hip", "femur", "knee"):
            a(f'    <actuatorfrc name="tau_leg{n}_{jn}" actuator="s{3*(n-1)+("hip","femur","knee").index(jn)+1:02d}_leg{n}_{jn}"/>')
    for n in range(1, 6):
        a(f'    <touch name="leg{n}_foot_touch" site="leg{n}_foot"/>')
    a('  </sensor>')
    qpos = [0, 0, base_z, 1, 0, 0, 0] + list(q0)
    a('  <keyframe>')
    a(f'    <key name="stand" qpos="{f(qpos, 7)}" ctrl="{f(q0, 7)}"/>')
    z0 = np.zeros(15)
    feet0 = K.fk_all(z0)
    a(f'    <key name="zero" qpos="{f([0, 0, -feet0[0,2] + K.FOOT_R + 0.002, 1, 0, 0, 0] + list(z0), 7)}" ctrl="{f(z0)}"/>')
    a('  </keyframe>')
    a('</mujoco>')
    open(os.path.join(HERE, "rocky.xml"), "w").write("\n".join(L) + "\n")
    tot = mp["base"]["mass"] + mp.get("camera", {"mass": 0.0})["mass"] + 5 * (mp["coxa"]["mass"] + mp["femur"]["mass"] + mp["tibia"]["mass"])
    print(f"wrote rocky.xml  total mass {tot:.3f} kg, stand base z {base_z:.4f} m")




def build_rough_scene(max_height=0.025, n=100, size=5.0, seed=7):
    """scene_rough.xml: the flat scene's look plus a random-bump heightfield (flat 0.5 m pad at the origin)."""
    rng = np.random.default_rng(seed)
    z = rng.random((n, n))
    for _ in range(3):                                   # smooth the noise a little
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    z = (z - z.min()) / (z.max() - z.min())
    xs = np.linspace(-size, size, n)
    X, Y = np.meshgrid(xs, xs)
    pad = np.clip((np.hypot(X, Y) - 0.4) / 0.4, 0, 1)    # flat start pad, blends to full bumps by 0.8 m
    z = z * pad
    flat = open(os.path.join(HERE, "scene.xml")).read()
    elev = " ".join(f"{v:.3f}" for v in z.ravel())
    s = flat.replace('<mujoco model="rocky_scene">', '<mujoco model="rocky_scene_rough">')
    s = s.replace("  </asset>", f'    <hfield name="rough" nrow="{n}" ncol="{n}" size="{size} {size} {max_height} 0.05" elevation="{elev}"/>\n  </asset>')
    s = s.replace('<geom name="floor" size="10 10 0.05" type="plane"', '<geom name="floor" type="hfield" hfield="rough" pos="0 0 0"')
    open(os.path.join(HERE, "scene_rough.xml"), "w").write(s)
    print(f"wrote scene_rough.xml ({n}x{n} heightfield, bumps up to {max_height*100:.1f} cm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-export", help="folder with the raw STLs + export_info.json from Fusion")
    ap.add_argument("--camera-export", help="folder with camera_oakd_lite.stl + camera_mount_cap.stl (v7)")
    a = ap.parse_args()
    if a.fusion_export:
        process_fusion(a.fusion_export)
    if a.camera_export or a.fusion_export:
        process_camera(a.camera_export or a.fusion_export)
    build_xml()
    build_rough_scene()
