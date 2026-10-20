"""
Pentapod walker ("Rocky") - v6: v5 mechanics with the module-based electronics and the modelled
wiring harness removed, ready for the new integrated control PCB.
5 legs x 3 DOF, Hiwonder HX-30HM bus servos, Raspberry Pi 5 + Active Cooler, 3S LiPo.

Run from Fusion: Utilities > Scripts and Add-Ins > + (add this file) > Run.
Creates a NEW design document. All dimensions in mm (converted to cm for the API).
Vendor STEP files are read from HW_DIR (Documents/Pentapod/hardware).

v6 changes (from v5)
  - removed every modelled wire / cable (leg branches, hip>femur and femur>knee stock cables,
    rail bridges, deck wiring, battery / fuse leads, USB-C pigtail)
  - removed the ESP32 carrier perfboard + 2x19 headers, ESP32-DevKitC, BusLinker V3.0, 6 x WAGO
    221-415 + printed hex hub holder, 1000 uF cap, LM2596 (HW-411), Pololu D36V50F5, inline ATM
    fuse holder, Adafruit BNO085. All of it moves onto the new integrated control PCB
    (ESP32 module, servo-bus driver, 5 V / 3.3 V regulators, fuse, bulk caps, battery sense, IMU).
  - kept: plates, deck, dome, posts, legs, 15 servos, Pi 5 + Active Cooler, battery, XT60 loop key.
  - V5_MOUNTS picks which v5 leftovers stay: hub-holder holes, LM2596 + Pololu bosses and deck holes A / C
    are gone; BusLinker, perfboard and BNO085 bosses (control-PCB mounts) and hole B (battery leads) stay.
  - BNO085 breakout back at its v5 x/y, on 6 mm spacers on top of the control PCB (screws into the v5 IMU bosses).
  - PCB_STEP: if that file exists in HW_DIR, stage_hardware places it on the deck (PCB_M).
"""
import math, os

try:
    import adsk.core, adsk.fusion
except ImportError:                      # offline checking with the OCP stand-in
    import adsk.core, adsk.fusion        # noqa

S = 0.1  # mm -> cm
tb = adsk.fusion.TemporaryBRepManager.get()
UNION = adsk.fusion.BooleanTypes.UnionBooleanType
DIFF = adsk.fusion.BooleanTypes.DifferenceBooleanType
INTER = adsk.fusion.BooleanTypes.IntersectionBooleanType

HW_DIR = os.path.join(os.path.expanduser('~'), 'Documents', 'Pentapod', 'hardware')


def P(x, y, z): return adsk.core.Point3D.create(x*S, y*S, z*S)
def V(x, y, z): return adsk.core.Vector3D.create(x, y, z)


def rbox(cx, cy, cz, lx, ly, lz, ang=0.0):
    a = math.radians(ang)
    obb = adsk.core.OrientedBoundingBox3D.create(
        P(cx, cy, cz), V(math.cos(a), math.sin(a), 0), V(-math.sin(a), math.cos(a), 0),
        lx*S, ly*S, lz*S)
    return tb.createBox(obb)


def bx(x0, x1, y0, y1, z0, z1):
    return rbox((x0+x1)/2, (y0+y1)/2, (z0+z1)/2, x1-x0, y1-y0, z1-z0)


def cyl(a, b, r, r2=None):
    return tb.createCylinderOrCone(P(*a), r*S, P(*b), (r if r2 is None else r2)*S)


def sph(c, r): return tb.createSphere(P(*c), r*S)


def U(t, *tools):
    for x in tools: tb.booleanOperation(t, x, UNION)
    return t


def D(t, *tools):
    for x in tools: tb.booleanOperation(t, x, DIFF)
    return t


def I(t, *tools):
    for x in tools: tb.booleanOperation(t, x, INTER)
    return t


def pol(r, deg):
    a = math.radians(deg)
    return (r*math.cos(a), r*math.sin(a))


def rot2(p, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return (p[0]*c - p[1]*s, p[0]*s + p[1]*c, p[2])


def rotz(deg):
    m = adsk.core.Matrix3D.create()
    m.setToRotation(math.radians(deg), V(0, 0, 1), adsk.core.Point3D.create(0, 0, 0))
    return m


def frame(origin_mm, ex, ey, ez):
    m = adsk.core.Matrix3D.create()
    m.setWithCoordinateSystem(P(*origin_mm), V(*ex), V(*ey), V(*ez))
    return m


def halfspace(deg_normal):
    a = math.radians(deg_normal)
    n = (math.cos(a), math.sin(a))
    return tb.createBox(adsk.core.OrientedBoundingBox3D.create(
        P(n[0]*500, n[1]*500, 0), V(n[0], n[1], 0), V(-n[1], n[0], 0), 1000*S, 1000*S, 1000*S))


def tube(pts, r):
    """Wire / cable: cylinders between waypoints with spheres at the bends."""
    body = None
    for a, b in zip(pts[:-1], pts[1:]):
        if sum((a[i]-b[i])**2 for i in range(3)) < 1e-6:
            continue
        seg = cyl(a, b, r)
        body = seg if body is None else U(body, seg)
    for p in pts[1:-1]:
        U(body, sph(p, r))
    return body


def arc(r, a0, a1, z, step=8.0):
    """Points on a horizontal arc (degrees), shortest way round."""
    d = (a1 - a0 + 540) % 360 - 180
    n = max(1, int(abs(d) // step))
    return [(r*math.cos(math.radians(a0 + d*i/n)), r*math.sin(math.radians(a0 + d*i/n)), z) for i in range(n+1)]


def ang_of(p): return math.degrees(math.atan2(p[1], p[0]))


def plen(pts): return sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:]))


# ---------------- rigid placements (pure python, mm): world = R . model + t ----------------
I3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def place(cols, ma, wa):
    """cols = world images of the model x, y, z axes; model point ma lands on world point wa."""
    cols = [tuple(float(v) for v in c) for c in cols]
    t = tuple(wa[i] - sum(cols[j][i]*ma[j] for j in range(3)) for i in range(3))
    return (cols, t)


def xf(M, p):
    cols, t = M
    return tuple(t[i] + sum(cols[j][i]*p[j] for j in range(3)) for i in range(3))


def zrot_cols(cols, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return [(v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]) for v in cols]


def fmatrix(M):
    cols, t = M
    m = adsk.core.Matrix3D.create()
    m.setWithArray([cols[0][0], cols[1][0], cols[2][0], t[0]*S,
                    cols[0][1], cols[1][1], cols[2][1], t[1]*S,
                    cols[0][2], cols[1][2], cols[2][2], t[2]*S,
                    0, 0, 0, 1])
    return m


# ---------------- servo (measured from the Hiwonder STEP) ----------------
SVL, SVW = 45.0, 24.6
SOFF = 9.75
HF = 17.35
HORN_R, HORN_PCD_R, HORN_SCREW_R = 9.7, 6.5, 1.1
HUB_R, HUB_D = 4.6, 0.7
CASE_TOP, CASE_BOT = 15.45, 16.45
SERVO_STEP_ORIGIN = (-12.75, -13.75, 0.0)
CASE_HOLES_MAIN = [(8.45, 10.0), (8.45, -10.0)]
CASE_HOLES_AUX = [(8.45, 10.0), (8.45, -10.0), (15.95, 9.3), (15.95, -9.3)]

# ---------------- layout ----------------
CLR = 0.4
R_HIP = 100.0
PT = 4.0
GZT = CASE_TOP + CLR/2
GZB = CASE_BOT + CLR/2
BOSS_R = 9.5
PLATE_HOLE_R = 11.0
WEB0, WEB1 = 126.0, 132.0
FS_X0 = WEB1 + CLR; FS_X1 = FS_X0 + SVW; PFX = (FS_X0 + FS_X1)/2
WALL0 = FS_X1 + CLR; WALL1 = WALL0 + 6.0
CAPZ0 = SVL/2 + CLR
ZA0 = CAPZ0; ZA1 = ZA0 + 5.0
PFZ = SVL/2 - SOFF
AY0 = HF; AY1 = AY0 + 4.0
ARM_R, ARM_LO = 11.5, 11.0
TAY0 = AY1 + 1.5; TAY1 = TAY0 + 4.0
L_FEM = 85.0; L_TIB = 126.0; TDX = 8.0
PKX = PFX + L_FEM; PKZ = PFZ
KS_X0 = PKX - (SVL - SOFF); KS_X1 = PKX + SOFF
CZ0 = SVW/2 + CLR; CZ1 = CZ0 + 4.2
LIP = 5.0
LIP_T = 2.5
LEG_ANG = [90 + 72*i for i in range(5)]
GAP_ANG = [a + 36 for a in LEG_ANG]           # 126, 198, 270, 342, 54

ZPT = GZT + PT                       # top-plate top face   (19.65)
ZPB = -GZB - PT                      # bottom-plate bottom face (-20.65)
# dome: rim z 44 / r 131.9, 70 mm tall
RIM, H_DOME = 44.0, 70.0
RR = math.sqrt(175.0**2 - 115.0**2)
RO = (RR**2 + H_DOME**2) / (2*H_DOME); RI = RO - 2.4; CZD = RIM + H_DOME - RO
APEX_R = 40.0
POST_R, POST_TOP = 6.0, 78.0
POST_RAD = 88.0
DECK_Z0, DECK_T, DECK_R = 52.0, 3.5, 82.0
DECK_Z1 = DECK_Z0 + DECK_T
# battery tray
BAY_X, BAY_Y = 56.5, 21.5
TRAY_W, TRAY_F, TRAY_D = 2.5, 3.0, 12.0
TRAY_Z_IN = ZPB - TRAY_D
TRAY_SCREWS = [(61.0, 0.0), (-61.0, 0.0), (0.0, 27.0), (0.0, -27.0)]
# loop key + fuse (between the plates, above the battery)
SWX, SWY = 0.0, -68.0
FUSE_C = (-20.0, 16.0, 0.0)          # wire axis centre of the inline ATM holder
# ---- top plate: v5 legacy features (only cut / added when V5_MOUNTS) ----
# v5 leftovers: True = still cut / added. The control PCB mounts on the BusLinker, perfboard and BNO085
# bosses (the BNO085 breakout is stacked on the PCB over its v5 bosses); hole B is the battery-lead entry.
V5_MOUNTS = {'hub_screws': False, 'lm2596_bosses': False, 'pololu_bosses': False,
             'buslinker_bosses': True, 'perfboard_bosses': True, 'imu_bosses': True,
             'hole_A': False, 'hole_B': True, 'hole_C': False}
SLOT_R0, SLOT_R1 = 53.5, 61.0        # branch slot in the top plate, on each leg axis (real servo cables)
BRK_R = 57.5                         # v5 branch break-out radius (reference only)
PWR_SLOT = (0.0, -56.0)              # battery leads come up here (12 x 8)
LM_C, LM_ANG = pol(70, 126), 126.0   # v5 LM2596 (HW-411) position (bosses only)
Z_LM = ZPT + 3.0
# ---- logic deck ----
PI_X0, PI_Y0 = -50.5, -28.0          # Pi 5 board corner (85 x 56)
PI_Z0 = DECK_Z1 + 5.0                # board underside
PI_HOLES = [(PI_X0 + 3.5, PI_Y0 + 3.5), (PI_X0 + 61.5, PI_Y0 + 3.5),
            (PI_X0 + 3.5, PI_Y0 + 52.5), (PI_X0 + 61.5, PI_Y0 + 52.5)]
# v5 deck bosses / holes for removed parts (positions kept for V5_MOUNTS)
PB_C = (2.0, 51.0)                   # v5 ESP32 carrier (3 x 7 cm perfboard)
PB_Z0 = DECK_Z1 + 5.0
PB_HOLES = [(PB_C[0] + sx*32.8, PB_C[1] + sy*12.8) for sx in (-1, 1) for sy in (-1, 1)]
BL_C, BL_Z0 = (0.0, -49.5), DECK_Z1 + 5.0                 # v5 BusLinker V3.0
POL_C, POL_Z = (-65.8, -9.3), DECK_Z1 + 6.57              # v5 Pololu D36V50F5
IMU_C, IMU_Z0 = (-48.3, 41.45), DECK_Z1 + 5.0             # v5 Adafruit BNO085
HOLE_A, HOLE_B, HOLE_C = (-58.0, -36.0, 6.0), (0.0, -71.0, 5.5), (-40.0, 60.0, 5.0)   # v5 wire holes
# new integrated control PCB (placed only if the STEP exists in HW_DIR)
PCB_STEP = 'rocky_control_pcb.step'
PCB_ORIGIN = (0.0, 0.0, PI_Z0)       # STEP origin (board centre, underside) lands here: z 60.5, on the deck bosses
PCB_ROT = 0.0                        # deg about z
PCB_T = 1.6
# BNO085 breakout: same x/y as v5, but on 6 mm spacers ON TOP of the control PCB (long M2.5 screws into
# the v5 IMU bosses through breakout, spacer and PCB)
IMU_Z_V6 = PCB_ORIGIN[2] + PCB_T + 6.0     # 68.1

app = adsk.core.Application.get()


def horn_mount(body, centre, axis, v_face, v_out, outward):
    def pt(a, b, v):
        p = list(centre)
        o = [i for i in range(3) if i != axis]
        p[o[0]] += a; p[o[1]] += b; p[axis] = centre[axis] + v
        return tuple(p)
    D(body, cyl(pt(0, 0, v_face - outward*0.5), pt(0, 0, v_face + outward*HUB_D), HUB_R),
          cyl(pt(0, 0, v_face - outward*5), pt(0, 0, v_out + outward), 2.8))
    for k in range(8):
        a, b = pol(HORN_PCD_R, 45*k)
        D(body, cyl(pt(a, b, v_face - outward), pt(a, b, v_out + outward), HORN_SCREW_R),
              cyl(pt(a, b, v_out - outward*1.5), pt(a, b, v_out + outward), 2.0))


def servo_tf(t, eu, ev, ew):
    c = SERVO_STEP_ORIGIN
    o = tuple(t[i] - (c[0]*eu[i] + c[1]*ev[i] + c[2]*ew[i]) for i in range(3))
    return frame(o, eu, ev, ew)


HIP_TF = ((R_HIP, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 1, 0))
FEM_TF = ((PFX, 0, PFZ), (0, 0, -1), (0, 1, 0), (1, 0, 0))
KNEE_TF = ((PKX, 0, PKZ), (-1, 0, 0), (0, 1, 0), (0, 0, -1))


def servo_box(t, eu, ev, ew):
    def L(u, v, w): return tuple(t[i] + u*eu[i] + v*ev[i] + w*ew[i] for i in range(3))
    pts = [L(u, v, w) for u in (-SOFF, SVL-SOFF) for v in (-CASE_BOT, CASE_TOP) for w in (-SVW/2, SVW/2)]
    lo = [min(p[i] for p in pts) for i in range(3)]; hi = [max(p[i] for p in pts) for i in range(3)]
    return U(bx(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), cyl(L(0, -HF, 0), L(0, HF, 0), HORN_R))


# ======================= leg =======================
# reference cable routes (NOT modelled in v6; kept for cable-length planning)
HIP_CONN = (R_HIP - (SVL - SOFF) - 1.0, 0.0, 0.0)          # hip servo far-end face (connectors)
DROP_R = 75.0
BOT_SLOT_L = (DROP_R*math.cos(math.radians(-20)), DROP_R*math.sin(math.radians(-20)))
Z_UNDER = -ZA1 - 2.6                                       # cable level under the coxa (-30.5)
ARM_SLOT_X = PFX + 30.0                                    # femur>knee lead passes the -y arm here

# leg-frame cable centre-lines (also used for the length report)
LEAD_HF_CHASSIS = [(62.5, -4, -8), (62.0, -14, -8), (BOT_SLOT_L[0], BOT_SLOT_L[1], -8),
                   (BOT_SLOT_L[0], BOT_SLOT_L[1], Z_UNDER), (90, -12, Z_UNDER), (R_HIP, 0, Z_UNDER)]
LEAD_HF_COXA = [(R_HIP, 0, Z_UNDER), (118, 4, Z_UNDER), (PFX, 4, Z_UNDER), (PFX, 4, -CAPZ0-1.5)]
LEAD_FK_COXA = [(PFX, -4, -CAPZ0-1.5), (PFX, -4, Z_UNDER-1), (PFX, -25, Z_UNDER-1), (PFX, -25, PFZ)]
LEAD_FK_FEMUR = [(PFX, -25, PFZ), (ARM_SLOT_X, -25, PFZ), (ARM_SLOT_X, -12, PFZ), (PFX+38, -4, PFZ),
                 (KS_X0-1, 0, PFZ)]
BRANCH_LEG = [(HIP_CONN[0], 4, -4), (59.0, 4, -4), (BRK_R, 2, 4), (BRK_R, 0, ZPT), (BRK_R, 0, ZPT + 6)]


def leg_geometry():
    out = {'Coxa': [], 'Femur': [], 'Tibia': [], 'Foot (TPU)': []}
    coxa = U(cyl((R_HIP, 0, -ZA1), (R_HIP, 0, -ZA0), 14),
             bx(R_HIP, WEB1, -14, 14, -ZA1, -ZA0), bx(116, WEB1, -15, 15, -ZA1, -ZA0),
             cyl((R_HIP, 0, -ZA0), (R_HIP, 0, -HF), BOSS_R),
             bx(WEB0, WEB1, -15, 15, -ZA1, ZA0),
             bx(WALL0, WALL1, -15, 15, -ZA1, ZA0),
             bx(WEB0, WALL1, -(CASE_BOT+CLR+LIP_T), CASE_TOP+CLR+LIP_T, -ZA1, -CAPZ0),
             bx(WEB0, FS_X1, CASE_TOP+CLR, CASE_TOP+CLR+LIP_T, -CAPZ0, -CAPZ0+LIP),
             bx(WEB0, FS_X1, -(CASE_BOT+CLR+LIP_T), -(CASE_BOT+CLR), -CAPZ0, -CAPZ0+LIP))
    horn_mount(coxa, (R_HIP, 0, 0), 2, -HF, -ZA1, -1)
    D(coxa, bx(PFX-7, PFX+7, -9, 9, -ZA1-1, -CAPZ0+1))
    TOPSCREWS = [((WEB0+WEB1)/2, 9), ((WEB0+WEB1)/2, -9), ((WALL0+WALL1)/2, 9), ((WALL0+WALL1)/2, -9)]
    for sx, sy in TOPSCREWS:
        D(coxa, cyl((sx, sy, ZA0-12), (sx, sy, ZA0+1), 1.25))
    for x0 in (112.0, 124.0):                                  # zip-tie anchors
        D(coxa, bx(x0-1.5, x0+1.5, -6, -3, -ZA1-1, -ZA0+1), bx(x0-1.5, x0+1.5, 3, 6, -ZA1-1, -ZA0+1))
    top = U(cyl((R_HIP, 0, ZA0), (R_HIP, 0, ZA1), 14), bx(R_HIP, WALL1, -14, 14, ZA0, ZA1),
            bx(WEB0, WALL1, -15, 15, ZA0, ZA1), cyl((R_HIP, 0, HF), (R_HIP, 0, ZA0), BOSS_R))
    horn_mount(top, (R_HIP, 0, 0), 2, HF, ZA1, 1)
    for sx, sy in TOPSCREWS:
        D(top, cyl((sx, sy, ZA0-1), (sx, sy, ZA1+1), 1.7))
    out['Coxa'] += [(coxa, 'Coxa'), (top, 'Coxa Top Arm')]

    fem = None
    for s in (1, -1):
        ya, yb = sorted((s*AY0, s*AY1))
        arm = U(cyl((PFX, s*AY0, PFZ), (PFX, s*AY1, PFZ), ARM_R), bx(PFX, KS_X0-CLR, ya, yb, PFZ-ARM_LO, PFZ+CZ0))
        fem = arm if fem is None else U(fem, arm)
    U(fem, bx(PFX+40, KS_X0-CLR, -AY1, AY1, PFZ-CZ1, PFZ+CZ0),
      bx(PFX+40, KS_X1+CLR+6, -15, 15, PFZ-CZ1, PFZ-CZ0),
      bx(KS_X1+CLR, KS_X1+CLR+6, -15, 15, PFZ-CZ1, PFZ+CZ0),
      bx(KS_X0-CLR, KS_X0+LIP, CASE_TOP+CLR, CASE_TOP+CLR+LIP_T, PFZ-CZ1, PFZ+CZ0),
      bx(KS_X0-CLR, KS_X0+LIP, -(CASE_BOT+CLR+LIP_T), -(CASE_BOT+CLR), PFZ-CZ1, PFZ+CZ0))
    for s in (1, -1):
        horn_mount(fem, (PFX, 0, PFZ), 1, s*AY0, s*AY1, s)
    D(fem, bx(PFX+39, KS_X0+0.5, -9, 9, PFZ-6, PFZ+6))
    D(fem, bx(ARM_SLOT_X-3.5, ARM_SLOT_X+3.5, -AY1-1, -AY0+1, PFZ-3.5, PFZ+3.5))   # lead slot, -y arm
    LIDSCREWS = [(PFX+44.7, 9), (PFX+44.7, -9), (KS_X1+CLR+3, 9), (KS_X1+CLR+3, -9)]
    for sx, sy in LIDSCREWS:
        D(fem, cyl((sx, sy, PFZ+CZ0-10), (sx, sy, PFZ+CZ0+1), 1.25))
    lid = bx(PFX+40, KS_X1+CLR+6, -15, 15, PFZ+CZ0, PFZ+CZ1)
    for sx, sy in LIDSCREWS:
        D(lid, cyl((sx, sy, PFZ+CZ0-1), (sx, sy, PFZ+CZ1+1), 1.7))
    out['Femur'] += [(fem, 'Femur'), (lid, 'Femur Lid')]

    tib = None
    for s in (1, -1):
        ya, yb = sorted((s*TAY0, s*TAY1))
        side = U(cyl((PKX, s*TAY0, PKZ), (PKX, s*TAY1, PKZ), 14),
                 bx(PKX-14, PKX+14, ya, yb, PKZ-10, PKZ),
                 bx(PKX-14+TDX, PKX+14+TDX, ya, yb, PKZ-58, PKZ-10),
                 cyl((PKX, s*HF, PKZ), (PKX, s*TAY0, PKZ), BOSS_R))
        tib = side if tib is None else U(tib, side)
    shank = cyl((PKX+TDX, 0, PKZ-54), (PKX+TDX, 0, PKZ-L_TIB+2), 11, 6.5)
    U(tib, bx(PKX-14+TDX, PKX+14+TDX, -TAY1, TAY1, PKZ-58, PKZ-50), tb.copy(shank))
    for s in (1, -1):
        horn_mount(tib, (PKX, 0, PKZ), 1, s*HF, s*TAY1, s)
    for dx in (-8, 8):
        D(tib, cyl((PKX+TDX+dx, -40, PKZ-54), (PKX+TDX+dx, 40, PKZ-54), 1.7))
    FZ = PKZ - L_TIB
    pin = cyl((PKX+TDX, -15, FZ+7), (PKX+TDX, 15, FZ+7), 1.1)
    D(tib, tb.copy(pin))
    tibA = I(tb.copy(tib), bx(PKX-60, PKX+60, 0, 60, PKZ-200, PKZ+60))
    tibB = I(tib, bx(PKX-60, PKX+60, -60, 0, PKZ-200, PKZ+60))
    out['Tibia'] += [(tibA, 'Tibia half A (+y)'), (tibB, 'Tibia half B (-y)')]
    foot = U(cyl((PKX+TDX, 0, FZ-2), (PKX+TDX, 0, FZ+12), 10), sph((PKX+TDX, 0, FZ-2), 10))
    D(foot, shank, pin)
    out['Foot (TPU)'] += [(foot, 'Foot')]
    return out


# ======================= chassis =======================
def lm_M():
    a = math.radians(LM_ANG)
    t = (-math.sin(a), math.cos(a), 0.0); n = (math.cos(a), math.sin(a), 0.0)
    return place([(-n[0], -n[1], 0.0), (0, 0, 1), t], (10.3, -0.05, -21.7), (LM_C[0], LM_C[1], Z_LM))


def chassis_geometry():
    items = []

    def plate(z0, z1, rz0, rz1, holes, top):
        pl = cyl((0, 0, z0), (0, 0, z1), 95)
        for a in LEG_ANG:
            x, y = pol(R_HIP, a); U(pl, cyl((x, y, z0), (x, y, z1), 24))
        for a in LEG_ANG:
            x, y = pol(R_HIP, a); D(pl, cyl((x, y, -40), (x, y, 40), PLATE_HOLE_R))
            x, y = pol(POST_RAD, a+36); D(pl, cyl((x, y, -40), (x, y, 40), 1.7))
            for u, w in holes:
                p = rot2((R_HIP - u, w, 0), a)
                D(pl, cyl((p[0], p[1], -40), (p[0], p[1], 40), 1.1))
            if top:   # branch slot right above the hip-servo connectors
                x, y = pol((SLOT_R0 + SLOT_R1)/2, a)
                D(pl, rbox(x, y, (z0+z1)/2, SLOT_R1 - SLOT_R0, 10, PT+2, a))
            else:     # hip -> femur lead goes under the bottom plate here
                x, y = pol(DROP_R, a - 20)
                D(pl, rbox(x, y, (z0+z1)/2, 10, 7, PT+2, a - 20))
        for a in LEG_ANG:
            cx, cy = pol(R_HIP - (SVL/2 - SOFF), a)
            ring = D(rbox(cx, cy, (rz0+rz1)/2, SVL+2*CLR+4.8, SVW+2*CLR+4.8, rz1-rz0, a),
                     rbox(cx, cy, (rz0+rz1)/2, SVL+2*CLR, SVW+2*CLR, rz1-rz0+0.4, a))
            U(pl, ring)
        return pl

    botp = plate(ZPB, -GZB, -GZB, -GZB+5, CASE_HOLES_AUX, False)
    D(botp, bx(-BAY_X, BAY_X, -BAY_Y, BAY_Y, ZPB-1, -GZB+1),
      rbox(SWX, SWY, -GZB-PT/2, 16.6, 8.6, PT+2))
    for x, y in TRAY_SCREWS:
        D(botp, cyl((x, y, ZPB-1), (x, y, -GZB+1), 2.0))                              # M3 heat-set
    topp = plate(GZT, ZPT, GZT-5, GZT, CASE_HOLES_MAIN, True)
    D(topp, bx(PWR_SLOT[0]-6, PWR_SLOT[0]+6, PWR_SLOT[1]-4, PWR_SLOT[1]+4, GZT-1, ZPT+1))
    if V5_MOUNTS['hub_screws']:
        for a in (30, 150, 270):                                                     # v5 hub holder screws
            x, y = pol(18, a); D(topp, cyl((x, y, GZT-1), (x, y, ZPT+1), 1.25))
    if V5_MOUNTS['lm2596_bosses']:
        Mlm = lm_M()
        for hx, hz in ((2.3, -6.3), (18.3, -36.8)):                                  # v5 LM2596 bosses
            p = xf(Mlm, (hx, 0, hz))
            U(topp, cyl((p[0], p[1], ZPT - 0.5), (p[0], p[1], Z_LM), 2.6))
            D(topp, cyl((p[0], p[1], ZPT - 2.5), (p[0], p[1], Z_LM + 1), 1.25))
    items += [(botp, 'Bottom Plate'), (topp, 'Top Plate')]
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36)
        items.append((D(cyl((x, y, -GZB), (x, y, GZT), 4.5), cyl((x, y, -40), (x, y, 40), 1.7)), 'Plate Standoff %d' % (i+1)))

    # battery tray
    tray = U(bx(-BAY_X-TRAY_W, BAY_X+TRAY_W, -BAY_Y-TRAY_W, BAY_Y+TRAY_W, TRAY_Z_IN-TRAY_F, ZPB-3),
             bx(-64, 64, -30, 30, ZPB-3, ZPB))
    D(tray, bx(-BAY_X, BAY_X, -BAY_Y, BAY_Y, TRAY_Z_IN, ZPB+1))
    for sx in (-30, 30):
        D(tray, bx(sx-1.75, sx+1.75, -11, 11, TRAY_Z_IN-TRAY_F-1, TRAY_Z_IN+1))          # strap slots
    for x, y in TRAY_SCREWS:
        D(tray, cyl((x, y, ZPB-5), (x, y, ZPB+1), 1.7))
    D(tray, rbox(SWX, SWY, ZPB-1.5, 18, 10, 4))
    items.append((tray, 'Battery Tray'))

    # split dome posts + logic deck, all clamped by one M3 rod per post
    deck = cyl((0, 0, DECK_Z0), (0, 0, DECK_Z1), DECK_R)
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36); nx, ny = pol(POST_RAD+3.5, a+36)
        low = D(cyl((x, y, ZPT), (x, y, DECK_Z0), POST_R), cyl((x, y, ZPT-1), (x, y, DECK_Z0+1), 1.7))
        up = D(cyl((x, y, DECK_Z1), (x, y, POST_TOP), POST_R),
               cyl((x, y, DECK_Z1-1), (x, y, POST_TOP-4), 1.7),
               cyl((x, y, POST_TOP-3.2), (x, y, POST_TOP+1), 4.1),
               rbox(nx, ny, POST_TOP-10, 7, 5.8, 2.7, a+36))
        items += [(low, 'Dome Post lower %d' % (i+1)), (up, 'Dome Post upper %d' % (i+1))]
        U(deck, cyl((x, y, DECK_Z0), (x, y, DECK_Z1), 11))
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36)
        D(deck, cyl((x, y, DECK_Z0-1), (x, y, DECK_Z1+1), 1.7))
    bosses = [(PI_HOLES, PI_Z0, 3.2, 1.1)]                                           # M2.5 thread-forming
    if V5_MOUNTS['perfboard_bosses']:                                               # control PCB, M2
        bosses.append((PB_HOLES, PB_Z0, 2.6, 0.8))
    if V5_MOUNTS['buslinker_bosses']:                                               # control PCB, M2.5
        bosses.append(([(BL_C[0] + s*18, BL_C[1] + t*14) for s in (-1, 1) for t in (-1, 1)], BL_Z0, 3.0, 1.1))
    if V5_MOUNTS['pololu_bosses']:
        bosses.append(([xf(pololu_M(), (hx, hy, 0))[:2] for hx, hy in ((2.16, 2.16), (2.16, 23.24), (23.24, 2.16))],
                       POL_Z - 1.62, 2.0, 0.8))
    if V5_MOUNTS['imu_bosses']:                                                     # BNO085 breakout under the PCB
        bosses.append(([(IMU_C[0] + s*10.16, IMU_C[1] + t*8.89) for s in (-1, 1) for t in (-1, 1)], IMU_Z0, 3.0, 1.1))
    for pts, ztop, rb, rp in bosses:
        for hx, hy in pts:
            U(deck, cyl((hx, hy, DECK_Z1), (hx, hy, ztop), rb))
            D(deck, cyl((hx, hy, DECK_Z1 - 2), (hx, hy, ztop + 1), rp))
    for key, (hx, hy, hr) in (('hole_A', HOLE_A), ('hole_B', HOLE_B), ('hole_C', HOLE_C)):   # wire pass-throughs
        if V5_MOUNTS[key]:
            D(deck, cyl((hx, hy, DECK_Z0-1), (hx, hy, DECK_Z1+1), hr))
    items.append((deck, 'Logic Deck'))
    return {'Chassis': items}


def dome_geometry():
    shell = D(sph((0, 0, CZD), RO), sph((0, 0, CZD), RI))
    I(shell, bx(-300, 300, -300, 300, RIM, 400))
    for a in LEG_ANG:
        x, y = pol(POST_RAD, a+36)
        U(shell, I(cyl((x, y, POST_TOP-6), (x, y, POST_TOP+14), POST_R+2.5), sph((0, 0, CZD), RO)))
        D(shell, cyl((x, y, POST_TOP-6.1), (x, y, POST_TOP), POST_R+0.25),
              cyl((x, y, POST_TOP-0.1), (x, y, POST_TOP+3.2), 4.1))
    apex = I(tb.copy(shell), cyl((0, 0, 0), (0, 0, 300), APEX_R))
    D(shell, cyl((0, 0, 0), (0, 0, 300), APEX_R))
    items = [(apex, 'Dome Apex Cap')]
    for k, a in enumerate(LEG_ANG):
        items.append((I(tb.copy(shell), halfspace(a - 36 + 90), halfspace(a + 36 - 90)), 'Dome Sector %d' % (k+1)))
    return {'Dome': items}


# ======================= bought parts: placements =======================
FLAT = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]       # model X->y, Y->z (up), Z->x


def pololu_M():   # flipped (big caps up), I/O row towards -y
    return place([(-1, 0, 0), (0, 1, 0), (0, 0, -1)], (12.7, 12.7, 0.0), (POL_C[0], POL_C[1], POL_Z))



def hw_list():
    """(file, name, placement) for every bought part that has a CAD model."""
    L = []
    def add(f, name, M): L.append((f, name, M))
    add('3s Turnigy Li-Po battery (simplified).STEP', '3S 2200mAh LiPo', place(FLAT, (0, -12, 54.5), (0, 0, TRAY_Z_IN)))
    add('XT60H-M_75mm wire.step', 'XT60H-M battery lead', place(FLAT, (0, -4.68, -32.15), (0, 0, TRAY_Z_IN + 24.3)))   # lies on the pack
    add('XT60_Hembra.STEP', 'XT60 loop-key socket', place([(1, 0, 0), (0, -1, 0), (0, 0, -1)], (0, 0, 17.16), (SWX, SWY, ZPB - 6)))
    add('XT60_Macho.STEP', 'XT60 loop-key plug', place(I3, (0, 0, 16.1), (SWX, SWY, ZPB - 6)))
    for i, (x, y) in enumerate(TRAY_SCREWS):
        add('M3X5.0X6.0_HeatSetScrew.STEP', 'M3 heat-set insert %d' % (i + 1),
            place([(0, 0, 1), (1, 0, 0), (0, 1, 0)], (-3.22, 0, 0), (x, y, ZPB)))
    add('RaspberryPi5 (simplified).step', 'Raspberry Pi 5', place(I3, (0, 0, 0), (PI_X0, PI_Y0, PI_Z0)))
    add('Pi5 Active Cooler (Pi frame).step', 'Raspberry Pi Active Cooler', place(I3, (0, 0, -1.6), (PI_X0, PI_Y0, PI_Z0)))   # seated on the SoC
    add('Adafruit BNO085 STEMMA QT.step', 'Adafruit BNO085 IMU', place(I3, (0, 0, 0), (IMU_C[0], IMU_C[1], IMU_Z_V6)))   # on the control PCB
    return L


def misc_geometry():
    out = []
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a + 36)
        out.append((cyl((x, y, POST_TOP - 3.0), (x, y, POST_TOP), 4.0), 'Magnet 8x3 (post) %d' % (i + 1)))
        out.append((cyl((x, y, POST_TOP + 0.1), (x, y, POST_TOP + 3.1), 4.0), 'Magnet 8x3 (dome) %d' % (i + 1)))
        out.append((cyl((x, y, ZPB - 3 - 2.4), (x, y, POST_TOP - 8.7), 1.5), 'M3 tie rod %d' % (i + 1)))
    return out




# ======================= Fusion assembly =======================
def add(comp, items):
    if not items: return
    bf = comp.features.baseFeatures.add()
    bf.startEdit()
    for b, n in items:
        nb = comp.bRepBodies.add(b, bf)
        nb.name = n
    bf.finishEdit()


def import_into(comp, path):
    if not os.path.isfile(path): return False
    opts = app.importManager.createSTEPImportOptions(path); opts.isViewFit = False
    app.importManager.importToTarget(opts, comp)
    return True


def place_part(parent, fname, name, M):
    occ = parent.occurrences.addNewComponent(fmatrix(M))
    occ.component.name = name
    if not import_into(occ.component, os.path.join(HW_DIR, fname)):
        occ.component.name = name + ' (STEP MISSING)'
    return occ


def pattern_part(parent, fname, name, mats):
    """Import one small STEP once, then copy its bodies to every placement (one component)."""
    occ = parent.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    tmp = occ.component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    if not import_into(tmp.component, os.path.join(HW_DIR, fname)):
        return occ
    src = []
    def collect(c):
        for b in c.bRepBodies: src.append(b)
        for o in c.occurrences: collect(o.component)
    collect(tmp.component)
    items = []
    for M in mats:
        m = fmatrix(M)
        for b in src:
            c = tb.copy(b); tb.transform(c, m); items.append((c, b.name or name))
    tmp.deleteMe()
    add(occ.component, items)
    return occ


def new_design():
    app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    design = adsk.fusion.Design.cast(app.activeProduct)
    design.designType = adsk.fusion.DesignTypes.ParametricDesignType
    return design.rootComponent


def root():
    return adsk.fusion.Design.cast(app.activeProduct).rootComponent


def find_occ(r, name):
    for o in r.occurrences:
        if o.component.name == name: return o
    return None


def stage_legs_a(ident):
    r = new_design()
    legOcc0 = r.occurrences.addNewComponent(rotz(LEG_ANG[0])); leg = legOcc0.component; leg.name = 'Leg'
    geo = leg_geometry()
    for name in ('Coxa', 'Femur'):
        o = leg.occurrences.addNewComponent(ident); o.component.name = name
        add(o.component, geo[name])


def stage_legs_b(ident):
    leg = find_occ(root(), 'Leg').component
    geo = leg_geometry()
    for name in ('Tibia', 'Foot (TPU)'):
        o = leg.occurrences.addNewComponent(ident); o.component.name = name
        add(o.component, geo[name])


def stage_servos(ident):
    leg = find_occ(root(), 'Leg').component
    srv = leg.occurrences.addNewComponent(ident).component; srv.name = 'Leg Servos (HX-30HM)'
    step = os.path.join(HW_DIR, 'Hiwonder HX-35HM v3.step')
    for name, tf in (('Hip servo', HIP_TF), ('Femur servo', FEM_TF), ('Knee servo', KNEE_TF)):
        mount = srv.occurrences.addNewComponent(servo_tf(*tf)); mount.component.name = name
        if not import_into(mount.component, step):
            add(srv, [(servo_box(*tf), name + ' (placeholder)')])


def stage_instances(ident):
    r = root(); leg = find_occ(r, 'Leg').component
    for a in LEG_ANG[1:]:
        r.occurrences.addExistingComponent(leg, rotz(a))


def stage_body(ident, which):
    r = root()
    if which == 'chassis':
        c = r.occurrences.addNewComponent(ident).component; c.name = 'Chassis'; add(c, chassis_geometry()['Chassis'])
    elif which == 'dome':
        c = r.occurrences.addNewComponent(ident).component; c.name = 'Dome'; add(c, dome_geometry()['Dome'])
    elif which == 'misc':
        c = r.occurrences.addNewComponent(ident).component; c.name = 'Magnets & tie rods'; add(c, misc_geometry())


def PCB_M():
    return place(zrot_cols(I3, PCB_ROT), (0, 0, 0), PCB_ORIGIN)


def stage_hardware(ident, names=None):
    r = root()
    hwc = find_occ(r, 'Bought parts')
    hwc = hwc.component if hwc else r.occurrences.addNewComponent(ident).component
    hwc.name = 'Bought parts'
    for f, n, M in hw_list():
        if names is None or any(n.startswith(x) for x in names):
            place_part(hwc, f, n, M)


def stage_pcb(ident):
    """Integrated control PCB from KiCad (STEP), only if it exists in HW_DIR."""
    path = os.path.join(HW_DIR, PCB_STEP)
    if not os.path.isfile(path):
        return None
    r = root()
    occ = r.occurrences.addNewComponent(fmatrix(PCB_M()))
    occ.component.name = 'Rocky control PCB'
    import_into(occ.component, path)
    return occ


def run(_context: str):
    ident = adsk.core.Matrix3D.create()
    stage_legs_a(ident); stage_legs_b(ident); stage_servos(ident); stage_instances(ident)
    for w in ('chassis', 'dome', 'misc'):
        stage_body(ident, w)
    stage_hardware(ident)
    stage_pcb(ident)
