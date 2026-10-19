"""
Pentapod walker ("Rocky") - v5: final controls spec, every bought part from a real CAD model,
full wiring harness down to the individual conductors.
5 legs x 3 DOF, Hiwonder HX-30HM bus servos, ESP32 + Hiwonder BusLinker V3.0, Raspberry Pi 5,
3S LiPo, star power hub of 6 x WAGO 221-415.

Run from Fusion: Utilities > Scripts and Add-Ins > + (add this file) > Run.
Creates a NEW design document. All dimensions in mm (converted to cm for the API).
Vendor STEP files are read from HW_DIR (Documents/Pentapod/hardware).

v5 changes (from v4)
  - servo bus: Hiwonder BusLinker V3.0 replaces the hand-built 74HCT126/74LVC1G125 buffer
  - power hub: 6 x WAGO 221-415 (V+, GND, SIG rails, 2 each) in a printed hexagonal holder in
    the middle of the top plate, star topology, one cut stock servo cable per leg
  - every leg branch drops through a slot right above its hip-servo connector, so the cut
    stock 20 cm cables reach; inside each leg the stock cables daisy-chain hip -> femur -> knee
    (femur -> knee lead now passes through a slot in the femur arm: 137 mm instead of 173 mm)
  - logic deck (r 82, was r 70): Raspberry Pi 5 + Active Cooler, ESP32 on a 3 x 7 cm perfboard,
    BusLinker, Pololu D36V50F5 for the Pi, Adafruit BNO085 IMU; LM2596 for the ESP32 on the
    top plate
  - 1000 uF bulk cap in the middle of the WAGO holder, soldered to the V+ and GND bridges
  - real CAD for every bought part (see HW list); the BusLinker is modelled from Hiwonder's
    dimension drawing because no CAD exists for it
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
# ---- top plate: WAGO hub, LM2596 ----
HOLDER_T = 2.0
Z_W = ZPT + HOLDER_T                 # WAGO underside
R_WF = 48.0                          # WAGO wire-entry face radius
WAGO_PORT_X = [3.6, 9.25, 14.85, 20.45, 26.1]
WAGO_PORT_Z = 3.0
RAIL_FACES = {'V+': (0, 180), 'GND': (60, 240), 'SIG': (120, 300)}   # opposite pairs, bridged
SLOT_R0, SLOT_R1 = 53.5, 61.0        # branch slot in the top plate, on each leg axis
BRK_R = 57.5                         # branch break-out radius (cable -> 3 conductors)
PWR_SLOT = (0.0, -56.0)              # battery leads come up here (12 x 8)
CAP_Z0 = ZPT + 5.0                   # 1000 uF can bottom (stands in the holder centre)
LM_C, LM_ANG = pol(70, 126), 126.0   # LM2596 (HW-411), long axis tangential
Z_LM = ZPT + 3.0
RING = (54.0, 57.5)                  # conductor ring radii on the top plate
Z_RING0, DZ_RING = ZPT + 7.5, 0.9
# ---- logic deck ----
PI_X0, PI_Y0 = -50.5, -28.0          # Pi 5 board corner (85 x 56)
PI_Z0 = DECK_Z1 + 5.0                # board underside
PI_HOLES = [(PI_X0 + 3.5, PI_Y0 + 3.5), (PI_X0 + 61.5, PI_Y0 + 3.5),
            (PI_X0 + 3.5, PI_Y0 + 52.5), (PI_X0 + 61.5, PI_Y0 + 52.5)]
PB_C = (2.0, 51.0)                   # ESP32 carrier (3 x 7 cm perfboard), long axis along x
PB_Z0 = DECK_Z1 + 5.0; PB_TOP = PB_Z0 + 1.8
PB_HOLES = [(PB_C[0] + sx*32.8, PB_C[1] + sy*12.8) for sx in (-1, 1) for sy in (-1, 1)]
ESP_C = (PB_C[0] - 0.88, PB_C[1] + 1.27, PB_TOP + 10.0)   # on 2 x 19 female headers
BL_C, BL_Z0 = (0.0, -49.5), DECK_Z1 + 5.0                 # BusLinker V3.0
POL_C, POL_Z = (-65.8, -9.3), DECK_Z1 + 6.57              # Pololu D36V50F5 (flipped, caps up)
IMU_C, IMU_Z0 = (-48.3, 41.45), DECK_Z1 + 5.0             # Adafruit BNO085
HOLE_A, HOLE_B, HOLE_C = (-58.0, -36.0, 6.0), (0.0, -71.0, 5.5), (-40.0, 60.0, 5.0)
Z_UNDECK = DECK_Z1 + 2.0             # wires under the perfboard
WIRE_BR, WIRE_C, WIRE_PWR, WIRE_PAIR, WIRE_SIG, WIRE_USB = 1.6, 0.7, 1.6, 1.2, 1.0, 1.6

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

    # ---- stock servo cables that move with the coxa / femur ----
    out['Coxa'] += [(tube(LEAD_HF_COXA, WIRE_BR), 'Wire - hip>femur stock cable (coxa part)'),
                    (tube(LEAD_FK_COXA, WIRE_BR), 'Wire - femur>knee stock cable (coxa part)')]
    out['Femur'] += [(tube(LEAD_FK_FEMUR, WIRE_BR), 'Wire - femur>knee stock cable (femur part)')]
    return out


# ======================= chassis =======================
def lm_M():
    a = math.radians(LM_ANG)
    t = (-math.sin(a), math.cos(a), 0.0); n = (math.cos(a), math.sin(a), 0.0)
    return place([(-n[0], -n[1], 0.0), (0, 0, 1), t], (10.3, -0.05, -21.7), (LM_C[0], LM_C[1], Z_LM))


def wago_M(f):
    a = math.radians(f)
    cols = [(-math.sin(a), math.cos(a), 0), (-math.cos(a), -math.sin(a), 0), (0, 0, 1)]
    return place(cols, (14.85, 0, 0), (R_WF*math.cos(a), R_WF*math.sin(a), Z_W))


def hexprism(apothem, z0, z1):
    h = None
    for a in (0, 60, 120):
        s = rbox(0, 0, (z0+z1)/2, 2*apothem, 400, z1-z0, a)
        h = s if h is None else I(h, s)
    return h


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
    for a in (30, 150, 270):                                                         # hub holder screws
        x, y = pol(18, a); D(topp, cyl((x, y, GZT-1), (x, y, ZPT+1), 1.25))
    Mlm = lm_M()
    for hx, hz in ((2.3, -6.3), (18.3, -36.8)):                                      # LM2596 bosses
        p = xf(Mlm, (hx, 0, hz))
        U(topp, cyl((p[0], p[1], ZPT - 0.5), (p[0], p[1], Z_LM), 2.6))
        D(topp, cyl((p[0], p[1], ZPT - 2.5), (p[0], p[1], Z_LM + 1), 1.25))
    items += [(botp, 'Bottom Plate'), (topp, 'Top Plate')]
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36)
        items.append((D(cyl((x, y, -GZB), (x, y, GZT), 4.5), cyl((x, y, -40), (x, y, 40), 1.7)), 'Plate Standoff %d' % (i+1)))

    # WAGO hub holder: floor, hexagonal back-stop with the bulk-cap pocket, dividing walls
    hold = U(cyl((0, 0, ZPT), (0, 0, Z_W), 47.0), hexprism(R_WF - 18.1 - 0.3, Z_W, ZPT + 7.0))
    for a in (30, 90, 150, 210):
        x, y = pol(38.0, a)
        U(hold, rbox(x, y, (Z_W + ZPT + 8.0)/2, 14.0, 1.6, ZPT + 8.0 - Z_W, a))
    D(hold, cyl((0, 0, Z_W), (0, 0, ZPT + 9), 5.6))                                   # cap pocket
    for a in (270, 330):                                                             # cap-lead grooves
        x, y = pol(20.0, a)
        D(hold, rbox(x, y, (Z_W + ZPT + 9)/2, 40.0, 2.4, ZPT + 9 - Z_W, a))
    for a in (30, 150, 270):
        x, y = pol(18, a)
        D(hold, cyl((x, y, ZPT - 1), (x, y, ZPT + 9), 1.7), cyl((x, y, ZPT + 4.5), (x, y, ZPT + 9), 3.0))
    items.append((hold, 'WAGO Hub Holder'))

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
    bosses = [(PI_HOLES, PI_Z0, 3.2, 1.1),                                           # M2.5 thread-forming
              (PB_HOLES, PB_Z0, 2.6, 0.8),                                           # M2
              ([(BL_C[0] + s*18, BL_C[1] + t*14) for s in (-1, 1) for t in (-1, 1)], BL_Z0, 3.0, 1.1),
              ([xf(pololu_M(), (hx, hy, 0))[:2] for hx, hy in ((2.16, 2.16), (2.16, 23.24), (23.24, 2.16))],
               POL_Z - 1.62, 2.0, 0.8),
              ([(IMU_C[0] + s*10.16, IMU_C[1] + t*8.89) for s in (-1, 1) for t in (-1, 1)], IMU_Z0, 3.0, 1.1)]
    for pts, ztop, rb, rp in bosses:
        for hx, hy in pts:
            U(deck, cyl((hx, hy, DECK_Z1), (hx, hy, ztop), rb))
            D(deck, cyl((hx, hy, DECK_Z1 - 2), (hx, hy, ztop + 1), rp))
    for hx, hy, hr in (HOLE_A, HOLE_B, HOLE_C):
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


def esp_M():      # USB end towards +x; header row "J2" (model -x) towards -y
    return place([(0, 1, 0), (0, 0, 1), (1, 0, 0)], (0, 0, 0), ESP_C)


def usb_M():      # right-angle plug into the Pi 5 USB-C, cable heading -x
    return place([(-1, 0, 0), (0, -1, 0), (0, 0, 1)], (8.6, -23.4, 3.2),
                 (PI_X0 + 11.2, PI_Y0 - 1.63 + 6.6, PI_Z0 + 2.9))


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
    add('Inline ATM fuse holder (closed, no wires).step', 'Inline ATM fuse holder (20 A)', place(I3, (-1.85, 5.46, 0), FUSE_C))
    for f in (0, 60, 120, 180, 240, 300):
        rail = [k for k, v in RAIL_FACES.items() if f in v][0]
        add('Wago 221-415.step', 'WAGO 221-415 %s (%d deg)' % (rail, f), wago_M(f))
    add('Cap radial D10 H20 black.step', '1000uF 25V bulk capacitor', place(I3, (0, 0, 0), (0, 0, CAP_Z0)))
    add('HW-411.STEP', 'LM2596 buck (HW-411) 5 V for ESP32', lm_M())
    add('RaspberryPi5 (simplified).step', 'Raspberry Pi 5', place(I3, (0, 0, 0), (PI_X0, PI_Y0, PI_Z0)))
    add('Pi5 Active Cooler (Pi frame).step', 'Raspberry Pi Active Cooler', place(I3, (0, 0, -1.6), (PI_X0, PI_Y0, PI_Z0)))   # seated on the SoC
    add('USB-C right angle pigtail (plug end).step', 'USB-C right-angle power pigtail', usb_M())
    add('Pololu D36V50Fx.step', 'Pololu D36V50F5 (5 V 5.5 A, Pi)', pololu_M())
    add('pcb_3x7.step', 'ESP32 carrier perfboard 3x7 cm', place(I3, (0, 0, -1.7), (PB_C[0], PB_C[1], PB_Z0)))
    add('ESP32-WROOM-32D-Assembly.STEP', 'ESP32-DevKitC', esp_M())
    add('Adafruit BNO085 STEMMA QT.step', 'Adafruit BNO085 IMU', place(I3, (0, 0, 0), (IMU_C[0], IMU_C[1], IMU_Z0)))
    return L


def header_socket_mats():
    """2 x 19 female header sockets for the ESP32 (single-socket model, patterned)."""
    out = []
    for row in (-11.43, 13.97):
        for j in range(19):
            out.append(place(I3, (0, 0, 0), (PB_C[0] - 23.74 + 2.54*j, PB_C[1] + row, PB_TOP + 7.5)))
    return out


SOCKET_STEP = 'Header-Female-round.step'


# ======================= BusLinker V3.0 (modelled from Hiwonder's dimension drawing) =======================
def bl_box(x0, x1, y0, y1, z0, z1):
    """Box in drawing coordinates (x right, y up, origin board centre, z up from board underside)."""
    return bx(BL_C[0] - y1, BL_C[0] - y0, BL_C[1] + x0, BL_C[1] + x1, BL_Z0 + z0, BL_Z0 + z1)


def bl_pt(xd, yd, z): return (BL_C[0] - yd, BL_C[1] + xd, BL_Z0 + z)


def buslinker_geometry():
    pcb = bl_box(-16.5, 16.5, -21, 21, 0, 1.6)
    for sx in (-14, 14):
        for sy in (-18, 18):
            p = bl_pt(sx, sy, 0)
            D(pcb, cyl((p[0], p[1], p[2] - 1), (p[0], p[1], p[2] + 3), 1.5))
    B = [(pcb, 'BusLinker V3.0 - PCB 33x42'),
         (bl_box(-18.0, -7.5, 7.5, 16.8, 1.6, 12.6), 'BusLinker V3.0 - DC jack 5.5x2.1'),
         (bl_box(-17.0, -9.0, -4.2, 5.9, 1.6, 11.6), 'BusLinker V3.0 - power terminal block'),
         (bl_box(-17.2, -9.7, -15.1, -6.1, 1.6, 4.8), 'BusLinker V3.0 - USB-C'),
         (bl_box(11.0, 16.5, 8.2, 15.5, 1.6, 8.0), 'BusLinker V3.0 - 5264 servo port'),
         (bl_box(11.0, 15.6, -3.4, 3.6, 1.6, 7.6), 'BusLinker V3.0 - PH2.0 servo port'),
         (bl_box(12.0, 15.0, -11.7, -7.3, 1.6, 5.0), 'BusLinker V3.0 - 1.25 servo port'),
         (bl_box(-4.0, 2.0, -13.0, -7.0, 1.6, 2.5), 'BusLinker V3.0 - USB-UART IC'),
         (bl_box(0.5, 8.8, -13.0, -9.2, 1.6, 4.1), 'BusLinker V3.0 - mode jumper header base'),
         (bl_box(0.5, 8.8, -19.7, -17.4, 1.6, 4.1), 'BusLinker V3.0 - TTL header base')]
    for k in range(3):
        for yd in (-12.4, -9.8, -18.55):
            p = bl_pt(1.9 + 2.7*k, yd, 4.1)
            B.append((cyl(p, (p[0], p[1], BL_Z0 + 10.1), 0.32), 'BusLinker V3.0 - pin'))
    return B


BL_TTL = [bl_pt(1.9 + 2.7*k, -18.55, 10.1) for k in range(3)]        # GND, TX, RX pin tops
BL_PWR = [bl_pt(-17.0, 3.3, 6.0), bl_pt(-17.0, -1.6, 6.0)]           # terminal + / - entries
BL_SERVO_S = bl_pt(13.75, 9.5, 8.0)                                  # 5264 "S" contact (plug from above)


def misc_geometry():
    out = []
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a + 36)
        out.append((cyl((x, y, POST_TOP - 3.0), (x, y, POST_TOP), 4.0), 'Magnet 8x3 (post) %d' % (i + 1)))
        out.append((cyl((x, y, POST_TOP + 0.1), (x, y, POST_TOP + 3.1), 4.0), 'Magnet 8x3 (dome) %d' % (i + 1)))
        out.append((cyl((x, y, ZPB - 3 - 2.4), (x, y, POST_TOP - 8.7), 1.5), 'M3 tie rod %d' % (i + 1)))
    return out


# ======================= wiring =======================
# ESP32-DevKitC pins (header row, pin 1..19 from the antenna end); J2 at model -x, J3 at model +x
def esp_pin(row, n, z_under=True):
    x = -12.78 if row == 'J2' else 12.78
    p = xf(esp_M(), (x, 0, -22.86 + 2.54*(n - 1)))
    return (p[0], p[1], PB_Z0 - 0.6) if z_under else p


def pololu_pin(x_model, y_model=1.27):
    return xf(pololu_M(), (x_model, y_model, 0.0))


POL_VIN, POL_VIN2, POL_GND, POL_GND2, POL_VOUT = (11.43, 1.27), (11.43, 3.81), (13.97, 1.27), (16.51, 3.81), (19.05, 1.27)
PI_GPIO = {n: (PI_X0 + 8.35 + 2.54*((n - 1)//2), PI_Y0 + (53.75 if n % 2 == 0 else 51.25), PI_Z0 + 9.9) for n in range(1, 41)}


def port_pts(f, k):
    M = wago_M(f); x = WAGO_PORT_X[k]
    return xf(M, (x, 3.0, WAGO_PORT_Z)), xf(M, (x, -4.0, WAGO_PORT_Z))


class Ring:
    """Assigns every hub conductor its own level on one of two ring radii above the top plate."""
    def __init__(self): self.i = 0

    def slot(self):
        r, z = RING[self.i % 2], Z_RING0 + DZ_RING*self.i
        self.i += 1
        return r, z


def from_port(f, k, ring, dest_ang, tail):
    """Conductor: inside a WAGO port -> out in front -> its ring level -> arc -> tail points."""
    pin, front = port_pts(f, k)
    r, z = ring.slot()
    a0 = ang_of(front)
    pts = [pin, front, (r*math.cos(math.radians(a0)), r*math.sin(math.radians(a0)), front[2])]
    pts += arc(r, a0, dest_ang, z)
    return pts + tail(r, z)


def harness_geometry(report=None):
    """All chassis-fixed wiring, world coordinates. report: dict filled with centre-line lengths."""
    W = []
    rep = report if report is not None else {}

    def w(pts, r, name):
        W.append((tube(pts, r), name)); rep[name] = plen(pts)

    ring = Ring()
    PORT = {}                                # (rail, owner) -> (face, port)
    # every leg conductor goes to the nearest WAGO of its rail (<= 90 deg round the ring) so the
    # cut 20 cm stock cables reach; battery / BusLinker / Pololu wires take the remaining ports
    alloc = {0: ['bridge', 306, 'busl', 18, 90], 180: [162, 'pololu', 'batt', 234, 'bridge'],
             60: ['bridge', 18, 'busl', 'pololu', 90], 240: [162, 'batt', 234, 306, 'bridge'],
             120: ['bridge', 90, 162, None, None], 300: [234, 'busl', 306, 18, 'bridge']}
    for f, owners in alloc.items():
        rail = [k for k, v in RAIL_FACES.items() if f in v][0]
        for k, o in enumerate(owners):
            if o is not None:
                PORT[(rail, o, f if o == 'bridge' else None)] = (f, k)

    def port(rail, owner):
        if owner == 'bridge':
            return [v for kk, v in PORT.items() if kk[0] == rail and kk[1] == 'bridge']
        return PORT[(rail, owner, None)]

    # ---- 5 leg branches: cut stock servo cable -> break-out -> 3 conductors to the rails ----
    for a in LEG_ANG:
        n = LEG_ANG.index(a) + 1
        cab = [rot2(p, a) for p in BRANCH_LEG]
        w(cab, WIRE_BR, 'Wire - leg %d branch (cut stock cable)' % n)
        brk = cab[-1]
        for rail in ('V+', 'GND', 'SIG'):
            f, k = port(rail, a % 360)
            pin, front = port_pts(f, k); r, z = ring.slot()
            a0 = ang_of(front)
            pts = [pin, front, (r*math.cos(math.radians(a0)), r*math.sin(math.radians(a0)), front[2])]
            pts += arc(r, a0, a, z) + [(brk[0], brk[1], z), brk]
            w(pts, WIRE_C, 'Wire - leg %d %s conductor' % (n, rail))
            rep['leg %d branch total (%s)' % (n, rail)] = plen(cab) + plen(pts)
        # hip -> femur stock cable, chassis part (under the bottom plate to the yaw axis)
        w([rot2(p, a) for p in LEAD_HF_CHASSIS], WIRE_BR, 'Wire - hip>femur stock cable (chassis part) %d' % n)

    # ---- battery side (between the plates) ----
    zb = TRAY_Z_IN + 24.3 + 4.7
    w([(51.5, 0, TRAY_Z_IN + 24 + 3.2), (53, 0, zb), (49.5, 0, zb)], WIRE_PAIR + 0.6, 'Wire - battery main lead (12 AWG pair)')
    fl, fr = (FUSE_C[0] - 22.05, FUSE_C[1], 0.0), (FUSE_C[0] + 22.05, FUSE_C[1], 0.0)
    w([(-50.5, 2.5, zb), (-53, 8, zb), (fl[0] - 8, fl[1], 0), (fl[0] + 0.5, fl[1], 0)], WIRE_PWR, 'Wire - +BATT to fuse (14 AWG)')
    sock = ZPB - 6 + 21.17
    w([(fr[0] - 0.5, fr[1], 0), (fr[0] + 4, fr[1], 0), (7, 8, 6), (12, -28, 6), (12, -50, 0), (3.5, -66, sock + 3),
       (3.5, -68, sock - 0.5)], WIRE_PWR, 'Wire - fuse to loop key (14 AWG)')
    # loop-key plug: wire loop between its solder cups
    zp = ZPB - 6 - 16.1 - 4.0
    w([(-3.5, -68, zp + 0.5), (-5, -68, zp - 6), (-2, -68, zp - 10), (2, -68, zp - 10), (5, -68, zp - 6), (3.5, -68, zp + 0.5)],
      WIRE_PWR, 'Wire - loop-key shorting loop (14 AWG)')

    def to_port(f, k):
        pin, front = port_pts(f, k)
        return lambda r, z: [(front[0]*r/math.hypot(front[0], front[1]), front[1]*r/math.hypot(front[0], front[1]), z),
                             (front[0]*r/math.hypot(front[0], front[1]), front[1]*r/math.hypot(front[0], front[1]), front[2]),
                             front, pin]

    def up_slot(x, z_start):
        return [(x, PWR_SLOT[1] - 2, z_start), (x, PWR_SLOT[1], 10.0), (x, PWR_SLOT[1], ZPT + 3)]

    # loop key -> hub V+ ; battery negative -> hub GND
    f, k = port('V+', 'batt'); r, z = ring.slot()
    pin, front = port_pts(f, k); af = ang_of(front)
    pts = [(-3.5, -68, sock - 0.5), (-3.5, -66, sock + 3)] + up_slot(-3.0, 4.0) + [(-3.0, PWR_SLOT[1], z)]
    pts += arc(r, ang_of((-3.0, PWR_SLOT[1])), af, z) + to_port(f, k)(r, z)[1:]
    w(pts, WIRE_PWR, 'Wire - loop key to hub V+ (14 AWG)')
    f, k = port('GND', 'batt'); r, z = ring.slot()
    pin, front = port_pts(f, k); af = ang_of(front)
    pts = [(-50.5, -2.5, zb), (-53, -10, zb), (-20, -22, 6), (3, -30, 8), (3, -50, 10)] + up_slot(3.0, 10.0)[1:]
    pts += [(3.0, PWR_SLOT[1], z)] + arc(r, ang_of((3.0, PWR_SLOT[1])), af, z) + to_port(f, k)(r, z)[1:]
    w(pts, WIRE_PWR, 'Wire - battery negative to hub GND (14 AWG)')

    # ---- rail bridges (opposite WAGOs) + bulk capacitor ----
    bridge_mid = {}
    for rail in ('V+', 'GND', 'SIG'):
        (fa, ka), (fb, kb) = sorted(port(rail, 'bridge'))
        pa, fa_pt = port_pts(fa, ka); pb, fb_pt = port_pts(fb, kb)
        r, z = ring.slot()
        a0, a1 = ang_of(fa_pt), ang_of(fb_pt)
        ra = (r*math.cos(math.radians(a0)), r*math.sin(math.radians(a0)))
        rb = (r*math.cos(math.radians(a1)), r*math.sin(math.radians(a1)))
        pts = [pa, fa_pt, (ra[0], ra[1], fa_pt[2]), (ra[0], ra[1], z)] + arc(r, a0, a1, z) + [(rb[0], rb[1], fb_pt[2]), fb_pt, pb]
        amid = a0 + ((a1 - a0 + 540) % 360 - 180)/2
        bridge_mid[rail] = (r*math.cos(math.radians(amid)), r*math.sin(math.radians(amid)), z)
        w(pts, WIRE_PWR if rail != 'SIG' else WIRE_C, 'Wire - %s rail bridge' % rail)
    for rail, a, sx in (('V+', 270, -2.5), ('GND', 330, 2.5)):
        m = bridge_mid[rail]
        g0, g1 = pol(6.0, a), pol(36.0, a)
        w([(sx, 0, CAP_Z0 - 2.6), (g0[0], g0[1], ZPT + 3.2), (g1[0], g1[1], ZPT + 3.2), (m[0]*0.93, m[1]*0.93, ZPT + 3.2),
           (m[0]*0.97, m[1]*0.97, m[2])], 0.6, 'Wire - bulk cap lead to %s bridge' % rail)

    # ---- hub -> deck: BusLinker power + signal (hole B), Pololu feed (hole A) ----
    def up_through(hole, z_from, z_to):
        return [(hole[0], hole[1], z_from), (hole[0], hole[1], z_to)]

    def hub_to_hole(rail, owner, hole, tail, r_w, name):
        f, k = port(rail, owner)
        pin, front = port_pts(f, k); r, z = ring.slot()
        a0, ah = ang_of(front), ang_of(hole)
        pts = [pin, front, (r*math.cos(math.radians(a0)), r*math.sin(math.radians(a0)), front[2])]
        pts += arc(r, a0, ah, z) + up_through(hole, z, Z_UNDECK) + tail
        w(pts, r_w, name)

    zt = BL_Z0 + 14.0
    hub_to_hole('V+', 'busl', HOLE_B, [(HOLE_B[0], HOLE_B[1], BL_PWR[0][2]), (BL_PWR[0][0], BL_PWR[0][1] - 4, BL_PWR[0][2]),
                                       (BL_PWR[0][0], BL_PWR[0][1] + 2, BL_PWR[0][2])], WIRE_SIG, 'Wire - hub V+ to BusLinker power +')
    hub_to_hole('GND', 'busl', HOLE_B, [(HOLE_B[0] + 1.5, HOLE_B[1], BL_PWR[1][2]), (BL_PWR[1][0], BL_PWR[1][1] - 4, BL_PWR[1][2]),
                                        (BL_PWR[1][0], BL_PWR[1][1] + 2, BL_PWR[1][2])], WIRE_SIG, 'Wire - hub GND to BusLinker power -')
    s = BL_SERVO_S
    hub_to_hole('SIG', 'busl', HOLE_B, [(HOLE_B[0] - 1.5, HOLE_B[1], zt), (s[0], HOLE_B[1] + 4, zt), (s[0], s[1], zt),
                                        (s[0], s[1], s[2] + 0.5)], WIRE_C, 'Wire - hub SIG to BusLinker servo port')
    vin, gnd = pololu_pin(*POL_VIN), pololu_pin(*POL_GND)
    for rail, p, dx, nm in (('V+', vin, 0.0, 'Wire - hub V+ to Pololu VIN (18 AWG)'), ('GND', gnd, 2.5, 'Wire - hub GND to Pololu GND (18 AWG)')):
        hub_to_hole(rail, 'pololu', (HOLE_A[0] + dx, HOLE_A[1]),
                    [(HOLE_A[0] + dx, HOLE_A[1], POL_Z + 4), (p[0], p[1] - 6, POL_Z + 4), (p[0], p[1], POL_Z + 1)], WIRE_PAIR, nm)

    # ---- LM2596 (top plate) fed from the Pololu input pins, 5 V up to the ESP32 ----
    Mlm = lm_M()
    lm_in = [xf(Mlm, (4.0, 1.65, -41.0)), xf(Mlm, (16.5, 1.65, -41.0))]
    lm_out = [xf(Mlm, (4.0, 1.65, -2.5)), xf(Mlm, (16.5, 1.65, -2.5))]
    vin2, gnd2 = pololu_pin(*POL_VIN2), pololu_pin(*POL_GND2)
    for i, (src, dst, nm) in enumerate(((vin2, lm_in[0], 'Wire - Pololu VIN tap to LM2596 IN+'),
                                        (gnd2, lm_in[1], 'Wire - Pololu GND tap to LM2596 IN-'))):
        r, z = ring.slot()
        hx = HOLE_A[0] - 2.5 + 1.8*i
        pts = [(src[0], src[1], POL_Z + 1), (src[0], src[1] - 4, POL_Z + 5.5), (hx, HOLE_A[1], POL_Z + 5.5),
               (hx, HOLE_A[1], Z_UNDECK), (hx, HOLE_A[1], z)]
        pts += arc(r, ang_of(HOLE_A), ang_of(dst), z) + [(dst[0], dst[1], z), (dst[0], dst[1], dst[2] + 0.5)]
        w(pts, WIRE_SIG, nm)
    e5, eg, eadc = esp_pin('J2', 19), esp_pin('J2', 14), esp_pin('J2', 5)
    zh = Z_LM + 16
    for i, (src, dst, nm) in enumerate(((lm_out[0], e5, 'Wire - LM2596 5V to ESP32 5V'), (lm_out[1], eg, 'Wire - LM2596 GND to ESP32 GND'),
                                        (lm_in[0], eadc, 'Wire - battery sense (LM2596 IN+) to ESP32 divider'))):
        hx = HOLE_C[0] - 2 + 2*i
        pts = [(src[0], src[1], src[2] + 0.5), (src[0], src[1], zh + 2*i), (hx, HOLE_C[1], zh + 2*i), (hx, HOLE_C[1], Z_UNDECK),
               (hx + 8, HOLE_C[1] - 8, Z_UNDECK), (dst[0] - 4, dst[1] + 2.5, Z_UNDECK), (dst[0], dst[1] + 2.5, Z_UNDECK), dst]
        w(pts, WIRE_SIG, nm)

    # ---- deck: Pololu -> USB-C pigtail -> Pi 5 ----
    vout = pololu_pin(*POL_VOUT)
    cut = xf(usb_M(), (29.5, -1.0, 3.2))
    w([cut, (cut[0] - 3, cut[1], cut[2]), (vout[0], cut[1] + 6, POL_Z + 3), (vout[0], vout[1] - 4, POL_Z + 3), (vout[0], vout[1], POL_Z + 1)],
      WIRE_USB, 'Wire - Pololu VOUT to USB-C pigtail')
    # ---- deck: ESP32 <-> BusLinker TTL, ESP32 <-> Pi UART, ESP32 <-> IMU ----
    tx = esp_pin('J3', 11)
    ttl = BL_TTL[1]
    w([tx, (tx[0], tx[1] + 4.5, Z_UNDECK), (40.5, tx[1] + 4.5, Z_UNDECK), (40.5, -40, Z_UNDECK), (24, -42, Z_UNDECK),
       (24, -42, ttl[2] + 4), (ttl[0], ttl[1], ttl[2] + 4), (ttl[0], ttl[1], ttl[2] + 0.5)], WIRE_PAIR, 'Wire - ESP32 UART2 to BusLinker TTL (3 wires)')
    pu = PI_GPIO[8]; eu = esp_pin('J2', 10)
    w([(pu[0], pu[1], pu[2] + 0.5), (pu[0], pu[1], pu[2] + 4), (-31, 32.5, pu[2] + 4), (-31, 32.5, Z_UNDECK),
       (eu[0], eu[1] - 3, Z_UNDECK), (eu[0], eu[1] - 0.5, Z_UNDECK), eu], WIRE_PAIR, 'Wire - Pi 5 UART (GPIO14/15) to ESP32 (3 wires)')
    q = (IMU_C[0] + 12.9, IMU_C[1], IMU_Z0 + 3.2)
    ei = esp_pin('J3', 6)
    w([q, (-34.5, q[1], q[2]), (-34.5, q[1], Z_UNDECK), (ei[0] - 3, ei[1] - 4, Z_UNDECK), (ei[0], ei[1] - 4, Z_UNDECK), ei],
      WIRE_SIG, 'Wire - BNO085 STEMMA QT cable to ESP32 I2C')
    return W


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
    elif which == 'harness':
        c = r.occurrences.addNewComponent(ident).component; c.name = 'Wiring Harness'; add(c, harness_geometry())
    elif which == 'misc':
        c = r.occurrences.addNewComponent(ident).component; c.name = 'Magnets & tie rods'; add(c, misc_geometry())
        c = r.occurrences.addNewComponent(ident).component; c.name = 'BusLinker V3.0 (modelled from Hiwonder drawing)'
        add(c, buslinker_geometry())


def stage_hardware(ident, names=None):
    r = root()
    hwc = find_occ(r, 'Bought parts')
    hwc = hwc.component if hwc else r.occurrences.addNewComponent(ident).component
    hwc.name = 'Bought parts'
    for f, n, M in hw_list():
        if names is None or any(n.startswith(x) for x in names):
            place_part(hwc, f, n, M)
    if names is None or 'sockets' in names:
        pattern_part(hwc, SOCKET_STEP, 'ESP32 female headers 2x19', header_socket_mats())


def run(_context: str):
    ident = adsk.core.Matrix3D.create()
    stage_legs_a(ident); stage_legs_b(ident); stage_servos(ident); stage_instances(ident)
    for w in ('chassis', 'dome', 'harness', 'misc'):
        stage_body(ident, w)
    stage_hardware(ident)
