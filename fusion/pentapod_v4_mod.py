
import math, os

try:
    import adsk.core, adsk.fusion
except ImportError:
    import adsk.core, adsk.fusion

S = 0.1
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
    d = (a1 - a0 + 540) % 360 - 180
    n = max(1, int(abs(d) // step))
    return [(r*math.cos(math.radians(a0 + d*i/n)), r*math.sin(math.radians(a0 + d*i/n)), z) for i in range(n+1)]


SVL, SVW = 45.0, 24.6
SOFF = 9.75
HF = 17.35
HORN_R, HORN_PCD_R, HORN_SCREW_R = 9.7, 6.5, 1.1
HUB_R, HUB_D = 4.6, 0.7
CASE_TOP, CASE_BOT = 15.45, 16.45
SERVO_STEP_ORIGIN = (-12.75, -13.75, 0.0)
CASE_HOLES_MAIN = [(8.45, 10.0), (8.45, -10.0)]
CASE_HOLES_AUX = [(8.45, 10.0), (8.45, -10.0), (15.95, 9.3), (15.95, -9.3)]

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

ZPT = GZT + PT
ZPB = -GZB - PT
RIM, H_DOME = 44.0, 70.0
RR = math.sqrt(175.0**2 - 115.0**2)
RO = (RR**2 + H_DOME**2) / (2*H_DOME); RI = RO - 2.4; CZD = RIM + H_DOME - RO
APEX_R = 40.0
POST_R, POST_TOP = 6.0, 78.0
POST_RAD = 88.0
DECK_Z0, DECK_T = 52.0, 3.5
DECK_Z1 = DECK_Z0 + DECK_T
BAY_X, BAY_Y = 56.5, 21.5
TRAY_W, TRAY_F, TRAY_D = 2.5, 3.0, 12.0
TRAY_Z_IN = ZPB - TRAY_D
TRAY_SCREWS = [(61.0, 0.0), (-61.0, 0.0), (0.0, 27.0), (0.0, -27.0)]
SWX, SWY = 0.0, -68.0
PI_CX, PI_CY = -8.0, 0.0
PI_STANDOFF = 5.0
PI_Z0 = DECK_Z1 + PI_STANDOFF
PI_HOLES = [(PI_CX - 42.5 + 3.5, PI_CY - 24.5), (PI_CX - 42.5 + 61.5, PI_CY - 24.5),
            (PI_CX - 42.5 + 3.5, PI_CY + 24.5), (PI_CX - 42.5 + 61.5, PI_CY + 24.5)]
PI_USBC = (PI_CX - 42.5 + 11.2, PI_CY - 28.0)
PI_GPIO = (PI_CX - 42.5 + 32.5, PI_CY + 24.5)
USBC_HOLE = (PI_USBC[0], -52.0)
UART_HOLE = (-18.0, 40.0)
Z_BRD = ZPT + 3.0
HUB_C, HUB_ANG = pol(50, 306), 36.0
BUF_C, BUF_ANG = pol(54, 54), 144.0
LM_C, LM_ANG = pol(55, 126), 216.0
POL_C, POL_ANG = pol(57, 234), 234.0
RING_R, DROP_R = 79.0, 75.0
Z_TRUNK = 47.5
WIRE_SERVO, WIRE_PWR, WIRE_PAIR, WIRE_SIG, WIRE_USB = 1.5, 2.25, 3.0, 1.25, 2.0

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


HIP_CONN = (R_HIP - (SVL - SOFF) - 1.0, 0.0, 0.0)
TOP_SLOT_L = (DROP_R*math.cos(math.radians(20)), DROP_R*math.sin(math.radians(20)))
BOT_SLOT_L = (DROP_R*math.cos(math.radians(-20)), DROP_R*math.sin(math.radians(-20)))
Z_UNDER = -ZA1 - 2.6


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
    for x0 in (112.0, 124.0):
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

    out['Coxa'] += [
        (tube([(R_HIP, 0, Z_UNDER), (118, 4, Z_UNDER), (PFX, 4, Z_UNDER), (PFX, 4, -CAPZ0-1.5)], WIRE_SERVO),
         'Wire - hip>femur servo lead (coxa part)'),
        (tube([(PFX, -4, -CAPZ0-1.5), (PFX, -4, Z_UNDER-1), (PFX, -25, Z_UNDER-1), (PFX, -25, PFZ)], WIRE_SERVO),
         'Wire - femur>knee servo lead (coxa part)')]
    out['Femur'] += [
        (tube([(PFX, -25, PFZ), (PFX+28, -25, PFZ), (PFX+28, -25, PFZ+16), (PFX+28, -8, PFZ+16),
               (PFX+28, -8, PFZ), (PFX+38, 0, PFZ), (KS_X0-1, 0, PFZ)], WIRE_SERVO),
         'Wire - femur>knee servo lead (femur part)')]
    return out


def chassis_geometry():
    items = []

    def plate(z0, z1, rz0, rz1, holes, slot_ang):
        pl = cyl((0, 0, z0), (0, 0, z1), 95)
        for a in LEG_ANG:
            x, y = pol(R_HIP, a); U(pl, cyl((x, y, z0), (x, y, z1), 24))
        for a in LEG_ANG:
            x, y = pol(R_HIP, a); D(pl, cyl((x, y, -40), (x, y, 40), PLATE_HOLE_R))
            x, y = pol(POST_RAD, a+36); D(pl, cyl((x, y, -40), (x, y, 40), 1.7))
            for u, w in holes:
                p = rot2((R_HIP - u, w, 0), a)
                D(pl, cyl((p[0], p[1], -40), (p[0], p[1], 40), 1.1))
            x, y = pol(DROP_R, a + slot_ang)
            D(pl, rbox(x, y, (z0+z1)/2, 10, 7, PT+2, a + slot_ang))
        for a in LEG_ANG:
            cx, cy = pol(R_HIP - (SVL/2 - SOFF), a)
            ring = D(rbox(cx, cy, (rz0+rz1)/2, SVL+2*CLR+4.8, SVW+2*CLR+4.8, rz1-rz0, a),
                     rbox(cx, cy, (rz0+rz1)/2, SVL+2*CLR, SVW+2*CLR, rz1-rz0+0.4, a))
            U(pl, ring)
        return pl

    botp = plate(ZPB, -GZB, -GZB, -GZB+5, CASE_HOLES_AUX, -20)
    D(botp, bx(-BAY_X, BAY_X, -BAY_Y, BAY_Y, ZPB-1, -GZB+1),
      rbox(SWX, SWY, -GZB-PT/2, 16.6, 8.6, PT+2))
    for x, y in TRAY_SCREWS:
        D(botp, cyl((x, y, ZPB-1), (x, y, -GZB+1), 2.0))
    topp = plate(GZT, ZPT, GZT-5, GZT, CASE_HOLES_MAIN, 20)
    D(topp, rbox(0, -40, GZT+PT/2, 30, 12, PT+2))
    items += [(botp, 'Bottom Plate'), (topp, 'Top Plate')]
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36)
        items.append((D(cyl((x, y, -GZB), (x, y, GZT), 4.5), cyl((x, y, -40), (x, y, 40), 1.7)), 'Plate Standoff %d' % (i+1)))
    tray = U(bx(-BAY_X-TRAY_W, BAY_X+TRAY_W, -BAY_Y-TRAY_W, BAY_Y+TRAY_W, TRAY_Z_IN-TRAY_F, ZPB-3),
             bx(-64, 64, -30, 30, ZPB-3, ZPB))
    D(tray, bx(-BAY_X, BAY_X, -BAY_Y, BAY_Y, TRAY_Z_IN, ZPB+1))
    for sx in (-30, 30):
        D(tray, bx(sx-1.75, sx+1.75, -11, 11, TRAY_Z_IN-TRAY_F-1, TRAY_Z_IN+1))
    for x, y in TRAY_SCREWS:
        D(tray, cyl((x, y, ZPB-5), (x, y, ZPB+1), 1.7))
    D(tray, rbox(SWX, SWY, ZPB-1.5, 18, 10, 4))
    items.append((tray, 'Battery Tray'))
    deck = cyl((0, 0, DECK_Z0), (0, 0, DECK_Z1), 70)
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36); nx, ny = pol(POST_RAD+3.5, a+36); mx, my = pol(74, a+36)
        low = D(cyl((x, y, ZPT), (x, y, DECK_Z0), POST_R), cyl((x, y, ZPT-1), (x, y, DECK_Z0+1), 1.7))
        up = D(cyl((x, y, DECK_Z1), (x, y, POST_TOP), POST_R),
               cyl((x, y, DECK_Z1-1), (x, y, POST_TOP-4), 1.7),
               cyl((x, y, POST_TOP-3.2), (x, y, POST_TOP+1), 4.1),
               rbox(nx, ny, POST_TOP-10, 7, 5.8, 2.7, a+36))
        items += [(low, 'Dome Post lower %d' % (i+1)), (up, 'Dome Post upper %d' % (i+1))]
        U(deck, rbox(mx, my, (DECK_Z0+DECK_Z1)/2, 28, 16, DECK_T, a+36), cyl((x, y, DECK_Z0), (x, y, DECK_Z1), 11))
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a+36)
        D(deck, cyl((x, y, DECK_Z0-1), (x, y, DECK_Z1+1), 1.7))
    for hx, hy in PI_HOLES:
        U(deck, cyl((hx, hy, DECK_Z1), (hx, hy, PI_Z0), 3.2))
        D(deck, cyl((hx, hy, DECK_Z1-1), (hx, hy, PI_Z0+1), 1.1))
    D(deck, cyl((USBC_HOLE[0], USBC_HOLE[1], DECK_Z0-1), (USBC_HOLE[0], USBC_HOLE[1], DECK_Z1+1), 7),
      cyl((UART_HOLE[0], UART_HOLE[1], DECK_Z0-1), (UART_HOLE[0], UART_HOLE[1], DECK_Z1+1), 6))
    for k in range(3):
        x, y = pol(52, 126 + 54*k + 54)
        D(deck, cyl((x, y, DECK_Z0-1), (x, y, DECK_Z1+1), 8))
    items.append((deck, 'Pi Deck'))
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


def _hub_pt(s, t, z):
    a = math.radians(HUB_ANG); ct, st = math.cos(a), math.sin(a)
    n = (st, -ct)
    return (HUB_C[0] + s*ct + t*n[0], HUB_C[1] + s*st + t*n[1], z)


def electronics_geometry():
    out = []
    zt = Z_BRD + 1.6
    out.append((bx(-30, 30, -17, 17, Z_BRD, zt), 'Perfboard - ESP32 carrier (2x19 female headers)'))
    out.append((rbox(BUF_C[0], BUF_C[1], Z_BRD + 0.8, 40, 30, 1.6, BUF_ANG), 'Perfboard - bus buffer (74HCT126 + 74LVC1G125)'))
    out.append((rbox(BUF_C[0], BUF_C[1], zt + 4, 24, 14, 8, BUF_ANG), 'Bus buffer ICs (envelope)'))
    hub = U(rbox(HUB_C[0], HUB_C[1], Z_BRD + 0.8, 36, 26, 1.6, HUB_ANG))
    out.append((hub, 'Perfboard - power hub'))
    cap = _hub_pt(-2, -1, zt)
    out.append((cyl(cap, (cap[0], cap[1], zt + 20), 6.5), '1000uF 25V capacitor'))
    for k in range(5):
        h = _hub_pt(-12 + 6*k, 10, zt + 3.25)
        out.append((rbox(h[0], h[1], h[2], 5, 6, 6.5, HUB_ANG), 'Molex 5267 header %d' % (k+1)))
    out.append((rbox(POL_C[0], POL_C[1], Z_BRD + 4.75, 25.4, 25.4, 9.5, POL_ANG), 'Pololu D36V50F5 5V 5.5A'))
    out.append((bx(-39, -1, 9, 21, -4.5, 4.5), 'ATM fuse holder (20 A)'))
    out.append((bx(SWX-8, SWX+8, SWY-4, SWY+4, ZPB-6-14, ZPB-6), 'XT60 loop-key plug'))
    out.append((bx(PI_USBC[0]-4.5, PI_USBC[0]+4.5, PI_USBC[1]-20, PI_USBC[1], PI_Z0+1.6, PI_Z0+1.6+6.5), 'USB-C plug'))
    for i, a in enumerate(LEG_ANG):
        x, y = pol(POST_RAD, a + 36)
        out.append((cyl((x, y, POST_TOP - 3.0), (x, y, POST_TOP), 4.0), 'Magnet 8x3 (post) %d' % (i + 1)))
        out.append((cyl((x, y, POST_TOP + 0.1), (x, y, POST_TOP + 3.1), 4.0), 'Magnet 8x3 (dome) %d' % (i + 1)))
        out.append((cyl((x, y, ZPB - 3 - 2.4), (x, y, POST_TOP - 8.7), 1.5), 'M3 tie rod %d' % (i + 1)))
    return out


ENVELOPES = {
    'Raspberry Pi 5': lambda: U(bx(PI_CX-42.5, PI_CX+42.5, PI_CY-28, PI_CY+28, PI_Z0, PI_Z0+1.6),
                                bx(PI_CX-42.5+2, PI_CX+45, PI_CY-26, PI_CY+26, PI_Z0+1.6, PI_Z0+17.6),
                                bx(PI_CX-42.5, PI_CX+42.5, PI_CY-26, PI_CY+26, PI_Z0-2, PI_Z0)),
    'ESP32-DevKitC V4': lambda: bx(-27.5, 27.5, -14.1, 14.1, ZPT + 10, ZPT + 24.3),
    '3S 2200mAh LiPo': lambda: bx(-54.5, 54.5, -17, 17, TRAY_Z_IN, TRAY_Z_IN + 24),
    'LM2596 buck (HW-411)': lambda: rbox(LM_C[0], LM_C[1], Z_BRD + 7, 43.4, 20.6, 14, LM_ANG),
    'XT60H-M battery lead': lambda: bx(-48, 48, -4.7, 4.7, TRAY_Z_IN + 24.3, TRAY_Z_IN + 24.3 + 9.4),
    'XT60 loop-key socket': lambda: bx(SWX-8, SWX+8, SWY-4, SWY+4, ZPB - 6, ZPB - 6 + 21.2),
}


def harness_geometry():
    W = []
    zb = TRAY_Z_IN + 24.3 + 4.7
    W.append((tube([(51.5, 0, TRAY_Z_IN + 24 + 3.2), (53, 0, zb), (49.5, 0, zb)], WIRE_PAIR),
              'Wire - battery main lead (12 AWG pair)'))
    W.append((tube([(-50.5, 2.5, zb), (-53, 8, zb), (-50, 15, 0), (-39.5, 15, 0)], WIRE_PWR), 'Wire - +BATT to fuse (14 AWG)'))
    sock_top = ZPB - 6 + 21.2 + 2.5
    W.append((tube([(0.5, 15, 0), (4, 15, 0), (7, 8, 6), (12, -28, 6), (12, -50, 0), (4.5, -64, sock_top)], WIRE_PWR),
              'Wire - fuse to loop key (14 AWG)'))
    za = Z_BRD + 1.6 + 3.0
    hub_in = _hub_pt(-18, -6, za)
    hub_in2 = _hub_pt(-18, 0, za)
    W.append((tube([(-4.5, -64, sock_top), (-6, -52, 6), (-6, -40, 10), (-6, -40, za), (-6, -51, za),
                    (hub_in2[0] - 5, hub_in2[1] - 2, za), hub_in2], WIRE_PWR), 'Wire - loop key to hub (14 AWG)'))
    W.append((tube([(-50.5, -2.5, zb), (-53, -10, zb), (-20, -22, 6), (3, -30, 8), (3, -40, 10), (3, -40, za),
                    (hub_in[0] - 3, hub_in[1] + 3, za), hub_in], WIRE_PWR), 'Wire - BATT negative to hub (14 AWG)'))
    order = sorted(LEG_ANG, key=lambda a: -abs(((a + 20) - 306 + 540) % 360 - 180))
    for k, a in enumerate(order):
        zk = 47.0 - 3.6*k
        h = _hub_pt(-12 + 6*order.index(a), 10, Z_BRD + 1.6 + 6.5 + 0.5)
        ang0 = math.degrees(math.atan2(h[1], h[0]))
        pts = [h, (h[0], h[1], zk)]
        pts += [(RING_R*math.cos(math.radians(ang0)), RING_R*math.sin(math.radians(ang0)), zk)]
        pts += arc(RING_R, ang0, a + 20, zk)[1:]
        sx, sy = pol(DROP_R, a + 20)
        pts += [(sx, sy, zk), (sx, sy, 0.0)]
        for p in [(62.5, 14, -2), (62.5, 4, -4), (HIP_CONN[0], 4, -4)]:
            pts.append(rot2(p, a))
        W.append((tube(pts, WIRE_SERVO), 'Wire - servo branch to leg %d' % (LEG_ANG.index(a) + 1)))
        q = [(62.5, -4, -8), (62.0, -14, -8), (BOT_SLOT_L[0], BOT_SLOT_L[1], -8),
             (BOT_SLOT_L[0], BOT_SLOT_L[1], Z_UNDER), (90, -12, Z_UNDER), (R_HIP, 0, Z_UNDER)]
        W.append((tube([rot2(p, a) for p in q], WIRE_SERVO), 'Wire - hip>femur servo lead (chassis part) %d' % (LEG_ANG.index(a) + 1)))
    p = _hub_pt(-10, -10, Z_BRD + 3)
    W.append((tube([p, (p[0], p[1], 45), (POL_C[0] + 10, POL_C[1] + 6, 45), (POL_C[0] + 10, POL_C[1] + 6, Z_BRD + 11)], WIRE_PAIR),
              'Wire - hub to Pi regulator (18 AWG pair)'))
    W.append((tube([(POL_C[0] - 2, POL_C[1] - 8, Z_BRD + 11), (USBC_HOLE[0], USBC_HOLE[1], 40), (USBC_HOLE[0], USBC_HOLE[1], DECK_Z1 + 3),
                    (PI_USBC[0], PI_USBC[1] - 21, PI_Z0 + 4.85)], WIRE_USB), 'Wire - 5V to Pi USB-C'))
    p = _hub_pt(-4, -10, Z_BRD + 3)
    lm_in = (LM_C[0] + 17.6*math.cos(math.radians(LM_ANG + 180)), LM_C[1] + 17.6*math.sin(math.radians(LM_ANG + 180)))
    W.append((tube([p, (p[0], p[1], Z_TRUNK), (lm_in[0], lm_in[1] - 2, Z_TRUNK), (lm_in[0], lm_in[1], Z_BRD + 16)], WIRE_SIG + 0.5),
              'Wire - hub to LM2596 (22 AWG pair)'))
    lm_out = (LM_C[0] + 17.6*math.cos(math.radians(LM_ANG)), LM_C[1] + 17.6*math.sin(math.radians(LM_ANG)))
    W.append((tube([(lm_out[0], lm_out[1], Z_BRD + 16), (-36, 22, Z_BRD + 16), (-29, 16.5, Z_BRD + 3)], WIRE_SIG + 0.5),
              'Wire - 5V to ESP32 (22 AWG pair)'))
    W.append((tube([(-20, 16.5, Z_BRD + 3), (-20, 18.5, Z_TRUNK), (UART_HOLE[0], UART_HOLE[1], Z_TRUNK),
                    (UART_HOLE[0], UART_HOLE[1], DECK_Z1 + 3), (PI_GPIO[0], PI_GPIO[1] + 6, PI_Z0 + 12),
                    (PI_GPIO[0], PI_GPIO[1] + 3.5, PI_Z0 + 1.6 + 8.5 + 2)], WIRE_SIG), 'Wire - ESP32 to Pi UART'))
    bi = (BUF_C[0] - 15*math.cos(math.radians(54)), BUF_C[1] - 15*math.sin(math.radians(54)))
    W.append((tube([(bi[0], bi[1], Z_BRD + 6), (20, 18.5, Z_BRD + 4)], WIRE_SIG), 'Wire - ESP32 to bus buffer'))
    hd = _hub_pt(12, -10, Z_BRD + 3)
    W.append((tube([(bi[0] + 6, bi[1] - 2, Z_BRD + 6), (bi[0] + 6, bi[1] - 2, Z_TRUNK), (44, -16, Z_TRUNK),
                    (hd[0], hd[1], Z_TRUNK), hd], WIRE_SIG), 'Wire - bus buffer to hub (servo data)'))
    return W


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


FLAT = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
INSERT_AX = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
PI_STEP = 'RaspberryPi5.step'
HARDWARE = [
    ('ESP32-WROOM-32D-Assembly.STEP', 'ESP32-DevKitC V4', FLAT, 0, 0.0, 0.0, ZPT + 10),
    ('3s Turnigy Li-Po battery (simplified).STEP', '3S 2200mAh LiPo', FLAT, 0, 0.0, 0.0, TRAY_Z_IN),
    ('HW-411.STEP', 'LM2596 buck (HW-411)', FLAT, LM_ANG, LM_C[0], LM_C[1], Z_BRD),
    ('XT60H-M_75mm wire.step', 'XT60H-M battery lead', FLAT, 0, 0.0, 0.0, TRAY_Z_IN + 24.3),
    ('XT60_Hembra.STEP', 'XT60 loop-key socket', [(1, 0, 0), (0, -1, 0), (0, 0, -1)], 0, SWX, SWY, ZPB - 6),
] + [('M3X5.0X6.0_HeatSetScrew.STEP', 'M3 heat-set insert (tray)', INSERT_AX, 0, x, y, ZPB) for x, y in TRAY_SCREWS]


def place_step(root, fname, name, cols, ang, cx, cy, zmin):
    path = os.path.join(HW_DIR, fname)
    if not os.path.isfile(path):
        return None
    n0 = root.occurrences.count
    import_into(root, path)
    occ = root.occurrences.item(root.occurrences.count - 1) if root.occurrences.count > n0 else None
    if occ is None: return None
    bb = occ.boundingBox
    lo = (bb.minPoint.x/S, bb.minPoint.y/S, bb.minPoint.z/S); hi = (bb.maxPoint.x/S, bb.maxPoint.y/S, bb.maxPoint.z/S)
    a = math.radians(ang); ca, sa = math.cos(a), math.sin(a)
    cols = [(c[0]*ca - c[1]*sa, c[0]*sa + c[1]*ca, c[2]) for c in cols]
    def R(p): return tuple(p[0]*cols[0][i] + p[1]*cols[1][i] + p[2]*cols[2][i] for i in range(3))
    pts = [R((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    mn = [min(p[i] for p in pts) for i in range(3)]; mx = [max(p[i] for p in pts) for i in range(3)]
    occ.transform2 = frame((cx - (mn[0]+mx[0])/2, cy - (mn[1]+mx[1])/2, zmin - mn[2]), *cols)
    occ.component.name = name
    return occ


def build_legs(root, ident):
    legOcc0 = root.occurrences.addNewComponent(rotz(LEG_ANG[0]))
    leg = legOcc0.component; leg.name = 'Leg'
    geo = leg_geometry()
    for name in ('Coxa', 'Femur', 'Tibia', 'Foot (TPU)'):
        o = leg.occurrences.addNewComponent(ident); o.component.name = name
        add(o.component, geo[name])
    srv = leg.occurrences.addNewComponent(ident).component; srv.name = 'Leg Servos'
    step = os.path.join(HW_DIR, 'Hiwonder HX-35HM v3.step')
    for name, tf in (('Hip servo', HIP_TF), ('Femur servo', FEM_TF), ('Knee servo', KNEE_TF)):
        mount = srv.occurrences.addNewComponent(servo_tf(*tf)); mount.component.name = name
        if not import_into(mount.component, step):
            add(srv, [(servo_box(*tf), name + ' (placeholder)')])
    for a in LEG_ANG[1:]:
        root.occurrences.addExistingComponent(leg, rotz(a))


def build_body(root, ident):
    for name, fn in (('Chassis', chassis_geometry), ('Dome', dome_geometry)):
        c = root.occurrences.addNewComponent(ident).component; c.name = name
        add(c, fn()[name])
    h = root.occurrences.addNewComponent(ident).component; h.name = 'Wiring Harness'
    add(h, harness_geometry())
    e = root.occurrences.addNewComponent(ident).component; e.name = 'Electronics & Hardware (simple)'
    add(e, electronics_geometry())


def build_hardware(root, ident):
    missing = []
    for hw in HARDWARE:
        if place_step(root, *hw) is None:
            missing.append(hw[1])
    pi = place_step(root, PI_STEP, 'Raspberry Pi 5', [(1, 0, 0), (0, 1, 0), (0, 0, 1)], 0, PI_CX, PI_CY, PI_Z0 - 2.0)
    if pi is None:
        missing.append('Raspberry Pi 5')
    if missing:
        e = root.occurrences.addNewComponent(ident).component; e.name = 'Envelopes (vendor STEP missing)'
        add(e, [(ENVELOPES[m](), m + ' (envelope)') for m in missing if m in ENVELOPES])
    return missing
