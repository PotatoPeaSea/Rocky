"""Hand-placed power copper (wide tracks, via fields) and GND / VSERVO via fan-out.

Runs on the placed board before autorouting. Coordinates are in the robot frame (see placement.py).
"""
import math
import os
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(__file__))
from gen_pcb import K, CX, CY, R_OUT, CUTOUT, NOTCH_X, NOTCH_TOP, inside_board  # noqa: E402

PCB = '/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb'
F, B, IN1, IN2 = pcbnew.F_Cu, pcbnew.B_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu
board = pcbnew.LoadBoard(PCB)
NET = {n: board.FindNet(n) for n in ['GND', '/VBAT_IN', '/VBAT_F', '/VBAT_P', '/VSERVO_SW', '/VSERVO', '/B5_SW', '+5V']}


def track(net, pts, w, layer=F):
    for a, b in zip(pts[:-1], pts[1:]):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(K(*a)); t.SetEnd(K(*b))
        t.SetWidth(pcbnew.FromMM(w)); t.SetLayer(layer); t.SetNet(NET[net])
        t.SetLocked(True)
        board.Add(t)


def via(net, x, y, d=0.8, drill=0.4):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(K(x, y))
    v.SetWidth(pcbnew.FromMM(d)); v.SetDrill(pcbnew.FromMM(drill))
    v.SetNet(NET[net]); v.SetLocked(True)
    board.Add(v)
    return v


# ---------------------------------------------------------------- 20 A battery path
# J1+ -> F1 pin 1 (both outer layers; J1 and F1 are THT)
for L in (F, B):
    track('/VBAT_IN', [(27.0, -58.8), (27.0, -51.0), (31.0, -47.0), (38.3, -46.96), (41.7, -46.96)], 4.0, L)
# F1 pin 2 -> Q1 drain (F.Cu), plus B.Cu return through a via row above the drain pad
track('/VBAT_F', [(38.3, -37.04), (41.7, -37.04), (46.5, -37.6)], 4.0, F)
track('/VBAT_F', [(38.3, -37.04), (41.7, -37.04), (44.6, -34.9), (49.0, -34.9)], 2.5, B)
for x in (45.6, 46.8, 48.0):
    via('/VBAT_F', x, -34.9)
track('/VBAT_F', [(45.6, -34.9), (48.0, -34.9)], 1.2, F)
track('/VBAT_F', [(46.8, -34.9), (46.8, -36.2)], 1.6, F)
# Q1 source -> Q2 source
track('/VBAT_P', [(51.29, -37.37), (51.29, -39.91)], 0.6, F)
track('/VBAT_P', [(51.29, -38.9), (59.9, -38.9)], 2.4, F)
track('/VBAT_P', [(58.59, -36.79), (61.13, -36.79)], 0.6, F)
track('/VBAT_P', [(59.9, -36.9), (59.9, -38.9)], 2.4, F)
for x in (53.6, 55.0, 56.4, 57.8):
    via('/VBAT_P', x, -38.9)
track('/VBAT_P', [(53.6, -38.9), (57.8, -38.9)], 2.4, B)
# Q2 drain -> shunt -> VSERVO via field (to the In2 VSERVO plane)
track('/VSERVO_SW', [(60.5, -32.0), (60.5, -29.5), (60.6, -27.6)], 2.0, F)
track('/VSERVO_SW', [(56.5, -29.55), (59.8, -29.3)], 0.5, F)
for (x, y) in [(61.5, -19.0), (62.7, -19.0), (63.9, -19.0), (61.5, -17.8), (62.7, -17.8), (63.9, -17.8),
               (64.7, -20.4), (64.7, -21.6), (64.7, -22.8)]:
    via('/VSERVO', x, y)
track('/VSERVO', [(62.13, -21.51), (62.7, -19.0), (62.7, -17.8)], 2.0, F)
track('/VSERVO', [(61.5, -19.0), (63.9, -19.0)], 1.0, F)
track('/VSERVO', [(61.5, -17.8), (63.9, -17.8)], 1.0, F)
track('/VSERVO', [(62.13, -21.51), (64.7, -21.6)], 1.6, F)
track('/VSERVO', [(64.7, -20.4), (64.7, -22.8)], 1.0, F)

# ---------------------------------------------------------------- 5 V buck power loop
track('/B5_SW', [(-7.56, -43.41), (-7.9, -45.9), (-9.6, -47.4)], 1.2, F)
track('/B5_SW', [(-10.0, -47.6), (-15.45, -47.0)], 2.5, F)
track('/B5_SW', [(-5.78, -45.6), (-7.8, -45.6)], 0.4, F)
track('+5V', [(-26.55, -47.0), (-26.55, -53.8), (-19.0, -53.8), (-17.4, -57.25)], 2.0, F)
track('+5V', [(-17.4, -57.25), (-9.3, -57.25)], 1.0, F)
track('+5V', [(-26.55, -53.8), (-28.96, -56.0), (-28.96, -59.5)], 2.0, F)


# ---------------------------------------------------------------- automatic fan-out (GND -> In1, VSERVO -> In2)
def bbox_dist(px, py, item_bb):
    x0, y0, x1, y1 = item_bb
    dx = max(x0 - px, 0, px - x1); dy = max(y0 - py, 0, py - y1)
    return math.hypot(dx, dy)


obstacles = []   # (bbox in robot coords, netcode)


def add_obstacle_pad(p):
    bb = p.GetBoundingBox()
    x0 = pcbnew.ToMM(bb.GetX()) - CX; x1 = pcbnew.ToMM(bb.GetRight()) - CX
    y1 = CY - pcbnew.ToMM(bb.GetY()); y0 = CY - pcbnew.ToMM(bb.GetBottom())
    obstacles.append(((x0, y0, x1, y1), p.GetNetCode(), p))


for fp in board.GetFootprints():
    for p in fp.Pads():
        add_obstacle_pad(p)
for t in board.GetTracks():
    bb = t.GetBoundingBox()
    obstacles.append(((pcbnew.ToMM(bb.GetX()) - CX, CY - pcbnew.ToMM(bb.GetBottom()),
                       pcbnew.ToMM(bb.GetRight()) - CX, CY - pcbnew.ToMM(bb.GetY())), t.GetNetCode(), t))
# mounting holes / bosses
for fp in board.GetFootprints():
    if fp.GetReference().startswith('H'):
        x, y = pcbnew.ToMM(fp.GetPosition().x) - CX, CY - pcbnew.ToMM(fp.GetPosition().y)
        obstacles.append(((x - 2.8, y - 2.8, x + 2.8, y + 2.8), -1, fp))
for (bx, by) in [(18.0, -63.5), (-18.0, -63.5)]:     # unused bosses: keep vias off them too
    obstacles.append(((bx - 3.2, by - 3.2, bx + 3.2, by + 3.2), -1, None))

VIA_R = 0.3


def free(x, y, net, own_pad, clr=0.28):
    if not inside_board(x, y):
        return False
    for bb, nc, item in obstacles:
        if item is own_pad:
            continue
        d = bbox_dist(x, y, bb)
        if nc == net and nc > 0:
            if isinstance(item, pcbnew.PCB_VIA) and d < 0.75:
                return False
            if d < 0.05 and isinstance(item, pcbnew.PAD) and item.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                return False      # never drop a via inside an SMD pad
            continue
        if d < VIA_R + clr:
            return False
    return True


def stub_free(x0, y0, x1, y1, net, own_pad, w=0.4):
    for k in range(1, 6):
        t = k / 6.0
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if not inside_board(x, y, 0.65):
            return False
        for bb, nc, item in obstacles:
            if item is own_pad or (nc == net and nc > 0):
                continue
            if bbox_dist(x, y, bb) < w / 2 + 0.22:
                return False
    return True


added = 0
failed = []
for fp in board.GetFootprints():
    for p in fp.Pads():
        net = p.GetNetname()
        if net not in ('GND', '/VSERVO') or p.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
            continue
        if not p.IsOnLayer(F):
            continue
        if fp.GetReference() == 'U7' and p.GetNumber() == '39':
            continue                       # ESP32 EP has its own via array
        if fp.GetReference() == 'U2' and p.GetNumber() == '9':
            continue
        if fp.GetReference() == 'R6':
            continue
        px, py = pcbnew.ToMM(p.GetPosition().x) - CX, CY - pcbnew.ToMM(p.GetPosition().y)
        sz = p.GetBoundingBox()
        hw = pcbnew.ToMM(sz.GetWidth()) / 2; hh = pcbnew.ToMM(sz.GetHeight()) / 2
        done = False
        for dist in (0.75, 1.1, 1.5, 2.0, 2.6):
            for ang in range(0, 360, 30):
                a = math.radians(ang)
                ex = abs(math.cos(a)) * hw + abs(math.sin(a)) * hh
                vx, vy = px + math.cos(a) * (ex + dist), py + math.sin(a) * (ex + dist)
                if free(vx, vy, p.GetNetCode(), p) and stub_free(px, py, vx, vy, p.GetNetCode(), p):
                    v = via(net if net == 'GND' else '/VSERVO', vx, vy, 0.6, 0.3)
                    track(net if net == 'GND' else '/VSERVO', [(px, py), (vx, vy)], 0.4, F)
                    obstacles.append(((vx - VIA_R, vy - VIA_R, vx + VIA_R, vy + VIA_R), p.GetNetCode(), v))
                    added += 1; done = True
                    break
            if done:
                break
        if not done:
            failed.append('%s.%s' % (fp.GetReference(), p.GetNumber()))
print('fan-out vias:', added, 'failed:', failed)
pcbnew.SaveBoard(PCB, board)
