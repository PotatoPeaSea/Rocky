"""Import the Freerouting session, add pours/planes and stitching vias, fill zones and save.

Usage: python3 finish_pcb.py [session.ses]
"""
import math
import os
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(__file__))
from gen_pcb import K, CX, CY, R_OUT, inside_board  # noqa: E402

PCB = '/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb'
F, B, IN1, IN2 = pcbnew.F_Cu, pcbnew.B_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu


def zone(board, net, layer, pts, prio=0, name='', clearance=0.25, solid_tht=False, minw=0.25):
    z = pcbnew.ZONE(board)
    z.SetLayer(layer)
    z.SetNet(board.FindNet(net))
    z.SetAssignedPriority(prio)
    z.SetZoneName(name or net)
    z.SetLocalClearance(pcbnew.FromMM(clearance))
    z.SetMinThickness(pcbnew.FromMM(minw))
    z.SetThermalReliefGap(pcbnew.FromMM(0.4))
    z.SetThermalReliefSpokeWidth(pcbnew.FromMM(0.6 if not solid_tht else 1.2))
    z.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
    z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    ol = z.Outline()
    ol.NewOutline()
    for p in pts:
        v = K(*p); ol.Append(v.x, v.y)
    board.Add(z)
    return z


def circle_pts(r, n=96):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def main(ses):
    board = pcbnew.LoadBoard(PCB)
    if ses:
        ok = pcbnew.ImportSpecctraSES(board, ses)
        print('SES import', ok)
    # remove old zones (idempotent re-runs)
    for z in list(board.Zones()):
        if not z.GetIsRuleArea():
            board.Remove(z)
    big = circle_pts(R_OUT + 2)
    zone(board, 'GND', IN1, big, 0, 'GND plane')
    zone(board, '/VSERVO', IN2, big, 0, 'VSERVO plane', solid_tht=True)
    zone(board, 'GND', F, big, 0, 'GND top')
    zone(board, 'GND', B, big, 0, 'GND bottom')
    # local high-current pours (higher priority than the GND pours)
    for L in (F, B):
        zone(board, '/VBAT_IN', L, [(24.0, -61.5), (30.0, -61.5), (30.0, -52.0), (33.5, -49.2), (44.0, -49.2),
                                    (44.0, -44.6), (30.0, -44.6), (24.0, -50.0)], 5, 'VBAT_IN', solid_tht=True)
        zone(board, '/VBAT_F', L, [(36.0, -33.6), (49.8, -33.6), (49.8, -40.6), (44.0, -40.6), (44.0, -40.0),
                                   (36.0, -40.0)], 5, 'VBAT_F', solid_tht=True)
    zone(board, '/VSERVO_SW', F, [(58.6, -35.0), (62.9, -35.0), (62.9, -29.2), (61.9, -29.2), (61.9, -26.4),
                                  (59.6, -26.4), (59.6, -28.7), (58.6, -28.7)], 5, 'VSERVO_SW')
    zone(board, '/VSERVO', F, [(60.9, -17.2), (65.3, -17.2), (65.3, -23.3), (60.9, -23.3)], 5, 'VSERVO top')
    zone(board, '/VBAT_P', F, [(50.75, -37.25), (60.0, -37.25), (60.0, -36.0), (61.6, -36.0), (61.6, -40.4),
                               (50.75, -40.4)], 5, 'VBAT_P')
    zone(board, '/VBAT_P', B, [(50.75, -36.0), (61.6, -36.0), (61.6, -41.0), (50.75, -41.0)], 5, 'VBAT_P bottom')
    zone(board, '+5V', F, [(-28.4, -44.4), (-24.7, -44.4), (-24.7, -52.6), (-19.6, -52.6), (-17.8, -56.6),
                           (-8.4, -56.6), (-8.4, -57.9), (-30.2, -57.9), (-30.2, -52.0), (-28.4, -50.5)], 5, '+5V')

    # high-current THT pads: solid connection to every pour
    for ref in ('J1', 'F1', 'J2', 'J5', 'J6', 'J7', 'J8', 'J9', 'C6', 'C7'):
        fp = board.FindFootprintByReference(ref)
        for p in fp.Pads():
            if p.GetNetname() in ('GND', '/VBAT_IN', '/VBAT_F', '/VSERVO', '+5V'):
                p.SetLocalZoneConnection(pcbnew.ZONE_CONNECTION_FULL)
    for ref in ('J3', 'U4', 'J12', 'J10'):
        for p in board.FindFootprintByReference(ref).Pads():
            if p.GetNetname() == 'GND':
                p.SetLocalZoneConnection(pcbnew.ZONE_CONNECTION_FULL)

    # GND stitching vias on a grid (connect top/bottom pours to In1)
    obst = []
    for fp in board.GetFootprints():
        for p in fp.Pads():
            obst.append((p.GetBoundingBox(), p.GetNetCode()))
        if fp.GetReference().startswith('H'):
            obst.append((fp.GetBoundingBox(False, False), -1))
    for t in board.GetTracks():
        obst.append((t.GetBoundingBox(), t.GetNetCode()))
    gnd = board.FindNet('GND')
    n = 0
    step = 5.0
    for i in range(-16, 17):
        for j in range(-16, 17):
            x, y = i * step + (step / 2 if j % 2 else 0), j * step
            if not inside_board(x, y, 1.2):
                continue
            if any(abs(x - bx) < 4 and abs(y - by) < 4 for bx, by in ((18.0, -63.5), (-18.0, -63.5))):
                continue
            v = K(x, y)
            ok = True
            for bb, nc in obst:
                bx0, by0, bx1, by1 = bb.GetX(), bb.GetY(), bb.GetRight(), bb.GetBottom()
                dx = max(bx0 - v.x, 0, v.x - bx1); dy = max(by0 - v.y, 0, v.y - by1)
                d = pcbnew.ToMM(math.hypot(dx, dy))
                if d < (0.75 if nc != gnd.GetNetCode() else 0.4):
                    ok = False; break
            if not ok:
                continue
            via = pcbnew.PCB_VIA(board)
            via.SetPosition(v); via.SetWidth(pcbnew.FromMM(0.6)); via.SetDrill(pcbnew.FromMM(0.3))
            via.SetNet(gnd)
            board.Add(via)
            n += 1
    print('stitching vias', n)
    for d in board.GetDrawings():
        if isinstance(d, pcbnew.PCB_TEXT) and d.GetText().startswith('github'):
            d.SetText('Rocky pentapod  |  M. Lin  |  2026-10')
    import silk
    silk.place_refs(board)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    pcbnew.SaveBoard(PCB, board)
    print('saved')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
