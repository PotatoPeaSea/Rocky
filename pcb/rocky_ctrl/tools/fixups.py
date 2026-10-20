"""Final hand fix-ups after autorouting: R20 +5V link, stitch any GND top-pour island without a via."""
import sys, os, math
import pcbnew
sys.path.insert(0, os.path.dirname(__file__))
from gen_pcb import K, CX, CY
PCB = '/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb'
b = pcbnew.LoadBoard(PCB)
def track(net, pts, w, layer=pcbnew.F_Cu):
    for a, c in zip(pts[:-1], pts[1:]):
        t = pcbnew.PCB_TRACK(b); t.SetStart(K(*a)); t.SetEnd(K(*c)); t.SetWidth(pcbnew.FromMM(w))
        t.SetLayer(layer); t.SetNet(b.FindNet(net)); b.Add(t)
dead = [t for t in b.GetTracks() if t.GetNetname() == '+5V' and t.GetWidth() == pcbnew.FromMM(0.5)
        and t.GetLayer() == pcbnew.F_Cu and -36 < pcbnew.ToMM(t.GetStart().x) - CX < -30 and -56.5 < CY - pcbnew.ToMM(t.GetStart().y) < -52]
for t in dead:
    b.Remove(t)
print('removed', len(dead))
v = pcbnew.PCB_VIA(b); v.SetPosition(K(-35.6, -52.5)); v.SetWidth(pcbnew.FromMM(0.6)); v.SetDrill(pcbnew.FromMM(0.3))
v.SetNet(b.FindNet('+5V')); b.Add(v)
track('+5V', [(-34.33, -52.5), (-35.6, -52.5)], 0.4)
track('+5V', [(-35.6, -52.5), (-35.6, -55.0), (-30.5, -59.5), (-28.96, -59.5)], 0.5, pcbnew.B_Cu)
gnd = b.FindNet('GND')
anchors = [t.GetPosition() for t in b.GetTracks() if isinstance(t, pcbnew.PCB_VIA) and t.GetNetCode() == gnd.GetNetCode()]
anchors += [p.GetPosition() for fp in b.GetFootprints() for p in fp.Pads()
            if p.GetNetCode() == gnd.GetNetCode() and p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH]
pcbnew.ZONE_FILLER(b).Fill(b.Zones())
added = 0
for z in b.Zones():
    if z.GetZoneName() not in ('GND top', 'GND bottom'):
        continue
    L = z.GetLayer()
    fp = z.GetFilledPolysList(L)
    for i in range(fp.OutlineCount()):
        sub = pcbnew.SHAPE_POLY_SET(); sub.AddOutline(fp.Outline(i))
        for j in range(fp.HoleCount(i)):
            sub.AddHole(fp.Hole(i, j))
        if any(sub.Contains(a) for a in anchors):
            continue
        # find a point well inside the island for a via
        bb = fp.Outline(i).BBox()
        best = None
        for gx in range(21):
            for gy in range(21):
                p = pcbnew.VECTOR2I(int(bb.GetX() + bb.GetWidth() * gx / 20), int(bb.GetY() + bb.GetHeight() * gy / 20))
                if sub.Contains(p):
                    d = sub.SquaredDistance(p) if hasattr(sub, 'SquaredDistance') else 0
                    # distance to island edge
                    dd = fp.Outline(i).SquaredDistance(p) if hasattr(fp.Outline(i), 'SquaredDistance') else 0
                    if best is None or dd > best[0]:
                        best = (dd, p)
        if best and math.sqrt(best[0]) > pcbnew.FromMM(0.45):
            v = pcbnew.PCB_VIA(b); v.SetPosition(best[1]); v.SetWidth(pcbnew.FromMM(0.6)); v.SetDrill(pcbnew.FromMM(0.3))
            v.SetNet(gnd); b.Add(v); anchors.append(best[1]); added += 1
            print('stitch via', z.GetZoneName(), pcbnew.ToMM(best[1].x) - CX, CY - pcbnew.ToMM(best[1].y))
print('island vias', added)
pcbnew.ZONE_FILLER(b).Fill(b.Zones())
pcbnew.SaveBoard(PCB, b)
