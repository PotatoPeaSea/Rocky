"""Drop a GND via into every GND pour fragment that has no via / PTH anchor (run after zone fill)."""
import math, pcbnew
P = '/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb'
b = pcbnew.LoadBoard(P)
pcbnew.ZONE_FILLER(b).Fill(b.Zones())
g = b.FindNet('GND').GetNetCode()
mm = pcbnew.ToMM
def anchors():
    a = [t.GetPosition() for t in b.GetTracks() if isinstance(t, pcbnew.PCB_VIA) and t.GetNetCode() == g]
    return a + [p.GetPosition() for fp in b.GetFootprints() for p in fp.Pads() if p.GetNetCode() == g and p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH]
def seg_d(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
def edge_d(chains, px, py):
    d = 1e9
    for ch in chains:
        n = ch.PointCount()
        for k in range(n):
            a, c = ch.CPoint(k), ch.CPoint((k + 1) % n)
            d = min(d, seg_d(px, py, mm(a.x), mm(a.y), mm(c.x), mm(c.y)))
    return d
obst = []
for t in b.GetTracks():
    if t.GetNetCode() == g: continue
    if isinstance(t, pcbnew.PCB_VIA):
        obst.append(('c', mm(t.GetPosition().x), mm(t.GetPosition().y), 0, 0, mm(t.GetWidth(pcbnew.F_Cu)) / 2))
    else:
        obst.append(('s', mm(t.GetStart().x), mm(t.GetStart().y), mm(t.GetEnd().x), mm(t.GetEnd().y), mm(t.GetWidth()) / 2))
for f in b.GetFootprints():
    for p in f.Pads():
        if p.GetNetCode() == g: continue
        bb = p.GetBoundingBox()
        obst.append(('r', mm(bb.GetX()), mm(bb.GetY()), mm(bb.GetRight()), mm(bb.GetBottom()), 0))
def clear(px, py, need=0.3 + 0.22):
    for k, a, c, d, e, r in obst:
        if k == 's': dist = seg_d(px, py, a, c, d, e) - r
        elif k == 'c': dist = math.hypot(px - a, py - c) - r
        else: dist = math.hypot(max(a - px, 0, px - d), max(c - py, 0, py - e))
        if dist < need: return False
    return True
added = 0
for z in list(b.Zones()):
    if z.GetNetCode() != g or z.GetLayer() not in (pcbnew.F_Cu, pcbnew.B_Cu):
        continue
    fp = z.GetFilledPolysList(z.GetLayer())
    anc = anchors()
    for i in range(fp.OutlineCount()):
        sub = pcbnew.SHAPE_POLY_SET(); sub.AddOutline(fp.Outline(i))
        chains = [fp.Outline(i)] + [fp.Hole(i, j) for j in range(fp.HoleCount(i))]
        for ch in chains[1:]:
            sub.AddHole(ch)
        if any(sub.Contains(a) for a in anc):
            continue
        bb = fp.Outline(i).BBox()
        best = None
        x0, y0, w, h = mm(bb.GetX()), mm(bb.GetY()), mm(bb.GetWidth()), mm(bb.GetHeight())
        for gx in range(int(w / 0.05) + 1):
            for gy in range(int(h / 0.05) + 1):
                px, py = x0 + gx * 0.05, y0 + gy * 0.05
                v = pcbnew.VECTOR2I(pcbnew.FromMM(px), pcbnew.FromMM(py))
                if not sub.Contains(v):
                    continue
                d = edge_d(chains, px, py)
                if d < 0.3 or not clear(px, py):
                    continue
                if best is None or d > best[0]:
                    best = (d, v)
        if best and best[0] >= 0.3:
            via = pcbnew.PCB_VIA(b); via.SetPosition(best[1]); via.SetWidth(pcbnew.FromMM(0.6)); via.SetDrill(pcbnew.FromMM(0.3))
            via.SetNet(b.FindNet('GND')); b.Add(via); added += 1
            print('via', z.GetZoneName(), round(mm(best[1].x) - 150, 2), round(150 - mm(best[1].y), 2), 'margin', round(best[0], 2))
        else:
            print('no room in fragment at', round(x0 - 150, 1), round(150 - y0, 1), best and round(best[0], 2))
pcbnew.ZONE_FILLER(b).Fill(b.Zones())
pcbnew.SaveBoard(P, b)
print('added', added)
