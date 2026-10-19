"""Move reference designators to a free spot next to each footprint (cosmetic silk clean-up)."""
import pcbnew
from gen_pcb import inside_board, CX, CY


def rect_of(bb, grow=0.0):
    return (bb.GetX() - pcbnew.FromMM(grow), bb.GetY() - pcbnew.FromMM(grow),
            bb.GetRight() + pcbnew.FromMM(grow), bb.GetBottom() + pcbnew.FromMM(grow))


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def place_refs(board):
    blockers = []
    for fp in board.GetFootprints():
        for p in fp.Pads():
            blockers.append(rect_of(p.GetBoundingBox(), 0.15))
        cy = fp.GetCourtyard(pcbnew.F_Cu)
        if cy.OutlineCount():
            pass
    for d in board.GetDrawings():
        if isinstance(d, pcbnew.PCB_TEXT):
            blockers.append(rect_of(d.GetBoundingBox(), 0.1))
    for t in board.GetTracks():
        if isinstance(t, pcbnew.PCB_VIA):
            blockers.append(rect_of(t.GetBoundingBox(), 0.05))
    moved = hidden = 0
    for fp in sorted(board.GetFootprints(), key=lambda f: -f.GetArea()):
        ref = fp.Reference()
        if fp.GetReference().startswith('H'):
            ref.SetVisible(False); continue
        ref.SetTextAngleDegrees(0)
        ref.SetKeepUpright(True)
        cy = fp.GetCourtyard(pcbnew.F_Cu)
        bb = cy.BBox() if cy.OutlineCount() else fp.GetBoundingBox(False, False)
        cx, cyy = bb.GetCenter().x, bb.GetCenter().y
        w = ref.GetBoundingBox().GetWidth(); h = ref.GetBoundingBox().GetHeight()
        gap = pcbnew.FromMM(0.5)
        cands = [(cx, bb.GetY() - gap - h // 2), (cx, bb.GetBottom() + gap + h // 2),
                 (bb.GetX() - gap - w // 2, cyy), (bb.GetRight() + gap + w // 2, cyy),
                 (bb.GetX() - gap - w // 2, bb.GetY() - gap - h // 2), (bb.GetRight() + gap + w // 2, bb.GetY() - gap - h // 2),
                 (bb.GetX() - gap - w // 2, bb.GetBottom() + gap + h // 2), (bb.GetRight() + gap + w // 2, bb.GetBottom() + gap + h // 2)]
        ok = False
        for (x, y) in cands:
            r = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
            corners = [(pcbnew.ToMM(r[0]) - CX, CY - pcbnew.ToMM(r[1])), (pcbnew.ToMM(r[2]) - CX, CY - pcbnew.ToMM(r[3]))]
            if not all(inside_board(px, py, 0.4) for px, py in corners + [(corners[0][0], corners[1][1]), (corners[1][0], corners[0][1])]):
                continue
            if any(overlap(r, bl) for bl in blockers):
                continue
            ref.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
            blockers.append(r)
            ok = True; moved += 1
            break
        if not ok:
            ref.SetVisible(False); hidden += 1
    print('refs placed', moved, 'hidden (no room; still on F.Fab)', hidden)
