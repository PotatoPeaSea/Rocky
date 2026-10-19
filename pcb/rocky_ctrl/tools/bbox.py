"""Print robot-frame courtyard boxes of placed footprints (filter by region)."""
import sys, pcbnew
b = pcbnew.LoadBoard('/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb')
x0, x1, y0, y1 = [float(v) for v in sys.argv[1:5]] if len(sys.argv) > 4 else (-99, 99, -99, 99)
for fp in sorted(b.GetFootprints(), key=lambda f: f.GetReference()):
    cy = fp.GetCourtyard(pcbnew.F_Cu)
    bb = cy.BBox() if cy.OutlineCount() else fp.GetBoundingBox(False, False)
    X0 = pcbnew.ToMM(bb.GetX()) - 150; X1 = pcbnew.ToMM(bb.GetRight()) - 150
    Y1 = 150 - pcbnew.ToMM(bb.GetY()); Y0 = 150 - pcbnew.ToMM(bb.GetBottom())
    if X1 < x0 or X0 > x1 or Y1 < y0 or Y0 > y1: continue
    print('%-5s x %6.1f..%6.1f  y %6.1f..%6.1f' % (fp.GetReference(), X0, X1, Y0, Y1))
