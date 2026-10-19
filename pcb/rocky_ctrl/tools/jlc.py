"""Write JLCPCB BOM + CPL from the board (top-side assembly; holes, test pads and solder jumpers excluded)."""
import csv
import pcbnew
P = '/home/claude/rocky/pcb/rocky_ctrl/'
b = pcbnew.LoadBoard(P + 'rocky_ctrl.kicad_pcb')
groups = {}
cpl = []
ox, oy = 150.0, 150.0
for fp in sorted(b.GetFootprints(), key=lambda f: f.GetReference()):
    ref = fp.GetReference()
    if ref.startswith(('H', 'TP', 'JP')):
        continue
    def fld(n):
        return fp.GetFieldText(n) if fp.HasFieldByName(n) else ''
    key = (fp.GetValue(), fp.GetFPIDAsString().split(':')[1], fld('LCSC'), fld('MPN'))
    groups.setdefault(key, []).append(ref)
    pos = fp.GetPosition()
    cpl.append((ref, '%.3fmm' % (pcbnew.ToMM(pos.x) - ox), '%.3fmm' % (oy - pcbnew.ToMM(pos.y)),
                'Top' if fp.GetLayer() == pcbnew.F_Cu else 'Bottom', '%.1f' % (fp.GetOrientationDegrees() % 360),
                'THT' if fp.GetAttributes() & pcbnew.FP_THROUGH_HOLE else 'SMD'))
with open(P + 'fab/rocky_ctrl_bom_jlc.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Comment', 'Designator', 'Footprint', 'LCSC', 'MPN', 'Qty'])
    for (val, fpn, lcsc, mpn), refs in sorted(groups.items(), key=lambda kv: kv[1][0]):
        w.writerow([val, ','.join(refs), fpn, lcsc, mpn, len(refs)])
with open(P + 'fab/rocky_ctrl_cpl_jlc.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Designator', 'Mid X', 'Mid Y', 'Layer', 'Rotation', 'Type'])
    w.writerows(cpl)
n_l = sum(1 for k in groups if k[2])
print('BOM lines', len(groups), 'with LCSC', n_l, 'placements', len(cpl))
