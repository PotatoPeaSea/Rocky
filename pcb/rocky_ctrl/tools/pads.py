import sys, pcbnew
b = pcbnew.LoadBoard('/home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb')
want = sys.argv[1:]
for fp in b.GetFootprints():
    for p in fp.Pads():
        n = p.GetNetname()
        if any(n == w or n == '/' + w for w in want):
            pos = p.GetPosition(); sz = p.GetSize()
            print('%-12s %-5s %-4s (%6.2f, %6.2f) %4.1fx%4.1f %s' % (n, fp.GetReference(), p.GetNumber(), pcbnew.ToMM(pos.x) - 150, 150 - pcbnew.ToMM(pos.y),
                  pcbnew.ToMM(sz.x), pcbnew.ToMM(sz.y), 'THT' if p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH else ''))
