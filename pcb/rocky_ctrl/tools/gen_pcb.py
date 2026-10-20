"""Build rocky_ctrl.kicad_pcb from the schematic netlist + placement.py (KiCad 9 pcbnew API).

Usage (inside KiCad's python):  python3 gen_pcb.py <netlist.net>
Produces an unrouted, placed board with outline, holes, keep-outs and net classes.
Power copper is added by power_route.py, signals by Freerouting (route.sh), pours by finish_pcb.py.
"""
import math
import os
import sys

import pcbnew

sys.path.insert(0, os.path.dirname(__file__))
from sexp import parse, find, find1  # noqa: E402
import design  # noqa: E402
import placement  # noqa: E402

PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FPDIR = os.environ.get('KICAD_FOOTPRINT_DIR', '/usr/share/kicad/footprints')
CX, CY = 150.0, 150.0          # KiCad position of the robot axis (board centre)
R_OUT = 78.0
CUTOUT = [(-53, 30), (40, 30), (40, -31), (-31, -31), (-31, -50), (-47, -50), (-47, -31), (-53, -31)]
NOTCH_X, NOTCH_TOP = 8.0, -60.0   # battery-lead notch over deck hole B (0, -71)


def K(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(CX + x), pcbnew.FromMM(CY - y))


def mm(v):
    return pcbnew.ToMM(v)


def robot(p):
    return (mm(p.x) - CX, CY - mm(p.y))


def inside_board(x, y, margin=0.9):
    r = math.hypot(x, y)
    if r > R_OUT - margin:
        return False
    # main Pi cut-out and the plug notch
    if -53 - margin < x < 40 + margin and -31 - margin < y < 30 + margin:
        return False
    if -47 - margin < x < -31 + margin and -50 - margin < y < -31 + margin:
        return False
    if -NOTCH_X - margin < x < NOTCH_X + margin and y < NOTCH_TOP + margin:
        return False
    if 74.8 < x and -2.5 < y < 22.5:      # ESP32 antenna keep-out
        return False
    return True


def read_netlist(path):
    net = parse(open(path).read())
    comps = {}
    for c in find(find1(net, 'components'), 'comp'):
        ref = find1(c, 'ref')[1]
        comps[ref] = dict(value=find1(c, 'value')[1], fp=find1(c, 'footprint')[1],
                          tstamp=find1(c, 'tstamps')[1] if find1(c, 'tstamps') else '',
                          fields={f[1]: (f[2] if len(f) > 2 else '') for f in find(find1(c, 'fields') or [], 'field')
                                  if isinstance(f[1], str)})
    pads = {}
    for n in find(find1(net, 'nets'), 'net'):
        name = find1(n, 'name')[1]
        for node in find(n, 'node'):
            pads[(find1(node, 'ref')[1], find1(node, 'pin')[1])] = name
    return comps, pads


def seg(board, a, b, layer=pcbnew.Edge_Cuts, w=0.1):
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_SEGMENT)
    s.SetStart(K(*a)); s.SetEnd(K(*b))
    s.SetLayer(layer); s.SetWidth(pcbnew.FromMM(w))
    board.Add(s)


def arc(board, start, mid, end, layer=pcbnew.Edge_Cuts, w=0.1):
    s = pcbnew.PCB_SHAPE(board)
    s.SetShape(pcbnew.SHAPE_T_ARC)
    s.SetArcGeometry(K(*start), K(*mid), K(*end))
    s.SetLayer(layer); s.SetWidth(pcbnew.FromMM(w))
    board.Add(s)


def outline(board):
    yb = -math.sqrt(R_OUT ** 2 - NOTCH_X ** 2)
    a0 = math.degrees(math.atan2(yb, NOTCH_X))           # ~ -84 deg
    a1 = math.degrees(math.atan2(yb, -NOTCH_X)) + 360    # ~ 264 deg
    am = (a0 + a1) / 2
    arc(board, (NOTCH_X, yb), placement.pol(R_OUT, am), (-NOTCH_X, yb))
    seg(board, (-NOTCH_X, yb), (-NOTCH_X, NOTCH_TOP))
    seg(board, (-NOTCH_X, NOTCH_TOP), (NOTCH_X, NOTCH_TOP))
    seg(board, (NOTCH_X, NOTCH_TOP), (NOTCH_X, yb))
    for i in range(len(CUTOUT)):
        seg(board, CUTOUT[i], CUTOUT[(i + 1) % len(CUTOUT)])


def text(board, s, x, y, size=1.5, layer=pcbnew.F_SilkS, angle=0, bold=False):
    t = pcbnew.PCB_TEXT(board)
    t.SetText(s)
    t.SetPosition(K(x, y))
    t.SetLayer(layer)
    t.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(size), pcbnew.FromMM(size)))
    t.SetTextThickness(pcbnew.FromMM(size * 0.15))
    t.SetTextAngleDegrees(angle)
    t.SetBold(bold)
    if layer == pcbnew.B_SilkS:
        t.SetMirrored(True)
    board.Add(t)


def keepout(board, pts, name, layers=None, tracks=True, vias=True, pours=True, footprints=False):
    z = pcbnew.ZONE(board)
    z.SetIsRuleArea(True)
    z.SetZoneName(name)
    z.SetDoNotAllowTracks(tracks); z.SetDoNotAllowVias(vias); z.SetDoNotAllowCopperPour(pours)
    z.SetDoNotAllowPads(False); z.SetDoNotAllowFootprints(footprints)
    ls = pcbnew.LSET()
    for l in (layers or [pcbnew.F_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu, pcbnew.B_Cu]):
        ls.AddLayer(l)
    z.SetLayerSet(ls)
    ol = z.Outline()
    ol.NewOutline()
    for p in pts:
        v = K(*p); ol.Append(v.x, v.y)
    board.Add(z)
    return z


def setup(board):
    board.SetCopperLayerCount(4)
    ds = board.GetDesignSettings()
    ds.SetBoardThickness(pcbnew.FromMM(1.6))
    ds.m_MinClearance = pcbnew.FromMM(0.15)
    ds.m_TrackMinWidth = pcbnew.FromMM(0.15)
    ds.m_ViasMinSize = pcbnew.FromMM(0.5)
    ds.m_MinThroughDrill = pcbnew.FromMM(0.2)
    ds.m_ViasMinAnnularWidth = pcbnew.FromMM(0.1)
    ds.m_HoleClearance = pcbnew.FromMM(0.25)
    ds.m_HoleToHoleMin = pcbnew.FromMM(0.25)
    ds.m_CopperEdgeClearance = pcbnew.FromMM(0.4)
    ds.m_SilkClearance = pcbnew.FromMM(0.0)
    board.SetLayerName(pcbnew.In1_Cu, 'In1.GND')
    board.SetLayerName(pcbnew.In2_Cu, 'In2.PWR')
    nc = ds.m_NetSettings
    dflt = nc.GetDefaultNetclass()
    dflt.SetClearance(pcbnew.FromMM(0.18)); dflt.SetTrackWidth(pcbnew.FromMM(0.25))
    dflt.SetViaDiameter(pcbnew.FromMM(0.6)); dflt.SetViaDrill(pcbnew.FromMM(0.3))
    for name, width, clr, via, drill in (('Power', 0.6, 0.2, 0.8, 0.4), ('HighCurrent', 1.0, 0.2, 0.8, 0.4),
                                         ('Plane', 0.5, 0.2, 0.8, 0.4), ('Manual', 1.0, 0.2, 0.8, 0.4),
                                         ('USB', 0.2, 0.15, 0.6, 0.3), ('USBPWR', 0.4, 0.15, 0.6, 0.3)):
        n = pcbnew.NETCLASS(name)
        n.SetTrackWidth(pcbnew.FromMM(width)); n.SetClearance(pcbnew.FromMM(clr))
        n.SetViaDiameter(pcbnew.FromMM(via)); n.SetViaDrill(pcbnew.FromMM(drill))
        nc.SetNetclass(name, n)
    for pat, cls in (('/VBAT_IN', 'Manual'), ('/VBAT_F', 'HighCurrent'), ('/VBAT_P', 'HighCurrent'),
                     ('/VSERVO', 'Plane'), ('GND', 'Plane'), ('/VSERVO_SW', 'Manual'), ('/B5_SW', 'Manual'), ('+5V', 'Power'), ('/V_LOGIC', 'Power'),
                     ('/B3_SW', 'Power'), ('/USB_VBUS', 'USBPWR'), ('+3V3', 'Power'), ('/VIO', 'Power'),
                     ('/USB_D?', 'USB')):
        nc.SetNetclassPatternAssignment(pat, cls)


MODEL_OVERRIDE = {
    'AMASS_XT60PW-M_1x02_P7.20mm_Horizontal': ('${KIPRJMOD}/3d/AMASS_XT60PW-M.step', None),
    'JST_VH_B3P-VH-B_1x03_P3.96mm_Vertical': ('${KIPRJMOD}/3d/JST_VH_B3P-VH-B.step', None),
    'Molex_SPOX_5267-03A_1x03_P2.50mm_Vertical': ('${KIPRJMOD}/3d/Molex_SPOX_5267-03A.step', None),
    'USB_C_Receptacle_HRO_TYPE-C-31-M-12': ('${KIPRJMOD}/3d/USB_C_HRO_TYPE-C-31-M-12.step', None),
    'L_Bourns_SRP1245A': ('${KIPRJMOD}/3d/L_Bourns_SRP1245A.step', None),
    'SW_Push_1P1T_XKB_TS-1187A': ('${KIPRJMOD}/3d/SW_XKB_TS-1187A.step', None),
    'CP_Radial_D12.5mm_P5.00mm': ('${KIPRJMOD}/3d/CP_Radial_D12.5mm_H20.step', None),
    'CP_Radial_D8.0mm_P3.50mm': ('${KIPRJMOD}/3d/CP_Radial_D8.0mm_H11.5.step', None),
    'Texas_R-PDSO-G8_EP2.95x4.9mm_Mask2.4x3.1mm_ThermalVias':
        ('${KICAD9_3DMODEL_DIR}/Package_SO.3dshapes/SOIC-8-1EP_3.9x4.9mm_P1.27mm_EP2.41x3.81mm.step', None),
    'R_Shunt_Vishay_WSK2512_6332Metric_T1.19mm': ('${KICAD9_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_2512_6332Metric.step', None),
}


def set_models(fp, name):
    if name not in MODEL_OVERRIDE:
        return
    path, _ = MODEL_OVERRIDE[name]
    models = fp.Models()
    if len(models):
        m = models[0]
        m.m_Filename = path
        while len(models) > 1:
            models.pop()
    else:
        m = pcbnew.FP_3DMODEL()
        m.m_Filename = path
        models.append(m)


def build(netfile, out):
    comps, padnets = read_netlist(netfile)
    board = pcbnew.BOARD()
    setup(board)
    nets = {}
    for name in sorted(set(padnets.values())):
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        nets[name] = ni
    holes = {r: xy for r, _, xy in design.HOLES}
    tp = placement.TP_PLACE
    missing = []
    for ref, c in sorted(comps.items()):
        lib, name = c['fp'].split(':')
        fp = pcbnew.FootprintLoad(os.path.join(FPDIR, lib + '.pretty'), name)
        if fp is None:
            raise RuntimeError('footprint not found ' + c['fp'])
        fp.SetReference(ref)
        fp.SetValue(c['value'])
        fp.SetFPIDAsString(c['fp'])
        if c['tstamp']:
            fp.SetPath(pcbnew.KIID_PATH('/' + c['tstamp']))
        for k, v in c['fields'].items():
            if k in ('Footprint', 'Datasheet', 'Description') or not v:
                continue
            fp.SetField(k, v)
            f = fp.GetFieldByName(k)
            if f:
                f.SetVisible(False)
        set_models(fp, name)
        board.Add(fp)
        for pad in fp.Pads():
            n = padnets.get((ref, pad.GetNumber()))
            if n:
                pad.SetNet(nets[n])
        if ref in holes:
            x, y = holes[ref]; rot = 0
        elif ref in tp:
            x, y = tp[ref]; rot = 0
        elif ref in placement.PLACE:
            x, y, rot = placement.PLACE[ref]
        else:
            missing.append(ref); x, y, rot = (0, 0, 0)
        fp.SetOrientationDegrees(rot)
        fp.SetPosition(K(x, y))
        numbered = [p for p in fp.Pads() if p.GetNumber()]
        if numbered and ref not in holes:
            sx = sum(p.GetPosition().x for p in numbered) / len(numbered)
            sy = sum(p.GetPosition().y for p in numbered) / len(numbered)
            tgt = K(x, y)
            fp.Move(pcbnew.VECTOR2I(int(tgt.x - sx), int(tgt.y - sy)))
        # silkscreen reference size
        fp.Reference().SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(0.8), pcbnew.FromMM(0.8)))
        fp.Reference().SetTextThickness(pcbnew.FromMM(0.12))
    if missing:
        print('UNPLACED:', missing)
    outline(board)
    # ESP32 antenna keep-out: no copper on any layer beyond the module pads (antenna end faces +x)
    u7 = board.FindFootprintByReference('U7')
    keepout(board, [(75.2, -2.0), (82, -2.0), (82, 22.0), (75.2, 22.0)], 'ESP32 antenna')
    # 0.45 mm track/via keep-out bands along every edge (the autorouter ignores copper-to-edge clearance)
    g = 0.45
    keepout(board, [(-53 - g, 30 + g), (40 + g, 30 + g), (40 + g, -31 - g), (-31 + g, -31 - g), (-31 + g, -50 - g),
                    (-47 - g, -50 - g), (-47 - g, -31 - g), (-53 - g, -31 - g)], 'edge band cut-out', pours=False)
    keepout(board, [(-NOTCH_X - g, NOTCH_TOP + g), (NOTCH_X + g, NOTCH_TOP + g), (NOTCH_X + g, -82), (-NOTCH_X - g, -82)],
            'edge band notch', pours=False)
    z = keepout(board, [placement.pol(R_OUT + 3, k * 3.75) for k in range(96)], 'edge band outer', pours=False)
    ol = z.Outline()
    ol.NewHole()
    for k in range(96):
        v = K(*placement.pol(R_OUT - g, k * 3.75)); ol.Append(v.x, v.y, -1, 0)
    # Board marking
    text(board, 'ROCKY CONTROL v1', -20.0, 70.0, 2.2, bold=True)
    text(board, 'Rocky pentapod  |  M. Lin  |  2026-10', -20.0, 66.5, 1.2)
    text(board, '3S LiPo 9-12.6V  20A ATM fuse', 30.0, -71.5, 1.2, angle=0)
    text(board, 'Pi 5', -2, 0, 6, layer=pcbnew.F_SilkS)  # sits in the cut-out: removed below
    # remove the stray silk inside the cut-out (keeps API usage simple above)
    for d in list(board.GetDrawings()):
        if isinstance(d, pcbnew.PCB_TEXT) and d.GetText() == 'Pi 5':
            board.Remove(d)
    for leg in range(1, 6):
        a = placement.LEG_CONN_ANG[leg]
        x, y = placement.pol(placement.LEG_CONN_R - 10.5, a)
        text(board, 'LEG %d' % leg, x, y, 1.6, angle=a - 90, bold=True)
    board.SetFileName(out)
    pcbnew.SaveBoard(out, board)
    print('saved', out, 'footprints', len(board.GetFootprints()))
    return board


if __name__ == '__main__':
    build(sys.argv[1], os.path.join(PRJ, 'rocky_ctrl.kicad_pcb'))
