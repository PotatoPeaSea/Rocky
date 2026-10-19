"""Generate simplified (envelope-accurate) STEP models for parts whose KiCad 3D models are not available offline.

Coordinates follow KiCad's 3D convention: origin at the footprint origin, +x right, +y = towards the TOP of the
footprint drawing (KiCad footprint y is negated), +z up from the board top surface. Dimensions are from the
manufacturers' drawings, rounded; these are mechanical envelopes, not detailed models.
Run with a Python that has cadquery:  python make_3d.py <outdir>
"""
import os
import sys

import cadquery as cq

OUT = sys.argv[1] if len(sys.argv) > 1 else '3d'
os.makedirs(OUT, exist_ok=True)


def box(x0, x1, y0, y1, z0, z1):
    """Box from footprint-frame extents (y given in footprint/KiCad coords, flipped here)."""
    return cq.Workplane('XY').box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, -y1, z0))


def save(name, parts):
    a = cq.Assembly()
    for i, (shape, rgb) in enumerate(parts):
        a.add(shape, name='%s_%d' % (name, i), color=cq.Color(*rgb))
    a.save(os.path.join(OUT, name + '.step'), exportType='STEP')
    print('wrote', name)


YEL, BLK, MET, WHT, GRY = (0.95, 0.8, 0.1), (0.1, 0.1, 0.1), (0.75, 0.75, 0.78), (0.92, 0.92, 0.88), (0.4, 0.4, 0.4)

# XT60PW-M: right-angle PCB male, pins at x=0/7.2, mouth towards footprint -y
xt = box(-4.3, 11.5, -16.4, 2.2, 0.0, 8.3)
for px in (0.0, 7.2):
    xt = xt.cut(cq.Workplane('XZ').center(px, 4.6).circle(2.35).extrude(-12.0).translate((0, 16.4 + 0.01, 0)))
pins = [cq.Workplane('XY').circle(0.9).extrude(3.5).translate((px, 0, -3.0)) for px in (0.0, 7.2)]
save('AMASS_XT60PW-M', [(xt, YEL)] + [(p, MET) for p in pins])

# JST VH B3P-VH-B vertical, 3.96 mm pitch, pins at x = 0, 3.96, 7.92
vh = box(-1.95, 9.87, -3.75, 4.85, 0.0, 11.5)
vh = vh.cut(box(-1.2, 9.12, -3.0, 4.1, 2.0, 11.6))
vpins = [box(px - 0.57, px + 0.57, -0.57, 0.57, -3.4, 10.0) for px in (0.0, 3.96, 7.92)]
save('JST_VH_B3P-VH-B', [(vh, WHT)] + [(p, MET) for p in vpins])

# Molex SPOX 5267-03A vertical header, 2.50 mm pitch
sp = box(-2.5, 7.5, -3.15, 1.85, 0.0, 7.6)
sp = sp.cut(box(-1.8, 6.8, -2.45, 1.15, 1.5, 7.7))
spins = [box(px - 0.32, px + 0.32, -0.32, 0.32, -3.0, 6.8) for px in (0.0, 2.5, 5.0)]
save('Molex_SPOX_5267-03A', [(sp, WHT)] + [(p, MET) for p in spins])

# HRO TYPE-C-31-M-12 top-mount receptacle, mouth towards footprint +y
usb = box(-4.47, 4.47, -3.4, 3.95, 0.0, 3.26)
usb = usb.cut(cq.Workplane('XZ').center(0, 1.63).slot2D(8.34, 2.56).extrude(6.5).translate((0, -3.95 + 0.01, 0)))
save('USB_C_HRO_TYPE-C-31-M-12', [(usb, MET)])

# Bourns SRP1245A 12.5 x 13.5 x 4.5 shielded inductor
ind = cq.Workplane('XY').box(13.5, 12.5, 4.5, centered=(True, True, False)).edges('|Z').fillet(1.0)
save('L_Bourns_SRP1245A', [(ind, GRY)])

# XKB TS-1187A 5.1 x 5.1 x 1.5 tact switch
sw = box(-2.55, 2.55, -2.55, 2.55, 0.0, 1.05)
act = cq.Workplane('XY').circle(1.5).extrude(0.45).translate((0, 0, 1.05))
save('SW_XKB_TS-1187A', [(sw, MET), (act, BLK)])

# Panasonic EEU-FR 12.5 x 20 radial electrolytic (pin 1 at origin, 5 mm pitch)
can = cq.Workplane('XY').circle(6.25).extrude(20.0).translate((2.5, 0, 0.3))
sleeve_top = cq.Workplane('XY').circle(5.6).extrude(0.2).translate((2.5, 0, 20.3))
save('CP_Radial_D12.5mm_H20', [(can, (0.15, 0.15, 0.2)), (sleeve_top, MET)])
# 8 x 11.5 radial (pin 1 at origin, 3.5 mm pitch)
can8 = cq.Workplane('XY').circle(4.0).extrude(11.5).translate((1.75, 0, 0.3))
save('CP_Radial_D8.0mm_H11.5', [(can8, (0.15, 0.15, 0.2))])
