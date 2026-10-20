"""
Pentapod walker ("Rocky") - v7: v6 + Luxonis OAK-D Lite depth camera on the dome apex.

Run on an open "Pentapod Walker v6" design (Utilities > Scripts and Add-Ins > + > Run), then
File > Save As "Pentapod Walker v7" so v6 stays as the baseline. Reuses the v6 helpers and
dimensions by loading pentapod_build_v6.py from this folder (FUSION_DIR).

v7 changes (from v6)
  - Luxonis OAK-D Lite (DM9095) from the vendor STEP (HW_DIR/CAM_STEP), facing -y (GAP 270 deg,
    between legs 3 and 4), pitched CAM_PITCH down. USB-C (bottom face) -> Pi 5 USB 3 port.
  - "Dome Apex Cap" is replaced by "Dome Apex Cap (camera mount)": the v6 cap + a pedestal under
    the camera + a back plate with 2 x M4 clearance holes on the camera's VESA pair (75 mm),
    + a USB-C plug pass-through into the dome. One printed part (PLA), no supports needed if printed
    on its back plate.

OAK-D Lite STEP frame (measured from the file): x along the 91 mm length (-4.5 .. 86.5),
y up (-3 bottom .. 25 top), z front-to-back (-8.96 cover glass .. 8.5 back face).
  stereo lenses (OV7251, global shutter) at x 3.5 / 78.5, y 16 (75 mm baseline); RGB (IMX214) at x 41, y 16
  M4 VESA holes on the back face at (3.5, 15.5) and (78.5, 15.5)
  USB-C opening on the bottom face: x 49.9 .. 63.2, z -6.9 .. 0.4 ; 1/4-20 tripod hole at x 11, z 4.25

Verified (2026-10-20, in Fusion): BRep intersection of the new cap, the camera shells and a USB-C
plug + cable envelope (12.2 x 6.5 x 20 mm overmold + 4.4 mm cable, 17 mm further down) against
every other body: zero volume everywhere. Cap + mount 26.7 cm3 (v6 cap ~12 cm3).
Optical centre (RGB / aligned depth) in the body frame: (0, -38.87, 133.03) mm, looking (0, -0.966, -0.259).
Wiring: OAK-D Lite peaks at ~5 W from USB; with the TPS54560 5 V rail (not the official PSU) set
usb_max_current_enable=1 in the Pi's config.txt so its USB ports supply 1.6 A.
"""
import importlib.util, math, os

import adsk.core, adsk.fusion

FUSION_DIR = os.path.join(os.path.expanduser('~'), 'Documents', 'Pentapod', 'rocky', 'fusion')
_spec = importlib.util.spec_from_file_location('pentapod_build_v6', os.path.join(FUSION_DIR, 'pentapod_build_v6.py'))
v6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v6)
from_v6 = ('P', 'V', 'bx', 'cyl', 'sph', 'U', 'D', 'I', 'place', 'xf', 'fmatrix', 'tb', 'app', 'add',
           'import_into', 'root', 'find_occ', 'HW_DIR', 'RO', 'RI', 'CZD', 'APEX_R')
globals().update({k: getattr(v6, k) for k in from_v6})

CAM_STEP = 'Luxonis OAK-D Lite enclosure (DM9095).stp'
CAM_NAME = 'Depth camera (Luxonis OAK-D Lite)'
CAM_PITCH = 15.0                     # deg nose-down; bottom ray at 15 + 29 = 44 deg clears the dome
CAM_C_STEP = (41.0, -3.0, 0.0)       # bottom-face centre (STEP), placed at CAM_C_BODY
CAM_C_BODY = (0.0, -25.3, 117.0)     # cover-glass bottom edge at y -34, ~2 mm above the cap; the USB-C plug
                                     # then drops inside the cap radius (at y -31.3 it clipped Dome Sector 3)
LENS_STEP = (41.0, 16.0, -8.96)      # RGB lens on the cover glass = depth frame origin (depth aligned to RGB)
VESA = [(3.5, 15.5), (78.5, 15.5)]   # M4, back face z = 8.5
USB_STEP = (56.55, -3.0, -3.25)      # USB-C opening centre on the bottom face
PLATE_T = 4.0                        # back plate thickness
M4_CLR_R = 2.25
USB_HOLE = (15.0, 9.5)               # pass-through for the plug overmold (x, z in STEP); typical USB-C overmold 12.2 x 6.5


def cam_M():
    """STEP frame -> body frame. Camera looks along -STEP z = (0, -cos p, -sin p) in the body."""
    p = math.radians(CAM_PITCH); c, s = math.cos(p), math.sin(p)
    cols = [(-1.0, 0.0, 0.0), (0.0, -s, c), (0.0, c, s)]       # images of STEP x, y, z
    return place(cols, CAM_C_STEP, CAM_C_BODY)


def lens_pose():
    """Depth/RGB optical centre (mm) and view direction in the body frame (for the MuJoCo camera)."""
    p = math.radians(CAM_PITCH)
    return xf(cam_M(), LENS_STEP), (0.0, -math.cos(p), -math.sin(p))


def sbox(x0, x1, y0, y1, z0, z1):
    """Box given in the camera STEP frame, returned in the body frame."""
    b = bx(x0, x1, y0, y1, z0, z1)
    tb.transform(b, fmatrix(cam_M()))
    return b


def scyl(a, b, r):
    return cyl(xf(cam_M(), a), xf(cam_M(), b), r)


def mount_geometry(cap):
    """v6 apex cap body (temporary BRep copy) -> cap + camera mount."""
    outer = sph((0, 0, CZD), RO)
    inner = sph((0, 0, CZD), RI)
    # pedestal: under the camera centre, from the camera bottom face down into the cap shell
    ped = sbox(22.0, 60.0 + 4.0, -40.0, -3.0, -8.5, 8.5 + PLATE_T)
    I(ped, cyl((0, 0, 0), (0, 0, 300), APEX_R))                  # stays on the cap
    D(ped, inner)                                                # nothing below the cap's inner skin
    # back plate: full camera width, from below the camera up to the top of the M4 holes
    plate = sbox(-4.5, 86.5, -12.0, 22.0, 8.5, 8.5 + PLATE_T)
    D(plate, tb.copy(outer))                                     # overhang clears the dome sectors
    gap = sph((0, 0, CZD), RO + 1.0)
    D(plate, I(gap, D(bx(-300, 300, -300, 300, 0, 400), cyl((0, 0, 0), (0, 0, 300), APEX_R))))
    U(cap, ped, plate)
    for x, y in VESA:                                            # M4 clearance + 7.5 mm head counterbore
        D(cap, scyl((x, y, 8.0), (x, y, 8.5 + PLATE_T + 1), M4_CLR_R),
               scyl((x, y, 8.5 + 1.5), (x, y, 8.5 + PLATE_T + 1), 3.75))
    ux, uz = USB_STEP[0], USB_STEP[2]
    D(cap, sbox(ux - USB_HOLE[0] / 2, ux + USB_HOLE[0] / 2, -60.0, -2.9, uz - USB_HOLE[1] / 2, uz + USB_HOLE[1] / 2))
    D(cap, sbox(-4.6, 86.6, -3.0, 25.1, -9.0, 8.55))             # camera pocket (0.05 mm clearance on the back)
    return cap


def stage_camera(ident=None):
    r = root()
    occ = find_occ(r, CAM_NAME)
    if occ is None:
        occ = r.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        occ.component.name = CAM_NAME
        import_into(occ.component, os.path.join(HW_DIR, CAM_STEP))
    occ.transform = fmatrix(cam_M())
    des = adsk.fusion.Design.cast(app.activeProduct)
    if des.snapshots.hasPendingSnapshot:          # capture the position, or the next recompute resets it
        des.snapshots.add()
    return occ


def stage_mount(ident=None):
    dome = find_occ(root(), 'Dome').component
    old = [b for b in dome.bRepBodies if b.name == 'Dome Apex Cap']
    new = [b for b in dome.bRepBodies if b.name == 'Dome Apex Cap (camera mount)']
    if new:
        raise RuntimeError('camera mount already built (undo it first)')
    if not old:
        raise RuntimeError('Dome Apex Cap body not found')
    cap = mount_geometry(tb.copy(old[0]))
    add(dome, [(cap, 'Dome Apex Cap (camera mount)')])
    # Remove feature, not deleteMe(): deleting one body of the v6 base feature drops all six dome bodies
    dome.features.removeFeatures.add(old[0])


def run(_context: str):
    stage_camera()
    stage_mount()
    o, f = lens_pose()
    print('lens (mm)', [round(v, 2) for v in o], 'view dir', [round(v, 4) for v in f])
