"""Render the stills and the gait clip in media/ (offscreen: MUJOCO_GL=egl or osmesa on Linux).

    python make_media.py
"""
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "media")
os.makedirs(OUT, exist_ok=True)

m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 540, 960)


def shot(name, key="stand", az=135, el=-25, dist=0.75, groups=None, z=0.07):
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, z]
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    for g, on in (groups or {}).items():
        opt.geomgroup[g] = on
    r.update_scene(d, cam, opt)
    imageio.imwrite(os.path.join(OUT, name), r.render())
    print("wrote", name)


shot("stand_iso.png")
shot("stand_side.png", az=90, el=-5, dist=0.7)
shot("no_dome_top.png", az=90, el=-70, dist=0.75, groups={1: 0})
shot("collision_geoms.png", az=135, el=-20, groups={1: 0, 2: 0, 3: 1})
subprocess.run([sys.executable, os.path.join(HERE, "gait_demo.py"), "--headless", "--seconds", "8",
                "--gait", "ripple", "--vy", "0.08", "--record", os.path.join(OUT, "gait_ripple.mp4")], check=True)
