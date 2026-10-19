"""Open Rocky in the interactive MuJoCo viewer.

    python view.py                 # flat ground, starts in the "stand" keyframe
    python view.py --rough         # heightfield terrain
    python view.py --hang          # base fixed in the air (gravity off): play with the servo sliders

In the viewer:
  * right panel > Control: one slider per servo (s01..s15 = servo IDs, target angle in rad)
  * left panel > Simulation > Key: pick "stand" or "zero", then "Load key" (Reset = model default)
  * left panel > Rendering > Group enable: 1 = dome, 2 = parts, 3 = collision geoms, 4 = sites (IMU, feet)
  * Space pauses, Backspace resets, double-click selects a body, Ctrl+right-drag pushes it
"""
import argparse
import os

import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--rough", action="store_true")
ap.add_argument("--hang", action="store_true", help="disable gravity so the legs can be posed freely")
ap.add_argument("--key", default="stand", choices=["stand", "zero"])
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene_rough.xml" if a.rough else "scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key(a.key).id)
if a.hang:
    m.opt.gravity[:] = 0
    d.qpos[2] += 0.15
mujoco.mj_forward(m, d)
mujoco.viewer.launch(m, d)
