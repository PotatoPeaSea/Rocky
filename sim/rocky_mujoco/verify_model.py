"""Sanity checks for rocky.xml - run after every change to build_model.py.

    python verify_model.py
"""
import math
import os

import mujoco
import numpy as np

import rocky_kinematics as K

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "scene.xml"))
d = mujoco.MjData(m)
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)


# ---- mass ----
print("body masses (g):", {m.body(i).name: round(m.body_mass[i] * 1e3, 1) for i in range(1, m.nbody) if "leg1" in m.body(i).name or m.body(i).name == "base"})
tot = m.body_mass.sum()
check(2.0 < tot < 3.2, f"total mass {tot:.3f} kg")

# ---- joint limits ----
spec = {"hip": (-75, 75), "femur": (-45, 85), "knee": (-60, 75)}
for j in range(1, m.njnt):
    nm = m.joint(j).name
    lo, hi = np.degrees(m.jnt_range[j])
    s = spec[nm.split("_")[1]]
    check(abs(lo - s[0]) < 0.01 and abs(hi - s[1]) < 0.01, f"{nm:12s} range {lo:+.1f} .. {hi:+.1f} deg") if j <= 3 else None
lims_all = np.degrees(m.jnt_range[1:])
check(np.allclose(lims_all, np.degrees(K.LIMITS)), "all 15 joint ranges equal rocky_kinematics.LIMITS")
check(all(m.actuator(i).name.startswith(f"s{i+1:02d}_") and m.actuator_trnid[i, 0] == i + 1 for i in range(15)),
      "actuator i drives joint i, named with servo ID i+1")

# ---- FK / IK vs MuJoCo ----
rng = np.random.default_rng(0)
err = 0.0
for _ in range(300):
    q = rng.uniform(K.LIMITS[:, 0], K.LIMITS[:, 1])
    d.qpos[:] = 0
    d.qpos[3] = 1
    d.qpos[7:] = q
    mujoco.mj_kinematics(m, d)
    fk = K.fk_all(q)
    for n in range(5):
        err = max(err, np.abs(d.site(f"leg{n+1}_foot").xpos - fk[n]).max())
check(err < 1e-6, f"FK matches MuJoCo foot sites over 300 random poses (max err {err:.2e} m)")
ik_err = 0.0
for _ in range(300):
    q = rng.uniform(K.LIMITS[:, 0] * 0.8, K.LIMITS[:, 1] * 0.8)
    q2 = K.ik_all(K.fk_all(q))
    ik_err = max(ik_err, np.abs(K.fk_all(q2) - K.fk_all(q)).max())
check(ik_err < 1e-9, f"IK(FK(q)) reproduces the foot positions (max err {ik_err:.2e} m)")

# ---- standing 10 s ----
mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
mujoco.mj_forward(m, d)
init_contacts = [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]]
check(d.ncon == 0, f"no contacts at the stand keyframe before settling ({init_contacts})")
p0 = d.qpos[:3].copy()
min_dist, max_tilt = 0.0, 0.0
while d.time < 10.0:
    mujoco.mj_step(m, d)
    if d.ncon:
        min_dist = min(min_dist, min(c.dist for c in d.contact[:d.ncon]))
    up = d.xmat[1].reshape(3, 3)[:, 2]
    max_tilt = max(max_tilt, math.degrees(math.acos(min(1.0, up[2]))))
drift = np.linalg.norm(d.qpos[:2] - p0[:2])
feet = sorted({m.geom(c.geom1).name if "foot" in m.geom(c.geom1).name else m.geom(c.geom2).name for c in d.contact[:d.ncon]})
check(drift < 1e-3 and max_tilt < 1.0, f"stands 10 s: xy drift {drift*1e3:.2f} mm, max tilt {max_tilt:.2f} deg, "
                                       f"z {p0[2]:.4f} -> {d.qpos[2]:.4f} m")
check(-min_dist < 0.002, f"max penetration {-min_dist*1e3:.2f} mm")
check(feet == [f"leg{n}_foot" for n in range(1, 6)], f"only the 5 feet touch the floor: {feet}")
print("     stand torques (N.m):", np.round(d.actuator_force[:3], 3), "stall", m.actuator_forcerange[0, 1].round(3))

# ---- drop test, servos limp then holding, random targets (no explosions) ----
mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
d.qpos[2] += 0.25
bad = False
for k in range(4000):
    if k % 25 == 0:
        d.ctrl[:] = rng.uniform(K.LIMITS[:, 0], K.LIMITS[:, 1])
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)) or np.abs(d.qvel).max() > 200:
        bad = True
        break
check(not bad and d.qpos[2] > -0.01, f"8 s random-target thrash after a 25 cm drop: finite, base z {d.qpos[2]:.3f} m, "
                                     f"max |qvel| {np.abs(d.qvel).max():.1f}")
warn = [m for m in d.warning if m.number > 0]
check(len(warn) == 0, f"no MuJoCo warnings ({[(i, w.number) for i, w in enumerate(d.warning) if w.number]})")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
