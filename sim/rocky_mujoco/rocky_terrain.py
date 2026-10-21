"""Terrain curriculum for Rocky: one heightfield made of square tiles, difficulty rows x terrain-type columns
(legged_gym style), plus a model builder with MJX-friendly collision shapes.

    T = generate()                       # numpy heights + tile centres
    m = build_model(T)                   # MjModel: scene.xml robot on the terrain heightfield
    python rocky_terrain.py              # writes media/terrain_overview.png and checks the tiles

Terrain types (columns), difficulty d = level / (levels - 1) in [0, 1]:
  rough    random rocks: smoothed noise + scattered round stones, up to 1 + 4 d cm high
  slope_up   pyramid, robot starts on top and walks down; slope 0 .. 15 deg
  slope_down inverted pyramid, robot starts in the pit and walks up
  stairs_up  concentric steps up to the centre: 1 + 5 d cm risers (6 cm max), 30 cm treads
  stairs_dn  concentric steps down to the centre (robot climbs out)
  blocks     random 30-50 cm blocks, 0 .. 6 d cm tall

Every tile has a flat 0.6 m spawn pad at its centre and meets its neighbours at height 0 on its
border, so a robot that walks off a tile does not hit a cliff.

Collision for MJX/Warp (Warp collides heightfields with spheres, capsules and cylinders, not reliably
with boxes or meshes): base = plate cylinder + battery-tray cylinder + dome cylinder + camera capsule;
legs = coxa / femur / tibia / shank capsules + the foot sphere. Robot parts collide with the terrain
only (no self-collision, as in the flat MJX env).
"""
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["rough", "slope_up", "slope_down", "stairs_up", "stairs_dn", "blocks"]
MAX_STEP = 0.06          # m, tallest stair riser / block at difficulty 1
MAX_SLOPE_DEG = 15.0
MAX_ROCK = 0.05          # m


class Terrain:
    def __init__(self, heights, res, tile, levels, types, centers):
        self.heights = heights            # (ny, nx) metres, row = y (MuJoCo hfield order)
        self.res, self.tile, self.levels, self.types = res, tile, levels, types
        self.centers = centers            # (levels, ntypes, 3) spawn point (x, y, ground z)
        self.ny, self.nx = heights.shape
        self.size_x, self.size_y = self.nx * res, self.ny * res

    def height_at(self, x, y):
        i = np.clip(np.round((y + self.size_y / 2) / self.res).astype(int), 0, self.ny - 1)
        j = np.clip(np.round((x + self.size_x / 2) / self.res).astype(int), 0, self.nx - 1)
        return self.heights[i, j]


def _smooth(z, n):
    for _ in range(n):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    return z


def _tile(kind, d, n, res, rng, pad):
    """One tile (n x n cells) of terrain type `kind` at difficulty d; border at 0, centre pad flat."""
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = (xx - c) * res, (yy - c) * res
    cheb = np.maximum(np.abs(dx), np.abs(dy))          # square "radius" from the centre
    half = n * res / 2
    z = np.zeros((n, n))
    if kind == "rough":
        amp = 0.01 + (MAX_ROCK - 0.01) * d
        z = _smooth(rng.random((n, n)), 2)
        z = (z - z.min()) / (np.ptp(z) + 1e-9) * amp * 0.6
        for _ in range(int(60 + 140 * d)):            # scattered stones
            sx, sy = rng.uniform(-half, half, 2)
            r = rng.uniform(0.03, 0.10)
            h = rng.uniform(0.3, 1.0) * amp
            dd = np.hypot(dx - sx, dy - sy)
            z = np.maximum(z, h * np.sqrt(np.clip(1 - (dd / r) ** 2, 0, 1)))
    elif kind in ("slope_up", "slope_down"):
        g = math.tan(math.radians(MAX_SLOPE_DEG * d))
        z = g * (half - np.maximum(cheb, pad / 2))        # top = pad, falls to 0 at the border
        if kind == "slope_down":
            z = -z
    elif kind in ("stairs_up", "stairs_dn"):
        h = 0.01 + (MAX_STEP - 0.01) * d
        tread = 0.30
        k = np.floor((half - np.maximum(cheb, pad / 2)) / tread)   # steps counted from the border
        z = h * k
        if kind == "stairs_dn":
            z = -z
    elif kind == "blocks":
        hmax = MAX_STEP * d
        for _ in range(40):
            sx, sy = rng.uniform(-half, half, 2)
            w = rng.uniform(0.30, 0.50)
            m = (np.abs(dx - sx) < w / 2) & (np.abs(dy - sy) < w / 2)
            z[m] = rng.uniform(0.0, hmax)
    # flat spawn pad, and a 3-cell border ramp to 0 so tiles never form cliffs against each other
    padm = cheb <= pad / 2
    if padm.any():
        z[padm] = z[padm].mean() if kind in ("rough", "blocks") else z[padm].max() if kind.endswith("up") else z[padm].min()
    edge = np.clip((half - cheb) / (3 * res), 0, 1)
    return z * edge


def generate(levels=10, types=TYPES, tile=4.0, res=0.025, pad=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(tile / res))
    H = np.zeros((levels * n, len(types) * n))
    centers = np.zeros((levels, len(types), 3))
    for li in range(levels):
        d = li / max(levels - 1, 1)
        for ti, kind in enumerate(types):
            z = _tile(kind, d, n, res, rng, pad)
            H[li * n:(li + 1) * n, ti * n:(ti + 1) * n] = z
            centers[li, ti] = [(ti + 0.5) * tile - len(types) * tile / 2, (li + 0.5) * tile - levels * tile / 2,
                               z[n // 2, n // 2]]
    return Terrain(H, res, tile, levels, list(types), centers)


# ---------------------------------------------------------------------------------------------------
# MJX collision primitives (body frame / link frames, metres)
BASE_COLLIDERS = [
    # name, type, size, pos, fromto(for capsule)
    ("t_plate", "cylinder", [0.120, 0.0202], [0, 0, 0.0], None),
    ("t_tray", "cylinder", [0.060, 0.0075], [0, 0, -0.0282], None),
    ("t_dome", "cylinder", [0.105, 0.035], [0, 0, 0.079], None),
    ("t_camera", "capsule", [0.013], None, [-0.040, -0.030, 0.130, 0.040, -0.030, 0.130]),
]
LEG_COLLIDERS = {   # body suffix -> list of (name, radius, fromto in that link frame)
    "coxa": [("t_coxa", 0.020, [-0.012, 0, 0, 0.062, 0, 0])],
    "femur": [("t_femur", 0.018, [-0.010, 0, 0, 0.100, 0, 0])],
    "tibia": [("t_tibia", 0.020, [0.004, 0, 0.014, 0.004, 0, -0.054]),
              ("t_shank", 0.009, [0.008, 0, -0.058, 0.008, 0, -0.112])],
}
TERRAIN_BIT, ROBOT_BIT = 1, 2


def build_spec(terrain, scene="scene.xml"):
    """MjSpec of the robot on the terrain heightfield, with MJX collision primitives and contact
    sensors (base parts, and every non-foot leg part, against the terrain)."""
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, scene))
    H = terrain.heights
    zmin, zmax = float(H.min()), float(H.max())
    span = max(zmax - zmin, 1e-3)
    spec.add_hfield(name="terrain", size=[terrain.size_x / 2, terrain.size_y / 2, span, 0.1],
                    nrow=terrain.ny, ncol=terrain.nx, userdata=((H - zmin) / span).ravel().tolist())
    floor = spec.geom("floor")
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "terrain"
    floor.pos = [0, 0, zmin]
    floor.contype, floor.conaffinity = TERRAIN_BIT, ROBOT_BIT
    # robot: switch off every existing collider except the feet, add the primitives
    for g in spec.geoms:
        if g.name != "floor" and not g.name.endswith("_foot"):
            g.contype = g.conaffinity = 0
    for g in spec.geoms:
        if g.name.endswith("_foot"):
            g.contype, g.conaffinity = ROBOT_BIT, TERRAIN_BIT
    def add(body, name, kind, size, pos=None, fromto=None):
        g = body.add_geom(name=name, group=3, rgba=[0.9, 0.4, 0.2, 0.5], contype=ROBOT_BIT, conaffinity=TERRAIN_BIT,
                          friction=[0.6, 0.005, 0.0001], density=0)
        g.type = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}[kind]
        g.size = list(size) + [0] * (3 - len(size))
        if fromto is not None:
            g.fromto = fromto
        else:
            g.pos = pos
        return g
    base = spec.body("base")
    for name, kind, size, pos, ft in BASE_COLLIDERS:
        add(base, name, kind, size, pos, ft)
    for n in range(1, 6):
        for link, cols in LEG_COLLIDERS.items():
            b = spec.body(f"leg{n}_{link}")
            for name, r, ft in cols:
                add(b, f"leg{n}_{name}", "capsule", [r], fromto=ft)
    # contact sensors: base hit (termination) and leg shin / link hits (penalty)
    for name, *_ in BASE_COLLIDERS:
        spec.add_sensor(name=f"{name}_hit", type=mujoco.mjtSensor.mjSENS_CONTACT, objtype=mujoco.mjtObj.mjOBJ_GEOM,
                        objname=name, reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="floor", intprm=[1, 0, 1])
    for n in range(1, 6):
        for link, cols in LEG_COLLIDERS.items():
            for name, *_ in cols:
                spec.add_sensor(name=f"leg{n}_{name}_hit", type=mujoco.mjtSensor.mjSENS_CONTACT,
                                objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=f"leg{n}_{name}",
                                reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="floor", intprm=[1, 0, 1])
    return spec


def build_model(terrain, scene="scene.xml"):
    return build_spec(terrain, scene).compile()


def build_render_model(terrain, tiles, scene="scene.xml"):
    """Render-only model: the same robot, but the ground is just the listed tiles [(type, level), ...],
    each its own small heightfield at its real position. Drawing the full 1.5 M-cell heightfield made
    every video frame ~10x slower; physics always uses build_model()."""
    spec = mujoco.MjSpec.from_file(os.path.join(HERE, scene))
    floor = spec.geom("floor")                    # kept (contact sensors name it) but hidden: geom group 5
    floor.group, floor.contype, floor.conaffinity = 5, 0, 0
    n = int(round(terrain.tile / terrain.res))
    for k, (kind, lv) in enumerate(tiles):
        ti = terrain.types.index(kind)
        z = terrain.heights[lv * n:(lv + 1) * n, ti * n:(ti + 1) * n]
        zmin, span = float(z.min()), max(float(np.ptp(z)), 1e-3)
        spec.add_hfield(name=f"tile{k}", size=[terrain.tile / 2, terrain.tile / 2, span, 0.05], nrow=n, ncol=n,
                        userdata=((z - zmin) / span).ravel().tolist())
        cx, cy = terrain.centers[lv, ti, 0], terrain.centers[lv, ti, 1]
        spec.worldbody.add_geom(name=f"tile{k}", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=f"tile{k}",
                                pos=[cx - terrain.res / 2, cy - terrain.res / 2, zmin], material="groundplane",
                                contype=0, conaffinity=0)
    return spec.compile()


def overview_png(terrain, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(terrain.heights, origin="lower", cmap="terrain",
                   extent=[-terrain.size_x / 2, terrain.size_x / 2, -terrain.size_y / 2, terrain.size_y / 2])
    for ti, kind in enumerate(terrain.types):
        ax.text(terrain.centers[0, ti, 0], -terrain.size_y / 2 - 0.8, kind, ha="center", fontsize=8)
    ax.set_ylabel("difficulty level 0 (bottom) .. %d (top)" % (terrain.levels - 1))
    fig.colorbar(im, ax=ax, label="height (m)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    T = generate()
    print(f"heightfield {T.nx} x {T.ny} cells, {T.size_x:.0f} x {T.size_y:.0f} m, z {T.heights.min():.3f} .. {T.heights.max():.3f} m")
    for ti, kind in enumerate(T.types):
        n = int(round(T.tile / T.res))
        tiles = [T.heights[li * n:(li + 1) * n, ti * n:(ti + 1) * n] for li in (0, T.levels // 2, T.levels - 1)]
        def maxstep(z):
            return max(np.abs(np.diff(z, axis=0)).max(), np.abs(np.diff(z, axis=1)).max())
        print(f"  {kind:10s} max height step per cell (level 0 / mid / max): "
              + " / ".join(f"{maxstep(z) * 100:4.1f} cm" for z in tiles)
              + "   relief " + " / ".join(f"{np.ptp(z) * 100:4.1f} cm" for z in tiles))
    m = build_model(T)
    print("model compiles:", m.ngeom, "geoms,", m.nsensor, "sensors, hfield", m.hfield_nrow[0], "x", m.hfield_ncol[0])
    overview_png(T, os.path.join(HERE, "media", "terrain_overview.png"))
    print("wrote media/terrain_overview.png")
