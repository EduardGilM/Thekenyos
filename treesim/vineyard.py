"""Seeded VSP (vertical shoot positioned) table-grape vineyard geometry.

Order 0 is trellis hardware (posts + catch/cordon wires), order 1 is old wood
(trunks + bilateral cordon arms), order 2 is compliant fruiting shoots.  The
builder fixes orders 0/1 and gives shoots its existing compliant joints.  One
rooted, connected tree without kinematic loops: the top catch wire chains post
to post, remaining posts hang down from it, and extra rows link at top-wire
height through the first post line.  All connections use parent endpoints,
matching TreeSkeleton's contract.
"""
import math
from dataclasses import dataclass

import numpy as np

from .config import FruitParams
from .grape_material import BERRY_RADIUS_RANGE, sample_cluster
from .lsystem import _frame_to_quat
from .skeleton import Segment, TreeSkeleton

POST_TOP = 1.9        # m, end-post / line-post height above ground
TOP_WIRE = 1.6        # m, top catch-wire height (the chained one)
MID_WIRE = 1.25       # m, second catch-wire height
SHOOT_DX = 0.25       # m, shoot spacing along each cordon arm


@dataclass
class GrapePlacement:
    parent_seg: int
    attach: np.ndarray          # world point on the shoot where the peduncle starts
    radius: float               # equatorial semi-axis [m]
    half_height: float          # legacy interface: axial semi-axis = radius + half_height
    color: tuple
    radii: np.ndarray           # xyz semi-axes of the cluster ellipsoid [m]
    mass: float                 # kg
    peduncle_diameter: float    # m
    peduncle_length: float      # m
    berries: np.ndarray         # (N,3) visual berry offsets from cluster centre [m]
    berry_radius: float         # m


def generate_vsp(rows: int = 1, row_length: float = 6.0, row_spacing: float = 2.4,
             vine_spacing: float = 1.5, cordon_height: float = 0.9,
             seed: int = 0) -> TreeSkeleton:
    """``rows`` VSP rows along x, centred on the origin; row i sits at
    y = (i - (rows-1)/2) * row_spacing.  Dimensions are initial scene
    parameters, not a calibrated vineyard survey."""
    if not isinstance(rows, (int, np.integer)) or rows < 1:
        raise ValueError("rows must be an integer >= 1")
    vals = dict(row_length=row_length, row_spacing=row_spacing,
                vine_spacing=vine_spacing, cordon_height=cordon_height)
    if not all(np.isfinite(v) for v in vals.values()):
        raise ValueError("vineyard dimensions must be finite")
    if row_length < 1.5:
        raise ValueError("row_length must be at least 1.5 m")
    if row_spacing < 1.0:
        raise ValueError("row_spacing must be at least 1.0 m")
    if vine_spacing <= 0.5:
        raise ValueError("vine_spacing must exceed 0.5 m")
    if not 0.5 <= cordon_height <= 1.5:
        raise ValueError("cordon_height must be within [0.5, 1.5] m")

    rng = np.random.default_rng(seed)
    segments = []

    def add(parent, end, radius, order, start=None):
        start = np.asarray(start if parent < 0 else segments[parent].end, float)
        end = np.asarray(end, float)
        heading = end - start
        heading /= np.linalg.norm(heading)
        ref = np.array([0., 0., 1.]) if abs(heading[2]) < 0.9 else np.array([1., 0., 0.])
        left = np.cross(ref, heading)
        left /= np.linalg.norm(left)
        idx = len(segments)
        segments.append(Segment(
            idx, parent, start.copy(), end, radius, radius, order,
            0 if parent < 0 else segments[parent].depth + 1,
            frame=_frame_to_quat(heading, left, np.cross(heading, left)),
        ))
        return idx

    x0, x1 = -row_length / 2.0, row_length / 2.0
    n_post = max(2, int(math.ceil(row_length / 3.0)) + 1)   # both ends + every ~3 m
    post_x = np.linspace(x0, x1, n_post)

    # Vine positions, offset half a spacing so vines never coincide with posts.
    vine_x = []
    vx = x0 + vine_spacing / 2.0
    while vx < x1 - 1e-9:
        if not np.isclose(vx, post_x).any():
            vine_x.append(vx)
        vx += vine_spacing

    n_arm = max(1, int(round(vine_spacing * 0.5 / SHOOT_DX)))
    arm_dx = vine_spacing * 0.5 / n_arm

    def hang_post(node, x, y):
        """Line post hung from its top-wire node: stub above, then a downward
        chain with nodes at MID_WIRE and cordon_height for the wire spans."""
        add(node, [x, y, POST_TOP], 0.05, 0)
        mid = add(node, [x, y, MID_WIRE], 0.05, 0)
        cor = add(mid, [x, y, cordon_height], 0.05, 0)
        add(cor, [x, y, 0.0], 0.05, 0)
        return mid, cor

    def vine(vseg, x, y):
        """Trunk down to the ground, bilateral cordon arms (a 0.25 m-node chain
        each way) and a pair of compliant shoots at every arm node."""
        add(vseg, [x, y, 0.0], 0.03, 1)                  # trunk (old wood)
        for sgn in (1.0, -1.0):
            cur = vseg
            for a in range(n_arm):
                cur = add(cur, [x + sgn * (a + 1) * arm_dx, y, cordon_height],
                          0.02, 1)
                # fruiting shoots at each 0.25 m node: one leaning +y and one
                # -y, so the fruit wall carries two shoots per node
                for lean_sign in (1.0, -1.0):
                    lean = math.radians(rng.uniform(0.0, 10.0))
                    h = np.array([0.0, lean_sign * math.sin(lean), math.cos(lean)])
                    h /= np.linalg.norm(h)
                    length = rng.uniform(1.0, 1.3)
                    add(cur, np.asarray(segments[cur].end) + h * length,
                        rng.uniform(0.004, 0.006), 2)

    prev = None          # segment ending at (x0, y_{i-1}, TOP_WIRE)
    for i in range(rows):
        y = (i - (rows - 1) / 2.0) * row_spacing
        top = [None] * n_post
        mid = [None] * n_post
        cor = [None] * n_post
        if i == 0:
            # rooted first post, grown upward as a chain with nodes at each
            # wire height (the only parent<0 segment in the skeleton)
            cor[0] = add(-1, [x0, y, cordon_height], 0.05, 0, start=[x0, y, 0.0])
            mid[0] = add(cor[0], [x0, y, MID_WIRE], 0.05, 0)
            top[0] = add(mid[0], [x0, y, TOP_WIRE], 0.05, 0)
            add(top[0], [x0, y, POST_TOP], 0.05, 0)
        else:
            # fixed cross-row link at top-wire height along the first post line
            top[0] = add(prev, [x0, y, TOP_WIRE], 0.0025, 0)
            mid[0], cor[0] = hang_post(top[0], x0, y)
        for k in range(n_post):
            if k:
                mid[k], cor[k] = hang_post(top[k], post_x[k], y)
            if k + 1 >= n_post:
                continue
            xn = post_x[k + 1]
            top[k + 1] = add(top[k], [xn, y, TOP_WIRE], 0.0025, 0)  # chained
            add(mid[k], [xn, y, MID_WIRE], 0.0025, 0)               # span, open far end
            cur = cor[k]                                            # cordon chain
            for vxx in vine_x:
                if post_x[k] + 1e-9 < vxx < xn - 1e-9:
                    cur = add(cur, [vxx, y, cordon_height], 0.0025, 0)
                    vine(cur, vxx, y)
            add(cur, [xn, y, cordon_height], 0.0025, 0)             # ends at next post
        prev = top[0]
    return TreeSkeleton(segments)


def generate(rows=2, row_length=12.0, row_spacing=2.4, vine_spacing=1.5,
             cordon_height=2.3, seed=0):
    if isinstance(rows, bool) or not isinstance(rows, (int, np.integer)) or rows < 2:
        raise ValueError("an overhead vineyard requires at least two support rows")
    values = (row_length, row_spacing, vine_spacing, cordon_height)
    if not np.isfinite(values).all() or row_length < 1.5 or row_spacing < 1.5 \
            or vine_spacing <= 0.5 or not 1.8 <= cordon_height <= 3.0:
        raise ValueError("invalid canopy dimensions: height 1.8..3 m, spacing >=1.5 m")
    rng = np.random.default_rng(seed)
    segments = []

    def add(parent, end, radius, order, start=None):
        start = np.asarray(start if parent < 0 else segments[parent].end, float)
        end = np.asarray(end, float)
        heading = (end - start) / np.linalg.norm(end - start)
        ref = np.array([0., 0., 1.]) if abs(heading[2]) < .9 else np.array([1., 0., 0.])
        left = np.cross(ref, heading)
        left /= np.linalg.norm(left)
        index = len(segments)
        segments.append(Segment(index, parent, start.copy(), end, radius, radius,
                                order, 0 if parent < 0 else segments[parent].depth + 1,
                                frame=_frame_to_quat(heading, left, np.cross(heading, left))))
        return index

    xs = np.linspace(-row_length / 2, row_length / 2,
                     max(2, int(np.ceil(row_length / vine_spacing)) + 1))
    previous = None
    for row in range(rows):
        y = (row - (rows - 1) / 2) * row_spacing
        top = add(-1, [xs[0], y, cordon_height], .045, 0, [xs[0], y, 0.]) \
            if previous is None else add(previous, [xs[0], y, cordon_height], .009, 0)
        previous = top
        if row:
            add(top, [xs[0], y, 0.], .045, 0)
        for bay, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
            for x in np.linspace(x0, x1, 5)[1:]:
                top = add(top, [x, y, cordon_height], .016, 1)
                directions = ([1] if row == 0 else [-1] if row == rows - 1 else [-1, 1])
                for direction in directions:
                    cane = top
                    for k in range(1, 4):
                        end = [x + rng.uniform(-.08, .08),
                               y + direction * row_spacing * k / 6,
                               cordon_height + .10 * np.sin(k * np.pi / 3)]
                        cane = add(cane, end, .009, 1)
                        add(cane, np.asarray(end) + [rng.uniform(-.12, .12),
                            direction * .12, rng.uniform(-.035, .025)], .006, 2)
            add(top, [x1, y, 0.], .045, 0)
            trunk = add(top, [x1 - .10, y + .045, cordon_height * .52], .025, 1)
            add(trunk, [x1 + .04, y + .06, 0.], .03, 1)
            if row < rows - 1:
                add(top, [x1, y + row_spacing, cordon_height], .006, 0)
    return TreeSkeleton(segments)


def build_ground(builder, config):
    import newton
    from PIL import Image
    rng = np.random.default_rng(config.seed + 911)
    p = config.lsystem
    width = (p.vy_rows - 1) * p.vy_row_spacing / 2 + 6
    length = p.vy_row_length / 2 + 6
    h, w = 768, 1024
    noise = np.zeros((h, w), np.float32)
    for size, weight in ((12, .40), (48, .28), (192, .18), (768, .14)):
        layer = Image.fromarray(rng.uniform(0, 1, (size, size)).astype(np.float32))
        noise += weight * np.asarray(layer.resize((w, h), Image.Resampling.BILINEAR))
    y = np.linspace(-width, width, h)[:, None]
    track = np.zeros((h, w))
    for row in range(p.vy_rows - 1):
        alley = (row + .5 - (p.vy_rows - 1) / 2) * p.vy_row_spacing
        distance = np.abs(np.abs(y - alley) - .36)
        track = np.maximum(track, np.clip((.27 - distance + (noise - .5) * .22) / .13, 0, 1))
    grass = np.array([.27, .34, .12])
    soil = np.array([.43, .35, .24])
    color = grass[None, None, :] * (1 - track[..., None]) + soil * track[..., None]
    color *= (.55 + noise * .9 + rng.normal(0, .055, (h, w)))[..., None]
    texture = np.uint8(np.clip(color * 255, 0, 255))
    mesh = newton.Mesh(np.array([[-length, -width, .004], [length, -width, .004],
                                 [length, width, .004], [-length, width, .004]], np.float32),
                       np.array([0, 1, 2, 0, 2, 3], np.int32),
                       uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                       texture=texture, compute_inertia=False, is_solid=False)
    cfg = builder.ShapeConfig(density=0., collision_group=0,
                              has_shape_collision=False, has_particle_collision=False)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg, color=(1., 1., 1.))


def place_leaves(skeleton, params, seed=0):
    from .foliage import LeafPlacement
    rng = np.random.default_rng(seed + 777)
    leaves = []
    for seg in skeleton:
        if seg.order != 2:
            continue
        for _ in range(params.leaves_per_terminal):
            angle = rng.uniform(0, 2 * np.pi)
            heading = np.array([np.cos(angle), np.sin(angle), rng.uniform(-.4, .4)])
            heading /= np.linalg.norm(heading)
            left = np.cross([0., 0., 1.], heading)
            left /= np.linalg.norm(left)
            position = seg.start + rng.uniform(-.22, .22, 3)
            position[2] = seg.start[2] + rng.uniform(-.12, .20)
            leaves.append(LeafPlacement(seg.index, position,
                _frame_to_quat(heading, left, np.cross(heading, left)),
                params.leaf_length, params.leaf_width))
    return leaves


def leaf_meshes(params):
    import newton
    meshes = []
    for scale in (.8, 1., 1.2):
        n = 40
        vertices = [[0., -.012, params.leaf_length * scale * .48]]
        for i in range(n):
            angle = 2 * np.pi * i / n
            lobe = 1 + .20 * np.cos(5 * angle) + .035 * (-1) ** i
            vertices.append([scale * params.leaf_width * .44 * np.sin(angle) * lobe,
                             .018 * np.cos(2 * angle),
                             scale * params.leaf_length * (.48 + .43 * np.cos(angle) * lobe)])
        indices = []
        for i in range(n):
            a, b = i + 1, (i + 1) % n + 1
            indices.extend([0, a, b])
        front = np.asarray(indices, np.int32).reshape(-1, 3)
        indices = np.concatenate((front, front[:, ::-1] + len(vertices))).ravel()
        vertices = np.concatenate((vertices, vertices)).astype(np.float32)
        meshes.append(newton.Mesh(vertices, indices,
                                  compute_inertia=False, is_solid=False))
    return meshes


def berry_positions(radii, radius, rng):
    layers = max(5, int(2 * (radii[2] - radius) / (1.7 * radius)) + 1)
    points = []
    phase = rng.uniform(0., 2 * np.pi)
    for layer, t in enumerate(np.linspace(0., 1., layers)):
        z = (radii[2] - radius) * (1 - 2 * t)
        width = (radii[0] - radius) * np.interp(t, [0, .2, .55, 1], [.55, 1, .72, .05])
        count = max(1, int(2 * np.pi * width / (1.85 * radius)))
        for j in range(count):
            angle = phase + j * 2 * np.pi / count + layer * 2.4
            radial = width * rng.uniform(.94, 1.02)
            points.append([radial * np.cos(angle), radial * np.sin(angle),
                           z + rng.uniform(-.12, .12) * radius])
        if width > 2 * radius:
            points.append([0., 0., z])
    return np.asarray(points)


def place_fruit(skeleton: TreeSkeleton, params: FruitParams,
                seed: int = 0) -> list[GrapePlacement]:
    """One cluster candidate per fruiting shoot, shuffled before the count cap;
    candidates whose bounding sphere would touch an accepted cluster are
    skipped so no two clusters intersect in their initial pose."""
    if params.max_count < 0:
        raise ValueError("fruit count must be nonnegative")
    rng = np.random.default_rng(seed + 4242)
    candidates = [(s, rng.uniform(0.08, 0.22)) for s in skeleton if s.order == 2]
    rng.shuffle(candidates)
    # posts are the only trellis wood that collides (wires/arms/trunks/shoots
    # are built non-colliding): reject clusters spawning inside a post
    support = [s for s in skeleton if s.mean_radius >= 0.04]
    out = []
    centres = []                       # (centre, bounding radius) of accepted
    for seg, t in candidates:
        if len(out) >= params.max_count:
            break
        radii, mass, ped_d, ped_l = sample_cluster(rng)
        attach = seg.start + t * seg.axis
        centre = attach - np.array([0.0, 0.0, ped_l + radii[2]])
        bound = float(radii[2])
        if any(np.linalg.norm(centre - c) < bound + cb for c, cb in centres):
            continue
        hit = False
        for s in support:
            u = np.clip(np.dot(centre - s.start, s.direction), 0.0, s.length)
            d = np.linalg.norm(centre - (s.start + u * s.direction))
            if d < bound + s.radius_end + 0.005:
                hit = True
                break
        if hit:
            continue
        color = tuple(params.colors[int(rng.integers(len(params.colors)))])
        br = float(np.clip(round(rng.uniform(*BERRY_RADIUS_RANGE) / 0.001) * 0.001,
                           *BERRY_RADIUS_RANGE))
        berries = berry_positions(radii, br, rng)
        centres.append((centre, bound))
        out.append(GrapePlacement(
            seg.index, attach, float(radii[0]), float(radii[2] - radii[0]),
            color, radii, mass, ped_d, ped_l, berries, br))
    return out
