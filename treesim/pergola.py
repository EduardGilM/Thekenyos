"""Seeded kiwi pergola geometry, independent of the physics runtime.

Order 0 is posts/beams, order 1 is support wires, and order 2 is fruiting
canes. Tied cane sections are fixed; free tips use the existing compliant
joints. All connections use parent endpoints, matching TreeSkeleton's contract.
"""
from dataclasses import dataclass

import numpy as np

from .config import FruitParams
from .lsystem import _frame_to_quat
from .skeleton import Segment, TreeSkeleton


@dataclass
class KiwiPlacement:
    parent_seg: int
    attach: np.ndarray
    radius: float
    half_height: float  # legacy bounding-size interface: axial semi-axis = r+h
    color: tuple
    radii: np.ndarray
    mass: float


def generate(height: float = 1.6, seed: int = 0, rows: int = 45,
             columns: int = 40, spacing: float = 5.0,
             ground_z=None, canopy_z=None) -> TreeSkeleton:
    """Generate a connected commercial plantation of kiwi pergola rows.

    ``rows`` and ``columns`` count structural post lines, not fruiting plants.
    The default 45 x 40 layout at 5 m centres is about 4.3 ha. Main beams,
    row connectors and posts are structural; paired long canes fill each
    corridor while keeping the grid connected to one rooted skeleton. Height
    denotes the support/cane centreline above the aisle plane. On level
    ground that is world z. When ``ground_z`` / ``canopy_z`` are provided
    they are ``(x, y) -> z`` samples so posts sit on the orchard floor and
    the wires follow the slope plane. Dimensions and material values are
    initial scene parameters, not a calibrated crop model.
    """
    if not np.isfinite(height) or height < 0.3:
        raise ValueError("canopy height must be finite and at least 0.3 m")
    if int(rows) != rows or int(columns) != columns or rows < 2 or columns < 2:
        raise ValueError("pergola rows and columns must be integers >= 2")
    if not np.isfinite(spacing) or not 4.5 <= spacing <= 5.0:
        raise ValueError("pergola structural spacing must be between 4.5 and 5 m")
    rows, columns = int(rows), int(columns)
    planted = ground_z is not None
    if ground_z is None:
        ground_z = lambda x, y: 0.0
    if canopy_z is None:
        canopy_z = lambda x, y: height
    rng = np.random.default_rng(seed)
    segments = []

    def add(parent, end, radius, order, start=None, supported=False):
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
            supported=supported,
        ))
        return idx

    def post_ends(x, y):
        from .orchard_terrain import POST_EMBED_M
        top = float(canopy_z(x, y))
        foot = float(ground_z(x, y)) - (POST_EMBED_M if planted else 0.0)
        if not np.isfinite(top) or not np.isfinite(foot):
            raise ValueError("terrain samples must be finite")
        if top - foot < 0.3:
            raise ValueError("post would be shorter than 0.3 m at this terrain")
        return foot, top

    def canopy(x, y) -> float:
        z = float(canopy_z(x, y))
        if not np.isfinite(z):
            raise ValueError("terrain samples must be finite")
        return z

    xs = (np.arange(columns) - 0.5 * (columns - 1)) * spacing
    ys = (np.arange(rows) - 0.5 * (rows - 1)) * spacing

    # One rooted, connected fixed frame avoids closed kinematic loops. The
    # row beams snake through the field; short end headers connect each row.
    direction = 1
    foot0, top0 = post_ends(xs[0], ys[0])
    root = add(-1, [xs[0], ys[0], top0], 0.045, 0, start=[xs[0], ys[0], foot0])
    previous_end = root
    for yi, y in enumerate(ys):
        if yi:
            # Continue from the actual end of the previous row. The traversal
            # direction has already flipped for this row.
            end_x = float(segments[previous_end].end[0])
            previous_end = add(previous_end, [end_x, y, canopy(end_x, y)], 0.025, 0)
        order = range(columns) if direction > 0 else range(columns - 1, -1, -1)
        nodes = []
        beam = previous_end
        for xi in order:
            if xi != (0 if direction > 0 else columns - 1):
                beam = add(beam, [xs[xi], y, canopy(xs[xi], y)], 0.025, 0)
            nodes.append((xi, beam))
            if not (yi == 0 and xi == 0):
                foot, _ = post_ends(xs[xi], y)
                add(beam, [xs[xi], y, foot], 0.045, 0)

        if yi < rows - 1:
            next_y = ys[yi + 1]
            for xi, node in nodes:
                # Three parallel fruiting wires spread vegetation through the
                # whole 5 m working corridor without turning every vine into a
                # separate physics articulation.
                for fraction in (0.25, 0.50, 0.75):
                    wire_y = y + fraction * (next_y - y)
                    wire = add(node, [xs[xi], wire_y, canopy(xs[xi], wire_y)], 0.004, 1)
                    # Transverse support meets the next fixed row wire; avoid
                    # a redundant closed-loop joint in the rigid frame.
                    if xi < columns - 1:
                        add(wire, [xs[xi+1], wire_y, canopy(xs[xi+1], wire_y)], 0.004, 1)
                    # Long paired canes make the canopy continuous between
                    # posts; their foliage remains massless/render-only.
                    inward = -1 if xi == columns - 1 else 1
                    for side in (inward, -inward):
                        if ((xi == 0 and side < 0) or
                                (xi == columns - 1 and side > 0)):
                            continue
                        length = rng.uniform(2.15, 2.40)
                        ex = xs[xi] + side * length
                        radius = rng.uniform(0.006, 0.009)
                        # Ideal rigid ties secure the main cane to the wire.
                        # Only the final 0.35 m horizontal span is compliant.
                        # Tie/wire compliance is an engineering simplification.
                        tx = ex - side * .35
                        tied = add(wire, [tx, wire_y, canopy(tx, wire_y)],
                                   radius, 2, supported=True)
                        add(tied, [ex, wire_y, canopy(ex, wire_y)], radius, 2)
        previous_end = beam
        direction *= -1
    return TreeSkeleton(segments)


def place_fruit(skeleton: TreeSkeleton, params: FruitParams,
                seed: int = 0) -> list[KiwiPlacement]:
    """Place separated pairs on canes, shuffled before applying the count cap.

    All generated fruit is harvest-ready; there is no maturity filtering.
    """
    if params.max_count < 0:
        raise ValueError("fruit count must be nonnegative")
    rng = np.random.default_rng(seed + 4242)
    candidates = [(s, t) for s in skeleton if s.order == 2 for t in (0.35, 0.80)]
    rng.shuffle(candidates)
    out = []
    for seg, t in candidates[:params.max_count]:
        from .kiwi_material import sample_hayward
        radii, mass = sample_hayward(rng)
        radius = float(radii[0])
        out.append(KiwiPlacement(
            seg.index, seg.start + t * seg.axis, radius, float(radii[2]-radius),
            tuple(params.colors[int(rng.integers(len(params.colors)))]), radii, mass,
        ))
    return out
