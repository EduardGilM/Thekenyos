"""Seeded kiwi pergola geometry, independent of the physics runtime.

Order 0 is posts/beams, order 1 is support wires, and order 2 is fruiting
canes. The builder fixes orders 0/1 and gives canes its existing compliant
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
             columns: int = 40, spacing: float = 5.0) -> TreeSkeleton:
    """Generate a connected commercial plantation of kiwi pergola rows.

    ``rows`` and ``columns`` count structural post lines, not fruiting plants.
    The default 45 x 40 layout at 5 m centres is about 4.3 ha. Main beams,
    row connectors and posts are structural; paired long canes fill each
    corridor while keeping the grid connected to one rooted skeleton. Height
    denotes the support/cane centreline above level ground. Dimensions and
    material values are initial scene parameters, not a calibrated crop model.
    """
    if not np.isfinite(height) or height < 0.3:
        raise ValueError("canopy height must be finite and at least 0.3 m")
    if int(rows) != rows or int(columns) != columns or rows < 2 or columns < 2:
        raise ValueError("pergola rows and columns must be integers >= 2")
    if not np.isfinite(spacing) or not 4.5 <= spacing <= 5.0:
        raise ValueError("pergola structural spacing must be between 4.5 and 5 m")
    rows, columns = int(rows), int(columns)
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

    xs = (np.arange(columns) - 0.5 * (columns - 1)) * spacing
    ys = (np.arange(rows) - 0.5 * (rows - 1)) * spacing

    # One rooted, connected fixed frame avoids closed kinematic loops. The
    # row beams snake through the field; short end headers connect each row.
    direction = 1
    root = add(-1, [xs[0], ys[0], height], 0.045, 0,
               start=[xs[0], ys[0], 0.])
    previous_end = root
    for yi, y in enumerate(ys):
        if yi:
            # Continue from the actual end of the previous row. The traversal
            # direction has already flipped for this row.
            end_x = segments[previous_end].end[0]
            previous_end = add(previous_end, [end_x, y, height], 0.025, 0)
        order = range(columns) if direction > 0 else range(columns - 1, -1, -1)
        nodes = []
        beam = previous_end
        for xi in order:
            if xi != (0 if direction > 0 else columns - 1):
                beam = add(beam, [xs[xi], y, height], 0.025, 0)
            nodes.append((xi, beam))
            if not (yi == 0 and xi == 0):
                add(beam, [xs[xi], y, 0.], 0.045, 0)

        if yi < rows - 1:
            next_y = ys[yi + 1]
            for xi, node in nodes:
                # Three parallel fruiting wires spread vegetation through the
                # whole 5 m working corridor without turning every vine into a
                # separate physics articulation.
                for fraction in (0.25, 0.50, 0.75):
                    wire_y = y + fraction * (next_y - y)
                    wire = add(node, [xs[xi], wire_y, height], 0.004, 1)
                    # Long paired canes make the canopy continuous between
                    # posts; their foliage remains massless/render-only.
                    inward = -1 if xi == columns - 1 else 1
                    for side in (inward, -inward):
                        if ((xi == 0 and side < 0) or
                                (xi == columns - 1 and side > 0)):
                            continue
                        length = rng.uniform(2.15, 2.40)
                        end = np.array([xs[xi] + side * length,
                                        wire_y + rng.uniform(-0.08, 0.08), height])
                        add(wire, end, rng.uniform(0.006, 0.009), 2)
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
