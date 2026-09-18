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


def generate(height: float = 1.6, seed: int = 0,
             ground_z=None, canopy_z=None) -> TreeSkeleton:
    """One 3 x 4 m bay, with 0.5 m wire spacing and randomized paired canes.

    Height denotes the support/cane centreline above the aisle plane. On
    level ground that is world z. When ``ground_z`` / ``canopy_z`` are
    provided they are ``(x, y) -> z`` samples so posts sit on the orchard
    floor and the wires follow the slope plane. Branch radii and leaves
    extend above the canopy. Fruit hangs below it. Dimensions and material
    values are initial scene parameters, not a calibrated crop model.
    """
    if not np.isfinite(height) or height < 0.3:
        raise ValueError("canopy height must be finite and at least 0.3 m")
    planted = ground_z is not None
    if ground_z is None:
        ground_z = lambda x, y: 0.0
    if canopy_z is None:
        canopy_z = lambda x, y: height
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

    def post_ends(x, y):
        from .orchard_terrain import POST_EMBED_M
        top = float(canopy_z(x, y))
        foot = float(ground_z(x, y)) - (POST_EMBED_M if planted else 0.0)
        if not np.isfinite(top) or not np.isfinite(foot):
            raise ValueError("terrain samples must be finite")
        if top - foot < 0.3:
            raise ValueError("post would be shorter than 0.3 m at this terrain")
        return foot, top

    # One rooted, connected fixed frame avoids closed kinematic loops.
    foot, top = post_ends(-1.5, -2.)
    root = add(-1, [-1.5, -2., top], 0.045, 0, start=[-1.5, -2., foot])
    beam = root
    for xi, x in enumerate(np.linspace(-1.5, 1.5, 7)):
        if xi:
            beam = add(beam, [x, -2., float(canopy_z(x, -2.))], 0.025, 0)
        wire = beam
        for yi, y in enumerate(np.linspace(-1.5, 2., 8)):
            wire = add(wire, [x, y, float(canopy_z(x, y))], 0.004, 1)
            # Each cane occupies its own grid cell, preventing overlapping
            # fruit pairs on adjacent wires.
            if xi < 6:
                length = rng.uniform(0.34, 0.44)
                ex, ey = x + length, y + rng.uniform(-0.06, 0.0)
                add(wire, [ex, ey, float(canopy_z(ex, ey))], rng.uniform(0.006, 0.009), 2)
        if xi in (0, 6):
            foot, _ = post_ends(x, 2.)
            add(wire, [x, 2., foot], 0.045, 0)
    foot, _ = post_ends(1.5, -2.)
    add(beam, [1.5, -2., foot], 0.045, 0)
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
