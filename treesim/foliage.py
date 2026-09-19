"""Foliage generation: leaf placement on the skeleton.

Leaves are placed on the thin outer twigs (branch order >= a threshold).  Each
leaf becomes, in the Newton builder, a light card (thin box) attached to its
parent twig by a compliant *petiole* joint so it flutters when the branch moves
or a force/wind is applied.  Rendering can replace the card with an
folded, curled elliptical blade mesh.

This module only computes placements (pure geometry); the actual bodies/joints
are added by :mod:`treesim.builder` so everything lives in one Model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FoliageParams
from .skeleton import TreeSkeleton


@dataclass
class LeafPlacement:
    parent_seg: int          # twig segment the leaf grows from
    attach: np.ndarray       # world position of the petiole base
    frame: np.ndarray        # xyzw quaternion: local +Z = leaf out-direction
    length: float
    width: float


def _rodrigues(v, axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


# --------------------------------------------------------------------------- #
# Leaf geometry: a folded, curled elliptical blade (double-sided mesh).
#
# All leaves of one size class share ONE Mesh object, so the GL viewer
# instances them exactly like the old identical boxes — a handful of size
# classes means a handful of draw batches regardless of leaf count.  ~40
# triangles per leaf mesh; leaves stay massless & non-colliding, so physics
# cost is still zero.
# --------------------------------------------------------------------------- #
def leaf_mesh(length: float, width: float, fold: float = 0.55,
              curl: float = 0.30, droop: float = 0.35, nseg: int = 5):
    """Return a :class:`newton.Mesh` leaf blade.

    Local frame matches the old cards: +Z along the blade from the petiole,
    +X across the blade.  The blade folds up along the midrib (``fold``),
    lifts/curls toward the tip (``curl``) and droops down overall (``droop``),
    so it catches light like a real leaf instead of a flat card.
    """
    import newton
    ts = np.linspace(0.0, 1.0, nseg + 1)
    verts: list[tuple] = []
    rows: list[tuple] = []       # (left, mid, right) vertex ids per row
    for t in ts:
        w = 0.5 * width * (np.sin(np.pi * min(t, 0.995) ** 0.8) ** 0.85 + 0.03)
        z = length * t
        y_rib = curl * length * t * t - droop * length * t * t * t
        y_edge = y_rib + fold * w
        i0 = len(verts)
        verts.append((-w, y_edge, z))
        verts.append((0.0, y_rib, z))
        verts.append((w, y_edge, z))
        rows.append((i0, i0 + 1, i0 + 2))
    idx: list[int] = []
    for r in range(nseg):
        l0, m0, r0 = rows[r]
        l1, m1, r1 = rows[r + 1]
        idx += [l0, m0, l1, m0, m1, l1,      # left strip
                m0, r0, m1, r0, r1, m1]      # right strip
    # double-sided: same triangles with flipped winding
    back = []
    for k in range(0, len(idx), 3):
        back += [idx[k], idx[k + 2], idx[k + 1]]
    return newton.Mesh(np.asarray(verts, dtype=np.float32),
                       np.asarray(idx + back, dtype=np.int32),
                       compute_inertia=False, is_solid=False)


# size classes: a few DISCRETE sizes -> a few instance batches (a continuous
# per-leaf size would make every leaf a unique geometry and tank the fps)
LEAF_SIZE_CLASSES = (0.72, 1.0, 1.35)


def leaf_meshes(fp: FoliageParams):
    """One shared blade mesh per size class for this config's leaf size."""
    return [leaf_mesh(fp.leaf_length * s, fp.leaf_width * s)
            for s in LEAF_SIZE_CLASSES]


def _frame_quat(H, L, U):
    from .lsystem import _frame_to_quat
    return _frame_to_quat(H / np.linalg.norm(H), L / np.linalg.norm(L), U / np.linalg.norm(U))


def place_canopy_leaves(skel: TreeSkeleton, fp: FoliageParams,
                        seed: int = 0) -> list[LeafPlacement]:
    spacing = fp.canopy_spacing_m
    if not np.isfinite(spacing) or spacing < .03:
        raise ValueError("canopy_spacing_m must be finite and at least 0.03 m")
    if fp.physics:
        raise ValueError("canopy infill is render-only")
    if not all(np.isfinite(v) and v > 0 for v in (fp.leaf_length, fp.leaf_width)):
        raise ValueError("canopy leaf dimensions must be finite and positive")
    canes = [s for s in skel if s.order == 2 and s.supported]
    if not canes:
        raise ValueError("canopy infill requires supported pergola canes")
    lo, hi = skel.bounds()
    nx, ny = (max(1, int(np.ceil(span / spacing))) for span in (hi - lo)[:2])
    if nx * ny > 100000:
        raise ValueError("canopy infill exceeds 100000 leaves; crop the pergola or increase spacing")
    rng = np.random.default_rng(seed + 1777)
    grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), axis=-1).reshape(-1, 2)
    cell_m = (hi - lo)[:2] / np.array([nx, ny])
    xy = lo[:2] + (grid + .5 + rng.uniform(-.3, .3, grid.shape)) * cell_m
    points = np.array([p for s in canes for p in (s.start, s.end)])
    plane = np.linalg.lstsq(np.column_stack((points[:, :2], np.ones(len(points)))),
                           points[:, 2], rcond=None)[0]
    z = xy @ plane[:2] + plane[2] + rng.uniform(.04, .18, len(xy))
    from scipy.spatial import cKDTree
    parents = cKDTree(np.array([s.midpoint[:2] for s in canes])).query(xy)[1]
    out = []
    for center, parent in zip(np.column_stack((xy, z)), parents):
        yaw = rng.uniform(0., 2 * np.pi)
        tilt = rng.uniform(-.2, .2)
        heading = np.array([np.cos(yaw) * np.cos(tilt),
                            np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        left = _rodrigues(left, heading, rng.uniform(-.25, .25))
        out.append(LeafPlacement(
            parent_seg=canes[int(parent)].index,
            attach=center - .5 * fp.leaf_length * heading,
            frame=_frame_quat(heading, left, np.cross(heading, left)),
            length=fp.leaf_length, width=fp.leaf_width,
        ))
    return out


def place_leaves(skel: TreeSkeleton, fp: FoliageParams,
                 seed: int = 0) -> list[LeafPlacement]:
    """Return leaf placements for all eligible twigs."""
    rng = np.random.default_rng(seed + 777)
    out: list[LeafPlacement] = []
    max_order = max(s.order for s in skel.segments)
    thr = min(fp.min_order_for_leaves, max_order)

    for seg in skel.segments:
        if seg.order < thr:
            continue
        # leaf-bearing twigs: terminal twigs, or any twig at/after threshold
        if not (seg.is_terminal or seg.order >= thr):
            continue
        H = seg.direction
        # build a frame off the twig direction
        ref = np.array([0.0, 0.0, 1.0]) if abs(H[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        L = np.cross(ref, H); L /= np.linalg.norm(L)
        U = np.cross(H, L)
        nleaf = fp.leaves_per_terminal if seg.is_terminal else max(1, fp.leaves_per_terminal // 2)
        for k in range(nleaf):
            # distribute along the twig and around it (phyllotaxis ~137.5 deg)
            t = (k + 1) / (nleaf + 1)
            base = seg.start + H * (t * seg.length)
            roll = np.deg2rad(137.5 * k + rng.uniform(0, 360))
            pitch = np.deg2rad(rng.uniform(45, 75))   # leaves splay outward/up
            Lr = _rodrigues(L, H, roll)
            Ur = _rodrigues(U, H, roll)
            outdir = _rodrigues(H, Lr, pitch)          # leaf points away from twig
            Uo = _rodrigues(Ur, Lr, pitch)
            jitter = 1.0 + rng.normal(0, 0.15)
            out.append(LeafPlacement(
                parent_seg=seg.index,
                attach=base.copy(),
                frame=_frame_quat(outdir, Lr, Uo),
                length=fp.leaf_length * max(0.4, jitter),
                width=fp.leaf_width * max(0.4, jitter),
            ))
    return out
