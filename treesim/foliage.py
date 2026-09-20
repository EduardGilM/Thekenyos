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
# Shared blade styles. Apple stays the coarser elliptic card; kiwi cordate
# uses more segments, a basal sinus and a serrated margin (artistic proxy).
LEAF_BLADE_STYLE = {
    "elliptic": dict(fold=0.55, curl=0.30, droop=0.35, nseg=5),
    "cordate": dict(fold=0.72, curl=0.12, droop=0.28, nseg=10),
}


def leaf_blade_style(shape: str = "elliptic") -> dict:
    if shape not in LEAF_BLADE_STYLE:
        raise ValueError("leaf shape must be elliptic or cordate")
    return dict(LEAF_BLADE_STYLE[shape], shape=shape)


def leaf_blade_arrays(length: float, width: float, fold: float = 0.55,
                     curl: float = 0.30, droop: float = 0.35, nseg: int = 5,
                     shape: str = "elliptic"):
    """Folded blade vertices/faces. ``shape`` is elliptic (apple) or cordate (kiwi).

    Cordate outline is an artistic Actinidia-style proxy, not a scanned cultivar.
    """
    if shape not in LEAF_BLADE_STYLE:
        raise ValueError("leaf shape must be elliptic or cordate")
    ts = np.linspace(0.0, 1.0, nseg + 1)
    verts: list[tuple] = []
    rows: list[tuple] = []
    across = 5 if shape == "cordate" else 3
    for t in ts:
        tt = min(float(t), 0.995)
        if shape == "cordate":
            sinus = float(np.exp(-(tt / 0.065) ** 2))
            lobe = float(np.sin(np.pi * min(tt / 0.34, 1.0)) ** 0.80)
            body = float(np.sin(np.pi * (tt ** 0.62)) ** 0.72)
            envelope = (0.88 * body + 0.42 * lobe * (1.0 - tt)) * (1.0 - 0.62 * sinus)
            envelope = max(envelope, 0.035 * (1.0 - tt) + 0.02)
            serration = 1.0
            if 0.08 < tt < 0.92:
                serration += 0.08 * float(np.sin(12.0 * np.pi * tt))
            w = 0.5 * width * envelope * serration
        else:
            w = 0.5 * width * (np.sin(np.pi * tt ** 0.8) ** 0.85 + 0.03)
        z = length * t
        y_rib = curl * length * t * t - droop * length * t * t * t
        i0 = len(verts)
        if across == 3:
            y_edge = y_rib + fold * w
            verts.extend(((-w, y_edge, z), (0.0, y_rib, z), (w, y_edge, z)))
        else:
            for u in (-1.0, -0.52, 0.0, 0.52, 1.0):
                cup = fold * w * u * u
                twist = 0.07 * fold * w * u * tt
                verts.append((u * w, y_rib + cup + twist, z))
        rows.append(tuple(range(i0, i0 + across)))
    idx: list[int] = []
    for r in range(nseg):
        a, b = rows[r], rows[r + 1]
        for k in range(across - 1):
            idx += [a[k], a[k + 1], b[k], a[k + 1], b[k + 1], b[k]]
    back = []
    for k in range(0, len(idx), 3):
        back += [idx[k], idx[k + 2], idx[k + 1]]
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(idx + back, dtype=np.int32))


def leaf_mesh(length: float, width: float, fold: float = 0.55,
              curl: float = 0.30, droop: float = 0.35, nseg: int = 5,
              shape: str = "elliptic"):
    """Return a :class:`newton.Mesh` leaf blade.

    Local frame matches the old cards: +Z along the blade from the petiole,
    +X across the blade.  The blade folds up along the midrib (``fold``),
    lifts/curls toward the tip (``curl``) and droops down overall (``droop``),
    so it catches light like a real leaf instead of a flat card.
    """
    import newton
    verts, faces = leaf_blade_arrays(length, width, fold, curl, droop, nseg, shape)
    return newton.Mesh(verts, faces, compute_inertia=False, is_solid=False)


# size classes: a few DISCRETE sizes -> a few instance batches (a continuous
# per-leaf size would make every leaf a unique geometry and tank the fps)
LEAF_SIZE_CLASSES = (0.72, 1.0, 1.35)


def leaf_meshes(fp: FoliageParams):
    """One shared blade mesh per size class for this config's leaf size."""
    style = leaf_blade_style(getattr(fp, "leaf_shape", "elliptic"))
    return [leaf_mesh(fp.leaf_length * s, fp.leaf_width * s, **style)
            for s in LEAF_SIZE_CLASSES]


def _frame_quat(H, L, U):
    from .lsystem import _frame_to_quat
    return _frame_to_quat(H / np.linalg.norm(H), L / np.linalg.norm(L), U / np.linalg.norm(U))


def rotate_xyzw(q, v) -> np.ndarray:
    """Rotate vec3 ``v`` by xyzw quaternion ``q``."""
    u = np.array([q[0], q[1], q[2]], dtype=float)
    w = float(q[3])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _horizontal_blade_frame(plane_n, prefer_heading, rng, tilt_rad=0.18,
                            yaw_rad=0.70, roll_rad=0.22) -> np.ndarray:
    """Blade in the canopy plane: +Z along the midrib, +Y ~ plane normal.

    Yaw/tilt/roll stay small so the roof looks a bit messy without standing
    the leaves on edge. This is an artistic kiwi-canopy proxy, not a scan.
    """
    n = np.asarray(plane_n, dtype=float)
    n = n / np.linalg.norm(n)
    h = np.asarray(prefer_heading, dtype=float)
    h = h - n * np.dot(h, n)
    if np.linalg.norm(h) < 1e-8:
        h = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(h) < 1e-8:
            h = np.cross(n, np.array([0.0, 1.0, 0.0]))
    h = h / np.linalg.norm(h)
    h = _rodrigues(h, n, float(rng.uniform(-yaw_rad, yaw_rad)))
    tilt_axis = np.cross(n, h)
    tn = np.linalg.norm(tilt_axis)
    if tn > 1e-8:
        h = _rodrigues(h, tilt_axis / tn, float(rng.uniform(-tilt_rad, tilt_rad)))
        h = h / np.linalg.norm(h)
    left = np.cross(n, h)
    left = left / np.linalg.norm(left)
    left = _rodrigues(left, h, float(rng.uniform(-roll_rad, roll_rad)))
    up = np.cross(h, left)
    up = up / np.linalg.norm(up)
    if np.dot(up, n) < 0.0:
        left, up = -left, -up
    return _frame_quat(h, left, up)


def _cane_samples(canes, step_m: float = 0.45):
    pts, dirs, owners = [], [], []
    for i, cane in enumerate(canes):
        n = max(2, int(np.ceil(cane.length / step_m)))
        for t in np.linspace(0.0, 1.0, n):
            pts.append(cane.start + t * (cane.end - cane.start))
            dirs.append(cane.direction)
            owners.append(i)
    return np.asarray(pts), np.asarray(dirs), np.asarray(owners, dtype=int)


def _roof_bar_samples(skel: TreeSkeleton, step_m: float = 0.45):
    """Tied wires and canes that define the leaf roof (not hanging laterals)."""
    bars = [s for s in skel if s.order == 1 or (s.order == 2 and s.supported)]
    if not bars:
        bars = [s for s in skel if s.order == 2]
    return _cane_samples(bars, step_m)


def _heights_at(xy, height_z, roof_pts) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if height_z is not None:
        return np.array([float(height_z(float(x), float(y))) for x, y in xy], dtype=float)
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    pts = np.asarray(roof_pts, dtype=float)
    z = LinearNDInterpolator(pts[:, :2], pts[:, 2])(xy)
    miss = ~np.isfinite(z)
    if np.any(miss):
        z = np.asarray(z, dtype=float)
        z[miss] = NearestNDInterpolator(pts[:, :2], pts[:, 2])(xy[miss])
    return np.asarray(z, dtype=float)


def _normals_at(xy, height_z, roof_pts, eps: float = 0.30) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    z0 = _heights_at(xy, height_z, roof_pts)
    zx = _heights_at(xy + np.array([eps, 0.0]), height_z, roof_pts)
    zy = _heights_at(xy + np.array([0.0, eps]), height_z, roof_pts)
    n = np.column_stack((-(zx - z0) / eps, -(zy - z0) / eps, np.ones(len(xy))))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return n


def place_canopy_leaves(skel: TreeSkeleton, fp: FoliageParams,
                        seed: int = 0, height_z=None) -> list[LeafPlacement]:
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
    roof_pts, _, _ = _roof_bar_samples(skel)
    z = _heights_at(xy, height_z, roof_pts) + rng.uniform(.04, .14, len(xy))
    normals = _normals_at(xy, height_z, roof_pts)
    cane_pts, cane_dirs, cane_owners = _cane_samples(canes)
    from scipy.spatial import cKDTree
    _, near = cKDTree(cane_pts[:, :2]).query(xy)
    out = []
    half = 0.5 * fp.leaf_length
    for i, center_xy in enumerate(xy):
        cane = canes[int(cane_owners[near[i]])]
        bar = cane_pts[near[i]]
        toward = np.array([center_xy[0] - bar[0], center_xy[1] - bar[1], 0.0])
        if np.linalg.norm(toward) < 1e-4:
            toward = np.cross(normals[i], cane_dirs[near[i]])
        frame = _horizontal_blade_frame(normals[i], toward, rng)
        heading = rotate_xyzw(frame, np.array([0.0, 0.0, 1.0]))
        center = np.array([center_xy[0], center_xy[1], z[i]])
        out.append(LeafPlacement(
            parent_seg=cane.index,
            attach=center - half * heading,
            frame=frame,
            length=fp.leaf_length, width=fp.leaf_width,
        ))
    return out


def place_leaves(skel: TreeSkeleton, fp: FoliageParams,
                 seed: int = 0, height_z=None) -> list[LeafPlacement]:
    """Return leaf placements for all eligible twigs."""
    rng = np.random.default_rng(seed + 777)
    out: list[LeafPlacement] = []
    max_order = max(s.order for s in skel.segments)
    thr = min(fp.min_order_for_leaves, max_order)
    kiwi = getattr(fp, "leaf_shape", "elliptic") == "cordate"
    roof_pts = _roof_bar_samples(skel)[0] if kiwi else None

    for seg in skel.segments:
        if seg.order < thr:
            continue
        # leaf-bearing twigs: terminal twigs, or any twig at/after threshold
        if not (seg.is_terminal or seg.order >= thr):
            continue
        H = seg.direction
        nleaf = fp.leaves_per_terminal if seg.is_terminal else max(1, fp.leaves_per_terminal // 2)
        if kiwi:
            along = np.array([H[0], H[1], 0.0])
            if np.linalg.norm(along) < 0.05:
                along = np.array([1.0, 0.0, 0.0])
            along = along / np.linalg.norm(along)
            for k in range(nleaf):
                t = (k + 1) / (nleaf + 1)
                p = seg.start + H * (t * seg.length)
                xy = np.array([[p[0], p[1]]])
                plane_n = _normals_at(xy, height_z, roof_pts)[0]
                side = np.cross(plane_n, along)
                sn = np.linalg.norm(side)
                side = side / sn if sn > 1e-8 else np.array([1.0, 0.0, 0.0])
                lateral = side if (k % 2 == 0) else -side
                z = float(_heights_at(xy, height_z, roof_pts)[0]
                          + rng.uniform(0.03, 0.10))
                base = np.array([p[0], p[1], z]) + lateral * rng.uniform(0.012, 0.038)
                out.append(LeafPlacement(
                    parent_seg=seg.index,
                    attach=base,
                    frame=_horizontal_blade_frame(plane_n, lateral, rng,
                                                  tilt_rad=0.20, yaw_rad=0.85),
                    length=fp.leaf_length * max(0.4, 1.0 + rng.normal(0, 0.15)),
                    width=fp.leaf_width * max(0.4, 1.0 + rng.normal(0, 0.15)),
                ))
            continue
        ref = np.array([0.0, 0.0, 1.0]) if abs(H[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        L = np.cross(ref, H); L /= np.linalg.norm(L)
        U = np.cross(H, L)
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
