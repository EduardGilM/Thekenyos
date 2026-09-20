#!/usr/bin/env python
"""Native MuJoCo cinematic still or animation of a kiwi street.

Fruit uses ``place_fruit`` hang and Hayward radii. The scene only adds
render-only leaf blades, timber visuals, a ground texture aligned to the
post rows and, for the animation, Spot robots driven by a *scripted*
kinematic trot with a smooth pseudo-random arm. Nothing here is the RELIC
gait policy, a learned behaviour or contact physics: poses are written
straight into ``qpos`` and the model is only used for forward kinematics
and rendering. Not Newton GL, not harvest.

    MUJOCO_GL=osmesa python scripts/render_kiwi_street.py \
        --snapshot output/kiwi-street.png
    MUJOCO_GL=osmesa python scripts/render_kiwi_street.py \
        --relic ../relic --video output/kiwi-street.mp4 --seconds 12
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from treesim.config import FoliageParams, FruitParams
from treesim.foliage import LEAF_SIZE_CLASSES, place_canopy_leaves, place_leaves
from treesim.gl_backend import bind_mujoco_gl
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit


BAYS = 5
AISLES = 3
ROWS = AISLES + 1
COLUMNS = BAYS + 1
SPACING_M = 5.0
CANOPY_Z_M = 1.6
POST_RADIUS_M = 0.095
BEAM_HALF_M = (0.070, 0.052)
FOOTER_RADIUS_M = 0.18
FOOTER_HALF_M = 0.055
LEAF_LENGTH_M = 0.22
LEAF_WIDTH_M = 0.17
# Still camera: azimuth 0 looks +X down the aisle.
CAMERA_LOOKAT = (2.4, 0.18, 0.48)
CAMERA_DISTANCE_M = 13.6
CAMERA_AZIMUTH_DEG = 18.0
CAMERA_ELEVATION_DEG = -11.0
# MuJoCo renders directional shadows from an orthographic light camera at the
# light's ``pos``: lateral half-size shadowclip*extent and depth range
# [0, shadowclip*extent] downstream of ``pos`` (measured with probes; casters
# upstream of ``pos`` are clipped). Track the camera's region of interest and
# back the eye off along the light direction so the roof stays in front.
SHADOW_HALF_M = 24.0
SHADOW_BACK_M = 12.0
SHADOW_FOCUS_Z_M = 0.8
SHADOW_MAP_PX = 8192
# Scripted arm: hand must stay under the beams. Planar FK constants from the
# RELIC Spot URDF (metres): shoulder above body, upper arm, forearm, hand.
ARM_SHOULDER_Z_M = 0.188
ARM_UPPER_M = (0.3385, 0.0)
ARM_FORE_M = (0.4033, 0.075)
ARM_HAND_M = (0.20, 0.015)
ARM_HAND_MAX_Z_M = CANOPY_Z_M - 0.30

SPOT_HOME = {
    "arm_sh0": 0.0, "arm_sh1": -0.9, "arm_el0": 1.8, "arm_el1": 0.0,
    "arm_wr0": -0.9, "arm_wr1": 0.0, "arm_f1x": -1.54,
    "fl_hx": 0.0, "fr_hx": 0.0, "hl_hx": 0.0, "hr_hx": 0.0,
    "fl_hy": 0.5, "fr_hy": 0.5, "hl_hy": 0.5, "hr_hy": 0.5,
    "fl_kn": -1.0, "fr_kn": -1.0, "hl_kn": -1.0, "hr_kn": -1.0,
}
SPOT_ARM = ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x")
# Centre and amplitude of the scripted arm wander, clipped to model ranges.
SPOT_ARM_WANDER = {
    "arm_sh0": (0.0, 0.70), "arm_sh1": (-0.85, 0.35), "arm_el0": (1.75, 0.40),
    "arm_el1": (0.0, 0.55), "arm_wr0": (-0.60, 0.50), "arm_wr1": (0.0, 0.9),
    "arm_f1x": (-0.8, 0.6),
}
SPOT_LEGS = ("fl", "fr", "hl", "hr")
SPOT_TROT_PHASE = {"fl": 0.0, "hr": 0.0, "fr": 0.5, "hl": 0.5}
SPOT_BODY_HEIGHT_M = 0.55
SPOT_FOOT_RADIUS_M = 0.035


def _rgba(rgb, a=1.0) -> str:
    return " ".join(f"{float(c):.3f}" for c in (*rgb, a))


def _png_bytes(pixels) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _quat_wxyz_from_z(direction) -> np.ndarray:
    """Rotate local +Z onto ``direction``. MuJoCo wxyz."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    xyz = np.cross(z, d)
    q = np.array([1.0 + c, xyz[0], xyz[1], xyz[2]])
    return q / np.linalg.norm(q)


def _mul_wxyz(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _euler_wxyz(roll, pitch, yaw) -> np.ndarray:
    def axis(angle, x, y, z):
        h = 0.5 * angle
        return np.array([np.cos(h), x * np.sin(h), y * np.sin(h), z * np.sin(h)])
    return _mul_wxyz(_mul_wxyz(axis(yaw, 0, 0, 1), axis(pitch, 0, 1, 0)), axis(roll, 1, 0, 0))


def leaf_card_quat(frame, extra_pitch_rad: float) -> str:
    """Leaf xyzw plus a local-X tilt so the roof is not a flat slab."""
    x, y, z, w = (float(v) for v in frame)
    base = np.array([w, x, y, z])
    half = 0.5 * float(extra_pitch_rad)
    tilt = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    q = _mul_wxyz(base, tilt)
    q /= np.linalg.norm(q)
    return f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"


def fruit_center(item) -> np.ndarray:
    """World fruit COM from ``place_fruit`` attach + Hayward radii."""
    return np.asarray(item.attach, dtype=np.float64) - np.array(
        [0.0, 0.0, STEM_LENGTH + float(item.radii[2])]
    )


def row_layout(rows: int = ROWS, columns: int = COLUMNS, spacing: float = SPACING_M):
    xs = (np.arange(columns) - 0.5 * (columns - 1)) * spacing
    ys = (np.arange(rows) - 0.5 * (rows - 1)) * spacing
    aisles = [0.5 * (ys[i] + ys[i + 1]) for i in range(rows - 1)]
    return [float(x) for x in xs], [float(y) for y in ys], [float(a) for a in aisles]


# --------------------------------------------------------------------------- #
# Procedural textures
# --------------------------------------------------------------------------- #
def _value_field(rng, shape, waves, dtype=np.float32):
    field = np.zeros(shape, dtype=dtype)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(dtype)
    for scale, weight in waves:
        gy, gx = max(int(shape[0] / scale), 2), max(int(shape[1] / scale), 2)
        grid = rng.standard_normal((gy, gx)).astype(dtype)
        y = yy / (shape[0] - 1) * (gy - 1)
        x = xx / (shape[1] - 1) * (gx - 1)
        i0 = np.floor(y).astype(int)
        j0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, gy - 1)
        j1 = np.minimum(j0 + 1, gx - 1)
        fy, fx = y - i0, x - j0
        top = grid[i0, j0] * (1 - fx) + grid[i0, j1] * fx
        bot = grid[i1, j0] * (1 - fx) + grid[i1, j1] * fx
        field += weight * (top + (bot - top) * fy)
    field -= field.min()
    peak = float(field.max()) or 1.0
    return field / peak


def _iso_field(rng, n: int, sigmas_px, weights, dtype=np.float32) -> np.ndarray:
    """Isotropic band-limited noise (blurred white noise); no lattice grid."""
    from scipy.ndimage import gaussian_filter
    field = np.zeros((n, n), dtype=dtype)
    for sigma, weight in zip(sigmas_px, weights):
        white = rng.standard_normal((n, n)).astype(dtype)
        field += weight * gaussian_filter(white, max(float(sigma), 0.5), mode="wrap")
    field -= field.min()
    peak = float(field.max()) or 1.0
    return field / peak


def _block_field(rng, shape, cell: int, dtype=np.float32) -> np.ndarray:
    """Nearest-neighbour clumps that survive hfield filtering."""
    gy = max(shape[0] // cell, 2)
    gx = max(shape[1] // cell, 2)
    grid = rng.random((gy, gx)).astype(dtype)
    y = (np.arange(shape[0]) * gy) // shape[0]
    x = (np.arange(shape[1]) * gx) // shape[1]
    return grid[y[:, None], x[None, :]]


def ground_texture(floor, xs, ys, seed: int, n: int = 4096) -> bytes:
    """Orchard sod with vine-row soil strips, wheel tracks and post pads.

    Mid-scale (0.1-2 m) structure carries the look: anything finer than a
    few centimetres is filtered away at a 12 m camera distance.
    """
    half = float(floor.half_extent_m)
    rng = np.random.default_rng(seed + 331)
    u = np.linspace(-half, half, n, dtype=np.float32)
    xx, yy = np.meshgrid(u, u)
    aisles = [0.5 * (ys[i] + ys[i + 1]) for i in range(len(ys) - 1)]
    m_per_px = 2.0 * half / n

    def px(metres):
        return max(2, int(round(metres / m_per_px)))

    broad = _iso_field(rng, n, (px(0.9), px(0.4)), (1.0, 0.5))
    patch = _iso_field(rng, n, (px(0.2), px(0.09)), (1.0, 0.5))
    tuft = _iso_field(rng, n, (px(0.045),), (1.0,))
    fine = _iso_field(rng, n, (px(0.02), px(0.01)), (1.0, 0.5))
    wander = _iso_field(rng, n, (px(0.25), px(0.1)), (1.0, 0.4))
    blades = _iso_field(rng, n, (1.2,), (1.0,))
    speck = rng.random((n, n), dtype=np.float32)

    # Olive/khaki orchard sod rather than a lime lawn; soil is a warm umber.
    dark = np.array([0.12, 0.17, 0.06], np.float32)
    mid = np.array([0.24, 0.28, 0.11], np.float32)
    light = np.array([0.39, 0.39, 0.18], np.float32)
    straw = np.array([0.52, 0.45, 0.25], np.float32)
    clover = np.array([0.10, 0.20, 0.08], np.float32)
    soil_wet = np.array([0.16, 0.11, 0.07], np.float32)
    soil_dry = np.array([0.47, 0.36, 0.23], np.float32)
    rut = np.array([0.22, 0.16, 0.09], np.float32)
    packed = np.array([0.48, 0.42, 0.31], np.float32)
    litter_col = np.array([0.44, 0.34, 0.14], np.float32)

    t = np.clip(0.42 * patch + 0.40 * broad + 0.18 * tuft, 0.0, 1.0)
    low = t < 0.5
    rgb = np.empty((n, n, 3), np.float32)
    a = (2.0 * t)[..., None]
    b = (2.0 * t - 1.0)[..., None]
    rgb[low] = ((1.0 - a) * dark + a * mid)[low]
    rgb[~low] = ((1.0 - b) * mid + b * light)[~low]
    rgb *= (0.86 + 0.28 * fine)[..., None]
    dry = np.clip((broad - 0.70) / 0.30, 0, 1) * np.clip((patch - 0.55) / 0.40, 0, 1)
    rgb = rgb * (1.0 - 0.45 * dry)[..., None] + (0.45 * dry)[..., None] * straw
    clo = np.clip((0.28 - patch) / 0.22, 0, 1) * np.clip((tuft - 0.45) / 0.5, 0, 1)
    rgb = rgb * (1.0 - 0.5 * clo)[..., None] + (0.5 * clo)[..., None] * clover
    bare = np.clip((tuft - 0.985) / 0.015, 0, 1) * np.clip((broad - 0.55) / 0.4, 0, 1)
    rgb = rgb * (1.0 - 0.6 * bare)[..., None] + (0.6 * bare)[..., None] * (soil_dry * 0.85)
    rgb *= (0.72 + 0.56 * blades)[..., None]

    soil_w = np.zeros((n, n), np.float32)
    for y in ys:
        edge = 0.14 + 0.30 * (wander - 0.5)
        soil_w = np.maximum(soil_w, np.clip((0.66 + edge - np.abs(yy - y)) / 0.12, 0, 1))
    clods = _value_field(rng, (n, n), ((px(0.12), 1.0), (px(0.05), 0.5)))
    crumb = np.clip(0.40 * fine + 0.25 * patch + 0.35 * clods, 0, 1)
    soil = (1.0 - crumb)[..., None] * soil_wet + crumb[..., None] * soil_dry
    soil *= (0.86 + 0.28 * blades)[..., None]
    for y in ys:
        moist = np.clip(1.0 - np.abs(yy - y) / 0.22, 0, 1) ** 1.3
        soil = soil * (1.0 - 0.45 * moist)[..., None] + (0.45 * moist)[..., None] * soil_wet
    edge_col = np.array([0.20, 0.16, 0.07], np.float32)
    w = soil_w
    lo = w < 0.5
    a = (2.0 * w)[..., None]
    b = (2.0 * w - 1.0)[..., None]
    out = np.empty_like(rgb)
    out[lo] = ((1.0 - a) * rgb + a * edge_col)[lo]
    out[~lo] = ((1.0 - b) * edge_col + b * soil)[~lo]
    rgb = out

    track = np.zeros((n, n), np.float32)
    for aisle_y in aisles:
        for side in (-0.64, 0.64):
            d = np.abs(yy - (aisle_y + side + 0.08 * (wander - 0.5)))
            track = np.maximum(track, np.clip(1.0 - d / 0.24, 0, 1) ** 1.2)
    track *= (0.55 + 0.45 * fine) * (1.0 - 0.5 * soil_w)
    rgb = rgb * (1.0 - 0.6 * track)[..., None] + (0.6 * track)[..., None] * rut

    pads = np.zeros((n, n), np.float32)
    for x in xs:
        for y in ys:
            pads = np.maximum(pads, np.clip(1.0 - np.hypot(xx - x, yy - y) / 0.36, 0, 1))
    rgb = rgb * (1.0 - 0.5 * pads)[..., None] + (0.5 * pads)[..., None] * packed

    stones = ((speck > 0.9975) & (soil_w > 0.4)).astype(np.float32)
    rgb = rgb * (1.0 - 0.7 * stones)[..., None] + (0.7 * stones)[..., None] * np.array(
        [0.42, 0.38, 0.30], np.float32)
    # Fallen leaves under the roof: sparse flecks plus a few drifted clumps.
    litter = ((speck < 0.010) & (soil_w < 0.4)).astype(np.float32)
    litter = np.maximum(litter, 0.6 * np.clip((tuft - 0.94) / 0.06, 0, 1)
                        * np.clip((broad - 0.45) / 0.4, 0, 1) * (soil_w < 0.4))
    rgb = rgb * (1.0 - 0.6 * litter)[..., None] + (0.6 * litter)[..., None] * litter_col
    return _png_bytes(np.flipud(np.clip(rgb, 0, 1)) * 255.0)


def wood_texture(seed: int) -> bytes:
    """Weathered treated-pine grain for round posts and sawn beams."""
    rng = np.random.default_rng(seed + 19)
    n = 512
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    noise = _value_field(rng, (n, n), ((40, 1.0), (14, 0.55), (5, 0.25)))
    grain = 0.5 + 0.5 * np.sin(xx * 0.55 + 3.2 * np.sin(yy * 0.04) + 1.6 * noise)
    cracks = (np.sin(xx * 2.4 + 12.0 * noise) > 0.92).astype(np.float32)
    weather = _block_field(rng, (n, n), 28)
    dark = np.array([0.16, 0.10, 0.06])
    mid = np.array([0.40, 0.26, 0.13])
    light = np.array([0.58, 0.42, 0.24])
    silver = np.array([0.42, 0.36, 0.26])
    rings = 0.5 + 0.5 * np.sin(xx * 0.12 + 1.2 * noise)
    rgb = dark + (mid - dark) * (0.45 * grain + 0.55 * rings)[..., None]
    rgb = rgb * (1.0 - 0.35 * weather)[..., None] + weather[..., None] * light
    rgb = rgb * (1.0 - 0.70 * cracks)[..., None] + cracks[..., None] * dark
    gray = np.clip((weather - 0.55), 0.0, 1.0) * 0.55
    rgb = rgb * (1.0 - gray)[..., None] + gray[..., None] * silver
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def leaf_texture(seed: int, sun: bool) -> bytes:
    """Actinidia blade map for the curved leaf mesh (u across, v petiole->tip)."""
    rng = np.random.default_rng(seed + (81 if sun else 77))
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    u = (xx / (n - 1)) * 2.0 - 1.0
    v = yy / (n - 1)
    mottling = _value_field(rng, (n, n), ((18, 1.0), (7, 0.5), (3, 0.25)))
    blotch = _block_field(rng, (n, n), 22)
    speckle = 0.88 + 0.12 * rng.random((n, n), dtype=np.float32)
    midrib = np.exp(-((u * 18.0) ** 2)) * (0.35 + 0.65 * (1.0 - v))
    veins = np.zeros((n, n), np.float32)
    for k in range(-6, 7):
        if k == 0:
            continue
        shift = 0.11 * k * (0.10 + 0.90 * v)
        slant = 0.22 * k * (v - 0.06)
        veins += np.exp(-(((u - shift - 0.12 * slant) * 22.0) ** 2)) * (
            0.85 * (1.0 - abs(k) / 8.0))
    veins *= 0.25 + 0.75 * v
    if sun:
        dark = np.array([0.14, 0.28, 0.08])
        light = np.array([0.31, 0.44, 0.15])
        rib = np.array([0.30, 0.40, 0.16])
    else:
        dark = np.array([0.07, 0.16, 0.05])
        light = np.array([0.15, 0.27, 0.09])
        rib = np.array([0.14, 0.24, 0.09])
    mix = np.clip(0.22 + 0.40 * mottling + 0.38 * blotch, 0.0, 1.0)
    rgb = dark + (light - dark) * mix[..., None]
    rgb *= speckle[..., None]
    rgb = rgb * (1.0 - 0.45 * midrib)[..., None] + midrib[..., None] * rib
    veins = np.clip(veins, 0, 1)
    rgb = rgb * (1.0 - 0.38 * veins)[..., None] + veins[..., None] * rib
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def kiwi_texture(seed: int) -> bytes:
    """Brown fuzz for the fruit skin; modulated by the placement colour."""
    rng = np.random.default_rng(seed + 9)
    n = 256
    fuzz = _value_field(rng, (n, n), ((6, 1.0), (2, 0.6)))
    hairs = (rng.random((n, n), dtype=np.float32) > 0.965).astype(np.float32)
    patch = _value_field(rng, (n, n), ((40, 1.0), (16, 0.5)))
    base = 0.72 + 0.34 * fuzz + 0.16 * (patch - 0.5)
    rgb = np.stack([base * 1.06, base * 0.98, base * 0.86], axis=-1)
    rgb = rgb * (1.0 - 0.35 * hairs)[..., None] + (0.35 * hairs)[..., None] * np.array(
        [0.55, 0.45, 0.32])
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def concrete_texture(seed: int) -> bytes:
    rng = np.random.default_rng(seed + 4)
    n = 256
    grit = _value_field(rng, (n, n), ((12, 1.0), (4, 0.5)))
    speckle = rng.random((n, n), dtype=np.float32)
    rgb = np.array([0.42, 0.40, 0.36]) * (0.82 + 0.28 * grit[..., None])
    rgb = rgb * (0.92 + 0.12 * speckle[..., None])
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


# --------------------------------------------------------------------------- #
# Leaf blade mesh (numpy port of treesim.foliage.leaf_mesh, single-sided;
# the renderer disables back-face culling)
# --------------------------------------------------------------------------- #
def leaf_blade(length: float, width: float, fold: float = 0.55, curl: float = 0.30,
               droop: float = 0.35, nseg: int = 6):
    ts = np.linspace(0.0, 1.0, nseg + 1)
    verts, uvs, faces = [], [], []
    for t in ts:
        w = 0.5 * width * (np.sin(np.pi * min(t, 0.995) ** 0.8) ** 0.85 + 0.03)
        z = length * t
        y_rib = curl * length * t * t - droop * length * t ** 3
        y_edge = y_rib + fold * w
        i0 = len(verts)
        verts += [(-w, y_edge, z), (0.0, y_rib, z), (w, y_edge, z)]
        uvs += [(0.0, t), (0.5, t), (1.0, t)]
        if t > 0.0:
            l0, m0, r0 = i0 - 3, i0 - 2, i0 - 1
            l1, m1, r1 = i0, i0 + 1, i0 + 2
            faces += [(l0, m0, l1), (m0, m1, l1), (m0, r0, m1), (r0, r1, m1)]
    return np.asarray(verts), np.asarray(uvs), np.asarray(faces)


def leaf_mesh_assets() -> str:
    out = []
    for k, scale in enumerate(LEAF_SIZE_CLASSES):
        v, uv, f = leaf_blade(LEAF_LENGTH_M * scale, LEAF_WIDTH_M * scale)
        out.append(
            f'    <mesh name="leaf{k}" inertia="shell" '
            f'vertex="{" ".join(f"{c:.4f}" for c in v.ravel())}" '
            f'texcoord="{" ".join(f"{c:.3f}" for c in uv.ravel())}" '
            f'face="{" ".join(str(int(i)) for i in f.ravel())}"/>'
        )
    return "\n".join(out)


def _leaf_class(length: float) -> int:
    ratio = length / LEAF_LENGTH_M
    return int(np.argmin([abs(ratio - s) for s in LEAF_SIZE_CLASSES]))


# --------------------------------------------------------------------------- #
# Scene MJCF
# --------------------------------------------------------------------------- #
def street_heights(floor, xs, ys) -> np.ndarray:
    """Keep sampled slope; add aisle ruts and post pads on the working lanes."""
    heights = np.asarray(floor.heights_m, dtype=np.float64).copy()
    x = np.asarray(floor.x_m, dtype=np.float64)
    y = np.asarray(floor.y_m, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    for i in range(len(ys) - 1):
        aisle_y = 0.5 * (ys[i] + ys[i + 1])
        for side in (-0.64, 0.64):
            dist = np.abs(yy - (aisle_y + side))
            heights -= 0.024 * np.clip(1.0 - dist / 0.16, 0.0, 1.0) ** 2
    for px in xs:
        for py in ys:
            pad = np.clip((0.26 - np.hypot(xx - px, yy - py)) / 0.10, 0.0, 1.0)
            iu = int(np.clip(round((px - x[0]) / (x[-1] - x[0]) * (x.size - 1)), 0, x.size - 1))
            iv = int(np.clip(round((py - y[0]) / (y[-1] - y[0]) * (y.size - 1)), 0, y.size - 1))
            heights = (1.0 - pad) * heights + pad * float(heights[iv, iu])
    return heights


def mjcf(floor, skeleton, fruit, leaves, xs, ys, with_spot_wrap: bool = False) -> str:
    heights = street_heights(floor, xs, ys)
    min_z = float(heights.min())
    elevation = max(float(heights.max()) - min_z, 1e-4)
    nrow, ncol = heights.shape
    half = float(floor.half_extent_m)
    geoms = []
    for seg in skeleton:
        mid = 0.5 * (seg.start + seg.end)
        axis = seg.end - seg.start
        length = float(np.linalg.norm(axis))
        if length < 1e-4:
            continue
        vertical = abs(axis[2]) > 0.75 * length and seg.order == 0
        if vertical:
            quat = _quat_wxyz_from_z(axis)
            foot = seg.start if seg.start[2] < seg.end[2] else seg.end
            geoms.append(
                f'    <geom type="cylinder" pos="{foot[0]:.5f} {foot[1]:.5f} '
                f'{float(floor.ground_z(foot[0], foot[1])) + FOOTER_HALF_M:.5f}" '
                f'size="{FOOTER_RADIUS_M:.3f} {FOOTER_HALF_M:.3f}" material="concrete" '
                f'rgba="0.62 0.60 0.54 1" contype="0" conaffinity="0"/>'
            )
            geoms.append(
                f'    <geom type="cylinder" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}" '
                f'size="{POST_RADIUS_M:.3f} {0.5 * length:.5f}" '
                f'quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}" '
                f'material="wood" rgba="0.58 0.42 0.24 1" contype="0" conaffinity="0"/>'
            )
            low_z = float(min(seg.start[2], seg.end[2]))
            for frac in (0.22, 0.48, 0.74):
                ring_z = low_z + frac * length
                geoms.append(
                    f'    <geom type="cylinder" pos="{foot[0]:.5f} {foot[1]:.5f} {ring_z:.5f}" '
                    f'size="{POST_RADIUS_M + 0.008:.3f} 0.014" material="wood" '
                    f'rgba="0.28 0.18 0.10 1" contype="0" conaffinity="0"/>'
                )
            top = seg.end if seg.end[2] > seg.start[2] else seg.start
            geoms.append(
                f'    <geom type="cylinder" pos="{top[0]:.5f} {top[1]:.5f} {top[2] - 0.04:.5f}" '
                f'size="{POST_RADIUS_M + 0.014:.3f} 0.018" material="steel" '
                f'rgba="0.34 0.34 0.34 1" contype="0" conaffinity="0"/>'
            )
        elif seg.order == 0:
            quat = _quat_wxyz_from_z(axis)
            geoms.append(
                f'    <geom type="box" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}" '
                f'size="{BEAM_HALF_M[0]:.3f} {BEAM_HALF_M[1]:.3f} {0.5 * length:.5f}" '
                f'quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}" '
                f'material="wood" rgba="0.42 0.26 0.12 1" contype="0" conaffinity="0"/>'
            )
        else:
            material = "steel" if seg.order == 1 else "vine"
            rgb = (0.42, 0.44, 0.46) if seg.order == 1 else (0.22, 0.12, 0.05)
            geoms.append(
                f'    <geom type="capsule" fromto="'
                f'{seg.start[0]:.5f} {seg.start[1]:.5f} {seg.start[2]:.5f} '
                f'{seg.end[0]:.5f} {seg.end[1]:.5f} {seg.end[2]:.5f}" '
                f'size="{seg.mean_radius:.5f}" material="{material}" '
                f'rgba="{_rgba(rgb)}" contype="0" conaffinity="0"/>'
            )
    for item in fruit:
        center = fruit_center(item)
        rx, ry, rz = (float(v) for v in item.radii)
        # The fuzz map averages ~0.85; lift the placement colour to compensate.
        skin = tuple(min(1.0, 1.25 * float(c)) for c in item.color)
        geoms.append(
            f'    <geom type="ellipsoid" pos="'
            f'{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}" '
            f'size="{rx:.5f} {ry:.5f} {rz:.5f}" material="kiwi" '
            f'rgba="{_rgba(skin)}" contype="0" conaffinity="0"/>'
        )
        geoms.append(
            f'    <geom type="capsule" fromto="'
            f'{item.attach[0]:.5f} {item.attach[1]:.5f} {item.attach[2]:.5f} '
            f'{center[0]:.5f} {center[1]:.5f} {center[2] + rz:.5f}" '
            f'size="0.0018" material="vine" rgba="0.20 0.10 0.04 1" '
            f'contype="0" conaffinity="0"/>'
        )
    leaf_rng = np.random.default_rng(19)
    for leaf in leaves:
        sun = bool(leaf_rng.random() < 0.40)
        stressed = bool(leaf_rng.random() < 0.06)
        pitch = float(leaf_rng.uniform(-0.15, 0.45))
        material = "leaf_sun" if sun else "leaf_shade"
        if stressed:
            tint = (leaf_rng.uniform(0.95, 1.12), leaf_rng.uniform(0.82, 0.95), 0.55)
        elif sun:
            tint = (leaf_rng.uniform(0.90, 1.08), leaf_rng.uniform(0.94, 1.06), 0.90)
        else:
            tint = (leaf_rng.uniform(0.82, 1.00), leaf_rng.uniform(0.88, 1.04), 0.92)
        geoms.append(
            f'    <geom type="mesh" mesh="leaf{_leaf_class(leaf.length)}" pos="'
            f'{leaf.attach[0]:.5f} {leaf.attach[1]:.5f} {leaf.attach[2]:.5f}" '
            f'quat="{leaf_card_quat(leaf.frame, pitch)}" material="{material}" '
            f'rgba="{tint[0]:.3f} {tint[1]:.3f} {tint[2]:.3f} 1" '
            f'contype="0" conaffinity="0"/>'
        )
    spot_assets = ""
    if with_spot_wrap:
        spot_assets = (
            '    <texture type="2d" name="spotwrap" file="bdaii_spot_wrap.png"/>\n'
            '    <material name="spotwrap" texture="spotwrap" texrepeat="2 2" '
            'texuniform="false" reflectance="0.10" shininess="0.45" specular="0.35"/>\n'
            '    <material name="spotdark" reflectance="0.06" shininess="0.30" '
            'specular="0.22" rgba="0.16 0.17 0.18 1"/>\n'
        )
    return f'''<mujoco model="kiwi_street">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="42"/>
    <headlight ambient=".30 .30 .27" diffuse=".20 .20 .19" specular=".04 .04 .03"/>
    <rgba haze=".86 .86 .82 1" fog=".84 .83 .78 1"/>
    <map znear=".004" zfar="6" shadowclip="0.5"/>
    <quality shadowsize="{SHADOW_MAP_PX}" offsamples="4"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".34 .50 .70" rgb2=".84 .86 .86"
             width="512" height="512"/>
    <texture type="2d" name="orchard" file="orchard_ground.png"/>
    <texture type="2d" name="wood" file="wood.png"/>
    <texture type="2d" name="leaf_sun" file="leaf_sun.png"/>
    <texture type="2d" name="leaf_shade" file="leaf_shade.png"/>
    <texture type="cube" name="kiwi" file="kiwi.png"/>
    <texture type="2d" name="concrete" file="concrete.png"/>
{spot_assets}    <material name="orchard" texture="orchard" texrepeat="1 1" texuniform="false"
              reflectance="0.02" shininess="0.04" specular="0.06" rgba="1 1 1 1"/>
    <material name="wood" texture="wood" texrepeat="2 1" texuniform="true"
              reflectance="0.08" shininess="0.18" specular="0.20"/>
    <material name="concrete" texture="concrete" texrepeat="2 2" texuniform="true"
              reflectance="0.06" shininess="0.10" specular="0.12"/>
    <material name="steel" reflectance="0.38" shininess="0.68" specular="0.48"
              rgba="0.38 0.38 0.40 1"/>
    <material name="vine" reflectance="0.05" shininess="0.10" specular="0.08"/>
    <material name="leaf_sun" texture="leaf_sun" texrepeat="1 1" texuniform="false"
              reflectance="0.02" shininess="0.10" specular="0.08" emission="0.16"/>
    <material name="leaf_shade" texture="leaf_shade" texrepeat="1 1" texuniform="false"
              reflectance="0.01" shininess="0.06" specular="0.05" emission="0.04"/>
    <material name="kiwi" texture="kiwi" texuniform="true" reflectance="0.04"
              shininess="0.12" specular="0.08"/>
    <hfield name="orchard_ground" nrow="{nrow}" ncol="{ncol}"
            size="{half} {half} {elevation:.5f} 0.08"/>
{leaf_mesh_assets()}
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="12 -16 12" dir="-0.34 0.58 -0.74"
           ambient=".34 .34 .31" diffuse=".78 .70 .56" specular=".20 .17 .12"
           castshadow="true"/>
    <light name="rim" directional="true" pos="-10 10 8" dir="0.40 -0.30 -0.86"
           castshadow="false" ambient=".07 .08 .10" diffuse=".24 .27 .32"
           specular=".06 .06 .08"/>
{chr(10).join(_aisle_lights(xs, ys))}
    <geom name="ground" type="hfield" hfield="orchard_ground" material="orchard"
          pos="0 0 {min_z:.5f}" rgba="1 1 1 1"
          friction="{float(floor.friction):.3f} 0.01 0.001"/>
{chr(10).join(geoms)}
  </worldbody>
</mujoco>
'''


def _aisle_lights(xs, ys):
    """Six soft down-lights (MuJoCo allows eight lights in total)."""
    aisles = [0.5 * (ys[i] + ys[i + 1]) for i in range(len(ys) - 1)]
    out = []
    span = xs[-1] - xs[0]
    for i, y in enumerate(aisles):
        for k, frac in enumerate((0.28, 0.72)):
            x = xs[0] + frac * span
            out.append(
                f'    <light name="aisle{i}{k}" pos="{x:.2f} {y:.2f} 1.40" dir="0 0 -1" '
                f'cutoff="88" exponent="0.4" attenuation="0.90 0.06 0.006" '
                f'castshadow="false" diffuse=".48 .52 .38" specular=".05 .05 .04"/>'
            )
    return out[:6]


def apply_hfield(model, floor, xs, ys) -> None:
    heights = street_heights(floor, xs, ys)
    min_z = float(heights.min())
    span = max(float(heights.max()) - min_z, 1e-4)
    model.hfield_data[:] = ((heights - min_z) / span).astype(np.float64).ravel()


# --------------------------------------------------------------------------- #
# Spot: URDF import plus scripted kinematics
# --------------------------------------------------------------------------- #
def spot_spec(relic: Path, workdir: Path):
    import mujoco
    asset = relic.resolve() / "source/relic/relic/assets/spot"
    urdf_path = asset / "spot_with_arm.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"RELIC Spot URDF not found: {urdf_path}")
    text = urdf_path.read_text()
    inject = (
        f'<mujoco><compiler meshdir="{asset}" balanceinertia="true" '
        f'discardvisual="false" strippath="false" fusestatic="false"/></mujoco>'
    )
    text, count = re.subn(r"(<robot\b[^>]*>)", lambda m: m.group(1) + inject, text, count=1)
    if count != 1:
        raise ValueError("could not inject MuJoCo compiler block into the URDF")
    local = workdir / "spot_with_arm.urdf"
    local.write_text(text)
    spec = mujoco.MjSpec.from_file(str(local))
    for body in spec.bodies:
        for geom in body.geoms:
            if geom.contype or geom.conaffinity:
                geom.group = 3
                geom.contype = 0
                geom.conaffinity = 0
            else:
                geom.group = 1
    return spec, asset / "meshes" / "bdaii_spot_wrap.png"


SPOT_WRAP_BODIES = ("body", "fl_uleg", "fr_uleg", "hl_uleg", "hr_uleg",
                    "arm_link_sh0", "arm_link_sh1")


def _rot_y(angle: float, v) -> np.ndarray:
    """Rotate an (x, z) vector about +Y by ``angle`` (MuJoCo right-handed)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c])


def leg_fk(v1, v2, hy: float, kn: float) -> np.ndarray:
    """Foot (x, z) in the hip-pitch frame for upper/lower link vectors."""
    return _rot_y(hy, np.asarray(v1, float) + _rot_y(kn, np.asarray(v2, float)))


def leg_ik(v1, v2, target, guess, hy_range, kn_range, iterations: int = 8) -> np.ndarray:
    """Damped Gauss-Newton on the planar two-link chain; clipped to ranges."""
    q = np.asarray(guess, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("leg IK target must be finite")
    for _ in range(iterations):
        err = target - leg_fk(v1, v2, q[0], q[1])
        if np.linalg.norm(err) < 1e-5:
            break
        eps = 1e-5
        base = leg_fk(v1, v2, q[0], q[1])
        j0 = (leg_fk(v1, v2, q[0] + eps, q[1]) - base) / eps
        j1 = (leg_fk(v1, v2, q[0], q[1] + eps) - base) / eps
        step = np.linalg.lstsq(np.column_stack((j0, j1)), err, rcond=None)[0]
        q += np.clip(step, -0.4, 0.4)
    q[0] = float(np.clip(q[0], *hy_range))
    q[1] = float(np.clip(q[1], *kn_range))
    return q


def trot_foot_offset(phase: float, stride: float, lift: float):
    """Foot x/z offset (body frame) for one gait phase in [0, 1).

    ``stride`` is the body travel per full gait cycle (speed / frequency).
    Stance occupies half the cycle, so the planted foot must move back by
    exactly ``stride / 2`` at body speed to stay locked on the ground; swing
    lifts on a half-sine and returns it forward.
    """
    phase = phase % 1.0
    half = 0.5 * stride
    if phase < 0.5:
        u = phase / 0.5
        return 0.5 * half - half * u, 0.0
    u = (phase - 0.5) / 0.5
    smooth = u * u * (3.0 - 2.0 * u)
    return -0.5 * half + half * smooth, lift * np.sin(np.pi * u)


def arm_hand_height(sh1: float, el0: float, wr0: float) -> float:
    """Hand tip height above the Spot body frame for pitch joints (rad).

    Planar chain about +Y (URDF link offsets); sh0/el1/wr1 do not change
    height. Used to bound the scripted wander under the pergola beams.
    """
    z = ARM_SHOULDER_Z_M
    z += _rot_y(sh1, ARM_UPPER_M)[1]
    z += _rot_y(sh1 + el0, ARM_FORE_M)[1]
    z += _rot_y(sh1 + el0 + wr0, ARM_HAND_M)[1]
    return float(z)


def arm_wander_max_hand_z(wander=SPOT_ARM_WANDER, body_z: float = SPOT_BODY_HEIGHT_M) -> float:
    """Worst-case hand height above ground over the wander box corners and
    a coarse interior grid (the sum-of-sines never leaves the box)."""
    best = -np.inf
    grids = []
    for name in ("arm_sh1", "arm_el0", "arm_wr0"):
        centre, amp = wander[name]
        grids.append(np.linspace(centre - amp, centre + amp, 7))
    for sh1 in grids[0]:
        for el0 in grids[1]:
            for wr0 in grids[2]:
                best = max(best, arm_hand_height(sh1, el0, wr0))
    return float(body_z + 0.012 + best)


def camera_forward(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit view direction of a MuJoCo free camera (azimuth 0 looks +X)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def shadow_focus(lookat, distance: float, azimuth: float, elevation: float,
                 ahead: float = 0.30) -> np.ndarray:
    """Centre of the directional shadow cube: a little past the look-at point
    along the horizontal view direction, near ground level."""
    f = camera_forward(azimuth, elevation)
    f[2] = 0.0
    n = float(np.linalg.norm(f))
    f = f / n if n > 1e-9 else np.zeros(3)
    focus = np.asarray(lookat, dtype=np.float64) + ahead * float(distance) * f
    focus[2] = SHADOW_FOCUS_Z_M
    return focus


def shadow_light_pos(focus, light_dir, back: float = SHADOW_BACK_M) -> np.ndarray:
    """Directional-light eye: ``back`` metres upstream of the focus so the
    depth window [0, SHADOW_HALF_M] brackets the roof and the aisle floor."""
    d = np.asarray(light_dir, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9 or not np.isfinite(d).all():
        raise ValueError("light direction must be finite and nonzero")
    return np.asarray(focus, dtype=np.float64) - back * d / n


def leaf_out_dir(frame) -> np.ndarray:
    """World direction of the local +Z blade axis for an xyzw quaternion."""
    x, y, z, w = (float(v) for v in frame)
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])


def upright_leaves(leaves, min_z: float = -0.15):
    """Drop shoot leaves that hang below the canes. On a pergola the foliage
    forms a roof above the wires and the fruit hangs free beneath it; leaves
    pointing down would hide the crop from the aisle."""
    return [leaf for leaf in leaves if leaf_out_dir(leaf.frame)[2] >= min_z]


def style_spot(scene) -> None:
    """Yellow wrap on chassis, upper legs and shoulder; dark ABS elsewhere."""
    for body in scene.bodies:
        if not body.name.startswith("spot"):
            continue
        part = body.name.split("_", 1)[1] if "_" in body.name else body.name
        wrap = part in SPOT_WRAP_BODIES
        for geom in body.geoms:
            if geom.group != 1:
                continue
            geom.material = "spotwrap" if wrap else "spotdark"
            geom.rgba = [1.0, 1.0, 1.0, 1.0] if wrap else [0.17, 0.18, 0.19, 1.0]


class SpotWalker:
    """Scripted trot along a lane plus a smooth pseudo-random arm.

    Joint values are written directly; there is no controller, contact or
    policy. This is an animation rig, not a gait result.
    """

    def __init__(self, model, prefix: str, lane_y: float, x0: float, direction: int,
                 speed_mps: float, ground_z, rng):
        if not np.isfinite([lane_y, x0, speed_mps]).all() or speed_mps <= 0:
            raise ValueError("walker lane, start and speed must be finite and positive")
        self.model = model
        self.prefix = prefix
        self.lane_y = float(lane_y)
        self.x0 = float(x0)
        self.direction = 1 if direction >= 0 else -1
        self.speed = float(speed_mps)
        self.ground_z = ground_z
        self.freq_hz = 1.55 + 0.25 * (self.speed - 0.6)
        self.stride = self.speed / self.freq_hz
        self.lift = 0.09
        body = model.body(f"{prefix}body")
        self.free_adr = int(model.jnt_qposadr[model.body_jntadr[body.id]])
        self.q = {}
        self.range = {}
        for name in SPOT_HOME:
            j = model.joint(f"{prefix}{name}")
            self.q[name] = int(model.jnt_qposadr[j.id])
            self.range[name] = tuple(float(v) for v in model.jnt_range[j.id])
        self.hip = {}
        self.v1 = {}
        self.v2 = {}
        for leg in SPOT_LEGS:
            hip = model.body_pos[model.body(f"{prefix}{leg}_hip").id]
            uleg = model.body_pos[model.body(f"{prefix}{leg}_uleg").id]
            self.hip[leg] = hip + uleg
            self.v1[leg] = model.body_pos[model.body(f"{prefix}{leg}_lleg").id][[0, 2]]
            self.v2[leg] = model.body_pos[model.body(f"{prefix}{leg}_foot").id][[0, 2]]
        self.ik_state = {leg: np.array([SPOT_HOME[f"{leg}_hy"], SPOT_HOME[f"{leg}_kn"]])
                         for leg in SPOT_LEGS}
        self.arm_waves = {}
        for name in SPOT_ARM:
            freqs = rng.uniform(0.07, 0.30, 3)
            phases = rng.uniform(0, 2 * np.pi, 3)
            weights = rng.uniform(0.4, 1.0, 3)
            weights /= weights.sum()
            self.arm_waves[name] = (freqs, phases, weights)
        self.phase0 = float(rng.uniform(0, 1))

    def _ik(self, leg, target):
        q = leg_ik(self.v1[leg], self.v2[leg], target, self.ik_state[leg],
                   self.range[f"{leg}_hy"], self.range[f"{leg}_kn"])
        self.ik_state[leg] = q
        return q

    def foot_offset(self, phase: float):
        return trot_foot_offset(phase, self.stride, self.lift)

    def position(self, t: float) -> np.ndarray:
        x = self.x0 + self.direction * self.speed * t
        return np.array([x, self.lane_y, float(self.ground_z(x, self.lane_y)) + SPOT_BODY_HEIGHT_M])

    def write(self, data, t: float) -> np.ndarray:
        x = self.x0 + self.direction * self.speed * t
        y = self.lane_y
        gait = self.freq_hz * t + self.phase0
        bob = 0.010 * np.cos(4.0 * np.pi * gait)
        roll = 0.012 * np.sin(2.0 * np.pi * gait)
        pitch = 0.008 * np.sin(4.0 * np.pi * gait + 0.6)
        z = float(self.ground_z(x, y)) + SPOT_BODY_HEIGHT_M + bob
        yaw = 0.0 if self.direction > 0 else np.pi
        adr = self.free_adr
        data.qpos[adr:adr + 3] = (x, y, z)
        data.qpos[adr + 3:adr + 7] = _euler_wxyz(roll, pitch, yaw)
        for leg in SPOT_LEGS:
            dx, dz = self.foot_offset(gait + SPOT_TROT_PHASE[leg])
            # Target in the hip-pitch frame (x forward, z up); hips sit at z=0.
            target = np.array([dx - 0.03,
                               -(SPOT_BODY_HEIGHT_M + bob) + SPOT_FOOT_RADIUS_M + dz])
            hy, kn = self._ik(leg, target)
            data.qpos[self.q[f"{leg}_hx"]] = 0.0
            data.qpos[self.q[f"{leg}_hy"]] = hy
            data.qpos[self.q[f"{leg}_kn"]] = kn
        for name in SPOT_ARM:
            centre, amp = SPOT_ARM_WANDER[name]
            freqs, phases, weights = self.arm_waves[name]
            value = centre + amp * float(np.sum(weights * np.sin(2 * np.pi * freqs * t + phases)))
            lo, hi = self.range[name]
            margin = 0.04 * (hi - lo)
            data.qpos[self.q[name]] = float(np.clip(value, lo + margin, hi - margin))
        return np.array([x, y, z])


def default_walkers(model, aisles, xs, ground_z, seed: int):
    rng = np.random.default_rng(seed + 2024)
    span = xs[-1] - xs[0]
    plan = [
        (0, 0.45, -0.36 * span, +1, 0.72),
        (0, -0.75, 0.30 * span, -1, 0.66),
        (1, 0.35, -0.30 * span, +1, 0.70),
        (1, -0.85, 0.15 * span, -1, 0.64),
        (2, 0.20, -0.12 * span, +1, 0.68),
    ]
    walkers = []
    for i, (aisle, dy, x0, direction, speed) in enumerate(plan):
        lane = aisles[min(aisle, len(aisles) - 1)] + dy
        walkers.append(SpotWalker(model, f"spot{i}_", lane, x0, direction, speed,
                                  ground_z, rng))
    return walkers


# --------------------------------------------------------------------------- #
# Camera, grading, encoding
# --------------------------------------------------------------------------- #
def _smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def camera_path(t: float, seconds: float, aisles, follow=None):
    """Two shots: lateral tracking of the first walker, then a low push down
    the middle aisle. Camera height = lookat z + distance*sin(-elevation);
    both shots stay under the 1.6 m roof."""
    cut = 0.42 * seconds
    mid = aisles[len(aisles) // 2]
    near = aisles[0]
    if t < cut:
        u = _smooth(t / max(cut, 1e-6))
        fx = follow(t)[0] if follow is not None else -6.0 + 0.7 * t
        lookat = (fx + 0.6, near - 0.2, 0.72)
        return lookat, 7.8 - 0.8 * u, 60.0 - 10.0 * u, -4.5
    u = _smooth((t - cut) / max(seconds - cut, 1e-6))
    lookat = (-1.5 + 4.5 * u, mid + 0.1, 0.62 + 0.08 * u)
    return lookat, 12.8 - 2.6 * u, 8.0 - 3.0 * u, -2.6 + 0.6 * u


def grade(frame: np.ndarray, letterbox: bool = True) -> np.ndarray:
    """Light cinematic grade: contrast, warmth, bloom, vignette, 2.39:1 bars."""
    from scipy.ndimage import gaussian_filter, zoom
    x = frame.astype(np.float32) / 255.0
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = lum[..., None] + 1.00 * (x - lum[..., None])
    curve = x * x * (3.0 - 2.0 * x)
    x = 0.62 * x + 0.38 * curve
    x = 0.025 + 0.975 * x
    x *= np.array([1.04, 1.00, 0.94], np.float32)
    x += (1.0 - lum)[..., None] * np.array([0.0, 0.006, 0.022], np.float32)
    small = x[::4, ::4]
    glow = np.clip(small - 0.72, 0.0, 1.0)
    glow = np.stack([gaussian_filter(glow[..., c], 6.0) for c in range(3)], axis=-1)
    glow = zoom(glow, (x.shape[0] / glow.shape[0], x.shape[1] / glow.shape[1], 1.0), order=1)
    x += 0.32 * glow[: x.shape[0], : x.shape[1]]
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - 0.5 * w) / (0.5 * w), (yy - 0.5 * h) / (0.5 * h))
    x *= (1.0 - 0.22 * np.clip(r / 1.35, 0.0, 1.0) ** 2.2)[..., None]
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if letterbox:
        bar = int(round(0.5 * (h - w / 2.39)))
        out[:bar] = 0
        out[h - bar:] = 0
    return out


def _encode(video: Path, width: int, height: int, fps: int, label: str):
    video.parent.mkdir(parents=True, exist_ok=True)
    safe = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf = (
        f"drawtext=text='{safe}':x=40:y=h-44:fontsize=20:fontcolor=white@0.80:"
        f"shadowcolor=black@0.7:shadowx=1:shadowy=1"
    )
    return subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-vf", vf,
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], stdin=subprocess.PIPE)


# --------------------------------------------------------------------------- #
def build_scene(args):
    cover = floor_kwargs_for_plantation(ROWS, COLUMNS, SPACING_M, margin_m=9.0)
    floor = sample_orchard_floor(
        args.seed, canopy_height_m=CANOPY_Z_M,
        slope_deg=0.6, noise_m=0.012, rut_depth_m=0.04, rut_width_m=0.38,
        friction=1.0, slope_azimuth_deg=12.0, **cover,
    )
    skeleton = generate(
        height=CANOPY_Z_M, seed=args.seed, rows=ROWS, columns=COLUMNS,
        spacing=SPACING_M, ground_z=floor.ground_z, canopy_z=floor.canopy_z,
    )
    fruit = place_fruit(skeleton, FruitParams(
        max_count=args.fruit_count, joint="free",
        colors=((0.39, 0.27, 0.12), (0.48, 0.34, 0.17)),
    ), seed=args.seed)
    foliage = FoliageParams(
        enabled=True, leaves_per_terminal=args.leaves, min_order_for_leaves=2,
        leaf_length=LEAF_LENGTH_M, leaf_width=LEAF_WIDTH_M, physics=False,
        canopy_spacing_m=args.canopy_spacing,
    )
    leaves = upright_leaves(place_leaves(skeleton, foliage, seed=args.seed))
    leaves.extend(sun_gaps(place_canopy_leaves(skeleton, foliage, seed=args.seed),
                           seed=args.seed, fraction=args.sun_gaps))
    xs, ys, aisles = row_layout()
    return floor, skeleton, fruit, leaves, xs, ys, aisles


def sun_gaps(leaves, seed: int, fraction: float, cell_m: float = 1.0):
    """Drop render-only infill leaves inside coherent low-frequency pools so
    the aisle gets metre-scale sun patches instead of confetti dapples."""
    if not leaves or fraction <= 0.0:
        return leaves
    if not 0.0 <= fraction < 0.6:
        raise ValueError("sun gap fraction must be in [0, 0.6)")
    xy = np.array([leaf.attach[:2] for leaf in leaves])
    lo = xy.min(axis=0) - cell_m
    span = xy.max(axis=0) - lo + cell_m
    n = 256
    rng = np.random.default_rng(seed + 515)
    field = _iso_field(rng, n, (n * cell_m / float(span.max()),), (1.0,))
    idx = np.clip(((xy - lo) / span * (n - 1)).astype(int), 0, n - 1)
    values = field[idx[:, 1], idx[:, 0]]
    threshold = float(np.quantile(values, 1.0 - fraction))
    return [leaf for leaf, v in zip(leaves, values) if v < threshold]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot", type=Path, default=Path("output/kiwi-street.png"))
    p.add_argument("--video", type=Path, help="write an MP4 animation instead of a still")
    p.add_argument("--relic", type=Path, help="external RELIC checkout (adds Spot walkers)")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--xml", type=Path)
    p.add_argument("--fruit-count", type=int, default=600)
    p.add_argument("--leaves", type=int, default=28)
    p.add_argument("--canopy-spacing", type=float, default=0.09)
    p.add_argument("--sun-gaps", type=float, default=0.14,
                   help="fraction of canopy infill removed in coherent sun pools")
    p.add_argument("--no-grade", action="store_true")
    p.add_argument("--gl", choices=("auto", "egl", "osmesa"), default="auto")
    p.add_argument("--require-gpu", action="store_true")
    args = p.parse_args()
    if args.fruit_count < 0 or args.leaves < 1:
        p.error("fruit/leaf counts must be valid")
    if not np.isfinite(args.canopy_spacing) or args.canopy_spacing < 0.03:
        p.error("--canopy-spacing must be finite and at least 0.03 m")
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.fps < 1:
        p.error("--seconds must be positive and --fps at least 1")
    if args.width % 2 or args.height % 2 or args.width < 320 or args.height < 180:
        p.error("--width/--height must be even and at least 320x180")

    gl_backend = bind_mujoco_gl(args.gl, require_gpu=args.require_gpu)
    import mujoco

    floor, skeleton, fruit, leaves, xs, ys, aisles = build_scene(args)
    with_spot = args.relic is not None
    xml = mjcf(floor, skeleton, fruit, leaves, xs, ys, with_spot_wrap=with_spot)
    if args.xml:
        args.xml.parent.mkdir(parents=True, exist_ok=True)
        args.xml.write_text(xml)

    assets = {
        "orchard_ground.png": ground_texture(floor, xs, ys, args.seed),
        "wood.png": wood_texture(args.seed),
        "leaf_sun.png": leaf_texture(args.seed, sun=True),
        "leaf_shade.png": leaf_texture(args.seed, sun=False),
        "kiwi.png": kiwi_texture(args.seed),
        "concrete.png": concrete_texture(args.seed),
    }
    walkers = []
    with tempfile.TemporaryDirectory(prefix="kiwi-street-") as tmp:
        if with_spot:
            spot, wrap_png = spot_spec(args.relic, Path(tmp))
            assets["bdaii_spot_wrap.png"] = wrap_png.read_bytes()
            scene = mujoco.MjSpec.from_string(xml, assets=assets)
            scene.copy_during_attach = True
            n_robots = 5
            for i in range(n_robots):
                frame = scene.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
                scene.attach(spot, prefix=f"spot{i}_", frame=frame)
                scene.body(f"spot{i}_body").add_freejoint()
            style_spot(scene)
            model = scene.compile()
        else:
            model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    apply_hfield(model, floor, xs, ys)
    # fogstart/fogend are multiples of stat.meansize; express them in metres.
    meansize = max(float(model.stat.meansize), 1e-3)
    model.vis.map.fogstart = 18.0 / meansize
    model.vis.map.fogend = 95.0 / meansize
    # shadowclip is a multiple of stat.extent; express the cube in metres.
    model.vis.map.shadowclip = SHADOW_HALF_M / max(float(model.stat.extent), 1e-3)
    key_light = model.light("key").id
    data = mujoco.MjData(model)
    if with_spot:
        walkers = default_walkers(model, aisles, xs, floor.ground_z, args.seed)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(model, height=args.height, width=args.width, max_geom=160000)
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 1
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SKYBOX)] = 1
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_HAZE)] = 1
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_CULL_FACE)] = 0
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_FOG)] = 1

    def render(t: float, lookat, distance, azimuth, elevation):
        for walker in walkers:
            walker.write(data, t)
        model.light_pos[key_light] = shadow_light_pos(
            shadow_focus(lookat, distance, azimuth, elevation), model.light_dir[key_light])
        # mj_forward also refreshes light_xpos/xdir; mj_kinematics alone does not.
        mujoco.mj_forward(model, data)
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        frame = np.array(renderer.render(), copy=True)
        return frame if args.no_grade else grade(frame)

    summary = {
        "bays": BAYS, "aisles": AISLES, "posts": f"{ROWS}x{COLUMNS}",
        "segments": len(skeleton), "fruit": len(fruit), "leaves": len(leaves),
        "robots": len(walkers), "geoms": int(model.ngeom), "gl": gl_backend,
        "fruit_hang": "place_fruit + STEM_LENGTH",
        "robot_motion": "scripted kinematic trot + sum-of-sines arm; no policy, no contact",
        "scope": "MuJoCo cinematic render; not harvest, not RELIC gait, not a training result",
    }
    if args.video:
        n_frames = int(round(args.seconds * args.fps))
        label = ("MuJoCo 3.8.1  |  Thekenyos kiwi street  |  animacion cinematica programada: "
                 "marcha y brazo guionizados, sin politica RELIC ni fisica de contacto")
        encoder = _encode(args.video, args.width, args.height, args.fps, label)
        try:
            for k in range(n_frames):
                t = k / args.fps
                follow = walkers[0].position if walkers else None
                lookat, distance, azimuth, elevation = camera_path(
                    t, args.seconds, aisles, follow)
                frame = render(t, lookat, distance, azimuth, elevation)
                encoder.stdin.write(frame.tobytes())
                if k % args.fps == 0:
                    print(f"[kiwi-street] frame {k}/{n_frames}", flush=True)
                if k == n_frames // 2 or k == int(0.2 * n_frames):
                    still = args.video.with_name(f"{args.video.stem}-frame{k:04d}.png")
                    Image.fromarray(frame).save(still)
        finally:
            encoder.stdin.close()
            encoder.wait()
        renderer.close()
        summary.update({"video": str(args.video), "frames": n_frames, "fps": args.fps})
    else:
        frame = render(0.0, CAMERA_LOOKAT, CAMERA_DISTANCE_M, CAMERA_AZIMUTH_DEG,
                       CAMERA_ELEVATION_DEG)
        renderer.close()
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.snapshot)
        summary["snapshot"] = str(args.snapshot)
    print(summary)


if __name__ == "__main__":
    main()
