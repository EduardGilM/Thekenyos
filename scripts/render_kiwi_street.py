#!/usr/bin/env python
"""Native MuJoCo still of a five-bay kiwi street.

Fruit uses ``place_fruit`` hang and Hayward radii. The still only adds
render-only leaf cards, timber visuals and a ground texture aligned to the
post rows. Not Newton GL, not Spot gait, not harvest.

    MUJOCO_GL=osmesa python scripts/render_kiwi_street.py \
        --snapshot output/kiwi-street-five-bays.png
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from treesim.config import FoliageParams, FruitParams
from treesim.foliage import place_canopy_leaves, place_leaves
from treesim.gl_backend import bind_mujoco_gl
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit


BAYS = 5
ROWS = 2
COLUMNS = BAYS + 1
SPACING_M = 5.0
CANOPY_Z_M = 1.6
POST_RADIUS_M = 0.082
BEAM_HALF_M = (0.062, 0.048)
FOOTER_RADIUS_M = 0.16
FOOTER_HALF_M = 0.045
# Three-quarter down the aisle so both post rows read as parallel lanes.
CAMERA_LOOKAT = (0.8, 0.06, 0.70)
CAMERA_DISTANCE_M = 16.6
CAMERA_AZIMUTH_DEG = 78.0
CAMERA_ELEVATION_DEG = -10.0


def _rgba(rgb, a=1.0) -> str:
    return " ".join(f"{float(c):.3f}" for c in (*rgb, a))


def _xyzw_to_wxyz(q) -> str:
    x, y, z, w = (float(v) for v in q)
    return f"{w:.5f} {x:.5f} {y:.5f} {z:.5f}"


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


def launch_pads(rows, columns, spacing):
    xs = (np.arange(columns) - 0.5 * (columns - 1)) * spacing
    ys = (np.arange(rows) - 0.5 * (rows - 1)) * spacing
    aisle_y = 0.5 * (ys[0] + ys[1])
    pads = [(0.5 * (xs[i] + xs[i + 1]), aisle_y, i + 1) for i in range(BAYS)]
    return pads, (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]), aisle_y)


def _value_field(rng, shape, waves):
    field = np.zeros(shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    for scale, weight in waves:
        gy, gx = max(int(shape[0] / scale), 2), max(int(shape[1] / scale), 2)
        grid = rng.standard_normal((gy, gx))
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


def fruit_center(item) -> np.ndarray:
    """World fruit COM from ``place_fruit`` attach + Hayward radii."""
    return np.asarray(item.attach, dtype=np.float64) - np.array(
        [0.0, 0.0, STEM_LENGTH + float(item.radii[2])]
    )


def ground_texture(floor, xs, ys, seed: int) -> bytes:
    """Hi-res grass aisle, vine-row soil and wheel tracks on the post rows."""
    n = 2048
    half = float(floor.half_extent_m)
    rng = np.random.default_rng(seed + 331)
    u = np.linspace(-half, half, n)
    v = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(u, v)
    aisle_y = 0.5 * (ys[0] + ys[-1])

    clump = _value_field(rng, (n, n), ((96, 1.0), (36, 0.55), (14, 0.28)))
    patch = _value_field(rng, (n, n), ((28, 1.0), (11, 0.50)))
    grit = _value_field(rng, (n, n), ((5, 1.0), (2, 0.45)))
    speck = rng.random((n, n))
    wander = _value_field(rng, (n, n), ((40, 1.0), (16, 0.4)))

    grass_dark = np.array([0.10, 0.22, 0.06])
    grass_mid = np.array([0.20, 0.36, 0.09])
    grass_sun = np.array([0.34, 0.48, 0.14])
    straw = np.array([0.46, 0.40, 0.16])
    soil_wet = np.array([0.18, 0.10, 0.05])
    soil_loam = np.array([0.42, 0.26, 0.12])
    soil_dust = np.array([0.58, 0.40, 0.20])
    rut_col = np.array([0.30, 0.20, 0.10])
    packed = np.array([0.40, 0.34, 0.22])

    t = np.clip(0.38 * clump + 0.34 * patch + 0.28 * wander, 0.0, 1.0)
    grass = (1.0 - t)[..., None] * grass_dark + t[..., None] * grass_mid
    grass = grass * (0.82 + 0.28 * grit[..., None])
    sun_w = np.clip((clump - 0.55) / 0.40, 0.0, 1.0) * np.clip((patch - 0.40) / 0.45, 0.0, 1.0)
    grass = grass * (1.0 - 0.55 * sun_w)[..., None] + sun_w[..., None] * grass_sun
    dry_w = np.clip((wander - 0.62) / 0.30, 0.0, 1.0) * np.clip(patch, 0.0, 1.0) * 0.55
    grass = grass * (1.0 - dry_w)[..., None] + dry_w[..., None] * straw
    # Mower stripes along the street (X), readable at cinematic distance.
    mow = 0.5 + 0.5 * np.sin(yy * (2.0 * np.pi / 0.16) + 0.35 * np.sin(xx * 0.45))
    aisle = np.clip(1.0 - np.abs(yy - aisle_y) / 1.65, 0.0, 1.0)
    grass = grass * (1.0 - 0.16 * aisle * (mow - 0.5))[..., None]

    soil_w = np.zeros((n, n))
    for y in ys:
        edge = 0.10 + 0.16 * (wander - 0.5)
        d = np.abs(yy - y)
        soil_w = np.maximum(soil_w, np.clip((0.55 + edge - d) / 0.18, 0.0, 1.0))
    soil_t = np.clip(0.40 * grit + 0.35 * patch + 0.25 * clump, 0.0, 1.0)
    soil = (1.0 - soil_t)[..., None] * soil_wet + soil_t[..., None] * soil_dust
    soil = soil * (0.90 + 0.18 * speck[..., None]) + 0.06 * soil_loam
    edge_col = np.array([0.16, 0.14, 0.06])
    w = np.clip(soil_w, 0.0, 1.0)
    rgb = np.empty((n, n, 3))
    low = w < 0.5
    hi = ~low
    t_low = (2.0 * w)[..., None]
    t_hi = (2.0 * w - 1.0)[..., None]
    rgb[low] = (1.0 - t_low[low]) * grass[low] + t_low[low] * edge_col
    rgb[hi] = (1.0 - t_hi[hi]) * edge_col + t_hi[hi] * soil[hi]

    track = np.zeros((n, n))
    for side in (-0.58, 0.58):
        d = np.abs(yy - (aisle_y + side))
        track = np.maximum(track, np.clip(1.0 - d / 0.16, 0.0, 1.0) ** 1.35)
    track *= (0.55 + 0.45 * grit) * (1.0 - 0.65 * soil_w)
    rgb = rgb * (1.0 - 0.72 * track)[..., None] + track[..., None] * rut_col

    pads = np.zeros((n, n))
    for x in xs:
        for y in ys:
            pads = np.maximum(pads, np.clip(1.0 - np.hypot(xx - x, yy - y) / 0.28, 0.0, 1.0))
    rgb = rgb * (1.0 - 0.55 * pads)[..., None] + pads[..., None] * packed

    stones = (speck > 0.992).astype(np.float64) * (0.45 + 0.55 * grit)
    rgb = rgb * (1.0 - 0.55 * stones)[..., None] + stones[..., None] * np.array([0.36, 0.32, 0.24])
    litter = ((speck < 0.006) & (soil_w < 0.4)).astype(np.float64)
    rgb = rgb * (1.0 - 0.35 * litter)[..., None] + litter[..., None] * np.array([0.30, 0.28, 0.08])
    rgb *= 0.78 + 0.36 * clump[..., None]
    return _png_bytes(np.flipud(np.clip(rgb, 0, 1)) * 255.0)


def wood_texture(seed: int) -> bytes:
    """Weathered treated-pine grain for round posts and sawn beams."""
    rng = np.random.default_rng(seed + 19)
    n = 512
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    noise = _value_field(rng, (n, n), ((40, 1.0), (14, 0.55), (5, 0.25)))
    grain = 0.5 + 0.5 * np.sin(xx * 0.18 + 5.2 * np.sin(yy * 0.028) + 2.4 * noise)
    checks = np.clip(np.sin(xx * 0.9 + 8.0 * noise) * 0.5 + 0.15, 0.0, 1.0) ** 2
    weather = _value_field(rng, (n, n), ((22, 1.0), (8, 0.4)))
    dark = np.array([0.22, 0.13, 0.07])
    mid = np.array([0.46, 0.28, 0.12])
    light = np.array([0.62, 0.42, 0.20])
    silver = np.array([0.38, 0.30, 0.20])
    rgb = dark + (mid - dark) * grain[..., None]
    rgb = rgb * (1.0 - 0.35 * weather)[..., None] + weather[..., None] * light
    rgb = rgb * (1.0 - 0.40 * checks)[..., None] + checks[..., None] * dark
    gray = np.clip((weather - 0.55) / 0.40, 0.0, 1.0) * 0.35
    rgb = rgb * (1.0 - gray)[..., None] + gray[..., None] * silver
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def leaf_texture(seed: int, sun: bool) -> bytes:
    """Full-card Actinidia blade: midrib, veins and mottling, no dark mattes."""
    rng = np.random.default_rng(seed + (81 if sun else 77))
    n = 384
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    u = (xx / (n - 1)) * 2.0 - 1.0
    v = yy / (n - 1)
    mottling = _value_field(rng, (n, n), ((18, 1.0), (7, 0.5), (3, 0.25)))
    speckle = 0.90 + 0.10 * rng.random((n, n))
    midrib = np.exp(-((u * 11.0) ** 2)) * (0.55 + 0.45 * v)
    veins = np.zeros((n, n))
    for k in range(-6, 7):
        if k == 0:
            continue
        shift = 0.10 * k * (0.12 + 0.88 * v)
        slant = 0.18 * k * (v - 0.08)
        veins += np.exp(-(((u - shift - 0.15 * slant) * 14.0) ** 2)) * (
            0.50 * (1.0 - abs(k) / 8.0)
        )
    veins *= 0.35 + 0.65 * v
    edge = np.clip(1.0 - (np.abs(u) ** 1.6) * 0.22 - (v - 0.5) ** 2 * 0.15, 0.55, 1.0)
    if sun:
        dark = np.array([0.22, 0.46, 0.10])
        light = np.array([0.42, 0.64, 0.16])
        rib = np.array([0.34, 0.52, 0.12])
    else:
        dark = np.array([0.12, 0.34, 0.07])
        light = np.array([0.26, 0.52, 0.12])
        rib = np.array([0.20, 0.40, 0.09])
    mix = np.clip(0.28 + 0.55 * mottling + 0.17 * v, 0.0, 1.0)
    rgb = dark + (light - dark) * mix[..., None]
    rgb *= speckle[..., None] * edge[..., None]
    rgb = rgb * (1.0 - 0.28 * midrib)[..., None] + midrib[..., None] * rib
    rgb = rgb * (1.0 - 0.16 * np.clip(veins, 0, 1))[..., None] + (
        np.clip(veins, 0, 1)[..., None] * (rib * 0.85)
    )
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def concrete_texture(seed: int) -> bytes:
    rng = np.random.default_rng(seed + 4)
    n = 256
    grit = _value_field(rng, (n, n), ((12, 1.0), (4, 0.5)))
    speckle = rng.random((n, n))
    rgb = np.array([0.42, 0.40, 0.36]) * (0.82 + 0.28 * grit[..., None])
    rgb = rgb * (0.92 + 0.12 * speckle[..., None])
    return _png_bytes(np.clip(rgb, 0, 1) * 255.0)


def street_heights(floor, xs, ys) -> np.ndarray:
    """Keep sampled slope; add aisle ruts and post pads on the working lane."""
    heights = np.asarray(floor.heights_m, dtype=np.float64).copy()
    x = np.asarray(floor.x_m, dtype=np.float64)
    y = np.asarray(floor.y_m, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    aisle_y = 0.5 * (ys[0] + ys[-1])
    for side in (-0.58, 0.58):
        dist = np.abs(yy - (aisle_y + side))
        heights -= 0.028 * np.clip(1.0 - dist / 0.14, 0.0, 1.0) ** 2
    for px in xs:
        for py in ys:
            pad = np.clip((0.26 - np.hypot(xx - px, yy - py)) / 0.10, 0.0, 1.0)
            iu = int(np.clip(round((px - x[0]) / (x[-1] - x[0]) * (x.size - 1)), 0, x.size - 1))
            iv = int(np.clip(round((py - y[0]) / (y[-1] - y[0]) * (y.size - 1)), 0, y.size - 1))
            heights = (1.0 - pad) * heights + pad * float(heights[iv, iu])
    return heights


def mjcf(floor, skeleton, fruit, leaves, pads, xs, ys) -> str:
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
                f'rgba="0.46 0.44 0.40 1" contype="0" conaffinity="0"/>'
            )
            geoms.append(
                f'    <geom type="cylinder" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}" '
                f'size="{POST_RADIUS_M:.3f} {0.5 * length:.5f}" '
                f'quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}" '
                f'material="wood" rgba="0.50 0.32 0.16 1" contype="0" conaffinity="0"/>'
            )
            top = seg.end if seg.end[2] > seg.start[2] else seg.start
            geoms.append(
                f'    <geom type="cylinder" pos="{top[0]:.5f} {top[1]:.5f} {top[2] - 0.04:.5f}" '
                f'size="{POST_RADIUS_M + 0.012:.3f} 0.016" material="steel" '
                f'rgba="0.30 0.30 0.30 1" contype="0" conaffinity="0"/>'
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
        geoms.append(
            f'    <geom type="ellipsoid" pos="'
            f'{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}" '
            f'size="{rx:.5f} {ry:.5f} {rz:.5f}" material="kiwi" '
            f'rgba="{_rgba(item.color)}" contype="0" conaffinity="0"/>'
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
        sun = bool(leaf_rng.random() < 0.42)
        tint = leaf_rng.uniform(0.88, 1.08)
        material = "leaf_sun" if sun else "leaf_shade"
        geoms.append(
            f'    <geom type="box" pos="'
            f'{leaf.attach[0]:.5f} {leaf.attach[1]:.5f} {leaf.attach[2]:.5f}" '
            f'size="{0.5 * leaf.width:.4f} 0.0014 {0.5 * leaf.length:.4f}" '
            f'quat="{_xyzw_to_wxyz(leaf.frame)}" material="{material}" '
            f'rgba="{tint:.3f} {tint:.3f} {min(1.0, tint * 0.96):.3f} 1" '
            f'contype="0" conaffinity="0"/>'
        )
    for x, y, _number in pads:
        z = float(floor.ground_z(x, y)) + 0.025
        geoms.append(
            f'    <geom type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'size="0.58 0.36 0.018" material="pad" rgba="0.12 0.13 0.14 1" '
            f'contype="0" conaffinity="0"/>'
        )
        geoms.append(
            f'    <geom type="box" pos="{x:.4f} {y:.4f} {z + 0.02:.4f}" '
            f'size="0.08 0.30 0.006" material="mark" rgba="0.90 0.68 0.10 1" '
            f'contype="0" conaffinity="0"/>'
        )
    return f'''<mujoco model="kiwi_street_five_bays">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="38"/>
    <headlight ambient=".20 .20 .18" diffuse=".40 .40 .36" specular=".10 .10 .09"/>
    <rgba haze=".60 .68 .74 1"/>
    <map fogstart="16" fogend="52" znear=".12" zfar="90"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".40 .56 .72" rgb2=".86 .90 .93"
             width="512" height="512"/>
    <texture type="2d" name="orchard" file="orchard_ground.png"/>
    <texture type="2d" name="wood" file="wood.png"/>
    <texture type="2d" name="leaf_sun" file="leaf_sun.png"/>
    <texture type="2d" name="leaf_shade" file="leaf_shade.png"/>
    <texture type="2d" name="concrete" file="concrete.png"/>
    <material name="orchard" texture="orchard" texrepeat="1 1" texuniform="false"
              reflectance="0.03" shininess="0.05" specular="0.08" rgba="1 1 1 1"/>
    <material name="wood" texture="wood" texrepeat="1 3" texuniform="true"
              reflectance="0.06" shininess="0.14" specular="0.16"/>
    <material name="concrete" texture="concrete" texrepeat="2 2" texuniform="true"
              reflectance="0.04" shininess="0.08" specular="0.10"/>
    <material name="steel" reflectance="0.38" shininess="0.68" specular="0.48"
              rgba="0.38 0.38 0.40 1"/>
    <material name="vine" reflectance="0.05" shininess="0.10" specular="0.08"/>
    <material name="leaf_sun" texture="leaf_sun" texrepeat="1 1" texuniform="false"
              reflectance="0.05" shininess="0.24" specular="0.18"/>
    <material name="leaf_shade" texture="leaf_shade" texrepeat="1 1" texuniform="false"
              reflectance="0.03" shininess="0.16" specular="0.12"/>
    <material name="kiwi" reflectance="0.07" shininess="0.38" specular="0.16"/>
    <material name="pad" reflectance="0.22" shininess="0.40" specular="0.30"/>
    <material name="mark" reflectance="0.14" shininess="0.30" specular="0.22"/>
    <hfield name="orchard_ground" nrow="{nrow}" ncol="{ncol}"
            size="{half} {half} {elevation:.5f} 0.08"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="16 -14 11" dir="-0.42 0.28 -0.86"
           diffuse=".90 .84 .70" specular=".22 .20 .14" castshadow="true"/>
    <light name="fill" pos="0 -7 7" diffuse=".26 .30 .28"/>
    <light name="rim" directional="true" pos="-10 8 7" dir="0.40 -0.18 -0.90"
           diffuse=".16 .22 .28"/>
    <light name="aisle0" pos="-6 0 1.20" diffuse=".44 .38 .26"/>
    <light name="aisle1" pos="0 0 1.20" diffuse=".46 .40 .28"/>
    <light name="aisle2" pos="6 0 1.20" diffuse=".40 .34 .22"/>
    <geom name="ground" type="hfield" hfield="orchard_ground" material="orchard"
          pos="0 0 {min_z:.5f}" rgba="1 1 1 1"
          friction="{float(floor.friction):.3f} 0.01 0.001"/>
{chr(10).join(geoms)}
  </worldbody>
</mujoco>
'''


def apply_hfield(model, floor, xs, ys) -> None:
    heights = street_heights(floor, xs, ys)
    min_z = float(heights.min())
    span = max(float(heights.max()) - min_z, 1e-4)
    model.hfield_data[:] = ((heights - min_z) / span).astype(np.float64).ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot", type=Path, default=Path("output/kiwi-street-five-bays.png"))
    p.add_argument("--xml", type=Path)
    p.add_argument("--fruit-count", type=int, default=600)
    p.add_argument("--leaves", type=int, default=28)
    p.add_argument("--canopy-spacing", type=float, default=0.08)
    p.add_argument("--gl", choices=("auto", "egl", "osmesa"), default="auto")
    p.add_argument("--require-gpu", action="store_true")
    args = p.parse_args()
    if args.fruit_count < 0 or args.leaves < 1:
        p.error("fruit/leaf counts must be valid")
    if not np.isfinite(args.canopy_spacing) or args.canopy_spacing < 0.03:
        p.error("--canopy-spacing must be finite and at least 0.03 m")

    gl_backend = bind_mujoco_gl(args.gl, require_gpu=args.require_gpu)
    import mujoco

    cover = floor_kwargs_for_plantation(ROWS, COLUMNS, SPACING_M, margin_m=4.0)
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
        leaf_length=0.22, leaf_width=0.17, physics=False,
        canopy_spacing_m=args.canopy_spacing,
    )
    leaves = place_leaves(skeleton, foliage, seed=args.seed)
    leaves.extend(place_canopy_leaves(skeleton, foliage, seed=args.seed))
    pads, extents = launch_pads(ROWS, COLUMNS, SPACING_M)
    xs = list(np.linspace(extents[0], extents[1], COLUMNS))
    ys = list(np.linspace(extents[2], extents[3], ROWS))
    xml = mjcf(floor, skeleton, fruit, leaves, pads, xs, ys)
    if args.xml:
        args.xml.parent.mkdir(parents=True, exist_ok=True)
        args.xml.write_text(xml)

    assets = {
        "orchard_ground.png": ground_texture(floor, xs, ys, args.seed),
        "wood.png": wood_texture(args.seed),
        "leaf_sun.png": leaf_texture(args.seed, sun=True),
        "leaf_shade.png": leaf_texture(args.seed, sun=False),
        "concrete.png": concrete_texture(args.seed),
    }
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    apply_hfield(model, floor, xs, ys)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE_M
    camera.azimuth = CAMERA_AZIMUTH_DEG
    camera.elevation = CAMERA_ELEVATION_DEG
    renderer = mujoco.Renderer(model, height=1080, width=1920, max_geom=80000)
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 1
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SKYBOX)] = 1
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_HAZE)] = 1
    renderer.update_scene(data, camera=camera)
    pixels = np.array(renderer.render(), copy=True)
    renderer.close()

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.snapshot)
    print({
        "snapshot": str(args.snapshot),
        "bays": BAYS,
        "posts": f"{ROWS}x{COLUMNS}",
        "segments": len(skeleton),
        "fruit": len(fruit),
        "leaves": len(leaves),
        "pads": len(pads),
        "geoms": int(model.ngeom),
        "extent_m": extents[:4],
        "gl": gl_backend,
        "fruit_hang": "place_fruit + STEM_LENGTH",
        "camera": {
            "lookat": CAMERA_LOOKAT,
            "distance_m": CAMERA_DISTANCE_M,
            "azimuth_deg": CAMERA_AZIMUTH_DEG,
            "elevation_deg": CAMERA_ELEVATION_DEG,
        },
        "scope": "MuJoCo cinematic still; not harvest, gait, or a five-robot demo",
    })


if __name__ == "__main__":
    main()
