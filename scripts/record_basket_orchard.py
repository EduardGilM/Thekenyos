#!/usr/bin/env python
"""Scripted six-fruit basket harvest inside the beauty-frames hillside orchard.

Reuses ``check_basket_kiwi.py``'s controller, frozen RELIC gait and the six
harvestable assisted fruit. The plantation look comes from
``cursor/mujoco-beauty-frames`` (cc65e82): rolling landform, tiled grass/soil,
cordate leaf roof, kiwis on tied canes as well as tips. Extra fruit and
foliage are visual only. This is not a physical-grasp validation and not
learned deposit behaviour.

The camera stays in the aisle and looks through Spot into the block. Storage
and wrist-RGB insets are composited on the same frame.

    MUJOCO_GL=egl python scripts/record_basket_orchard.py --relic ../relic \
      --output output/basket-orchard-six --picks 4 --video
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from treesim.assisted_kiwi_env import PREFIX
from treesim.basket import CENTER, SIZE
from treesim.basket_kiwi_env import BasketKiwiEnv, BasketTask, SCOPE
from treesim.visual_kiwi_env import RGB_FOVY_DEG, add_hand_cameras_xml
from treesim.config import FoliageParams, FruitParams
from treesim.foliage import (
    LEAF_SIZE_CLASSES, leaf_blade_arrays, leaf_blade_style, place_canopy_leaves, place_leaves,
)
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import (
    POST_EMBED_M, earth_cut_png_bytes, floor_kwargs_for_plantation,
    sample_orchard_floor, shade_under_canopy, tiled_floor_rgb,
)
from treesim.pergola import generate as generate_pergola, place_fruit

from scripts.check_basket_kiwi import ScriptedBasketController


HARVEST_XY = np.array([[x, y] for y in (-.65, .65) for x in (-1., 0., 1.)])
CANOPY_CLEARANCE_M = 1.6
WOOD = ((0.45, 0.28, 0.12), (0.35, 0.22, 0.10), (0.28, 0.42, 0.14))
FRUIT_COLORS = ((0.42, 0.28, 0.10), (0.55, 0.38, 0.14), (0.33, 0.22, 0.08))
SUN_DIR = np.array([0.26, 0.42, -1.0], dtype=float)
SUN_DIR /= float(np.linalg.norm(SUN_DIR))
BANNER = ('SCRIPTED MULTI-KIWI CHECK | hillside orchard | assisted attachment | '
          'free basket release')
VIDEO_SIZE = (1920, 1080)
CONTROL_HZ = 10


@dataclass
class ExtraKiwi:
    attach: np.ndarray
    center: np.ndarray
    radii: np.ndarray
    color: tuple


@dataclass
class OrchardScenery:
    floor: object
    skeleton: object
    leaves: list
    extra_fruit: list
    harvest_positions: np.ndarray
    hfield_normalized: np.ndarray
    ground_png: Path
    earth_png: Path
    metrics: dict
    canopy_png: Path | None = None
    leaf_draw: str = 'mesh'
    canopy_tiles: list | None = None

    def ground_z(self, x, y) -> float:
        return float(self.floor.ground_z(x, y))

    def canopy_z(self, x, y) -> float:
        return CANOPY_CLEARANCE_M + self.ground_z(x, y)


def _rgba(rgb, a=1.) -> str:
    return ' '.join(f'{float(c):.3f}' for c in (*rgb, a))


def _quat_wxyz(xyzw) -> str:
    x, y, z, w = (float(v) for v in xyzw)
    return f'{w:.6f} {x:.6f} {y:.6f} {z:.6f}'


def _cordate_outline(length: float, width: float, n: int = 28) -> np.ndarray:
    """2-D cordate blade in the leaf plane, same envelope as ``leaf_blade_arrays``."""
    ts = np.linspace(0.0, 1.0, n)
    right = []
    for t in ts:
        tt = min(float(t), 0.995)
        sinus = float(np.exp(-(tt / 0.065) ** 2))
        lobe = float(np.sin(np.pi * min(tt / 0.34, 1.0)) ** 0.80)
        body = float(np.sin(np.pi * (tt ** 0.62)) ** 0.72)
        envelope = (0.88 * body + 0.42 * lobe * (1.0 - tt)) * (1.0 - 0.62 * sinus)
        envelope = max(envelope, 0.035 * (1.0 - tt) + 0.02)
        serration = 1.0
        if 0.08 < tt < 0.92:
            serration += 0.08 * float(np.sin(12.0 * np.pi * tt))
        w = 0.5 * width * envelope * serration
        right.append((w, length * t))
    left = [(-x, z) for x, z in reversed(right[1:-1])]
    return np.asarray(right + left, dtype=np.float64)


def kiwi_canopy_png_bytes(seed: int = 42, size: int = 512) -> bytes:
    """Opaque tiled kiwi-leaf albedo for MuJoCo's textured-material shader.

    This is not a custom GLSL foliage shader. Native MuJoCo only samples a 2-D
    texture on a geom. The stamp uses the same cordate outline as the mesh leaves.
    """
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed + 91)
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[:] = (18, 48, 14)
    image = Image.fromarray(pixels, mode='RGB')
    draw = ImageDraw.Draw(image)
    outline = _cordate_outline(0.22, 0.17)
    span = 1.35
    for _ in range(220):
        cx, cy = rng.uniform(0.0, span, 2)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        scale = float(rng.uniform(0.78, 1.18))
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        xy = (outline * scale) @ rot.T + np.array([cx, cy])
        uv = (xy / span) * size
        uv[:, 0] %= size
        shade = int(rng.integers(-18, 16))
        color = (max(12, 36 + shade), max(40, 92 + shade), max(10, 26 + shade // 2))
        draw.polygon([(float(x), float(y)) for x, y in uv], fill=color)
    buf = __import__('io').BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def _canopy_tiles(skeleton, canopy_z, tile_m: float = 2.5) -> list:
    lo, hi = skeleton.bounds()
    xs = np.arange(float(lo[0]) + 0.5 * tile_m, float(hi[0]) + 1e-6, tile_m)
    ys = np.arange(float(lo[1]) + 0.5 * tile_m, float(hi[1]) + 1e-6, tile_m)
    half = 0.52 * tile_m
    tiles = []
    for x in xs:
        for y in ys:
            tiles.append((float(x), float(y), float(canopy_z(x, y)) + 0.06, half))
    if not tiles:
        raise RuntimeError('textured canopy produced no tiles')
    return tiles


def harvest_positions_on_floor(floor) -> np.ndarray:
    out = []
    for x, y in HARVEST_XY:
        z = CANOPY_CLEARANCE_M + float(floor.ground_z(x, y)) - 0.18
        out.append([float(x), float(y), z])
    return np.asarray(out, dtype=float)


def build_scenery(*, seed: int, extra_fruit: int, rows: int, columns: int, spacing: float,
                  slope_deg: float, slope_azimuth_deg: float, noise_m: float, rut_depth_m: float,
                  landform_m: float, landform_wavelength_m: float, canopy_spacing: float,
                  asset_dir: Path, leaf_draw: str = 'mesh') -> OrchardScenery:
    if extra_fruit < 0 or rows < 2 or columns < 2:
        raise ValueError('extra fruit must be nonnegative and the pergola at least 2x2')
    if leaf_draw not in ('mesh', 'texture'):
        raise ValueError('leaf_draw must be mesh or texture')
    asset_dir.mkdir(parents=True, exist_ok=True)
    cover = floor_kwargs_for_plantation(rows, columns, spacing, margin_m=8.)
    floor = sample_orchard_floor(
        seed, slope_deg=slope_deg, slope_azimuth_deg=slope_azimuth_deg,
        noise_m=noise_m, rut_depth_m=rut_depth_m, rut_width_m=0.40, friction=1.0,
        canopy_height_m=CANOPY_CLEARANCE_M, landform_m=landform_m,
        landform_wavelength_m=landform_wavelength_m, **cover)
    if float(floor.heights_m.max() - floor.heights_m.min()) < 0.04:
        raise RuntimeError('Orchard floor is too flat for an uneven-terrain recording')
    skeleton = generate_pergola(
        height=CANOPY_CLEARANCE_M, seed=seed, rows=rows, columns=columns, spacing=spacing,
        ground_z=floor.ground_z, canopy_z=lambda x, y: CANOPY_CLEARANCE_M + floor.ground_z(x, y))
    height_z = lambda x, y: CANOPY_CLEARANCE_M + floor.ground_z(x, y)
    leaves = []
    tiles = []
    canopy_png = None
    if leaf_draw == 'mesh':
        fp = FoliageParams(
            enabled=True, leaves_per_terminal=8, min_order_for_leaves=2,
            leaf_length=0.22, leaf_width=0.17, leaf_shape='cordate',
            leaf_color=(0.14, 0.36, 0.10), canopy_spacing_m=float(canopy_spacing))
        leaves = place_leaves(skeleton, fp, seed=seed, height_z=height_z)
        if fp.canopy_spacing_m:
            leaves = list(leaves) + place_canopy_leaves(skeleton, fp, seed=seed, height_z=height_z)
    else:
        canopy_png = asset_dir / 'kiwi_canopy.png'
        canopy_png.write_bytes(kiwi_canopy_png_bytes(seed))
        tiles = _canopy_tiles(skeleton, height_z)
    harvest = harvest_positions_on_floor(floor)
    extra = []
    for fruit in place_fruit(skeleton, FruitParams(max_count=int(extra_fruit), joint='free',
                                                   colors=FRUIT_COLORS), seed=seed):
        center = fruit.attach - np.array([0., 0., STEM_LENGTH + float(fruit.radii[2])])
        if np.min(np.linalg.norm(center[:2] - harvest[:, :2], axis=1)) < .28:
            continue
        extra.append(ExtraKiwi(attach=np.asarray(fruit.attach, float), center=center,
                               radii=np.asarray(fruit.radii, float), color=tuple(fruit.color)))
    from io import BytesIO
    from PIL import Image
    look = shade_under_canopy(tiled_floor_rgb(floor), floor, skeleton, sun_dir=SUN_DIR, seed=seed)
    buf = BytesIO()
    Image.fromarray(np.clip(np.flipud(look) * 255.0, 0, 255).astype(np.uint8), mode='RGB').save(buf, format='PNG')
    ground_png = asset_dir / 'orchard_ground.png'
    ground_png.write_bytes(buf.getvalue())
    earth_png = asset_dir / 'earth_cut.png'
    earth_png.write_bytes(earth_cut_png_bytes(seed))
    heights = np.asarray(floor.heights_m, dtype=np.float64)
    span = max(float(heights.max() - heights.min()), 1e-4)
    return OrchardScenery(
        floor=floor, skeleton=skeleton, leaves=leaves, extra_fruit=extra,
        harvest_positions=harvest,
        hfield_normalized=((heights - heights.min()) / span).astype(np.float64),
        ground_png=ground_png, earth_png=earth_png,
        metrics=dict(floor.metrics(), extra_fruit=len(extra), leaves=len(leaves),
                     segments=len(skeleton), harvest_fruit=len(HARVEST_XY),
                     canopy_clearance_m=CANOPY_CLEARANCE_M, visual_only_extra_fruit=True,
                     post_embed_m=POST_EMBED_M, beauty_frames_commit='cc65e82',
                     leaf_draw=leaf_draw, canopy_tiles=len(tiles)),
        canopy_png=canopy_png, leaf_draw=leaf_draw, canopy_tiles=tiles)


def _leaf_assets(fp: FoliageParams) -> list:
    style = leaf_blade_style(fp.leaf_shape)
    assets = []
    for i, scale in enumerate(LEAF_SIZE_CLASSES):
        verts, faces = leaf_blade_arrays(fp.leaf_length * scale, fp.leaf_width * scale, **style)
        assets.append((f'kiwi_leaf_{i}', verts, faces))
    return assets


def add_storage_camera(chassis):
    """Chassis camera above the open rear basket, looking down at the floor."""
    # Floor is CENTER; walls rise SIZE[2]. Sit above the rim and look at the
    # liner so deposited fruit stay in frame as they accumulate.
    eye = CENTER + np.array([0.06, 0.12, SIZE[2] + 0.38])
    target = CENTER + np.array([0.0, 0.0, 0.06])
    look = target - eye
    look /= max(float(np.linalg.norm(look)), 1e-9)
    z_axis = -look
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    norm = float(np.linalg.norm(x_axis))
    x_axis = np.array([0.0, 1.0, 0.0]) if norm < 1e-8 else x_axis / norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
    ET.SubElement(chassis, 'camera', name='basket_cam',
                  pos=' '.join(f'{v:.5f}' for v in eye),
                  xyaxes=' '.join(f'{v:.5f}' for v in np.r_[x_axis, y_axis]),
                  fovy='55')


def apply_scenery(root, scenery: OrchardScenery) -> None:
    """Replace the four-post fixture with the beauty-frames hillside orchard."""
    assets = root.find('asset')
    world = root.find('worldbody')
    visual = root.find('visual')
    for geom in world.findall('geom'):
        kind = geom.get('type')
        if kind in ('capsule', 'ellipsoid'):
            geom.set('rgba', '0 0 0 0')
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
        elif kind == 'plane':
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            geom.set('rgba', '0 0 0 0')
    for i, position in enumerate(scenery.harvest_positions):
        body = world.find(f"body[@name='assisted_kiwi_{i}']")
        if body is not None:
            body.set('pos', ' '.join(f'{v:.5f}' for v in position))
        anchor = world.find(f"site[@name='anchor_{i}']")
        if anchor is not None:
            anchor.set('pos', ' '.join(f'{v:.5f}' for v in position))
    sky = assets.find("texture[@name='sky']")
    if sky is not None:
        sky.set('rgb1', '.42 .62 .86')
        sky.set('rgb2', '.90 .93 .96')
        sky.set('width', '512')
        sky.set('height', '512')
    if visual.find('rgba') is None:
        ET.SubElement(visual, 'rgba', haze='.70 .78 .86 1')
    mapping = visual.find('map')
    if mapping is None:
        mapping = ET.SubElement(visual, 'map')
    mapping.set('fogstart', '18')
    mapping.set('fogend', '90')
    mapping.set('znear', '.05')
    mapping.set('zfar', '220')
    quality = visual.find('quality')
    if quality is None:
        quality = ET.SubElement(visual, 'quality')
    quality.set('shadowsize', '256')
    quality.set('offsamples', '4')
    glob = visual.find('global')
    if glob is not None:
        glob.set('offwidth', str(VIDEO_SIZE[0]))
        glob.set('offheight', str(VIDEO_SIZE[1]))
        glob.set('fovy', '52')
    headlight = visual.find('headlight')
    if headlight is not None:
        headlight.set('ambient', '.22 .24 .20')
        headlight.set('diffuse', '.16 .17 .14')
        headlight.set('specular', '0 0 0')
    for light in list(world.findall('light')):
        world.remove(light)
    heights = np.asarray(scenery.floor.heights_m, dtype=np.float64)
    min_z, max_z = float(heights.min()), float(heights.max())
    elevation = max(max_z - min_z, 1e-4)
    half = float(scenery.floor.half_extent_m)
    bulk = max(6.0, 0.35 * elevation)
    sun_pos = np.array([0.0, 0.0, float(scenery.canopy_z(0.0, 0.0))]) - 28.0 * SUN_DIR
    ET.SubElement(world, 'light', name='sun', directional='true', castshadow='false',
                  pos=f'{sun_pos[0]:.3f} {sun_pos[1]:.3f} {sun_pos[2]:.3f}',
                  dir=f'{SUN_DIR[0]:.4f} {SUN_DIR[1]:.4f} {SUN_DIR[2]:.4f}',
                  diffuse='.88 .80 .62', specular='.08 .06 .04', ambient='.10 .11 .09')
    ET.SubElement(world, 'light', name='fill', directional='true', castshadow='false',
                  pos=f'{-0.4 * half:.3f} {0.3 * half:.3f} {max(18.0, 0.7 * half):.3f}',
                  dir='-0.12 -0.08 -1', diffuse='.22 .26 .22', specular='0 0 0')
    ET.SubElement(assets, 'texture', name='orchard', type='2d', file=str(scenery.ground_png.resolve()))
    ET.SubElement(assets, 'texture', name='earth_cut', type='2d', file=str(scenery.earth_png.resolve()))
    ET.SubElement(assets, 'material', name='orchard', texture='orchard', texrepeat='1 1',
                  texuniform='false', emission='0.28', reflectance='0.0', specular='0.02',
                  shininess='0.04', roughness='0.95', metallic='0.0')
    ET.SubElement(assets, 'material', name='earth_cut', texture='earth_cut', texrepeat='8 8',
                  texuniform='true', reflectance='0.0', specular='0.03', shininess='0.05',
                  roughness='0.95', metallic='0.0')
    nrow, ncol = heights.shape
    ET.SubElement(assets, 'hfield', name='orchard_ground', nrow=str(nrow), ncol=str(ncol),
                  size=f'{half} {half} {elevation:.5f} {bulk:.3f}')
    ET.SubElement(world, 'geom', name='orchard_ground', type='hfield', hfield='orchard_ground',
                  material='orchard', pos=f'0 0 {min_z:.5f}', friction=f'{scenery.floor.friction:.3f} 0.01 0.001')
    skirt = 0.45
    ET.SubElement(world, 'geom', name='earth_mass', type='box',
                  size=f'{half + 0.8:.3f} {half + 0.8:.3f} {bulk:.3f}',
                  pos=f'0 0 {min_z - bulk - 0.15:.4f}', material='earth_cut',
                  contype='0', conaffinity='0')
    for name, sx, sy, px, py in (
            ('earth_x_pos', skirt, half, half, 0.0),
            ('earth_x_neg', skirt, half, -half, 0.0),
            ('earth_y_pos', half, skirt, 0.0, half),
            ('earth_y_neg', half, skirt, 0.0, -half)):
        ET.SubElement(world, 'geom', name=name, type='box',
                      size=f'{sx:.3f} {sy:.3f} {(elevation + bulk) * 0.5:.3f}',
                      pos=f'{px:.4f} {py:.4f} {min_z - bulk + 0.5 * (elevation + bulk):.4f}',
                      material='earth_cut', contype='0', conaffinity='0')
    for seg in scenery.skeleton:
        rgb = WOOD[min(seg.order, 2)]
        size = max(float(seg.mean_radius), 0.003 if seg.order else 0.04)
        ET.SubElement(world, 'geom', type='capsule',
                      fromto=' '.join(f'{v:.5f}' for v in (*seg.start, *seg.end)),
                      size=f'{size:.5f}', rgba=_rgba(rgb), contype='0', conaffinity='0', group='2')
    if scenery.leaf_draw == 'texture':
        if scenery.canopy_png is None or not scenery.canopy_tiles:
            raise RuntimeError('textured canopy is missing its albedo or tiles')
        ET.SubElement(assets, 'texture', name='kiwi_canopy', type='2d',
                      file=str(scenery.canopy_png.resolve()))
        ET.SubElement(assets, 'material', name='kiwi_canopy', texture='kiwi_canopy',
                      texrepeat='5 5', texuniform='false', emission='0.22',
                      reflectance='0.0', specular='0.02', shininess='0.04',
                      roughness='0.95', metallic='0.0')
        for i, (x, y, z, half) in enumerate(scenery.canopy_tiles):
            ET.SubElement(world, 'geom', name=f'canopy_tile_{i}', type='box',
                          size=f'{half:.4f} {half:.4f} 0.018',
                          pos=f'{x:.4f} {y:.4f} {z:.4f}',
                          material='kiwi_canopy', contype='0', conaffinity='0',
                          group='2', mass='0')
    else:
        fp = FoliageParams(leaf_length=0.22, leaf_width=0.17, leaf_shape='cordate',
                           leaf_color=(0.14, 0.36, 0.10))
        for name, verts, faces in _leaf_assets(fp):
            ET.SubElement(assets, 'mesh', name=name, inertia='shell',
                          vertex=' '.join(f'{float(v):.5f}' for v in np.asarray(verts).ravel()),
                          face=' '.join(str(int(i)) for i in np.asarray(faces).ravel()))
        rgba = _rgba(fp.leaf_color)
        nominal = max(float(fp.leaf_length), 1e-9)
        for leaf in scenery.leaves:
            cls = int(np.argmin([abs(leaf.length / nominal - s) for s in LEAF_SIZE_CLASSES]))
            ET.SubElement(world, 'geom', type='mesh', mesh=f'kiwi_leaf_{cls}',
                          pos=' '.join(f'{v:.4f}' for v in leaf.attach), quat=_quat_wxyz(leaf.frame),
                          rgba=rgba, contype='0', conaffinity='0', group='2', mass='0')
    wrist = root.find(f'.//body[@name="{PREFIX}arm_link_wr1"]')
    if wrist is not None and wrist.find("camera[@name='ee_cam']") is None:
        add_hand_cameras_xml(wrist)
    chassis = root.find(f'.//body[@name="{PREFIX}body"]')
    if chassis is not None and chassis.find("camera[@name='basket_cam']") is None:
        add_storage_camera(chassis)
    for i, fruit in enumerate(scenery.extra_fruit):
        top = fruit.center + np.array([0., 0., float(fruit.radii[2])])
        ET.SubElement(world, 'geom', type='capsule',
                      fromto=' '.join(f'{v:.5f}' for v in (*fruit.attach, *top)),
                      size='.002', rgba='.28 .38 .12 1', contype='0', conaffinity='0', group='2')
        ET.SubElement(world, 'geom', name=f'visual_kiwi_{i}', type='ellipsoid',
                      pos=' '.join(f'{v:.5f}' for v in fruit.center),
                      size=' '.join(f'{v:.5f}' for v in fruit.radii),
                      rgba=_rgba(fruit.color), contype='0', conaffinity='0', group='2')


def dress_model(model, scenery: OrchardScenery):
    with tempfile.TemporaryDirectory(prefix='basket-orchard-') as directory:
        path = Path(directory) / 'scene.xml'
        mujoco.mj_saveLastXML(str(path), model)
        root = ET.parse(path).getroot()
        apply_scenery(root, scenery)
        dressed = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    dressed.hfield_data[:] = scenery.hfield_normalized.ravel()
    return dressed


def mjv_from_eye_target(eye, target):
    lookat = np.asarray(target, dtype=float)
    delta = np.asarray(eye, dtype=float) - lookat
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance < 1e-6:
        raise ValueError('camera eye and target must be distinct')
    azimuth = math.degrees(math.atan2(delta[0], -delta[1]))
    elevation = math.degrees(math.asin(float(np.clip(delta[2] / distance, -1.0, 1.0))))
    return lookat, distance, azimuth, elevation


def free_camera_eye(lookat, distance, azimuth, elevation) -> np.ndarray:
    """World-space eye of a MuJoCo free camera (matches ``mjv_updateCamera``)."""
    az = math.radians(float(azimuth))
    el = math.radians(float(elevation))
    d = float(distance)
    lookat = np.asarray(lookat, dtype=float)
    return lookat + d * np.array([
        math.cos(el) * math.sin(az),
        -math.cos(el) * math.cos(az),
        math.sin(el),
    ])


def landscape_camera(env):
    """Aisle working shot: stay under the leaf roof, Spot readable, orchard ahead."""
    chassis = np.asarray(env.data.xpos[env.chassis], dtype=float)
    ground = float(env.scenery.ground_z(chassis[0], chassis[1]))
    # South-west in the grass aisle. Do not climb above the canopy or the earth
    # skirt fills the frame and Spot collapses to a speck.
    eye = np.array([chassis[0] - 1.4, chassis[1] - 5.8,
                    min(ground + 1.16, float(env.scenery.canopy_z(chassis[0], chassis[1])) - 0.42)])
    eye[2] = max(eye[2], float(env.scenery.ground_z(eye[0], eye[1])) + 0.95)
    eye[2] = min(eye[2], float(env.scenery.canopy_z(eye[0], eye[1])) - 0.28)
    target = chassis + np.array([0.7, 2.2, 0.12])
    target[2] = float(np.clip(
        target[2], eye[2] - 0.08,
        float(env.scenery.canopy_z(target[0], target[1])) - 0.25))
    return mjv_from_eye_target(eye, target)


class OrchardBasketShowEnv(BasketKiwiEnv):
    """Same harvest task as ``BasketKiwiEnv``, with recording-only orchard dressing."""

    def __init__(self, relic, scenery: OrchardScenery, **kwargs):
        self.scenery = scenery
        super().__init__(relic, **kwargs)

    def _build(self):
        import treesim.assisted_kiwi_env as assisted
        original = assisted.build_scene

        def wrapped(relic, physics_hz):
            model, constants, positions = original(relic, physics_hz)
            dressed = dress_model(model, self.scenery)
            return dressed, constants, self.scenery.harvest_positions.copy()

        assisted.build_scene = wrapped
        try:
            super()._build()
        finally:
            assisted.build_scene = original
        self.model.hfield_data[:] = self.scenery.hfield_normalized.ravel()
        self.fruit_positions = self.scenery.harvest_positions.copy()
        self.ground_geoms = set(np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD))
        if not self.ground_geoms:
            raise RuntimeError('Orchard heightfield was not attached')
        try:
            self.ee_cam = self.model.camera('ee_cam').id
            self.basket_cam = self.model.camera('basket_cam').id
        except KeyError as error:
            raise RuntimeError('Wrist RGB or basket storage camera is missing from the dressed model') from error
        if not np.allclose(self.model.cam_fovy[self.ee_cam], RGB_FOVY_DEG):
            raise RuntimeError('ee_cam fovy must match the Boston Dynamics gripper specification')

    def reset(self, *, seed=None, options=None):
        real = mujoco.mj_forward

        def lift(model, data):
            if getattr(self, 'base_q', None) is None:
                return real(model, data)
            q = data.qpos
            index = self.base_q
            if abs(float(q[index + 2]) - .6) < 1e-9:
                q[index + 2] = .6 + self.scenery.ground_z(float(q[index]), float(q[index + 1]))
            return real(model, data)

        mujoco.mj_forward = lift
        try:
            return super().reset(seed=seed, options=options)
        finally:
            mujoco.mj_forward = real

    def render(self):
        if self.render_mode != 'rgb_array' or self.model is None:
            raise RuntimeError('Reset an rgb_array environment first')
        width, height = VIDEO_SIZE
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width,
                                            max_geom=max(50000, int(self.model.ngeom) + 2048))
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
        lookat, distance, azimuth, elevation = landscape_camera(self)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        scene = self.renderer.scene
        for i, body in enumerate(self.fruit_bodies):
            if not self.data.eq_active[self.stems[i]]:
                continue
            stem = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(stem, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                               np.eye(3).ravel(), np.array([.3, .4, .12, 1.], np.float32))
            hang = self.fruit_positions[i] + np.array([0., 0., STEM_LENGTH + 0.04])
            mujoco.mjv_connector(stem, mujoco.mjtGeom.mjGEOM_CAPSULE, .003,
                                self.data.xpos[body] + [0., 0., .04], hang)
            scene.ngeom += 1
        return self.renderer.render().copy()


def _scene_option():
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    return option


@contextmanager
def _close_near_plane(model, near_m=0.005):
    """Orchard extent is tens of metres; the default near plane clips the basket."""
    original = float(model.vis.map.znear)
    model.vis.map.znear = float(near_m) / max(float(model.stat.extent), 1e-6)
    try:
        yield
    finally:
        model.vis.map.znear = original


def _inset_renderer(env, width, height):
    key = (int(width), int(height))
    current = getattr(env, '_inset_renderer', None)
    if current is None or getattr(env, '_inset_size', None) != key:
        env._inset_renderer = mujoco.Renderer(
            env.model, height=key[1], width=key[0],
            max_geom=max(20000, int(env.model.ngeom) + 2048))
        env._inset_size = key
        env._inset_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        env._inset_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        env._inset_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    return env._inset_renderer


def basket_camera(env):
    """Look down into the open rear basket from above the rim."""
    rotation = env.data.xmat[env.chassis].reshape(3, 3)
    eye = env.data.xpos[env.chassis] + rotation @ (CENTER + [0.06, 0.12, SIZE[2] + 0.38])
    target = env.data.xpos[env.chassis] + rotation @ (CENTER + [0.0, 0.0, 0.06])
    lookat, distance, azimuth, elevation = mjv_from_eye_target(eye, target)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _paste_inset(image, draw, font, rgb, origin, title):
    from PIL import Image
    inset = Image.fromarray(rgb).resize((origin[2], origin[3]))
    x, y, w, h = origin
    image.paste(inset, (x, y))
    draw.rectangle((x, y, x + w, y + 22), fill=(18, 24, 32))
    draw.text((x + 8, y + 2), title, fill='white', font=font)


def frame(env, label, basket_inset=True, gripper_inset=True):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(env.render())
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 18 if image.height <= 720 else 22)
    draw.rectangle((0, 0, image.width, 84), fill=(18, 24, 32))
    draw.text((18, 8), BANNER, fill='white', font=font)
    info = env._info()
    draw.text((18, 32), f"{label} | t={info['elapsed_s']:.1f}s | target {info['target_index']} | {info['phase']}",
              fill='white', font=font)
    draw.text((18, 56), f"deposited {info['deposited_count']}/{env.task.picks} | {info['outcome']} | NOT learned deposit",
              fill='white', font=font)
    iw, ih = (320, 180) if image.height <= 720 else (400, 225)
    inset = _inset_renderer(env, iw, ih)
    with _close_near_plane(env.model):
        if basket_inset:
            inset.update_scene(env.data, camera='basket_cam', scene_option=_scene_option())
            basket = inset.render().copy()
            _paste_inset(image, draw, font, basket,
                         (image.width - iw - 16, image.height - ih - 16, iw, ih),
                         'Storage: rear basket')
        if gripper_inset:
            inset.update_scene(env.data, camera='ee_cam', scene_option=_scene_option())
            grip = inset.render().copy()
            _paste_inset(image, draw, font, grip, (16, image.height - ih - 16, iw, ih),
                         'Gripper wrist RGB')
    return image


def encode_triple(path: Path) -> None:
    faster = path.with_name(path.stem + '-3x.mp4')
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(path),
                           '-filter:v', 'setpts=PTS/3', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', str(faster)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11, help='Harvest episode seed; default matches six-1000hz')
    parser.add_argument('--canopy-seed', type=int, default=42)
    parser.add_argument('--picks', type=int, default=6)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=int, default=10, choices=(1, 2, 5, 10),
                        help='Recorded video fps; control stays at 10 Hz')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--basket-inset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gripper-inset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--episode-seconds', type=float, default=120.)
    parser.add_argument('--extra-fruit', type=int, default=800)
    parser.add_argument('--pergola-rows', type=int, default=9)
    parser.add_argument('--pergola-columns', type=int, default=7)
    parser.add_argument('--pergola-spacing', type=float, default=5.0)
    parser.add_argument('--slope-deg', type=float, default=3.5)
    parser.add_argument('--slope-azimuth-deg', type=float, default=38.0)
    parser.add_argument('--noise-m', type=float, default=0.04)
    parser.add_argument('--rut-depth-m', type=float, default=0.05)
    parser.add_argument('--landform-m', type=float, default=0.55,
                        help='Rolling landform amplitude [m]; milder than the static hillside preview so Spot can walk')
    parser.add_argument('--landform-wavelength-m', type=float, default=18.0)
    parser.add_argument('--canopy-spacing', type=float, default=0.15)
    parser.add_argument('--leaf-draw', choices=('mesh', 'texture'), default='mesh',
                        help='mesh = cordate blade geoms; texture = MuJoCo material albedo on canopy tiles')
    parser.add_argument('--preview-frames', type=int, default=0,
                        help='If >0, record only this many 10 Hz frames and do not require success')
    parser.add_argument('--bench-renders', type=int, default=0,
                        help='If >0, time this many landscape renders after reset and exit')
    args = parser.parse_args()
    global VIDEO_SIZE
    if args.width < 320 or args.height < 180:
        raise ValueError('video size must be at least 320x180')
    VIDEO_SIZE = (int(args.width), int(args.height))
    task = BasketTask(picks=args.picks, stage=args.stage, physics_hz=args.physics_hz,
                      time_limit_s=args.episode_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    scenery = build_scenery(
        seed=args.canopy_seed, extra_fruit=args.extra_fruit, rows=args.pergola_rows,
        columns=args.pergola_columns, spacing=args.pergola_spacing, slope_deg=args.slope_deg,
        slope_azimuth_deg=args.slope_azimuth_deg, noise_m=args.noise_m, rut_depth_m=args.rut_depth_m,
        landform_m=args.landform_m, landform_wavelength_m=args.landform_wavelength_m,
        canopy_spacing=args.canopy_spacing, asset_dir=args.output / 'assets',
        leaf_draw=args.leaf_draw)
    print(json.dumps(scenery.metrics, default=str), flush=True)
    source = Path(__file__).resolve().parents[1]
    names = ('scripts/record_basket_orchard.py', 'scripts/check_basket_kiwi.py',
             'treesim/basket_kiwi_env.py', 'treesim/assisted_kiwi_env.py', 'treesim/spot.py',
             'treesim/basket.py', 'treesim/foliage.py', 'treesim/pergola.py', 'treesim/orchard_terrain.py')
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    gait = args.relic / 'source/relic/relic/assets/spot/pretrained/policy.onnx'
    gait_hash = hashlib.sha256(gait.read_bytes()).hexdigest()
    env = OrchardBasketShowEnv(
        args.relic, scenery, task=task,
        render_mode='rgb_array' if args.video or args.bench_renders else None)
    encoder, events = None, []
    try:
        _, info = env.reset(seed=args.seed)
        if args.bench_renders:
            import time
            env.render()
            times = []
            for _ in range(int(args.bench_renders)):
                t0 = time.perf_counter()
                env.render()
                times.append(time.perf_counter() - t0)
            report = dict(
                ms_per_frame=1e3 * float(np.mean(times)),
                ms_per_frame_min=1e3 * float(np.min(times)),
                ms_per_frame_max=1e3 * float(np.max(times)),
                ngeom=int(env.model.ngeom),
                nmesh=int(env.model.nmesh),
                leaves=int(scenery.metrics['leaves']),
                extra_fruit=int(scenery.metrics['extra_fruit']),
                width=VIDEO_SIZE[0], height=VIDEO_SIZE[1],
            )
            print(json.dumps(report), flush=True)
            (args.output / 'render-bench.json').write_text(json.dumps(report, indent=2) + '\n')
            return
        controller = ScriptedBasketController()
        record_every = max(1, CONTROL_HZ // int(args.fps))
        if args.video:
            frame(env, 'initial state', basket_inset=args.basket_inset,
                  gripper_inset=args.gripper_inset).save(args.output / 'start.png')
            width, height = VIDEO_SIZE
            encoder = subprocess.Popen(
                ['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-movflags', '+faststart',
                 str(args.output / 'sequence.mp4')], stdin=subprocess.PIPE)
        limit = args.preview_frames if args.preview_frames else int(task.time_limit_s * CONTROL_HZ)
        for step in range(limit):
            action = controller.predict(env)
            _, reward, term, trunc, info = env.step(action)
            events.extend(info['events'])
            if step % 50 == 0 or info['events'] or term or trunc:
                print(json.dumps({k: v for k, v in info.items() if k != 'reward_terms'}), flush=True)
            if encoder and step % record_every == 0:
                encoder.stdin.write(np.asarray(frame(env, f'seed {args.seed}',
                                                     basket_inset=args.basket_inset,
                                                     gripper_inset=args.gripper_inset)).tobytes())
            if term or trunc:
                break
        if args.video:
            frame(env, 'final state', basket_inset=args.basket_inset,
                  gripper_inset=args.gripper_inset).save(args.output / 'final.png')
        frozen = hashlib.sha256(gait.read_bytes()).hexdigest() == gait_hash
        report = dict(scope=SCOPE, controller='scripted, not learned',
                      dressing='beauty-frames hillside orchard; extra visual fruit; uneven heightfield',
                      task=asdict(task), seed=args.seed, canopy_seed=args.canopy_seed,
                      scenery=scenery.metrics, events=events, source_sha256=hashes,
                      frozen_gait_sha256=gait_hash, frozen_gait_unchanged=frozen, final=info)
        np.savez(args.output / 'final-state.npz', qpos=env.data.qpos, qvel=env.data.qvel,
                 eq_active=env.data.eq_active, target_index=env.target_index, deposited=env.deposited)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')
        if not frozen:
            raise RuntimeError('Frozen RELIC weights changed')
        if not args.preview_frames and not info['success']:
            raise RuntimeError(f"Scripted basket sequence failed: {info['outcome']}")
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Video encoding failed')
        env.close()
    if args.video:
        encode_triple(args.output / 'sequence.mp4')


if __name__ == '__main__':
    main()
