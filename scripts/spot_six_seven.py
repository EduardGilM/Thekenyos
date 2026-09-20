#!/usr/bin/env python
"""Put Spot on its hind legs and run a scripted 6-7.

This writes RELIC joint targets kinematically, plants the hind feet on the
floor after each ``mj_forward``, and records an MP4. It is not Newton gait,
not contact-balanced bipedalism, not a weld-as-grasp, and not harvest.

    MUJOCO_GL=egl python scripts/spot_six_seven.py --relic ../relic \
        --video output/spot-six-seven.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treesim.gl_backend import bind_mujoco_gl
from treesim.six_seven import (
    ARM, LEGS, duration_s, pose_at, rpy_to_wxyz, stand_joints,
)

# 5x7 caps so SIX/SEVEN is visible without a font dependency.
_FONT = {
    'S': (0b01110, 0b10000, 0b01100, 0b00010, 0b11100, 0b00000, 0b00000),
    'I': (0b01110, 0b00100, 0b00100, 0b00100, 0b01110, 0b00000, 0b00000),
    'X': (0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000, 0b00000),
    'E': (0b11110, 0b10000, 0b11100, 0b10000, 0b11110, 0b00000, 0b00000),
    'V': (0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00000, 0b00000),
    'N': (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b00000, 0b00000),
}

_YELLOW = (0.93, 0.76, 0.12, 1.0)
_BLACK = (0.13, 0.13, 0.14, 1.0)


def _stamp_caption(pixels: np.ndarray, text: str, colour) -> None:
    scale = 14
    x0, y0 = 40, 16
    shadow = (0, 0, 0)
    for i, ch in enumerate(text):
        glyph = _FONT[ch]
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    for dx, dy, rgb in ((2, 2, shadow), (0, 0, colour)):
                        ys = slice(y0 + row * scale + dy, y0 + (row + 1) * scale + dy)
                        xs = slice(
                            x0 + (i * 6 + col) * scale + dx,
                            x0 + (i * 6 + col + 1) * scale + dx,
                        )
                        pixels[ys, xs] = rgb


def _visual_rgba(filename: str) -> tuple[float, float, float, float]:
    path = filename.replace('\\', '/').lower()
    if '/arm/' in path or '/gripper/' in path or 'hip' in path:
        return _BLACK
    return _YELLOW


def _opaque_obj_copy(src: Path, cache: Path) -> Path:
    """Drop mtllib/usemtl so a missing wrap PNG cannot punch holes in the mesh."""
    rel = Path(*src.parts[-3:]) if len(src.parts) >= 3 else Path(src.name)
    dst = cache / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding='utf-8', errors='ignore')
    kept = [line for line in text.splitlines(True)
            if not line.startswith(('mtllib', 'usemtl'))]
    dst.write_text(''.join(kept), encoding='utf-8')
    return dst


def _add_solid_core(body, kind: str, size, pos, rgba) -> None:
    """Opaque filler so CAD vents and backfaces cannot show the stage through Spot."""
    import mujoco
    geom = body.add_geom()
    geom.name = f'core_{body.name}_{kind}'
    geom.type = (mujoco.mjtGeom.mjGEOM_BOX if kind == 'box'
                 else mujoco.mjtGeom.mjGEOM_CAPSULE)
    geom.size[:len(size)] = size
    geom.pos[:] = pos
    geom.group = 2
    geom.contype = 0
    geom.conaffinity = 0
    geom.rgba[:] = rgba


def _attach_urdf_visuals(spec, urdf: Path) -> int:
    """Keep collision geoms off-screen and attach the URDF visual meshes.

    ``MjSpec.from_file`` compiles RELIC's URDF with ``discardvisual=true``,
    which drops the yellow body/leg meshes and the arm links whose collision
    blocks are commented out. Those visuals are not a harvest or contact check.
    """
    import mujoco
    spec.compiler.discardvisual = False
    spec.compiler.fusestatic = False
    for geom in spec.geoms:
        geom.group = 3
    cache = Path('/tmp/spot-six-seven-meshes')
    cache.mkdir(parents=True, exist_ok=True)
    attached = 0
    for link in ET.parse(urdf).getroot().findall('link'):
        name = link.get('name')
        if not name:
            continue
        try:
            body = spec.body(name)
        except Exception:
            continue
        for visual in link.findall('visual'):
            mesh_el = visual.find('geometry/mesh')
            if mesh_el is None or not mesh_el.get('filename'):
                continue
            mesh_path = (urdf.parent / mesh_el.get('filename')).resolve()
            if not mesh_path.is_file():
                continue
            mesh_name = f'vis_{name}_{attached}'
            mesh = spec.add_mesh()
            mesh.name = mesh_name
            mesh.file = str(_opaque_obj_copy(mesh_path, cache))
            geom = body.add_geom()
            geom.name = mesh_name
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name
            geom.group = 2
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba[:] = _visual_rgba(str(mesh_path))
            origin = visual.find('origin')
            if origin is not None:
                xyz = [float(v) for v in (origin.get('xyz') or '0 0 0').split()]
                rpy = [float(v) for v in (origin.get('rpy') or '0 0 0').split()]
                geom.pos[:] = xyz
                geom.quat[:] = rpy_to_wxyz(*rpy)
            attached += 1
    yellow = _YELLOW
    black = _BLACK
    _add_solid_core(spec.body('body'), 'box', (0.22, 0.08, 0.055), (0.0, 0.0, -0.01), yellow)
    for leg in ('fl', 'fr', 'hl', 'hr'):
        _add_solid_core(spec.body(f'{leg}_uleg'), 'capsule', (0.032, 0.11), (0.0, 0.0, -0.16), yellow)
        _add_solid_core(spec.body(f'{leg}_lleg'), 'capsule', (0.022, 0.12), (0.0, 0.0, -0.16), yellow)
        _add_solid_core(spec.body(f'{leg}_hip'), 'box', (0.025, 0.025, 0.025), (0.0, 0.0, 0.0), black)
    return attached


def build_model(urdf: Path):
    """Floating-base Spot plus a floor. Gravity stays off: this is a gag, not balance."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.option.gravity[:] = 0.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.global_.fovy = 48.0
    spec.visual.headlight.ambient[:] = (0.32, 0.32, 0.30)
    spec.visual.headlight.diffuse[:] = (0.55, 0.55, 0.50)
    visuals = _attach_urdf_visuals(spec, urdf)
    if visuals < 16:
        raise RuntimeError(f'expected Spot visual meshes, attached {visuals}')
    free = spec.body('body').add_freejoint()
    free.name = 'root'
    floor = spec.worldbody.add_geom()
    floor.name = 'floor'
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[0], floor.size[1], floor.size[2] = 4.0, 4.0, 0.05
    floor.rgba[:] = (0.22, 0.24, 0.20, 1.0)
    stage = spec.worldbody.add_geom()
    stage.name = 'stage'
    stage.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    stage.pos[:] = (0.0, 0.0, 0.008)
    stage.size[0], stage.size[1] = 1.15, 0.008
    stage.rgba[:] = (0.10, 0.10, 0.10, 1.0)
    stage.contype = 0
    stage.conaffinity = 0
    light = spec.worldbody.add_light()
    light.name = 'key'
    light.pos[:] = (1.6, -2.4, 3.6)
    light.dir[:] = (-0.25, 0.45, -1.0)
    light.diffuse[:] = (0.95, 0.88, 0.72)
    fill = spec.worldbody.add_light()
    fill.name = 'fill'
    fill.pos[:] = (-1.8, 2.0, 2.8)
    fill.dir[:] = (0.35, -0.25, -1.0)
    fill.diffuse[:] = (0.35, 0.40, 0.48)
    return spec.compile()


def _qpos_addr(model, name: str) -> int:
    return int(model.jnt_qposadr[model.joint(name).id])


def _foot_geom_ids(model, names: tuple[str, ...]) -> list[int]:
    ids = []
    for name in names:
        try:
            bid = int(model.body(name).id)
        except Exception:
            continue
        for i in range(model.ngeom):
            if int(model.geom_bodyid[i]) == bid:
                ids.append(i)
    return ids


def _configure_renderer(renderer) -> None:
    """Draw both triangle sides so CAD shells do not look like missing panels."""
    import mujoco
    renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_CULL_FACE)] = 0


def _hind_mid_xy(model, data) -> np.ndarray | None:
    hind = _foot_geom_ids(model, ('hl_foot', 'hr_foot'))
    if not hind:
        return None
    return np.mean(np.stack([data.geom_xpos[i, :2] for i in hind], axis=0), axis=0)


def _plant_feet(model, data, hind_only: bool, plant_xy: np.ndarray | None) -> np.ndarray | None:
    import mujoco
    hind = _foot_geom_ids(model, ('hl_foot', 'hr_foot'))
    front = _foot_geom_ids(model, ('fl_foot', 'fr_foot'))
    support = hind if hind_only else hind + front
    if not support:
        return plant_xy
    adr = int(model.jnt_qposadr[model.joint('root').id])
    zs = [float(data.geom_xpos[i, 2]) for i in support]
    data.qpos[adr + 2] -= min(zs) - 0.002
    mujoco.mj_forward(model, data)
    if plant_xy is None:
        plant_xy = _hind_mid_xy(model, data)
    if plant_xy is not None and hind_only:
        mid = _hind_mid_xy(model, data)
        if mid is not None:
            data.qpos[adr:adr + 2] += plant_xy - mid
            mujoco.mj_forward(model, data)
            zs = [float(data.geom_xpos[i, 2]) for i in hind]
            data.qpos[adr + 2] -= min(zs) - 0.002
            mujoco.mj_forward(model, data)
    return plant_xy


def apply_pose(model, data, pose: dict, home: dict[str, float], plant_xy=None):
    """Write the freejoint and RELIC joints, then drop support feet onto the floor."""
    import mujoco
    xyz = np.asarray(pose['xyz_m'], dtype=np.float64)
    quat = rpy_to_wxyz(*pose['rpy_rad'])
    adr = int(model.jnt_qposadr[model.joint('root').id])
    data.qpos[adr:adr + 3] = xyz
    data.qpos[adr + 3:adr + 7] = quat
    joints = pose['joints']
    for name in (*LEGS, *ARM):
        data.qpos[_qpos_addr(model, name)] = float(joints.get(name, home[name]))
    mujoco.mj_forward(model, data)
    return _plant_feet(model, data, hind_only=bool(pose['hind_support']), plant_xy=plant_xy)


def look_at(model, data, cam, *, distance=2.7) -> None:
    """Three-quarter view so the hind-leg columns and the waving hands both read."""
    body = np.array(data.xpos[model.body('body').id], dtype=np.float64)
    cam.lookat[:] = body
    cam.distance = distance
    cam.azimuth = 52.0
    cam.elevation = -8.0


def run(args) -> None:
    asset = args.relic.resolve() / 'source/relic/relic/assets/spot'
    urdf = asset / 'spot_with_arm.urdf'
    if not urdf.is_file():
        raise SystemExit(f'missing Spot URDF: {urdf}')
    constants = {}
    exec((asset / 'constants.py').read_text(encoding='utf-8'), constants)
    home = dict(constants['SPOT_DEFAULT_JOINT_POS'])
    bind_mujoco_gl(require_gpu=args.require_gpu)
    import mujoco
    model = build_model(urdf)
    data = mujoco.MjData(model)
    duration = duration_s()
    fps = int(args.fps)
    n_frames = int(round(duration * fps))
    renderer = None
    encoder = None
    captions = []
    pitches = []
    plant_xy = None
    nose_up = []
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        _configure_renderer(renderer)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        encoder = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', '1280x720', '-r', str(fps), '-i', '-',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             str(args.video)],
            stdin=subprocess.PIPE,
        )
    for frame in range(n_frames):
        t = frame / fps
        pose = pose_at(t, home)
        plant_xy = apply_pose(model, data, pose, home, plant_xy)
        captions.append(pose['caption'])
        pitches.append(float(pose['rpy_rad'][1]))
        # Body +X in world z: positive means the nose points up.
        body = model.body('body').id
        nose_up.append(float(data.xmat[body].reshape(3, 3)[2, 0]))
        if renderer is None:
            continue
        look_at(model, data, cam)
        renderer.update_scene(data, camera=cam)
        _configure_renderer(renderer)
        pixels = np.array(renderer.render(), copy=True, dtype=np.uint8)
        if pose['caption']:
            colour = (255, 220, 40) if pose['caption'] == 'SIX' else (80, 210, 255)
            _stamp_caption(pixels, pose['caption'], colour)
        encoder.stdin.write(np.ascontiguousarray(pixels))
    if encoder is not None:
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed')
    if renderer is not None:
        renderer.close()
    print({
        'frames': n_frames,
        'duration_s': duration,
        'min_pitch_rad': float(min(pitches)),
        'max_pitch_rad': float(max(pitches)),
        'min_forward_z': float(min(nose_up)),
        'max_forward_z': float(max(nose_up)),
        'six_frames': int(sum(c == 'SIX' for c in captions)),
        'seven_frames': int(sum(c == 'SEVEN' for c in captions)),
        'video': str(args.video) if args.video else None,
        'scope': 'Kinematic RELIC Spot gag; not harvest, gait, or a learned policy',
        'stand_joints': {k: stand_joints(home)[k] for k in ('hl_hy', 'fl_hy')},
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic', required=True, type=Path)
    p.add_argument('--video', type=Path)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--require-gpu', action='store_true')
    args = p.parse_args()
    if args.fps < 10 or args.fps > 60:
        p.error('--fps must be in [10, 60]')
    run(args)


if __name__ == '__main__':
    main()
