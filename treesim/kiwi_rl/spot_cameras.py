"""Nominal Spot gripper cameras from the pinned RELIC URDF.

The Boston Dynamics sensor frame has +Z along the optical axis, +X along
image columns and +Y down the image rows. MuJoCo cameras look along -Z with
+Y up, so the optical axes are converted before export. These frames are the
factory-nominal comments in ``spot_with_arm.urdf``, not per-unit calibration.

Body fisheye extrinsics are not in that URDF. Do not invent a mast or other
off-robot cameras to improve a view.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

# Published maximum FOV, horizontal × vertical.
# https://dev.bostondynamics.com/docs/concepts/arm/arm_specification.html
COLOR_FOV_DEG = (60.2, 46.4)
DEPTH_FOV_DEG = (55.9, 44.0)

GRIPPER_FRAMES = ('hand_color_sensor', 'hand_depth_sensor')
# Copied from the pinned RELIC URDF comments. load_relic_gripper_cameras
# rejects a checkout whose frames differ.
NOMINAL_OPTICAL_FRAMES = {
    'hand_color_sensor': dict(urdf_link='arm_link_wr1', rpy=(-1.41372, 0.0, -1.5708),
                              xyz=(0.13806, 0.0202, 0.02452)),
    'hand_depth_sensor': dict(urdf_link='arm_link_wr1', rpy=(0.0, 1.41372, 0.0),
                              xyz=(0.13495, 0.0, 0.00799)),
}
# 180 deg about X: BD +Z look / +Y down → MuJoCo -Z look / +Y up.
_OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])
_FRAME = re.compile(
    r'<frame\s+link="(?P<link>[^"]+)"\s+name="(?P<name>[^"]+)"'
    r'\s+rpy="(?P<rpy>[^"]+)"\s+xyz="(?P<xyz>[^"]+)"\s*/>'
)


def _floats(text: str) -> tuple[float, float, float]:
    values = tuple(float(part) for part in text.split())
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f'Expected three finite numbers, got {text!r}')
    return values  # type: ignore[return-value]


def rpy_matrix(rpy) -> np.ndarray:
    """URDF RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(x) for x in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_wxyz(rotation: np.ndarray) -> list[float]:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Expected a finite 3x3 rotation')
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        quat = [0.25 / s,
                (rotation[2, 1] - rotation[1, 2]) * s,
                (rotation[0, 2] - rotation[2, 0]) * s,
                (rotation[1, 0] - rotation[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k]))
        quat = [0.0, 0.0, 0.0, 0.0]
        quat[0] = (rotation[k, j] - rotation[j, k]) / s
        quat[i + 1] = 0.25 * s
        quat[j + 1] = (rotation[j, i] + rotation[i, j]) / s
        quat[k + 1] = (rotation[k, i] + rotation[i, k]) / s
    quat = np.asarray(quat, dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat = -quat
    return [float(x) for x in quat]


def optical_to_mujoco(rpy, xyz) -> dict[str, Any]:
    """Convert a BD optical frame in a parent body into a MuJoCo camera pose."""
    optical = rpy_matrix(rpy)
    mujoco = optical @ _OPTICAL_TO_MUJOCO
    if abs(float(np.linalg.det(mujoco)) - 1.0) > 1e-6:
        raise RuntimeError('Camera rotation is not a proper rotation')
    return dict(position_m=[float(x) for x in xyz],
                quaternion_wxyz=matrix_to_wxyz(mujoco),
                rotation=mujoco, optical_rotation=optical)


def parse_urdf_frames(urdf_text: str) -> dict[str, dict[str, Any]]:
    """Read ``<frame>`` elements, including those inside XML comments."""
    frames = {}
    for match in _FRAME.finditer(urdf_text):
        name = match.group('name')
        frames[name] = dict(name=name, urdf_link=match.group('link'),
                            rpy=_floats(match.group('rpy')), xyz=_floats(match.group('xyz')))
    return frames


def _fov(name: str) -> tuple[tuple[float, float], float]:
    if name == 'hand_color_sensor':
        return COLOR_FOV_DEG, COLOR_FOV_DEG[1]
    if name == 'hand_depth_sensor':
        return DEPTH_FOV_DEG, DEPTH_FOV_DEG[1]
    raise ValueError(f'No published FOV for {name!r}')


def mujoco_gripper_cameras() -> list[dict[str, Any]]:
    """MuJoCo specs for the nominal gripper RGB and ToF sensors."""
    cameras = []
    for name in GRIPPER_FRAMES:
        frame = NOMINAL_OPTICAL_FRAMES[name]
        pose = optical_to_mujoco(frame['rpy'], frame['xyz'])
        fov_deg, fovy = _fov(name)
        cameras.append(dict(
            name=name, urdf_link=frame['urdf_link'], urdf_name=name,
            position_m=pose['position_m'], quaternion_wxyz=pose['quaternion_wxyz'],
            fovy_degrees=float(fovy), fov_horizontal_vertical_deg=list(fov_deg),
            parent_role='wrist',
            calibration=('RELIC spot_with_arm.urdf nominal gripper frame; '
                         'BD optical axes converted to MuJoCo; published max FOV; '
                         'not per-unit calibration'),
            source='relic-urdf-frame-comment',
        ))
    return cameras


def load_relic_gripper_cameras(relic: str | Path) -> list[dict[str, Any]]:
    """Return MuJoCo camera specs after checking the pinned RELIC URDF."""
    urdf = Path(relic).resolve() / 'source/relic/relic/assets/spot/spot_with_arm.urdf'
    if not urdf.is_file():
        raise FileNotFoundError(f'missing {urdf} (external RELIC checkout)')
    frames = parse_urdf_frames(urdf.read_text())
    missing = [name for name in GRIPPER_FRAMES if name not in frames]
    if missing:
        raise ValueError('Pinned RELIC URDF is missing gripper camera frames: ' + ', '.join(missing))
    for name, expected in NOMINAL_OPTICAL_FRAMES.items():
        frame = frames[name]
        if frame['urdf_link'] != expected['urdf_link']:
            raise ValueError(f'{name} parent link changed in RELIC URDF')
        if np.linalg.norm(np.subtract(frame['xyz'], expected['xyz'])) > 1e-8:
            raise ValueError(f'{name} origin changed in RELIC URDF')
        if np.linalg.norm(np.subtract(frame['rpy'], expected['rpy'])) > 1e-8:
            raise ValueError(f'{name} optical RPY changed in RELIC URDF')
    return mujoco_gripper_cameras()


def parent_body(robot: dict, urdf_link: str) -> str:
    if urdf_link == 'arm_link_wr1':
        return robot['wrist']
    raise ValueError(f'{urdf_link} is not a RELIC gripper camera parent')


def require_gripper_camera_names(names) -> tuple[str, ...]:
    """Reject invented mounts such as the 0.55 m body mast."""
    names = tuple(names)
    extra = [name for name in names if name not in GRIPPER_FRAMES]
    missing = [name for name in GRIPPER_FRAMES if name not in names]
    if extra or missing:
        raise RuntimeError(
            'Scene cameras must be the nominal RELIC gripper frames '
            f'{GRIPPER_FRAMES}; missing={missing} extra={extra}')
    return names


def require_mujoco_gripper_cameras(model, robot=None) -> tuple[str, ...]:
    """Check exported MuJoCo cameras against the pinned URDF frames and FOV."""
    names = require_gripper_camera_names(model.camera(i).name for i in range(model.ncam))
    for name, expected in NOMINAL_OPTICAL_FRAMES.items():
        camera = model.camera(name)
        pose = optical_to_mujoco(expected['rpy'], expected['xyz'])
        if np.linalg.norm(np.asarray(camera.pos, dtype=float) - expected['xyz']) > 1e-6:
            raise RuntimeError(f'{name} origin is not the pinned RELIC URDF frame')
        expected_quat = np.asarray(pose['quaternion_wxyz'], dtype=float)
        actual_quat = np.asarray(camera.quat, dtype=float)
        if min(np.linalg.norm(actual_quat - expected_quat),
               np.linalg.norm(actual_quat + expected_quat)) > 1e-5:
            raise RuntimeError(f'{name} optical axes are not the MuJoCo conversion of the URDF frame')
        _, fovy = _fov(name)
        if abs(float(camera.fovy) - fovy) > 1e-3:
            raise RuntimeError(f'{name} fovy is not the published maximum FOV')
        if robot is not None:
            parent = int(model.body(parent_body(robot, expected['urdf_link'])).id)
            if int(model.cam_bodyid[camera.id]) != parent:
                raise RuntimeError(f'{name} is not attached to the gripper wrist')
    return names


def attach_mujoco_cameras(bodies: dict, robot: dict, cameras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nest MuJoCo cameras under the matching robot bodies. No extra bodies."""
    attached = []
    for camera in cameras:
        if camera.get('name') not in GRIPPER_FRAMES or camera.get('urdf_link') != 'arm_link_wr1':
            raise ValueError(f'{camera.get("name")} is not a RELIC gripper camera on arm_link_wr1')
        body = parent_body(robot, camera['urdf_link'])
        if body not in bodies:
            raise ValueError(f'Camera parent body {body!r} is missing')
        offset = np.asarray(camera['position_m'], dtype=float)
        if float(np.linalg.norm(offset)) > 0.25 or abs(float(offset[2])) > 0.35:
            raise ValueError(f'{camera["name"]} offset {camera["position_m"]} is not a gripper mount')
        element = bodies[body]
        existing = [child for child in list(element) if child.tag == 'camera' and child.get('name') == camera['name']]
        for child in existing:
            element.remove(child)
        from xml.etree.ElementTree import SubElement
        SubElement(element, 'camera', name=camera['name'],
                   pos=' '.join(map(str, camera['position_m'])),
                   quat=' '.join(map(str, camera['quaternion_wxyz'])),
                   fovy=str(camera['fovy_degrees']))
        record = dict(camera, body=body)
        attached.append(record)
    names = [camera['name'] for camera in attached]
    require_gripper_camera_names(names)
    if any('mast' in (camera.get('calibration') or '').lower() for camera in attached):
        raise RuntimeError('Invented mast cameras are not allowed')
    return attached
