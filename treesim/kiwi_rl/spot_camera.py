"""Nominal Spot wrist camera frame from the pinned RELIC URDF."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np


URDF_COMMIT = '27f8033c5064d32f049a17accb71cd1091422878'
URDF_SOURCE = 'source/relic/relic/assets/spot/spot_with_arm.urdf'
SOURCE_URL = 'https://dev.bostondynamics.com/docs/concepts/arm/arm_specification.html'
POSITION_M = np.array([.13806, .0202, .02452], dtype=float)
OPTICAL_RPY_RAD = np.array([-1.41372, 0., -1.5708], dtype=float)
HORIZONTAL_FOV_DEG = 60.2
VERTICAL_FOV_DEG = 46.4
RESOLUTION_PX = (640, 480)


def _urdf_rpy_matrix(rpy):
    """URDF fixed-axis XYZ RPY matrix, written without a SciPy dependency."""
    roll, pitch, yaw = map(float, rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1., 0., 0.), (0., cr, -sr), (0., sr, cr)))
    ry = np.array(((cp, 0., sp), (0., 1., 0.), (-sp, 0., cp)))
    rz = np.array(((cy, -sy, 0.), (sy, cy, 0.), (0., 0., 1.)))
    return rz @ ry @ rx


def install_spot_gripper_camera(root: ET.Element, robot: dict) -> list[dict]:
    """Install the nominal published Spot hand color camera into an MJCF root.

    The URDF optical convention is +Z forward and +Y down. MuJoCo cameras use
    -Z forward and +Y up, so the returned quaternion represents
    ``R_urdf @ diag(1, -1, -1)``.
    """
    import mujoco

    if not isinstance(root, ET.Element):
        raise TypeError('root must be an xml.etree.ElementTree.Element')
    wrist = robot.get('wrist') if isinstance(robot, dict) else None
    if not wrist:
        raise ValueError("robot must provide a nonempty 'wrist' body name")
    parent = next((body for body in root.iter('body') if body.get('name') == wrist), None)
    if parent is None:
        raise ValueError(f'Missing Spot wrist body {wrist!r}')

    parents = {child: owner for owner in root.iter() for child in owner}
    for camera in list(root.iter('camera')):
        if camera.get('name') in ('body_camera', 'hand_camera'):
            parents[camera].remove(camera)

    # Flip the two optical axes while preserving the right-handed frame.
    rotation = _urdf_rpy_matrix(OPTICAL_RPY_RAD) @ np.diag([1., -1., -1.])
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
    hfov, vfov = map(math.radians, (HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG))
    focal = (1. / (2. * math.tan(hfov / 2.)), 1. / (2. * math.tan(vfov / 2.)))
    ET.SubElement(parent, 'camera', name='hand_camera',
                  pos=' '.join(format(float(v), '.17g') for v in POSITION_M),
                  quat=' '.join(format(float(v), '.17g') for v in quaternion),
                  sensorsize='1 1',
                  focal=' '.join(format(float(v), '.17g') for v in focal),
                  principal='0 0', resolution=f'{RESOLUTION_PX[0]} {RESOLUTION_PX[1]}')
    return [dict(name='hand_camera', body=wrist, position_m=POSITION_M.tolist(),
        optical_rpy_rad=OPTICAL_RPY_RAD.tolist(), mujoco_quaternion_wxyz=quaternion.tolist(),
        horizontal_fov_deg=HORIZONTAL_FOV_DEG, vertical_fov_deg=VERTICAL_FOV_DEG,
        resolution_px=list(RESOLUTION_PX), source_urdf=URDF_SOURCE,
        source_commit=URDF_COMMIT, source_url=SOURCE_URL,
        calibration='nominal model and published FOV; not per-unit calibration',
        render_approximation='sensor render omits sealed wrist visual housing; collisions unchanged',
        depth='ideal registered geometric depth, not calibrated ToF')]

