"""Nominal Spot wrist camera frame from the pinned RELIC URDF."""

from __future__ import annotations

import math
from pathlib import Path
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
APERTURE_RADIUS_M = .008
APERTURE_DEPTH_M = (-.010, .020)


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


def _quat_matrix(quat):
    """Return a MuJoCo wxyz quaternion as a 3x3 rotation matrix."""
    w, x, y, z = map(float, quat)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                      2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                      2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w),
                      1 - 2 * (x * x + y * y))))


def _mesh_data(mesh, base_dir=Path('.')):
    """Read a MuJoCo inline or OBJ mesh without adding an asset dependency."""
    if mesh.get('vertex') is not None:
        vertices = np.fromstring(mesh.get('vertex'), sep=' ', dtype=float).reshape(-1, 3)
        faces = np.fromstring(mesh.get('face', ''), sep=' ', dtype=int).reshape(-1, 3)
        return vertices, faces
    filename = mesh.get('file')
    if not filename:
        return None, None
    path = Path(filename)
    if not path.is_absolute():
        path = base_dir / path
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in ('v', 'f'):
            continue
        if fields[0] == 'v':
            vertices.append(tuple(map(float, fields[1:4])))
        else:
            indices = [int(field.split('/')[0]) for field in fields[1:]]
            indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
            for index in range(1, len(indices) - 1):
                faces.append((indices[0], indices[index], indices[index + 1]))
    if not vertices or not faces:
        raise ValueError(f'Mesh {path} has no vertices or triangles')
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _open_aperture(vertices, faces, center, geom_pos, geom_quat, mesh_scale):
    """Subtract a local 16-sided lens aperture; split boundary triangles."""
    scale = np.asarray(mesh_scale, dtype=float)
    geom_rotation = _quat_matrix(geom_quat)
    optical = _urdf_rpy_matrix(OPTICAL_RPY_RAD)
    body = (vertices * scale) @ geom_rotation.T + np.asarray(geom_pos)
    local = (body - center) @ optical
    angles = np.arange(16) * (2 * np.pi / 16)
    # Inscribed polygon stays inside the stated radius.
    planes = [(np.array([np.cos(a), np.sin(a), 0.]), APERTURE_RADIUS_M * np.cos(np.pi / 16)) for a in angles]
    planes += [(np.array([0., 0., 1.]), APERTURE_DEPTH_M[1]),
               (np.array([0., 0., -1.]), -APERTURE_DEPTH_M[0])]
    out_vertices, out_faces = [], []

    def split(poly, normal, bound, inside):
        result = []
        for i, start in enumerate(poly):
            end = poly[(i + 1) % len(poly)]
            ds, de = start @ normal - bound, end @ normal - bound
            keep_s, keep_e = (ds <= 0, de <= 0) if inside else (ds >= 0, de >= 0)
            if keep_s:
                result.append(start)
            if keep_s != keep_e:
                result.append(start + (end - start) * (ds / (ds - de)))
        return result

    def emit(poly):
        if len(poly) < 3:
            return
        offset = len(out_vertices)
        out_vertices.extend(poly)
        for i in range(1, len(poly) - 1):
            if np.linalg.norm(np.cross(poly[i] - poly[0], poly[i + 1] - poly[0])) > 1e-14:
                out_faces.append((offset, offset + i, offset + i + 1))

    for face in faces:
        poly = list(local[face])
        if any(np.all(np.asarray(poly) @ n > bound) for n, bound in planes):
            emit(poly)
            continue
        for normal, bound in planes:
            emit(split(poly, normal, bound, False))
            poly = split(poly, normal, bound, True)
            if len(poly) < 3:
                break
        # The remaining polygon lies inside the aperture and is omitted.
    local_out = np.asarray(out_vertices).reshape(-1, 3)
    mesh_out = ((local_out @ optical.T + center - geom_pos) @ geom_rotation) / scale
    return mesh_out, np.asarray(out_faces, dtype=int).reshape(-1, 3), scale


def _install_visual_aperture(root, parent):
    """Replace only the Spot wrist visual mesh with a mesh containing an opening."""
    asset = root.find('asset')
    if asset is None:
        return False
    for geom in parent.findall('geom'):
        if geom.get('type', 'mesh' if geom.get('mesh') else 'plane') != 'mesh' or geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
            continue
        mesh = next((node for node in asset.findall('mesh') if node.get('name') == geom.get('mesh')), None)
        if mesh is None or 'arm_link_wr1' not in mesh.get('name', ''):
            continue
        if mesh.get('name', '').endswith('_aperture'):
            return True
        vertices, faces = _mesh_data(mesh)
        if vertices is None:
            continue
        geom_pos = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
        geom_quat = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
        mesh_scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        aperture_vertices, aperture_faces, scale = _open_aperture(
            vertices, faces, POSITION_M, geom_pos, geom_quat, mesh_scale)
        if len(aperture_faces) == len(faces):
            continue
        name = f'{mesh.get("name")}_aperture'
        ET.SubElement(asset, 'mesh', name=name,
                      vertex=' '.join(format(float(value), '.17g') for value in (aperture_vertices * scale).ravel()),
                      face=' '.join(str(int(value)) for value in aperture_faces.ravel()))
        geom.set('mesh', name)
        return True
    return False


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

    aperture = _install_visual_aperture(root, parent)

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
        render_approximation=('nominal 8 mm radius local aperture in the wrist visual mesh '
                              '(depth -10 to +20 mm along optical +Z); '
                              'collisions unchanged' if aperture else
                              'wrist visual mesh unavailable; collisions unchanged'),
        depth='ideal registered geometric depth, not calibrated ToF')]
