import xml.etree.ElementTree as ET

import numpy as np


def collision_vertices(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def normalize_collision_meshes(xml, tolerance_m=1e-6):
    import mujoco
    from scipy.spatial import cKDTree
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 1e-6:
        raise ValueError('Mesh normalization tolerance must be in (0, 1 micrometre]')
    root = ET.fromstring(xml)
    original = mujoco.MjModel.from_xml_string(xml)
    asset = root.find('asset')
    geoms = {g.get('name'): g for g in root.findall('.//geom') if g.get('name')}
    candidates = [g for g in range(original.ngeom)
                  if original.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and (original.geom_contype[g] or original.geom_conaffinity[g])]
    normalized = {}
    changed = []
    for geom in candidates:
        mesh = int(original.geom_dataid[geom])
        if np.linalg.norm(original.mesh_pos[mesh]) < 1e-8:
            continue
        name = original.geom(geom).name
        if name not in geoms:
            raise ValueError('Collision mesh normalization requires named geoms')
        if mesh not in normalized:
            mesh_name = f'kiwi_centered_mesh_{mesh}'
            if asset is None or any(m.get('name') == mesh_name for m in asset.findall('mesh')):
                raise ValueError('Missing assets or a conflicting normalized mesh name')
            va, vn = original.mesh_vertadr[mesh], original.mesh_vertnum[mesh]
            fa, fn = original.mesh_faceadr[mesh], original.mesh_facenum[mesh]
            ET.SubElement(asset, 'mesh', name=mesh_name,
                vertex=' '.join(format(float(x), '.17g') for x in original.mesh_vert[va:va + vn].ravel()),
                face=' '.join(str(int(x)) for x in original.mesh_face[fa:fa + fn].ravel()))
            normalized[mesh] = mesh_name
        element = geoms[name]
        for orientation in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
            element.attrib.pop(orientation, None)
        element.set('mesh', normalized[mesh])
        element.set('pos', ' '.join(format(float(x), '.17g') for x in original.geom_pos[geom]))
        element.set('quat', ' '.join(format(float(x), '.17g') for x in original.geom_quat[geom]))
        changed.append(name)
    result_xml = ET.tostring(root, encoding='unicode')
    result = mujoco.MjModel.from_xml_string(result_xml)
    if (original.nq, original.nv, original.nbody, original.ngeom) != (result.nq, result.nv, result.nbody, result.ngeom):
        raise RuntimeError('Mesh normalization changed model topology')
    for field in ('body_mass', 'body_inertia', 'body_ipos'):
        if not np.allclose(getattr(original, field), getattr(result, field), rtol=1e-5, atol=1e-8):
            raise RuntimeError(f'Mesh normalization changed {field}')
    for body in range(original.nbody):
        tensors = []
        for model in (original, result):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
            rotation = rotation.reshape(3, 3)
            tensors.append(rotation @ np.diag(model.body_inertia[body]) @ rotation.T)
        if not np.allclose(tensors[0], tensors[1], rtol=1e-5, atol=1e-8):
            raise RuntimeError('Mesh normalization changed the physical inertia tensor')
    before, after = mujoco.MjData(original), mujoco.MjData(result)
    mujoco.mj_forward(original, before)
    mujoco.mj_forward(result, after)
    maximum_error = 0.
    for geom in candidates:
        other = result.geom(original.geom(geom).name).id
        a, b = collision_vertices(original, before, geom), collision_vertices(result, after, other)
        maximum_error = max(maximum_error, float(cKDTree(a).query(b)[0].max()),
                            float(cKDTree(b).query(a)[0].max()))
        if maximum_error > tolerance_m:
            raise RuntimeError(f'Mesh normalization changed the collision surface by {maximum_error} m')
    return result_xml, dict(changed_geoms=changed, max_surface_error_m=maximum_error,
                            scope='Coordinate-frame normalization; collision surfaces and material unchanged')
