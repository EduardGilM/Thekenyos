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


def assemble_deformable_scene(directory, *, count=9, timestep_s=.00002, contact_time_s=.001,
                              integrator='Euler', solver='CG', iterations=1000, tolerance=1e-9,
                              stem_segments=4):
    import hashlib
    import json
    from pathlib import Path
    import mujoco
    from treesim.kiwi_material import STEM_LENGTH, XUXIANG
    from treesim.native_kiwi import flesh_mass, RADII_M, FLEX_RADIUS_M, FLESH_DENSITY_KG_M3
    from treesim.native_stem import add_stem
    directory = Path(directory)
    xml = (directory / 'base.xml').read_text()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'training-base-scene/v1' or hashlib.sha256(xml.encode()).hexdigest() != manifest.get('model_sha256'):
        raise ValueError('Invalid or corrupted base-scene artifact')
    if count not in (5, 7, 9, 11) or not np.isfinite([timestep_s, contact_time_s, tolerance]).all():
        raise ValueError('Invalid deformable discretization')
    if not isinstance(stem_segments, int) or not 1 <= stem_segments <= 8:
        raise ValueError('stem_segments must be an integer in [1, 8]')
    if not 0 < timestep_s <= .001 or not .0001 <= contact_time_s <= .01 or not 1e-12 <= tolerance <= 1e-6:
        raise ValueError('Numerical parameters outside diagnostic bounds')
    if integrator not in ('Euler', 'implicitfast', 'discrete') or solver not in ('CG', 'Newton') or not 1 <= iterations <= 1000:
        raise ValueError('Unsupported numerical configuration')
    base = mujoco.MjModel.from_xml_string(xml)
    reference = mujoco.MjData(base)
    mujoco.mj_resetDataKeyframe(base, reference, base.key('home').id)
    mujoco.mj_forward(base, reference)
    if base.nflex or not manifest['anchors']:
        raise ValueError('Expected an unassembled base with canopy attachment markers')
    root = ET.fromstring(xml)
    root.remove(root.find('keyframe'))
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    for name, value in dict(timestep=timestep_s, integrator=integrator, solver=solver,
                            iterations=iterations, tolerance=tolerance, jacobian='sparse').items():
        option.set(name, str(value))
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('midphase', 'enable')
    bodies = {body.get('name'): body for body in root.findall('.//body')}
    for name in ('arm_link_wr1', 'arm_link_fngr', 'arm_link_jaw'):
        for geom in bodies[manifest['robot']['prefix'] + name].findall('geom'):
            if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
                geom.set('solref', f'{contact_time_s} 1')
                geom.set('friction', '.44 .005 .0001')
    positions = []
    for index, anchor in enumerate(manifest['anchors']):
        position = reference.site_xpos[base.site(anchor['site']).id].copy()
        if np.linalg.norm(position - anchor['world_position_m']) > 1e-5:
            raise RuntimeError('Base-scene attachment frames changed across backends')
        center = position - [0., 0., STEM_LENGTH + RADII_M[2]]
        positions.append(center)
        flex = ET.SubElement(root.find('worldbody'), 'flexcomp', name=f'fruit_{index}', type='ellipsoid',
            dof='full', dim='3', count=f'{count} {count} {count}',
            spacing=' '.join(str(2 * r / (count - 1)) for r in RADII_M),
            pos=' '.join(map(str, center)), mass=str(flesh_mass(count)), radius=str(FLEX_RADIUS_M), rgba='.45 .29 .11 1')
        tissue = XUXIANG['flesh']
        ET.SubElement(flex, 'elasticity', young=str(tissue.young), poisson=str(tissue.poisson), damping='.00001')
        ET.SubElement(flex, 'contact', selfcollide='none', internal='false', condim='3',
                      friction='.44 .005 .0001', solref=f'{contact_time_s} 1')
    intermediate = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(intermediate)
    _set_initial_joints(intermediate, data, manifest['robot'])
    mujoco.mj_forward(intermediate, data)
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    attachments = []
    for index, anchor in enumerate(manifest['anchors']):
        flex = mujoco.mj_name2id(intermediate, mujoco.mjtObj.mjOBJ_FLEX, f'fruit_{index}')
        if flex < 0:
            raise RuntimeError('Compiled deformable fruit is missing')
        start, number = intermediate.flex_vertadr[flex], intermediate.flex_vertnum[flex]
        vertices = data.flexvert_xpos[start:start + number]
        pole = positions[index] + [0., 0., RADII_M[2]]
        local_vertex = int(np.argmin(np.linalg.norm(vertices - pole, axis=1)))
        if np.linalg.norm(vertices[local_vertex] - pole) > 1e-6:
            raise RuntimeError('Discretization does not resolve the stem attachment pole')
        node = intermediate.body(int(intermediate.flex_vertbodyid[start + local_vertex])).name
        temporary = ET.fromstring('<mujoco><worldbody/><equality/></mujoco>')
        add_stem(temporary, node, [0., 0., 0.], vertices[local_vertex], segments=stem_segments)
        prefix = f'fruit_{index}_'
        names = {element.get('name'): prefix + element.get('name') for element in temporary.iter() if element.get('name')}
        for element in temporary.iter():
            for key, value in list(element.attrib.items()):
                if value in names:
                    element.set(key, names[value])
        parent_id = intermediate.body(anchor['parent_body']).id
        parent_rotation = data.xmat[parent_id].reshape(3, 3)
        stem_root = temporary.find('worldbody/body')
        root_position = vertices[local_vertex] + [0., 0., STEM_LENGTH]
        local_position = parent_rotation.T @ (root_position - data.xpos[parent_id])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(parent_rotation.T).ravel())
        stem_root.set('pos', ' '.join(map(str, local_position)))
        stem_root.set('quat', ' '.join(map(str, quaternion)))
        for geom in stem_root.findall('.//geom'):
            geom.set('solref', f'{contact_time_s} 1')
        bodies[anchor['parent_body']].append(stem_root)
        for constraint in temporary.find('equality'):
            equality.append(constraint)
        ET.SubElement(contact, 'exclude', body1=prefix + 'stem_0', body2=anchor['parent_body'])
        attachments.append(dict(flex=f'fruit_{index}', surface_vertex=local_vertex, node_body=node,
                                root_body=prefix + 'stem_0', tip_site=prefix + 'stem_tip',
                                equality=prefix + 'abscission', parent_body=anchor['parent_body']))
    xml, normalization = normalize_collision_meshes(ET.tostring(root, encoding='unicode'))
    root = ET.fromstring(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    _set_initial_joints(model, data, manifest['robot'])
    mujoco.mj_forward(model, data)
    if model.nflex != len(attachments) or any(model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD for i in range(model.neq)):
        raise RuntimeError('Unexpected fruit topology or artificial weld in training scene')
    for attachment in attachments:
        node, tip = model.body(attachment['node_body']).id, model.site(attachment['tip_site']).id
        if np.linalg.norm(data.xpos[node] - data.site_xpos[tip]) > 1e-6:
            raise RuntimeError('Stem attachment does not coincide with its material node')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    visual_map = visual.find('map')
    if visual_map is None:
        visual_map = ET.SubElement(visual, 'map')
    visual_map.set('znear', str(.005 / model.stat.extent))
    visual_map.set('zfar', str(10. / model.stat.extent))
    keys = ET.SubElement(root, 'keyframe')
    ET.SubElement(keys, 'key', name='home', qpos=' '.join(map(str, data.qpos)))
    xml = ET.tostring(root, encoding='unicode')
    manifest = dict(manifest, schema='deformable-training-scene/v1', base_model_sha256=manifest['model_sha256'],
        model_sha256=hashlib.sha256(xml.encode()).hexdigest(), attachments=attachments,
        requires_deformable_assembly=False, training_ready=False, mesh_normalization=normalization,
        assembler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), mujoco_version=mujoco.__version__,
        asset_sha256=asset_digests(root), render_clip_m=[.005, 10.],
        numerical_profile=dict(timestep_s=timestep_s, contact_time_s=contact_time_s, integrator=integrator,
                               solver=solver, iterations=iterations, tolerance=tolerance),
        material=dict(scope='Homogeneous elastic flesh, not calibrated whole fruit', mesh_count=count,
                      stem_segments=stem_segments,
                      radii_m=list(RADII_M), mass_kg=flesh_mass(count), density_kg_m3=FLESH_DENSITY_KG_M3,
                      young_Pa=XUXIANG['flesh'].young, poisson=XUXIANG['flesh'].poisson,
                      stem_model='He2024 beam reduction; ideal point abscission connection',
                      contact_scope='0.44 provisional hand/fruit friction; basket properties unchanged',
                      uncalibrated=['skin/core', 'plasticity', 'creep', 'wet friction', 'abscission torque']),
        scope='Floating-base Spot with collidable stems and deformable fruit; dynamics and learning unvalidated')
    return xml, manifest


def _set_initial_joints(model, data, robot):
    for name, value in robot['initial_position_rad'].items():
        data.qpos[model.jnt_qposadr[model.joint(robot['prefix'] + name).id]] = value


def asset_digests(root):
    import hashlib
    from pathlib import Path
    if root.findall('.//include'):
        raise ValueError('Scene artifacts must be self-contained XML without includes')
    result = {}
    for element in root.findall('./asset/*'):
        filename = element.get('file')
        if filename:
            path = Path(filename)
            if not path.is_absolute():
                raise ValueError('External scene assets need explicit absolute paths')
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_scene_artifact(directory):
    import hashlib
    import json
    from pathlib import Path
    import mujoco
    directory = Path(directory)
    xml = (directory / 'scene.xml').read_text()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'deformable-training-scene/v1' or hashlib.sha256(xml.encode()).hexdigest() != manifest.get('model_sha256'):
        raise ValueError('Invalid or corrupted deformable scene')
    if asset_digests(ET.fromstring(xml)) != manifest.get('asset_sha256'):
        raise ValueError('Scene asset content changed')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    if model.nflex != len(manifest['attachments']) or not np.isfinite(data.qpos).all():
        raise RuntimeError('Invalid restored scene state')
    return model, data, manifest
