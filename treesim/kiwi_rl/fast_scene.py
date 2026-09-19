"""Small rigid-fruit scene assembled from a ``training-base-scene/v1`` export.

This is a training approximation: each kiwi is an independent free ellipsoid
and its canopy connection is a runtime-toggleable MuJoCo ``connect`` equality.
It deliberately has no flexcomp, segmented stalk, or hand weld.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .scene import _set_initial_joints, asset_digests, normalize_collision_meshes


SCHEMA = 'fast-training-scene/v1'
FRUIT_MASS_KG = 0.105
DEFAULT_TIMESTEP_S = 0.005
CONTACT_SOLREF = '.02 1'


def _read_base(directory: Path):
    xml = (directory / 'base.xml').read_text()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'training-base-scene/v1':
        raise ValueError('Expected a training-base-scene/v1 artifact')
    if hashlib.sha256(xml.encode()).hexdigest() != manifest.get('model_sha256'):
        raise ValueError('Invalid or corrupted base-scene artifact')
    return xml, manifest


def _fixed_canopy(root: ET.Element, manifest: dict) -> None:
    """Remove joints from support bodies while leaving the Spot hierarchy intact."""
    robot = manifest.get('robot', {})
    prefix = robot.get('prefix', '')
    bodies = {b.get('name'): b for b in root.findall('.//body') if b.get('name')}
    parent = {}
    for p in root.iter('body'):
        for child in p.findall('body'):
            parent[child.get('name')] = p.get('name')
    support = set()
    for anchor in manifest['anchors']:
        name = anchor['parent_body']
        while name and name not in support:
            support.add(name)
            name = parent.get(name)
    for name in support:
        if prefix and name.startswith(prefix):
            continue
        for joint in list(bodies[name].findall('joint')):
            bodies[name].remove(joint)


def _option(root: ET.Element, timestep_s: float) -> ET.Element:
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep=str(timestep_s), integrator='implicitfast', solver='Newton',
                        iterations='20', tolerance='1e-6')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('midphase', 'enable')
    return option


def _fruit_inertia(mass: float) -> tuple[float, float, float]:
    rx, ry, rz = (float(v) for v in __import__('treesim.native_kiwi', fromlist=['RADII_M']).RADII_M)
    return tuple(mass * (b * b + c * c) / 5.0 for b, c in ((ry, rz), (rx, rz), (rx, ry)))


def assemble_fast_scene(directory, *, fruit_count: int | None = None,
                        timestep_s: float = DEFAULT_TIMESTEP_S,
                        visual_stalk: bool = True):
    """Return ``(scene_xml, manifest)`` for a bounded rigid-fruit training scene."""
    import mujoco
    from treesim.native_kiwi import RADII_M
    from treesim.kiwi_material import STEM_LENGTH

    directory = Path(directory)
    xml, base_manifest = _read_base(directory)
    if fruit_count is None:
        fruit_count = len(base_manifest.get('anchors', []))
    if not isinstance(fruit_count, int) or not 1 <= fruit_count <= len(base_manifest.get('anchors', [])):
        raise ValueError('fruit_count must select 1..number of base anchors')
    if not np.isfinite(timestep_s) or timestep_s not in (0.002, 0.005):
        raise ValueError('timestep_s must be 0.002 (500 Hz) or 0.005 (200 Hz)')
    if not base_manifest.get('anchors'):
        raise ValueError('Base scene has no canopy anchors')

    base = mujoco.MjModel.from_xml_string(xml)
    base_data = mujoco.MjData(base)
    if base.nflex:
        raise ValueError('Base scene must not contain flex objects')
    if base.nkey:
        mujoco.mj_resetDataKeyframe(base, base_data, base.key('home').id)
    _set_initial_joints(base, base_data, base_manifest['robot'])
    mujoco.mj_forward(base, base_data)

    root = ET.fromstring(xml)
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)
    _option(root, timestep_s)
    _fixed_canopy(root, base_manifest)
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError('Base scene has no worldbody')
    _, _, rz = RADII_M
    inertia = _fruit_inertia(FRUIT_MASS_KG)
    fruits = []
    for index, anchor in enumerate(base_manifest['anchors'][:fruit_count]):
        anchor_id = base.site(anchor['site']).id
        anchor_position = base_data.site_xpos[anchor_id].copy()
        body_name, geom_name, site_name = f'fruit_{index}', f'fruit_{index}_geom', f'fruit_{index}_site'
        equality_name = f'fruit_{index}_connect'
        # Keep the original canopy-to-fruit stem length.  The fruit site is
        # outside the ellipsoid so the connect equality reaches the canopy.
        center = anchor_position - np.array([0., 0., STEM_LENGTH + rz])
        body = ET.SubElement(worldbody, 'body', name=body_name, pos=' '.join(map(str, center)))
        ET.SubElement(body, 'freejoint', name=f'{body_name}_free')
        ET.SubElement(body, 'inertial', mass=str(FRUIT_MASS_KG), pos='0 0 0',
                      diaginertia=' '.join(map(str, inertia)))
        ET.SubElement(body, 'geom', name=geom_name, type='ellipsoid',
                      size=' '.join(map(str, RADII_M)), density='0',
                      contype='1', conaffinity='1', condim='3',
                      friction='.44 .005 .0001', solref=CONTACT_SOLREF,
                      solimp='.9 .95 .001', rgba='.45 .29 .11 1')
        ET.SubElement(body, 'site', name=site_name, pos=f'0 0 {STEM_LENGTH + rz}', size='.001',
                      rgba='0 0 0 0', group='5')
        ET.SubElement(equality, 'connect', name=equality_name, site1=site_name,
                      site2=anchor['site'], active='true', solref=CONTACT_SOLREF,
                      solimp='.9 .95 .001')
        if visual_stalk:
            parent = root.find(f".//body[@name='{anchor['parent_body']}']")
            if parent is None:
                raise ValueError(f"Missing anchor parent body {anchor['parent_body']!r}")
            site = root.find(f".//site[@name='{anchor['site']}']")
            pos = np.fromstring(site.get('pos', '0 0 0'), sep=' ')
            parent_id = base.body(anchor['parent_body']).id
            parent_rotation = base_data.xmat[parent_id].reshape(3, 3)
            local_down = parent_rotation.T @ np.array([0., 0., -float(STEM_LENGTH)])
            start = pos
            end = pos + local_down
            ET.SubElement(parent, 'geom', name=f'{body_name}_visual_stalk', type='capsule',
                          fromto=' '.join(map(str, np.r_[start, end])), size='.002', density='0',
                          contype='0', conaffinity='0', group='2', rgba='.27 .16 .07 1')
        fruits.append(dict(body=body_name, geom=geom_name, site=site_name,
                           equality=equality_name, anchor_site=anchor['site']))

    # Normalize authored collision meshes after adding the fruit, preserving the
    # existing GPU/MJWarp coordinate-frame workaround and physical invariants.
    assembled_xml, normalization = normalize_collision_meshes(ET.tostring(root, encoding='unicode'))
    model = mujoco.MjModel.from_xml_string(assembled_xml)
    data = mujoco.MjData(model)
    # Copy home coordinates by joint name. Fixed canopy support can remove
    # canopy joints, so a raw qpos prefix is not a safe correspondence.
    qpos_width = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    for joint_id in range(base.njnt):
        name = base.joint(joint_id).name
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if target >= 0:
            source_width = qpos_width[int(base.jnt_type[joint_id])]
            target_width = qpos_width[int(model.jnt_type[target])]
            if source_width != target_width:
                continue
            source_slice = slice(base.jnt_qposadr[joint_id], base.jnt_qposadr[joint_id] + source_width)
            target_slice = slice(model.jnt_qposadr[target], model.jnt_qposadr[target] + target_width)
            data.qpos[target_slice] = base_data.qpos[source_slice]
    _set_initial_joints(model, data, base_manifest['robot'])
    mujoco.mj_forward(model, data)
    if model.nflex or any(model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD for i in range(model.neq)):
        raise RuntimeError('Fast scene contains flex objects or a hand weld')
    if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
        raise RuntimeError('Invalid initial fast-scene state')
    final_root = ET.fromstring(assembled_xml)
    keys = ET.SubElement(final_root, 'keyframe')
    ET.SubElement(keys, 'key', name='home', qpos=' '.join(map(str, data.qpos)))
    assembled_xml = ET.tostring(final_root, encoding='unicode')
    # Recompile once so the home keyframe is checked and serialized exactly.
    model = mujoco.MjModel.from_xml_string(assembled_xml)
    if model.nflex or model.nu != base.nu:
        raise RuntimeError('Fast scene changed base topology')
    root = ET.fromstring(assembled_xml)
    manifest = dict(base_manifest, schema=SCHEMA, base_model_sha256=base_manifest['model_sha256'],
        model_sha256=hashlib.sha256(assembled_xml.encode()).hexdigest(), fruits=fruits,
        requires_deformable_assembly=False, training_ready=False,
        canopy_dynamics='fixed support', mesh_normalization=normalization,
        numerical_profile=dict(timestep_s=timestep_s, frequency_hz=1. / timestep_s,
                               integrator='implicitfast', solver='Newton', iterations=20,
                               tolerance=1e-6, contact_solref=CONTACT_SOLREF),
        approximation=dict(model='independent rigid ellipsoid', mass_kg=FRUIT_MASS_KG,
                           radii_m=list(RADII_M), connection='runtime-toggleable point connect',
                           visual_stalk=bool(visual_stalk),
                           omitted=['flexible flesh', 'segmented stem springs', 'hand weld'],
                           caveat='Rigid contact is a training approximation, not tissue calibration'),
        asset_sha256=asset_digests(root),
        assembler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return assembled_xml, manifest


def load_fast_scene(directory):
    """Load and verify a fast scene artifact, including its home state."""
    import mujoco
    directory = Path(directory)
    xml = (directory / 'scene.xml').read_text()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or hashlib.sha256(xml.encode()).hexdigest() != manifest.get('model_sha256'):
        raise ValueError('Invalid or corrupted fast scene')
    root = ET.fromstring(xml)
    if asset_digests(root) != manifest.get('asset_sha256', {}):
        raise ValueError('Scene asset content changed')
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nflex or any(model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD for i in range(model.neq)):
        raise ValueError('Fast scene must contain no flex objects or weld equalities')
    if model.nkey == 0:
        raise ValueError('Fast scene has no home keyframe')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
        raise RuntimeError('Invalid restored fast-scene state')
    return model, data, manifest
