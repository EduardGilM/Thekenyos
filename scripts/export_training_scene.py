import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


PINNED_RELIC = '27f8033c5064d32f049a17accb71cd1091422878'


def export_scene(args):
    import mujoco
    from scripts.assisted_harvest_cycle import build
    relic = args.relic.resolve()
    commit = subprocess.check_output(['git', '-C', str(relic), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_RELIC:
        raise ValueError('Use the pinned external RELIC checkout')
    if args.output.exists():
        raise FileExistsError('Preserve existing scene export directories')
    args.output.mkdir(parents=True)
    args.fixed_base, args.freeze_canopy = False, args.static_canopy
    manifest = {}
    build(args, metadata=manifest)
    root = ET.parse(args.output / 'scene.xml').getroot()
    parent_of = {child: parent for parent in root.iter() for child in parent}
    removed = set()
    fruit_names = {entry['source_fruit_body'] for entry in manifest['anchors']}
    for body in list(root.findall('.//body')):
        if body.get('name') in fruit_names or body.get('name') == 'stem_0':
            removed.update(child.get('name') for child in body.iter() if child.get('name'))
            parent_of[body].remove(body)
    for section in ('contact', 'equality'):
        element = root.find(section)
        if element is not None:
            for entry in list(element):
                if entry.get('name') in ('grip_assist', 'abscission') or any(value in removed for value in entry.attrib.values()):
                    element.remove(entry)
    for site in root.findall('.//site'):
        if site.get('name') == 'grip_frame':
            parent_of[site].remove(site)
    robot = manifest['robot']
    actuator = root.find('actuator')
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, 'actuator')
    joints = robot['legs'] + robot['arm']
    for name, limit in zip(joints, robot['controller_torque_limit_Nm']):
        ET.SubElement(actuator, 'motor', name=name, joint=robot['prefix'] + name,
                      ctrllimited='true', ctrlrange=f'{-limit} {limit}',
                      forcelimited='true', forcerange=f'{-limit} {limit}')
    bodies = {body.get('name'): body for body in root.findall('.//body')}
    ET.SubElement(bodies[robot['wrist']], 'site', name='hand_tcp',
                  pos=' '.join(map(str, robot['tcp_offset_m'])), size='.001', rgba='0 0 0 0', group='5')
    from treesim.kiwi_rl.spot_cameras import attach_mujoco_cameras, load_relic_gripper_cameras
    cameras = attach_mujoco_cameras(bodies, robot, load_relic_gripper_cameras(relic))
    keyframes = root.find('keyframe')
    if keyframes is not None:
        root.remove(keyframes)
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body = model.body(robot['chassis']).id
    free_joint = None
    while body:
        for joint in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                free_joint = joint
        body = model.body_parentid[body]
    if free_joint is None or model.nu != 19:
        raise RuntimeError('Export must retain a floating base and all 19 torque actuators')
    for name, value in robot['initial_position_rad'].items():
        joint = model.joint(robot['prefix'] + name).id
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
        raise RuntimeError('Invalid initial exported state')
    position_error, rotation_error = 0., 0.
    for name, reference in manifest['reference_body_poses'].items():
        body = model.body(name).id
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(reference['quaternion_wxyz']))
        position_error = max(position_error, float(np.linalg.norm(data.xpos[body] - reference['position_m'])))
        rotation_error = max(rotation_error, float(np.max(np.abs(data.xmat[body] - rotation))))
    if position_error > 1e-5 or rotation_error > 1e-4:
        raise RuntimeError(f'Export changed body frames: {position_error} m, rotation error {rotation_error}')
    manifest['home_geometry_check'] = dict(max_position_error_m=position_error, max_rotation_matrix_error=rotation_error)
    for anchor in manifest['anchors']:
        anchor['world_position_m'] = data.site_xpos[model.site(anchor['site']).id].tolist()
    if len(manifest['anchors']) != args.fruit_count:
        raise RuntimeError('Generator did not supply the requested number of fruit anchors')
    keys = ET.SubElement(root, 'keyframe')
    ET.SubElement(keys, 'key', name='home', qpos=' '.join(map(str, data.qpos)))
    xml = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    mujoco.mj_resetDataKeyframe(model, mujoco.MjData(model), model.key('home').id)
    manifest.update(schema='training-base-scene/v1', seed=args.seed, relic_commit=commit,
        canopy_dynamics='static support' if args.static_canopy else 'compliant',
        model_sha256=hashlib.sha256(xml.encode()).hexdigest(), cameras=cameras,
        requires_deformable_assembly=True, training_ready=False,
        scope='Floating-base robot, basket, canopy, visual foliage and attachment markers; no fruit yet',
        producer_sources_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ('scripts/export_training_scene.py', 'scripts/assisted_harvest_cycle.py',
                         'treesim/harvest_env.py', 'treesim/kiwi_rl/spot_cameras.py')},
        producer_versions={name: importlib.metadata.version(name) for name in ('newton', 'warp-lang', 'mujoco')})
    (args.output / 'base.xml').write_text(xml)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relic', type=Path)
    parser.add_argument('--base-scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh-count', type=int, default=9)
    parser.add_argument('--contact-time', type=float, default=.001)
    parser.add_argument('--integrator', choices=('Euler', 'implicitfast', 'discrete'), default='Euler')
    parser.add_argument('--solver', choices=('CG', 'Newton'), default='CG')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fruit-count', type=int, default=1)
    parser.add_argument('--stem-segments', type=int, default=4,
                        help='Collidable stem segments per fruit (1-8)')
    parser.add_argument('--static-canopy', action='store_true', help='Keep the support canopy fixed; robot and deformable fruit remain dynamic')
    parser.add_argument('--foliage-density', type=float, default=.6)
    parser.add_argument('--timestep', type=float, default=.00002)
    args = parser.parse_args()
    if not 1 <= args.fruit_count <= 128 or not np.isfinite([args.foliage_density, args.timestep]).all():
        parser.error('Invalid scene generation parameters')
    if not 0 <= args.foliage_density <= 1 or not 0 < args.timestep <= .001:
        parser.error('Invalid foliage density or timestep')
    if args.base_scene is not None:
        from treesim.kiwi_rl.scene import assemble_deformable_scene
        if args.output.exists():
            parser.error('Preserve existing scene directories')
        xml, result = assemble_deformable_scene(args.base_scene, count=args.mesh_count,
            timestep_s=args.timestep, contact_time_s=args.contact_time, integrator=args.integrator,
            solver=args.solver, iterations=args.iterations, tolerance=args.tolerance,
            stem_segments=args.stem_segments)
        args.output.mkdir(parents=True)
        (args.output / 'scene.xml').write_text(xml)
        (args.output / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    else:
        if args.relic is None:
            parser.error('Base export requires --relic; deformable assembly requires --base-scene')
        result = export_scene(args)
    print(json.dumps(dict(schema=result['schema'], anchors=len(result['anchors']),
                          restored_canopy_visuals=result['restored_canopy_visuals'],
                          model_sha256=result['model_sha256'], training_ready=False)), flush=True)


if __name__ == '__main__':
    main()
