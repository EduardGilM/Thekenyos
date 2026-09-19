import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def make_scene(case, relic, dt, count, integrator, solver, iterations=100, contact_time=.004, tolerance=1e-9):
    from treesim.native_kiwi import flesh_mass
    if case in ('mesh', 'grip'):
        from scripts.check_spot_gripper import scene
        root = ET.fromstring(scene(relic / 'source/relic/relic/assets/spot', dt, count=count))
    else:
        root = ET.fromstring('<mujoco><option gravity="0 0 -9.81"/><size memory="256M"/><worldbody><geom name="floor" type="plane" size="1 1 .01" friction=".44 .005 .0001"/></worldbody></mujoco>')
        for i in range(2 if case == 'pair' else 1):
            flex = ET.SubElement(root.find('worldbody'), 'flexcomp',
                name=f'kiwi_{i}', type='ellipsoid', dim='3', count=f'{count} {count} {count}',
                spacing=' '.join(str(v / (count - 1)) for v in (.054, .054, .072)),
                pos=f'0 0 {.06 + i * .08}', mass=str(flesh_mass(count)), radius='.0003')
            ET.SubElement(flex, 'elasticity', young='1570000', poisson='.4', damping='.00001')
            ET.SubElement(flex, 'contact', selfcollide='none', internal='false',
                          condim='3', friction='.44 .005 .0001')
    for geom in root.findall('.//geom'):
        geom.set('solref', f'{contact_time} 1')
    for contact in root.findall('.//flexcomp/contact'):
        contact.set('solref', f'{contact_time} 1')
    option = root.find('option')
    for name, value in dict(timestep=dt, integrator=integrator, solver=solver,
                            jacobian='sparse', iterations=iterations, tolerance=tolerance).items():
        option.set(name, str(value))
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('nativeccd', 'disable')
    return ET.tostring(root, encoding='unicode')


def volumes(model, data):
    from treesim.kiwi_rl.physics import tetrahedra
    tetra = data.flexvert_xpos[tetrahedra(model)]
    return np.linalg.det(tetra[:, 1:] - tetra[:, :1]) / 6


def sample(model, data, rest_volumes, metrics):
    import mujoco
    from treesim.kiwi_rl.physics import contact_flex_ids
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError('Nonfinite physics state')
    if any(w.number for w in data.warning):
        raise RuntimeError('MuJoCo warning; automatic numerical reset is not a valid rollout')
    ratio = float(np.min(volumes(model, data) / rest_volumes))
    metrics['minimum_volume_ratio'] = min(metrics['minimum_volume_ratio'], ratio)
    if ratio <= .1:
        raise RuntimeError(f'Collapsed or inverted element: volume ratio {ratio}')
    force = np.zeros(6)
    for index, c in enumerate(data.contact):
        if np.any((c.geom >= 0) & (c.flex >= 0)):
            examples = metrics.setdefault('ambiguous_contact_examples', [])
            if len(examples) < 4:
                examples.append(dict(time_s=float(data.time), geom=c.geom.tolist(), flex=c.flex.tolist(),
                    names=[model.geom(int(g)).name if g >= 0 else None for g in c.geom]))
        flex_ids = contact_flex_ids(c.geom, c.flex)
        if not np.any(flex_ids >= 0):
            continue
        metrics['flex_contacts'] += 1
        overlap = max(0., -float(c.dist))
        metrics['max_penetration_m'] = max(metrics['max_penetration_m'], overlap)
        if np.all(flex_ids >= 0) and flex_ids[0] != flex_ids[1]:
            metrics['flex_flex_contacts'] += 1
        for g in c.geom:
            if g >= 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                metrics['mesh_flex_contacts'] += 1
                metrics['max_hand_penetration_m'] = max(metrics['max_hand_penetration_m'], overlap)
        mujoco.mj_contactForce(model, data, index, force)
        metrics['peak_contact_force_N'] = max(metrics['peak_contact_force_N'], float(np.linalg.norm(force[:3])))


def run_case(args, case, dt, backend):
    import mujoco
    from treesim.kiwi_rl.physics import DeformableMonitor, FlexContactObserver, NativeFlexContactObserver
    xml = make_scene(case, args.relic.resolve(), dt, args.count, args.integrator, args.solver,
                     args.iterations, args.contact_time, args.tolerance)
    normalization = None
    if args.normalize_meshes:
        from treesim.kiwi_rl.scene import normalize_collision_meshes
        xml, normalization = normalize_collision_meshes(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    gripper = case in ('mesh', 'grip')
    if gripper:
        data.qpos[0] = data.ctrl[0] = -1.
    mujoco.mj_forward(model, data)
    if any(c.dist < -.001 and np.any((c.flex >= 0) & (c.geom < 0)) for c in data.contact):
        raise RuntimeError('Initial fruit pose penetrates a collider')
    rest = volumes(model, data)
    if not len(rest) or np.any(rest == 0):
        raise RuntimeError('Invalid tetrahedral rest mesh')
    initial_qpos, initial_ctrl = data.qpos.copy(), data.ctrl.copy()
    worlds = args.worlds if backend == 'gpu' else 1
    metrics = dict(case=case, backend=backend, timestep_s=dt, integrator=args.integrator,
        solver=args.solver, iterations=args.iterations, tolerance=args.tolerance, contact_time_s=args.contact_time,
        mesh_count=args.count, vertices=model.nflexvert, dofs=model.nv, worlds=worlds,
        minimum_volume_ratio=1., max_penetration_m=0., max_hand_penetration_m=0., peak_contact_force_N=0.,
        flex_contacts=0, flex_flex_contacts=0, mesh_flex_contacts=0,
        scene_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Backend contact screen; not full-robot, calibrated tissue, or training-readiness validation',
        report_kind='deformable-backend-screen/v2', training_ready=False, mesh_normalization=normalization)
    steps = math.ceil(args.seconds / dt)
    block = min(32, steps)
    steps = math.ceil(steps / block) * block
    sample_blocks = max(1, round(.005 / (dt * block)))
    metrics['control_interval_s'] = block * dt
    metrics['observation_interval_s'] = sample_blocks * block * dt
    gravity_force = np.zeros((model.nbody, 6), dtype=np.float32)
    fruit_bodies = np.unique(model.flex_vertbodyid)
    gravity_force[fruit_bodies, 2] = -9.81 * model.body_mass[fruit_bodies]
    loaded = False
    hold_origin = None
    max_hold_motion = np.zeros(worlds)
    bilateral_samples = np.zeros(worlds)
    hold_samples = 0
    post_release_load = np.zeros(worlds)

    def jaw_target(t):
        if case == 'grip' and t >= 2.8:
            u = float(np.clip((t - 2.8) / .6, 0., 1.))
            return -u * u * (3. - 2. * u)
        u = float(np.clip(t - .2, 0., 1.))
        return -1. + u * u * (3. - 2. * u)

    def observe(t, contact_report, positions):
        nonlocal hold_origin, hold_samples
        sample(model, data, rest, metrics)
        if case != 'grip':
            return
        loads = np.asarray(contact_report['jaw_palm_load_N'])[:, 0, :2]
        if 1.7 <= t <= 2.7:
            centers = positions.mean(axis=1)
            if hold_origin is None:
                hold_origin = centers.copy()
            np.maximum(max_hold_motion, np.linalg.norm(centers - hold_origin, axis=1), out=max_hold_motion)
            bilateral_samples[:] += np.min(loads, axis=1) > .1
            hold_samples += 1
        if t >= 3.6:
            np.maximum(post_release_load, np.max(loads, axis=1), out=post_release_load)

    started = time.monotonic()
    if backend == 'gpu':
        import warp as wp
        import mujoco_warp as mjw
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            gpu_model = mjw.put_model(model)
            gpu_data = mjw.put_data(model, data, nworld=worlds, nconmax=4096, njmax=8192)
            monitor = DeformableMonitor(model, data, gpu_data)
            observer = FlexContactObserver(model, gpu_data)
            mjw.step(gpu_model, gpu_data)
            monitor.record()
            observer.record()
            wp.synchronize()
            monitor.check()
            observer.check()
            with wp.ScopedCapture() as capture:
                for _ in range(block):
                    mjw.step(gpu_model, gpu_data)
                    observer.record()
                    mjw.kinematics(gpu_model, gpu_data)
                    mjw.flex(gpu_model, gpu_data)
                    monitor.record()
            mjw.reset_data(gpu_model, gpu_data)
            monitor.reset()
            observer.reset()
            gpu_data.qpos.assign(np.tile(initial_qpos, (worlds, 1)).astype(np.float32))
            if model.nu:
                gpu_data.ctrl.assign(np.tile(initial_ctrl, (worlds, 1)).astype(np.float32))
            mjw.forward(gpu_model, gpu_data)
            metrics['setup_seconds'] = time.monotonic() - started
            run_start = time.monotonic()
            for k in range(steps // block):
                t = k * block * dt
                if gripper:
                    gpu_data.ctrl.assign(np.full((worlds, model.nu), jaw_target(t), dtype=np.float32))
                if case == 'grip' and t >= 1.5 and not loaded:
                    gpu_data.xfrc_applied.assign(np.tile(gravity_force[None], (worlds, 1, 1)))
                    loaded = True
                wp.capture_launch(capture.graph)
                metrics['gpu_numerical'] = monitor.check()
                contact_report = observer.check()
                if k % sample_blocks == 0 or k == steps // block - 1:
                    mjw.get_data_into(data, model, gpu_data, world_id=0)
                    observe((k + 1) * block * dt, contact_report, gpu_data.flexvert_xpos.numpy())
            wp.synchronize()
            metrics['integration_seconds'] = time.monotonic() - run_start
            metrics['finite_all_worlds'] = bool(np.isfinite(gpu_data.qpos.numpy()).all() and np.isfinite(gpu_data.qvel.numpy()).all())
            if not metrics['finite_all_worlds']:
                raise RuntimeError('Nonfinite state in a batched world')
    else:
        observer = NativeFlexContactObserver(model, data)
        metrics['setup_seconds'] = time.monotonic() - started
        run_start = time.monotonic()
        for k in range(steps // block):
            t = k * block * dt
            if gripper:
                data.ctrl[0] = jaw_target(t)
            if case == 'grip' and t >= 1.5 and not loaded:
                data.xfrc_applied[:] = gravity_force
                loaded = True
            for _ in range(block):
                mujoco.mj_step(model, data)
                observer.record()
                if any(w.number for w in data.warning):
                    raise RuntimeError('MuJoCo warning during a CPU substep')
            if k % sample_blocks == 0 or k == steps // block - 1:
                observe((k + 1) * block * dt, observer.check(), data.flexvert_xpos[None])
        metrics['integration_seconds'] = time.monotonic() - run_start
    contacts = observer.check()
    metrics['contacts'] = contacts
    metrics['simulated_seconds'] = float(data.time)
    metrics['control_time_s'] = steps * dt
    metrics['final_vertices_m'] = data.flexvert_xpos.tolist()
    metrics['final_centers_m'] = [data.flexvert_xpos[start:start + count].mean(axis=0).tolist()
                                  for start, count in zip(model.flex_vertadr, model.flex_vertnum)]
    if gripper:
        metrics['final_jaw_rad'] = float(data.qpos[0])
        metrics['final_actuator_force'] = data.actuator_force.tolist()
    required = 'mesh_flex_contacts' if gripper else 'flex_flex_contacts' if case == 'pair' else 'flex_contacts'
    metrics['contact_interpretation'] = 'Valid geom IDs take precedence over stale flex IDs, matching the solver'
    metrics['passed'] = bool(metrics[required] > 0 and metrics['minimum_volume_ratio'] > .1)
    if case == 'grip':
        ground = np.asarray(contacts['first_ground_step'])[:, 0]
        fraction = bilateral_samples / max(hold_samples, 1)
        metrics.update(hold_samples=hold_samples, sampled_bilateral_fraction=fraction.tolist(),
                       max_hold_motion_m=max_hold_motion.tolist(), post_release_jaw_load_N=post_release_load.tolist())
        metrics['passed'] &= bool(hold_samples and np.all(fraction >= .95) and np.all(max_hold_motion < .02)
                                 and np.all(ground * dt >= 2.8) and np.all(post_release_load < .01)
                                 and metrics['max_hand_penetration_m'] < .001)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('fall', 'pair', 'mesh', 'grip'), default='fall')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--timestep', type=float, default=.00002)
    parser.add_argument('--seconds', type=float, default=.2)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--worlds', type=int, default=2)
    parser.add_argument('--integrator', choices=('Euler', 'implicitfast', 'discrete'), default='Euler')
    parser.add_argument('--solver', choices=('CG', 'Newton'), default='CG')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--contact-time', type=float, default=.004)
    parser.add_argument('--normalize-meshes', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error('Use 1..1000 solver iterations')
    if not np.isfinite([args.timestep, args.seconds, args.contact_time, args.tolerance]).all():
        parser.error('Use finite numerical parameters')
    if not 0 < args.timestep <= .001 or not 0 < args.seconds <= 10:
        parser.error('Invalid duration or timestep')
    if not .0001 <= args.contact_time <= .01 or not 1e-12 <= args.tolerance <= 1e-6:
        parser.error('Contact time or tolerance outside diagnostic bounds')
    if args.case == 'grip' and args.seconds < 4:
        parser.error('Grip screen needs at least four seconds for hold and release')
    if not 4 <= args.count <= 11 or not 1 <= args.worlds <= 16:
        parser.error('Mesh count or world count outside smoke-test bounds')
    if args.output.exists():
        parser.error('Output already exists; preserve each probe result')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_case(args, args.case, args.timestep, args.backend)
    except Exception as exc:
        result = dict(passed=False, training_ready=False, case=args.case, backend=args.backend,
                      error=f'{type(exc).__name__}: {exc}', timestep_s=args.timestep)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        raise
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'final_vertices_m'}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
