import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def render_views(model, data, output):
    import mujoco
    from PIL import Image
    result = {}
    with mujoco.Renderer(model, height=128, width=128) as renderer:
        for name in ('hand_color_sensor', 'hand_depth_sensor'):
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=name)
            rgb = renderer.render().copy()
            Image.fromarray(rgb).save(output / f'{name}.png')
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            np.save(output / f'{name}-depth.npy', depth)
            finite = depth[np.isfinite(depth)]
            result[name] = dict(rgb_std=float(rgb.std()), depth_min_m=float(finite.min()) if len(finite) else None,
                                depth_max_m=float(finite.max()) if len(finite) else None,
                                finite_depth_fraction=float(np.isfinite(depth).mean()))
    camera = mujoco.MjvCamera()
    chassis = model.body('spot_with_arm_body').id
    camera.lookat[:] = data.xpos[chassis] + [0., 0., .35]
    camera.distance, camera.azimuth, camera.elevation = 3., 135., -20.
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / 'scene.png')
    return result


def run(args):
    import mujoco
    from treesim.kiwi_rl.scene import load_scene_artifact
    from treesim.kiwi_rl.control import NativeSpotControl, load_gait_artifact
    from treesim.kiwi_rl.physics import NativeFlexContactObserver, tetrahedra
    model, data, manifest = load_scene_artifact(args.scene)
    from treesim.kiwi_rl.spot_cameras import require_mujoco_gripper_cameras
    require_mujoco_gripper_cameras(model, manifest['robot'])
    actor = load_gait_artifact(args.gait_checkpoint) if args.gait_checkpoint else None
    controller = NativeSpotControl(model, manifest['robot'], actor)
    observer = NativeFlexContactObserver(model, data)
    elements = tetrahedra(model)
    vertices = data.flexvert_xpos[elements]
    rest = np.linalg.det(vertices[:, 1:] - vertices[:, :1])
    if not len(rest) or not np.isfinite(rest).all() or np.any(np.abs(rest) < 1e-18):
        raise RuntimeError('Invalid reference tissue mesh')
    result = dict(scope='Full-scene numerical and rendering diagnostic; not a gait benchmark or harvesting trainer',
        controller='pretrained RELIC inference' if actor else 'scripted joint-target PD',
        new_policy_training=False, training_ready=False, numerical_screen_passed=False,
        model_sha256=manifest['model_sha256'], nflex=model.nflex, nv=model.nv,
        timestep_s=float(model.opt.timestep), minimum_volume_ratio=1.,
        minimum_base_height_m=float(data.xpos[controller.chassis, 2]), maximum_tilt_rad=0.,
        max_abs_motor_torque_Nm=0.)
    steps = math.ceil(args.seconds / model.opt.timestep)
    gait_stride = max(1, round(.02 / model.opt.timestep))
    started = time.monotonic()
    try:
        for step in range(steps):
            if actor is not None and step % gait_stride == 0:
                mujoco.mj_comPos(model, data)
                mujoco.mj_comVel(model, data)
                controller.update_gait(data, [0., 0., 0.])
            torque = controller.apply(data)
            result['max_abs_motor_torque_Nm'] = max(result['max_abs_motor_torque_Nm'], float(np.abs(torque).max()))
            mujoco.mj_step(model, data)
            observer.record()
            if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Nonfinite state or native solver warning')
            mujoco.mj_kinematics(model, data)
            mujoco.mj_flex(model, data)
            vertices = data.flexvert_xpos[elements]
            ratio = float(np.min(np.linalg.det(vertices[:, 1:] - vertices[:, :1]) / rest))
            result['minimum_volume_ratio'] = min(result['minimum_volume_ratio'], ratio)
            if not np.isfinite(ratio) or ratio <= .1:
                raise RuntimeError('Collapsed or inverted tissue element')
            result['minimum_base_height_m'] = min(result['minimum_base_height_m'], float(data.xpos[controller.chassis, 2]))
            tilt = math.acos(float(np.clip(data.xmat[controller.chassis].reshape(3, 3)[2, 2], -1., 1.)))
            result['maximum_tilt_rad'] = max(result['maximum_tilt_rad'], tilt)
        result['numerical_screen_passed'] = True
        if args.render:
            mujoco.mj_camlight(model, data)
            result['rendering'] = render_views(model, data, args.output)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result.update(simulated_seconds=float(data.time), elapsed_seconds=time.monotonic() - started,
                  contacts=observer.check())
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez(args.output / 'diagnostic-state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, eq_active=data.eq_active)
    return result


def run_gpu(args):
    from PIL import Image
    from treesim.kiwi_rl.runtime import BatchedDeformableRuntime
    from treesim.kiwi_rl.control import gait_cpu_inference, load_gait_artifact
    result = dict(scope='Batched GPU scene diagnostic; not physical training acceptance',
                  training_ready=False, new_policy_training=False, numerical_screen_passed=False,
                  minimum_base_height_m=None, maximum_tilt_rad=0.)
    runtime = None
    started = time.monotonic()
    try:
        runtime = BatchedDeformableRuntime(args.scene, worlds=args.worlds,
                                           resolution=(128, 128) if args.render else None)
        actor = load_gait_artifact(args.gait_checkpoint) if args.gait_checkpoint else None
        callback = (lambda obs: gait_cpu_inference(actor, obs)) if actor is not None else None
        result['controller'] = 'pretrained RELIC inference' if actor else 'scripted joint-target PD'
        for _ in range(math.ceil(args.seconds / .02)):
            runtime.advance(.02, callback)
            positions = runtime.data.xpos.numpy()[:, runtime.control.chassis]
            minimum = float(positions[:, 2].min())
            result['minimum_base_height_m'] = minimum if result['minimum_base_height_m'] is None else min(result['minimum_base_height_m'], minimum)
            rotations = runtime.data.xmat.numpy()[:, runtime.control.chassis]
            result['maximum_tilt_rad'] = max(result['maximum_tilt_rad'], float(np.arccos(np.clip(rotations[:, 2, 2], -1., 1.)).max()))
        result['numerical_screen_passed'] = True
        if args.render:
            runtime.capture()
            result['rendering'] = {}
            for i, name in enumerate(runtime.rig.cameras):
                rgb, depth = runtime.rig.rgb[i].numpy(), runtime.rig.depth[i].numpy()
                valid = runtime.rig.valid[i].numpy()
                Image.fromarray((np.clip(rgb[0], 0., 1.) * 255).astype(np.uint8)).save(args.output / f'{name}.png')
                np.save(args.output / f'{name}-depth.npy', depth)
                result['rendering'][name] = dict(rgb_std=float(rgb.std()), valid_fraction=float(valid.mean()),
                    depth_max_m=float(depth.max()), depth_min_valid_m=float(depth[valid > 0].min()) if valid.any() else None)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    if runtime is not None:
        result.update(control_time_s=runtime.step_index * runtime.dt,
                      numerical=runtime.monitor.report(), contacts=runtime.contacts.report())
    result['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gait-checkpoint', type=Path)
    parser.add_argument('--seconds', type=float, default=.1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--worlds', type=int, default=2)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 5:
        parser.error('Use a finite diagnostic duration in (0, 5] seconds')
    if args.output.exists():
        parser.error('Preserve existing diagnostic directories')
    args.output.mkdir(parents=True)
    result = run_gpu(args) if args.backend == 'gpu' else run(args)
    print(json.dumps(result), flush=True)
    if not result['numerical_screen_passed'] or result.get('error'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
