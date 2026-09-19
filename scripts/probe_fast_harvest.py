"""Unassisted scripted grasp/carry diagnostic in the actual fast training runtime.

Uses privileged IK waypoints, never a hand weld or fruit-state edit. This is a
feasibility probe, not a learned policy or a source of accepted demonstrations.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(a):
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    from scipy.optimize import least_squares
    from treesim.basket import CENTER, SIZE
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    rt = FastRuntime(a.scene, worlds=1, camera=None, arm_speed_rad_s=.5,
                     solver_iterations=100)
    if a.jaw_cap is not None:
        rt.control.set_jaw_caps(np.array([a.jaw_cap], np.float32))
    gait = load_gait_artifact(a.gait_checkpoint).cuda().eval()
    model = rt.model
    data = mujoco.MjData(model)
    robot = rt.manifest['robot']
    wrist = model.body(robot['prefix'] + 'arm_link_wr1').id
    qids = np.asarray(rt.control.contract.qids)[12:18]
    joints = np.asarray(rt.control.contract.joints)[12:18]
    limits = model.jnt_range[joints]
    data.qpos[:] = rt.data.qpos.numpy()[0]
    mujoco.mj_forward(model, data)
    rotation = data.xmat[wrist].reshape(3, 3).copy()
    # Grasp centre from the existing actual-mesh fixed-wrist bench.
    offset = np.array([.195, -.005, 0.])
    fruit_start = data.xpos[rt.fruit_body].copy()
    rows, states = [], []
    phases = [('approach', 5.), ('close', 3.), ('pull', 3.),
              ('carry', 10.), ('release', 4.)]
    target_q = data.qpos[qids].copy()
    for phase, duration in phases:
        for tick in range(round(duration / rt.control_dt)):
            data.qpos[:] = rt.data.qpos.numpy()[0]
            mujoco.mj_forward(model, data)
            desired = fruit_start.copy()
            if phase == 'pull':
                desired[2] -= .12
            elif phase in ('carry', 'release'):
                chassis_rotation = data.xmat[rt.chassis].reshape(3, 3)
                desired = data.xpos[rt.chassis] + chassis_rotation @ (
                    np.asarray(CENTER) + [0., 0., SIZE[2] + .12])
            if tick % 5 == 0:
                home = data.qpos[qids].copy()
                def residual(q):
                    data.qpos[qids] = q
                    mujoco.mj_kinematics(model, data)
                    actual_rotation = data.xmat[wrist].reshape(3, 3)
                    point = data.xpos[wrist] + actual_rotation @ offset
                    return np.r_[8 * (point - desired),
                                 .3 * (actual_rotation - rotation).ravel(), .01 * (q - home)]
                result = least_squares(residual, np.clip(home, limits[:, 0], limits[:, 1]),
                                       bounds=(limits[:, 0], limits[:, 1]), max_nfev=35)
                # Integrate the measured joint correction into motor targets;
                # absolute IK angles alone leave the PD arm sagging under gravity.
                target_q = np.clip(rt.control.targets.numpy()[0, 12:18] + result.x - home,
                                   limits[:, 0], limits[:, 1])
            jaw = -1.57 if phase in ('approach', 'release') else 0.
            target = torch.tensor(np.r_[target_q, jaw], dtype=torch.float32, device='cuda')[None]
            action = ((target - wp.to_torch(rt.control.targets)[:, 12:]) /
                      (rt.arm_speed_rad_s * rt.control_dt)).clamp(-1, 1)
            with torch.no_grad():
                rt.set_gait_actions(gait(rt.observe()))
                _, _, done, info = rt.step(action)
            rt.check()
            row = dict(phase=phase, time_s=len(rows)*rt.control_dt,
                       distance_m=float(info['distance_m'][0]),
                       load_N=float(wp.to_torch(rt.task.hand_load)[0]),
                       finger_load_N=float(wp.to_torch(rt.task.finger_load)[0]),
                       jaw_load_N=float(wp.to_torch(rt.task.jaw_load)[0]),
                       palm_load_N=float(wp.to_torch(rt.task.palm_load)[0]),
                       stem_force_N=float(wp.to_torch(rt.task.stem_force)[0]),
                       **{k:bool(info[k][0]) for k in ('stable_grasp', 'ever_grasped', 'detached', 'success', 'failed')})
            rows.append(row)
            if len(rows) % 2 == 0:
                states.append(rt.data.qpos.numpy()[0].copy())
            if bool(done[0]):
                break
        print(json.dumps(rows[-1]), flush=True)
        if bool(done[0]):
            break
    report = dict(scope=__doc__, scene=str(a.scene), jaw_cap_Nm=float(rt.control.jaw_cap.numpy()[0]),
                  final=rows[-1], ever_grasped=any(r['ever_grasped'] for r in rows),
                  peak_hand_load_N=max(r['load_N'] for r in rows), numerical=rt.check())
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.output/'measurements.json').write_text(json.dumps(rows)+'\n')
    np.savez_compressed(a.output/'states.npz', qpos=np.asarray(states))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'gait-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--jaw-cap', type=float, help='Diagnostic torque cap override; no default change')
    args = p.parse_args()
    if args.jaw_cap is not None and not 0 < args.jaw_cap <= 2:
        p.error('Diagnostic jaw cap must be in (0, 2] Nm')
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        run(args)
