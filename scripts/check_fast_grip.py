"""Run the rigid Spot-jaw contact fixture through MJWarp at fast-training rates.

This checks rigid contact, hold, and release only. It does not test Spot motion,
deformable fruit, or task success. Each case writes a JSON result as it finishes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_case(relic: Path, dt: float, torque: float) -> dict:
    import mujoco_warp as mjw
    import torch
    import warp as wp
    from scripts import check_spot_gripper as bench
    from treesim.kiwi_rl.scene import normalize_collision_meshes
    from treesim.kiwi_rl.fast_runtime import _latch_state, _configure_epa_horizon
    from mujoco_warp._src import collision_convex

    _configure_epa_horizon(mjw, collision_convex)

    bench.CONTACT_TIME_S = .02
    asset = relic / 'source/relic/relic/assets/spot'
    xml = bench.scene(asset, dt, rigid=True, torque=torque, count=9)
    xml, normalization = normalize_collision_meshes(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = data.ctrl[0] = -1.
    mujoco.mj_forward(model, data)
    kiwi = model.body('kiwi').id
    fruit_geom = model.geom('kiwi').id
    floor_geom = model.geom('floor').id
    fixed = {i for i in range(model.ngeom)
             if 'arm_link_jaw_collision' in (model.geom(i).name or '')}
    moving = {i for i in range(model.ngeom)
              if 'arm_link_fngr_collision' in (model.geom(i).name or '')}
    if not fixed or not moving:
        raise RuntimeError('Could not identify both jaw collision mesh groups')

    wp.init()
    stream = torch.cuda.Stream()
    hold_origin = None
    max_motion = 0.
    bilateral, hold_samples = 0, 0
    peak_load = np.zeros(2)
    ground_before_release = False
    ground_after_release = False
    post_release_load = 0.
    force = np.zeros(6)
    applied = np.zeros((1, model.nbody, 6), dtype=np.float32)
    sample_every = max(1, round(.02 / dt))
    steps = round(4. / dt)

    with (torch.cuda.stream(stream),
          wp.ScopedStream(wp.stream_from_torch(stream)),
          wp.ScopedDevice('cuda:0')):
        gpu_model = mjw.put_model(model)
        gpu_data = mjw.put_data(model, data, nworld=1, nconmax=128, njmax=256)
        numerical_flags = wp.zeros(1, dtype=int)
        for step in range(steps):
            t = step * dt
            close = min(max(t - .2, 0.), 1.)
            release = min(max((t - 2.8) / .6, 0.), 1.)
            target = (-1. if t < .2 else
                      -1. + close * close * (3. - 2. * close) if t < 1.2 else
                      0. if t < 2.8 else
                      -release * release * (3. - 2. * release))
            gpu_data.ctrl.assign(np.array([[target]], dtype=np.float32))
            applied.fill(0.)
            if t >= 1.5:
                applied[0, kiwi, 2] = -9.81 * model.body_mass[kiwi]
            gpu_data.xfrc_applied.assign(applied)
            mjw.step(gpu_model, gpu_data)
            wp.launch(_latch_state, dim=(1, max(model.nq, model.nv)),
                      inputs=[gpu_data.qpos, gpu_data.qvel, gpu_data.overflow, numerical_flags])
            if step % sample_every:
                continue
            mjw.get_data_into(data, model, gpu_data)
            loads = np.zeros(2)
            floor = False
            for index, contact in enumerate(data.contact[:data.ncon]):
                geoms = set(map(int, contact.geom))
                if fruit_geom not in geoms:
                    continue
                mujoco.mj_contactForce(model, data, index, force)
                normal = abs(float(force[0]))
                floor |= floor_geom in geoms and normal > .01
                if geoms & fixed:
                    loads[0] += normal
                if geoms & moving:
                    loads[1] += normal
            peak_load = np.maximum(peak_load, loads)
            center = data.xpos[kiwi].copy()
            if t < 2.8:
                ground_before_release |= floor
            if 1.7 < t < 2.7:
                if hold_origin is None:
                    hold_origin = center
                max_motion = max(max_motion, float(np.linalg.norm(center - hold_origin)))
                bilateral += int(min(loads) > .1)
                hold_samples += 1
            if t > 3.4:
                ground_after_release |= floor
            if t >= 3.6:
                post_release_load = max(post_release_load, float(loads.max()))
        wp.synchronize()
        mjw.get_data_into(data, model, gpu_data)

    warnings = [int(w.number) for w in data.warning]
    result = dict(
        timestep_s=dt, frequency_hz=1. / dt, jaw_cap_Nm=torque,
        simulated_s=float(data.time), final_jaw_rad=float(data.qpos[0]),
        final_fruit_position_m=data.xpos[kiwi].tolist(),
        hold_samples=hold_samples,
        hold_bilateral_fraction=bilateral / max(hold_samples, 1),
        hold_max_displacement_m=max_motion,
        sampled_peak_jaw_load_N=peak_load.tolist(),
        ground_before_release=bool(ground_before_release),
        ground_after_release=bool(ground_after_release),
        max_post_release_jaw_load_N=post_release_load,
        warnings=warnings, final_contacts=int(data.ncon),
        numerical_flags=numerical_flags.numpy().tolist(),
        mesh_normalization=normalization,
        passed=bool(hold_samples and bilateral / hold_samples >= .95
                    and max_motion < .02 and not ground_before_release
                    and ground_after_release and post_release_load < .01
                    and not any(warnings) and not numerical_flags.numpy().any()),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh path to preserve prior results')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(scope=__doc__, backend='MJWarp', worlds=1, cases=[])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for dt in (.005, .002):
        for torque in (.3, 1.):
            result = run_case(args.relic.resolve(), dt, torque)
            report['cases'].append(result)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
