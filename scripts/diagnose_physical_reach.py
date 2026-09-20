"""Capture the first failing world in a deterministic physical reach replay."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _names(model, qpos_width, qvel_width, ctrl_width):
    qpos, qvel, ctrl = [], [], []
    for joint in range(model.njnt):
        name = model.joint(joint).name or f"joint_{joint}"
        q0, q1 = int(model.jnt_qposadr[joint]), int(model.jnt_qposadr[joint + 1]) if joint + 1 < model.njnt else qpos_width
        v0 = int(model.jnt_dofadr[joint])
        v1 = int(model.jnt_dofadr[joint + 1]) if joint + 1 < model.njnt else qvel_width
        qpos.extend(f"{name}.qpos[{i}]" for i in range(q0, q1))
        qvel.extend(f"{name}.qvel[{i}]" for i in range(v0, v1))
    for actuator in range(ctrl_width):
        name = model.actuator(actuator).name or f"actuator_{actuator}"
        ctrl.append(name)
    return qpos, qvel, ctrl


def _dump(runtime, output: Path, step: int, error: Exception):
    output.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(runtime.data.qpos.numpy())
    qvel = np.asarray(runtime.data.qvel.numpy())
    ctrl = np.asarray(runtime.data.ctrl.numpy())
    qpos_names, qvel_names, ctrl_names = _names(runtime.model, qpos.shape[1], qvel.shape[1], ctrl.shape[1])
    failures = []
    for world in range(runtime.worlds):
        indices = {
            "qpos": np.flatnonzero(~np.isfinite(qpos[world])).tolist(),
            "qvel": np.flatnonzero(~np.isfinite(qvel[world])).tolist(),
            "ctrl": np.flatnonzero(~np.isfinite(ctrl[world])).tolist(),
        }
        if any(indices.values()):
            np.savez(output / f"state-world-{world}.npz", qpos=qpos[world], qvel=qvel[world], ctrl=ctrl[world])
        failures.append({
            "world": world,
            "nonfinite_indices": indices,
            "qpos_names": [qpos_names[i] for i in indices["qpos"] if i < len(qpos_names)],
            "qvel_names": [qvel_names[i] for i in indices["qvel"] if i < len(qvel_names)],
            "ctrl_names": [ctrl_names[i] for i in indices["ctrl"] if i < len(ctrl_names)],
        })
    report = {"step": step, "error": f"{type(error).__name__}: {error}",
              "monitor": runtime.monitor.report(), "worlds": failures}
    (output / "failure.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def run(args):
    import torch
    from treesim.kiwi_rl.ppo import load_checkpoint
    from scripts.train_physical_smoke import build_policy, collect
    from treesim.kiwi_rl.control import gait_cpu_inference, load_gait_artifact
    from treesim.kiwi_rl.physical_rollout import BatchedPhysicalRollout, RuntimeReachReward
    from treesim.kiwi_rl.runtime import BatchedDeformableRuntime

    if args.output.exists():
        raise FileExistsError('Preserve existing diagnostics')
    torch.set_num_threads(1)
    runtime = BatchedDeformableRuntime(args.scene, worlds=args.worlds, resolution=(64, 64))
    policy = build_policy().to("cuda:0")
    saved = load_checkpoint(args.checkpoint, {'student': policy}, expected_meta={
        'schema': 'physical-reach-rgbd-r84/v1', 'camera': args.camera, 'control_dt_s': args.control_dt,
        'gait_sha256': hashlib.sha256(args.gait_checkpoint.read_bytes()).hexdigest()})
    policy.eval()
    gait_actor = load_gait_artifact(args.gait_checkpoint)
    gait = lambda obs: gait_cpu_inference(gait_actor, obs)
    adapter = BatchedPhysicalRollout(runtime, RuntimeReachReward(runtime), camera=args.camera)
    try:
        rows, _ = collect(adapter, policy, args.steps, gait, deterministic=True, control_dt=args.control_dt)
    except Exception as exc:
        report = _dump(runtime, args.output, runtime.step_index, exc)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1
    args.output.mkdir(parents=True)
    report = dict(failed=False, steps=len(rows), scope='sensor-only learned reach evaluation; no teacher',
        checkpoint=str(args.checkpoint), checkpoint_model_sha256=saved['meta']['model_sha256'],
        evaluated_model_sha256=runtime.manifest['model_sha256'],
        distance_m=[row['distance'].tolist() for row in rows],
        fallen=rows[-1]['terminated'].cpu().tolist(), numerical=runtime.monitor.check())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'states.npz', qpos=np.stack([row['qpos'] for row in rows]),
                        control_dt_s=args.control_dt, state_time='post-action')
    from PIL import Image
    frames = [Image.fromarray((np.clip(row['rgbd'][0, :3].cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)) for row in rows]
    frames[0].save(args.output / 'policy-input.gif', save_all=True, append_images=frames[1:],
                   duration=round(args.control_dt * 1000), loop=0)
    print(json.dumps(report), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gait-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--control-dt", type=float, default=.04)
    parser.add_argument("--camera", choices=("hand_camera", "body_camera"), default="body_camera")
    args = parser.parse_args()
    if args.steps < 1 or args.worlds < 1:
        parser.error("steps and worlds must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
