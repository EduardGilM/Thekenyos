#!/usr/bin/env python
"""Queue estimated-target PPO after existing trainers finish.

Waits for train_assisted_kiwi.py and queue_visual_harvest.py. Does not kill
other processes. Warm-starts the privileged 7-action policy; actor fruit XYZ
comes from cameras, not simulator coordinates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from scripts.queue_visual_harvest import emit, gpu_compute_apps, wait_for_free_gpu, wait_for_pids
except ImportError:
    from queue_visual_harvest import emit, gpu_compute_apps, wait_for_free_gpu, wait_for_pids


def harvest_job_pids(exclude=()):
    found = []
    exclude = {os.getpid(), *(int(pid) for pid in exclude)}
    for path in Path('/proc').glob('*/cmdline'):
        try:
            text = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            pid = int(path.parent.name)
        except (OSError, ValueError):
            continue
        if pid in exclude:
            continue
        if 'scripts/train_assisted_kiwi.py' in text or 'scripts/queue_visual_harvest.py' in text:
            found.append(pid)
    return sorted(found)


def wait_until_idle(log, poll_s=30, exclude=()):
    while True:
        pids = harvest_job_pids(exclude)
        if pids:
            wait_for_pids(pids, log, poll_s)
            continue
        wait_for_free_gpu(log, poll_s)
        if not harvest_job_pids(exclude) and sum(memory for _, memory in gpu_compute_apps()) <= 512:
            emit(log, dict(event='idle', note='No harvest trainers and GPU memory is free; starting estimated-target PPO'))
            return
        time.sleep(poll_s)


def train_args(relic, output, checkpoint, camera_size=96):
    if camera_size not in (64, 96, 192):
        raise ValueError('Unsupported camera size')
    return (
        '--relic', str(relic), '--output', str(output), '--estimate',
        '--camera-size', str(camera_size), '--warm-start', str(checkpoint),
        '--workspace-weight', '1', '--stage', '0', '--episode-seconds', '30',
        '--steps', '32768', '--num-envs', '4', '--rollout-steps', '128', '--batch-size', '128',
        '--learning-rate', '.0001', '--entropy', '.002', '--exploration-std', '.3',
        '--gripper-std', '.15', '--eval-every', '4096', '--eval-episodes', '8',
        '--heldout-seed', '90000', '--policy-device', 'cuda', '--record-video',
    )


def main():
    parser = argparse.ArgumentParser(description='Queue estimated-target assisted PPO after existing trainers')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--camera-size', type=int, choices=(64, 96, 192), default=96)
    parser.add_argument('--wait-pid', type=int, nargs='*', default=())
    parser.add_argument('--wait-for-jobs', action='store_true',
                        help='Wait for train_assisted_kiwi.py and queue_visual_harvest.py; never signals them')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('Poll interval must be at least 5 seconds')
    if not args.checkpoint.is_file() or not args.checkpoint.with_suffix('.json').is_file():
        parser.error('Privileged 7-action checkpoint and sidecar JSON are required')
    if args.output.exists():
        parser.error(f'Training output already exists: {args.output}')
    args.queue_dir.mkdir(parents=True, exist_ok=False)
    log = args.queue_dir/'progress.jsonl'
    extra = list(args.wait_pid)
    if args.wait_for_jobs:
        extra.extend(pid for pid in harvest_job_pids() if pid not in extra)
    command = [str(args.python), '-u', '-B', 'scripts/train_assisted_kiwi.py',
               *train_args(args.relic.resolve(), args.output, args.checkpoint.resolve(), args.camera_size)]
    (args.queue_dir/'plan.json').write_text(json.dumps(dict(
        scope='Estimated-target wrap of assisted PPO; camera fruit XYZ; artificial grip weld',
        checkpoint=str(args.checkpoint.resolve()), output=str(args.output.resolve()),
        wait_pids=extra, command=command,
        note='Does not load a visual CNN; fruit coordinates in the actor are camera estimates'), indent=2)+'\n')
    emit(log, dict(event='queued', wait_pids=extra, output=str(args.output), command=command))
    try:
        if extra or args.wait_for_jobs:
            wait_until_idle(log, args.poll_seconds, exclude=())
        env = os.environ.copy()
        env.setdefault('MUJOCO_GL', 'egl')
        env['PYTHONUNBUFFERED'] = '1'
        emit(log, dict(event='stage_start', command=command))
        completed = subprocess.run(command, cwd=str(args.source.resolve()), env=env)
        emit(log, dict(event='stage_finished', returncode=completed.returncode,
                       summary_exists=(args.output/'summary.json').exists()))
        if completed.returncode:
            raise RuntimeError(f'Estimated-target training failed with return code {completed.returncode}')
        emit(log, dict(event='finished', output=str(args.output)))
    except Exception as exc:
        emit(log, dict(event='error', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
