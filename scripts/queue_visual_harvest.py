#!/usr/bin/env python
"""Queue camera walking-grab, then 1-fruit collect, then 3-fruit collect.

Waits for already-running trainers to exit. Does not kill other processes.
The standing 4-action camera policy is encoder-only input; a later stage gets a
full weight warm-start only if held-out evaluation actually succeeded.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def utcnow():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def emit(path, record):
    record = dict(record, time=utcnow())
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
    print(json.dumps(record, allow_nan=False), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def trainer_pids(exclude=()):
    found = []
    exclude = {int(pid) for pid in exclude}
    for path in Path('/proc').glob('*/cmdline'):
        try:
            text = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            pid = int(path.parent.name)
        except (OSError, ValueError):
            continue
        if 'scripts/train_assisted_kiwi.py' in text and pid not in exclude:
            found.append(pid)
    return sorted(found)


def wait_for_pids(pids, log, poll_s=30):
    remaining = [int(pid) for pid in pids if alive(pid)]
    emit(log, dict(event='waiting', pids=remaining, poll_s=poll_s,
                   note='Waiting for existing trainers; no processes are signalled'))
    while remaining:
        time.sleep(poll_s)
        remaining = [pid for pid in remaining if alive(pid)]
        emit(log, dict(event='wait_heartbeat', remaining_pids=remaining))
    emit(log, dict(event='wait_complete', remaining_pids=[]))


def gpu_compute_apps():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'nvidia-smi failed')
    apps = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, memory = [part.strip() for part in line.split(',')]
        apps.append((int(pid), int(memory.split()[0])))
    return apps


def wait_for_free_gpu(log, poll_s=30, free_mib=512):
    while True:
        apps = gpu_compute_apps()
        used = sum(memory for _, memory in apps)
        emit(log, dict(event='gpu_check', compute_apps=[{'pid': pid, 'used_mib': memory} for pid, memory in apps],
                       used_mib=used))
        if used <= free_mib:
            return
        emit(log, dict(event='gpu_busy', used_mib=used, poll_s=poll_s,
                       note='GPU in use by another process; waiting rather than killing it'))
        time.sleep(poll_s)


def heldout_succeeded(summary):
    return any(int(item.get('trained_successes', 0)) > 0 for item in summary.get('heldout', []))


def select_load_args(previous_dir, stationary):
    if previous_dir is not None:
        summary_path = previous_dir/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            checkpoint = previous_dir/summary['recommended_checkpoint']
            if heldout_succeeded(summary) and checkpoint.is_file():
                return ['--warm-start', str(checkpoint.resolve())]
    return ['--encoder-warm-start', str(stationary.resolve())]


def stages():
    shared = ('--vision', '--num-envs', '4', '--rollout-steps', '128', '--batch-size', '128',
              '--learning-rate', '.0001', '--entropy', '.002', '--exploration-std', '.3',
              '--gripper-std', '.4', '--policy-device', 'cuda', '--record-video')
    return (
        dict(name='walk-grab', output='visual-walk-grab-03',
             args=('--visual-lesson', 'grab', '--start-phase', 'pick', '--episode-seconds', '30',
                   '--steps', '32768', '--eval-every', '4096', '--eval-episodes', '8',
                   '--heldout-seed', '87000', *shared)),
        dict(name='collect-1', output='visual-collect-1fruit-01',
             args=('--visual-lesson', 'collect', '--start-phase', 'pick', '--picks', '1',
                   '--episode-seconds', '90', '--steps', '32768', '--eval-every', '8192',
                   '--eval-episodes', '4', '--heldout-seed', '88000', *shared)),
        dict(name='collect-3', output='visual-collect-3fruit-01',
             args=('--visual-lesson', 'collect', '--start-phase', 'pick', '--picks', '3',
                   '--episode-seconds', '90', '--steps', '65536', '--eval-every', '8192',
                   '--eval-episodes', '4', '--heldout-seed', '89000', *shared)),
    )


def run_stage(python, source, relic, output, extra, load_args, log):
    if output.exists():
        raise FileExistsError(output)
    command = [str(python), '-u', '-B', 'scripts/train_assisted_kiwi.py',
               '--relic', str(relic), '--output', str(output), *extra, *load_args]
    emit(log, dict(event='stage_start', output=str(output), load_args=load_args, command=command))
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    env['PYTHONUNBUFFERED'] = '1'
    completed = subprocess.run(command, cwd=str(source), env=env)
    summary = output/'summary.json'
    record = dict(event='stage_finished', output=str(output), returncode=completed.returncode,
                  summary_exists=summary.exists())
    if summary.exists():
        saved = json.loads(summary.read_text())
        record.update(recommended_checkpoint=saved.get('recommended_checkpoint'),
                      heldout=saved.get('heldout'),
                      heldout_succeeded=heldout_succeeded(saved))
    emit(log, record)
    if completed.returncode:
        raise RuntimeError(f'Training failed for {output} with return code {completed.returncode}')
    return output


def main():
    parser = argparse.ArgumentParser(description='Queue walking camera grab then basket collect after existing trainers finish')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--stationary-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--wait-pid', type=int, nargs='*', default=())
    parser.add_argument('--wait-for-trainers', action='store_true',
                        help='Wait for train_assisted_kiwi.py processes already running at launch')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('Poll interval must be at least 5 seconds')
    if not args.stationary_checkpoint.is_file():
        parser.error('Stationary checkpoint is missing')
    wait_pids = list(args.wait_pid)
    if args.wait_for_trainers:
        wait_pids.extend(pid for pid in trainer_pids() if pid not in wait_pids)
    args.output_root.mkdir(parents=True, exist_ok=True)
    queue_dir = args.queue_dir or (args.output_root/'visual-harvest-queue-01')
    queue_dir.mkdir(parents=True, exist_ok=False)
    log = queue_dir/'progress.jsonl'
    plan = [dict(name=stage['name'], output=str((args.output_root/stage['output']).resolve())) for stage in stages()]
    (queue_dir/'plan.json').write_text(json.dumps(dict(
        scope='Camera walking grab, then one-fruit basket collect, then three-fruit collect',
        stationary_checkpoint=str(args.stationary_checkpoint.resolve()),
        wait_pids=wait_pids,
        plan=plan,
        note='Standing 4-action weights are never loaded as a 10-action walking/deposit policy'), indent=2)+'\n')
    emit(log, dict(event='queued', plan=plan, wait_pids=wait_pids))
    wait_for_pids(wait_pids, log, args.poll_seconds)
    previous = None
    try:
        for stage in stages():
            wait_for_free_gpu(log, args.poll_seconds)
            load_args = select_load_args(previous, args.stationary_checkpoint)
            previous = run_stage(args.python, args.source.resolve(), args.relic.resolve(),
                                 args.output_root/stage['output'], stage['args'], load_args, log)
            if stage['name'] == 'collect-1':
                summary = json.loads((previous/'summary.json').read_text())
                if not heldout_succeeded(summary):
                    emit(log, dict(event='skip_collect_3',
                                   reason='One-fruit collect had no held-out basket success; not jumping to three fruit'))
                    break
        emit(log, dict(event='finished', outputs=[item['output'] for item in plan]))
    except Exception as exc:
        emit(log, dict(event='error', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
