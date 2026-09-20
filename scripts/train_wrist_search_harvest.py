#!/usr/bin/env python
"""Train wrist-camera search, then gated one- and three-fruit collection.

The first lesson uses temporary hand RGB/ToF search rewards from SEARCH_* in
treesim/visual_kiwi_env.py. Edit those terms before launching. Every later lesson
has view shaping disabled and starts only after its parent has a held-out
success. Existing trainers are waited for and never signalled.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .queue_visual_harvest import (
        emit, heldout_succeeded, run_stage, trainer_pids, wait_for_free_gpu, wait_for_pids)
else:
    from queue_visual_harvest import (
        emit, heldout_succeeded, run_stage, trainer_pids, wait_for_free_gpu, wait_for_pids)


def collect_stages():
    shared = ('--vision', '--num-envs', '4', '--rollout-steps', '128', '--batch-size', '128',
              '--learning-rate', '.0001', '--entropy', '.002', '--exploration-std', '.3',
              '--policy-device', 'cuda', '--record-video')
    deposit = ('--deposit-mix', 'pick:0.5,carry:0.3,release:0.2', '--deposit-shaping', '1',
               '--view-weight', '0', '--gripper-std', '.25')
    return (
        dict(name='collect-1', output='wrist-collect-1fruit-03',
             args=('--visual-lesson', 'collect', '--start-phase', 'pick', '--picks', '1',
                   '--episode-seconds', '90', '--steps', '65536', '--eval-every', '8192',
                   '--eval-episodes', '8', '--heldout-seed', '95100', *deposit, *shared)),
        dict(name='collect-3', output='wrist-collect-3fruit-03',
             args=('--visual-lesson', 'collect', '--start-phase', 'pick', '--picks', '3',
                   '--episode-seconds', '120', '--steps', '131072', '--eval-every', '8192',
                   '--eval-episodes', '8', '--heldout-seed', '96100', *deposit, *shared)),
    )


def stages(grab_only=False):
    shared = ('--vision', '--num-envs', '4', '--rollout-steps', '128', '--batch-size', '128',
              '--learning-rate', '.0001', '--entropy', '.002', '--exploration-std', '.3',
              '--policy-device', 'cuda', '--record-video')
    grab = dict(name='search-grab', output='wrist-search-inview-grab-01',
                args=('--visual-lesson', 'grab', '--start-phase', 'pick', '--episode-seconds', '8',
                      '--steps', '32768', '--eval-every', '4096', '--eval-episodes', '8',
                      '--heldout-seed', '97000', '--search-rewards', '--view-weight', '1',
                      '--fixed-stage', '--gripper-std', '.25', *shared))
    if grab_only:
        return (grab,)
    return (grab,)+collect_stages()


def encoder_is_trained(path):
    meta = path.with_suffix('.json')
    if not meta.is_file():
        return False
    saved = json.loads(meta.read_text())
    return int(saved.get('steps', 0)) > 0


def successful_checkpoint(directory):
    summary_path = directory/'summary.json'
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    checkpoint = directory/summary.get('recommended_checkpoint', '')
    return checkpoint if heldout_succeeded(summary) and checkpoint.is_file() else None


def main():
    parser = argparse.ArgumentParser(description='Gated wrist-only visual search and kiwi collection curriculum')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--stationary-checkpoint', type=Path)
    parser.add_argument('--grab-only', action='store_true',
                        help='Train only the short in-FOV grab lesson; skip collect stages')
    parser.add_argument('--continue-from', type=Path,
                        help='Skip grab; warm-start basket stages from this completed grab directory')
    parser.add_argument('--initial-warm-start', type=Path,
                        help='Weight-only start for the first queued collect stage; requires --continue-from')
    parser.add_argument('--one-fruit', action='store_true',
                        help='Stop after one-fruit collect; skip the three-fruit stage')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--wait-for-trainers', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('Poll interval must be at least 5 seconds')
    if args.grab_only and args.continue_from:
        parser.error('Use either --grab-only or --continue-from')
    if args.initial_warm_start is not None and args.continue_from is None:
        parser.error('--initial-warm-start requires --continue-from')
    if args.initial_warm_start is not None and not args.initial_warm_start.is_file():
        parser.error('initial-warm-start checkpoint is missing')
    if args.one_fruit and args.grab_only:
        parser.error('--one-fruit applies to basket collect stages')
    if args.stationary_checkpoint is not None and not args.stationary_checkpoint.is_file():
        parser.error('Wrist-only stationary checkpoint is missing')
    previous = None
    if args.continue_from is not None:
        parent = successful_checkpoint(args.continue_from)
        if parent is None:
            parser.error('continue-from needs a held-out success and the recommended checkpoint')
        previous = args.initial_warm_start.resolve() if args.initial_warm_start is not None else parent
        plan_stages = collect_stages()
    else:
        plan_stages = stages(grab_only=args.grab_only)
    if args.one_fruit:
        plan_stages = tuple(stage for stage in plan_stages if stage['name'] != 'collect-3')

    args.output_root.mkdir(parents=True, exist_ok=True)
    queue_dir = args.queue_dir or args.output_root/'wrist-search-harvest-queue-01'
    queue_dir.mkdir(parents=True, exist_ok=False)
    log = queue_dir/'progress.jsonl'
    wait_pids = trainer_pids() if args.wait_for_trainers else []
    plan = [dict(name=stage['name'], output=str((args.output_root/stage['output']).resolve()))
            for stage in plan_stages]
    encoder = None
    if args.stationary_checkpoint is not None and encoder_is_trained(args.stationary_checkpoint):
        encoder = args.stationary_checkpoint.resolve()
    (queue_dir/'plan.json').write_text(json.dumps(dict(
        scope='In-FOV wrist RGB+ToF grab, then optional release/carry/collect',
        actor_cameras=['ee_cam', 'ee_depth'],
        head_camera=False,
        stationary_checkpoint=str(encoder) if encoder else None,
        encoder_warm_start=bool(encoder),
        grab_only=args.grab_only,
        one_fruit=args.one_fruit,
        continue_from=str(args.continue_from.resolve()) if args.continue_from else None,
        parent_checkpoint=str(previous) if previous else None,
        initial_warm_start=str(args.initial_warm_start.resolve()) if args.initial_warm_start else None,
        wait_pids=wait_pids,
        plan=plan,
        gate='Every parent needs at least one fresh held-out success; failed stages stop the queue',
        reward='Collect: training-only deposit-mix pick/carry/release, hold-while-far and '
               'open-near-basket, softer dropped penalty; evaluations stay pick with shaping off',
    ), indent=2)+'\n')
    emit(log, dict(event='queued', plan=plan, wait_pids=wait_pids, head_camera=False,
                   encoder_warm_start=bool(encoder), grab_only=args.grab_only,
                   one_fruit=args.one_fruit, continue_from=str(args.continue_from) if args.continue_from else None,
                   initial_warm_start=str(args.initial_warm_start) if args.initial_warm_start else None))
    wait_for_pids(wait_pids, log, args.poll_seconds)

    try:
        for index, stage in enumerate(plan_stages):
            wait_for_free_gpu(log, args.poll_seconds)
            if previous is None:
                load_args = ['--encoder-warm-start', str(encoder)] if encoder else []
            else:
                load_args = ['--warm-start', str(previous.resolve())]
            output = run_stage(args.python, args.source.resolve(), args.relic.resolve(),
                               args.output_root/stage['output'], stage['args'], load_args, log)
            previous = successful_checkpoint(output)
            if previous is None:
                emit(log, dict(event='curriculum_stopped', stage=stage['name'],
                               reason='No held-out success; later lessons were not started'))
                break
        emit(log, dict(event='finished', last_checkpoint=str(previous) if previous else None))
    except Exception as exc:
        emit(log, dict(event='error', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
