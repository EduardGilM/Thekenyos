"""PPO from scratch: progressively longer physical harvesting sequences, no CTI."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_fast import update
from treesim.kiwi_rl.sequence_curriculum import (Curriculum, CHECKPOINTS, SEQUENCE_SCHEMA,
    SEQUENCE_GAMMA, LIVE_CREDIT, TIME_BUDGET, FAILURE_COST, POSITION_HOLD_S, CARRY_REWARD_VERSION)
from treesim.kiwi_rl.reward_graph import SEQUENCE_PROFILE

# Full jaw torque of 1 Nm loads the pads to the 15 N damage limit, so a
# closed command on the fruit would always fail; 0.6 Nm squeezes at about 9 N.
JAW_CAP_NM = .6


def evaluate(runtime, policy, gait, args, level, seed):
    import torch
    from treesim.kiwi_rl.harvest_training import HarvestCollector, episode_metrics
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        collector = HarvestCollector(runtime, role='teacher', reward_profile=SEQUENCE_PROFILE,
            curriculum_stage=level, stall_seconds=args.stall_seconds,
            max_episode_seconds=args.max_episode_seconds)
        completed = {}
        for _ in range(round(args.max_episode_seconds / runtime.control_dt) // args.steps + 2):
            _, _, episodes = collector.collect(policy, gait, args.steps, deterministic=True, store=False)
            for episode in episodes:
                completed.setdefault(episode['world'], episode)
            if len(completed) == runtime.worlds:
                return episode_metrics(list(completed.values()))
        raise RuntimeError('Evaluation did not finish every varied starting pose')


def run(args):
    import torch
    import warp as wp
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.fast_teacher import build_privileged_policy
    from treesim.kiwi_rl.harvest_training import HarvestCollector, episode_metrics
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import save_checkpoint, load_checkpoint
    from treesim.kiwi_rl.training_log import TrainingLog
    from treesim.kiwi_rl.graph_training import save_acceptance, graph_rank
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.resume_from and (args.output/'report.json').exists():
        raise ValueError('Completed runs cannot resume in place')
    args.output.mkdir(parents=True, exist_ok=bool(args.resume_from))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(role='teacher', reward_profile=SEQUENCE_PROFILE, reward_gamma=SEQUENCE_GAMMA,
        initialization='random', cti=False, solver_iterations=100, jaw_cap_Nm=JAW_CAP_NM, absolute_jaw=True, initial_jaw_rad=args.initial_jaw_rad, jaw_rate_rad_s=args.jaw_rate_rad_s,
        position_hold_s=POSITION_HOLD_S, stall_terminal=True, carry_reward=CARRY_REWARD_VERSION, fruit_damping=args.fruit_damping, max_level=args.max_level, fruit_jitter_m=list(args.fruit_jitter_m), fruit_reach_fraction=list(args.fruit_reach_fraction), fruit_sector_deg=args.fruit_sector_deg,
        deposit='scripted joint-space carry and release after a valid held extraction (treesim.kiwi_rl.scripted_deposit)',
        source_dir=str(Path(__file__).resolve().parents[1]),
        actor_inputs='privileged simulator state, R84 and episode checkpoint history; the unlocked objective is not an input',
        freeze_legs=True, leg_mode='fixed robot base and settled leg targets; arm and fruit remain dynamic',
        reward_accounting='fixed once-per-checkpoint payments plus changes in live quality; finite deadline',
        checkpoint_bonus=1., live_credit_limit=LIVE_CREDIT, episode_time_cost_limit=TIME_BUDGET,
        failure_cost=FAILURE_COST, earlier_objective_world_fraction=.2, timeout_bootstrap=False,
        optimizer_mode='shuffled recurrent world minibatches', ppo_epochs=3, target_kl=.02,
        promotion=dict(episode_success=.9, worlds_per_trial=32, trials_per_evaluation=2,
                       consecutive_evaluations=2, scope='fresh randomized training-distribution resets'),
        approximations='rigid fruit, 200 Hz; uncalibrated 8 N stem and 15 N damage thresholds')
    log = TrainingLog(args.output, config, wandb_mode=args.wandb_mode, wandb_run_id=args.wandb_run_id,
                      wandb_project='Thekenyos', wandb_entity='juampab', wandb_name=args.output.name)
    curriculum = Curriculum(max_level=args.max_level)
    index = transitions = total_episodes = total_optimizer_steps = eval_round = 0
    elapsed_offset = 0.
    best = None
    best_path = None
    try:
        gait = load_gait_artifact(args.gait_checkpoint).cuda().eval()
        runtimes = [FastRuntime(scene, worlds=worlds, camera=None, task_profile=SEQUENCE_PROFILE,
            arm_speed_rad_s=args.arm_speed_rad_s, solver_iterations=100, jaw_cap_Nm=JAW_CAP_NM, absolute_jaw=True,
            initial_jaw_rad=args.initial_jaw_rad, jaw_rate_rad_s=args.jaw_rate_rad_s, fruit_damping=args.fruit_damping,
            fruit_jitter_m=tuple(args.fruit_jitter_m), fruit_reach_fraction=tuple(args.fruit_reach_fraction), fruit_sector_deg=args.fruit_sector_deg)
            for scene, worlds in ((args.scene, args.worlds), (args.scene, args.eval_worlds),
                                  (args.eval_scene, args.eval_worlds))]
        runtime = runtimes[0]
        if not all(target.manifest.get('fixed_base') for target in runtimes):
            raise ValueError('Sequence training requires exported fixed-base scenes for frozen legs')
        if runtime.manifest['model_sha256'] == runtimes[2].manifest['model_sha256']:
            raise ValueError('Use a distinct held-out scene')
        for target in runtimes:
            target.prepare_settled_reset(gait)
            target.freeze_legs = True
            target.reset_jitter_rad = args.reset_jitter_rad
        policy = build_privileged_policy(sequence=True).cuda()
        with torch.no_grad():
            torch.nn.init.normal_(policy.mean.weight, std=.01)
            policy.mean.bias.zero_()
            policy.logstd.fill_(-1.)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        if args.resume_from:
            saved = load_checkpoint(args.resume_from, {'teacher': policy}, {'teacher': optimizer},
                expected_meta={'schema': SEQUENCE_SCHEMA, 'model_sha256': runtime.manifest['model_sha256']})
            old = saved['meta']['config']
            # Reset distribution (arm jitter, fruit jitter) may widen on resume: that is how a
            # single-position policy is fine-tuned toward generalisation.
            for key in ('scene','eval_scene','worlds','steps','minibatch_worlds','learning_rate','gae_lambda',
                        'entropy_coef','arm_speed_rad_s','train_seconds','seed'):
                if old[key] != config[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            state = saved['meta']['training_state']
            curriculum = Curriculum(**dict(state['curriculum'], max_level=args.max_level))
            index, transitions = state['update'], state['transitions']
            total_episodes, total_optimizer_steps = state['episodes'], state['optimizer_steps']
            elapsed_offset, eval_round = state['elapsed_seconds'], state['eval_round']
            best, best_path = state['best_rank'], state['best_checkpoint']
            best = tuple(best) if best is not None else None
            history = [json.loads(line) for line in (args.output/'training.jsonl').read_text().splitlines()]
            if max(row['step'] for row in history) != index:
                raise ValueError('Resume must use the latest logged checkpoint')
            torch.set_rng_state(saved['rng']['torch'])
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        collector = HarvestCollector(runtime, role='teacher', reward_profile=SEQUENCE_PROFILE,
            curriculum_stage=curriculum.level, stall_seconds=args.stall_seconds,
            max_episode_seconds=args.max_episode_seconds, rehearse=True)
        started = time.monotonic() - elapsed_offset
        last_eval = time.monotonic()
        def checkpoint(render_objective=None):
            path = args.output / f'checkpoint-{index:06d}.pt'
            if path.exists():
                return str(path)
            state = dict(curriculum=curriculum.state(), update=index, transitions=transitions,
                episodes=total_episodes, optimizer_steps=total_optimizer_steps,
                elapsed_seconds=time.monotonic()-started, eval_round=eval_round,
                best_rank=best, best_checkpoint=best_path)
            save_checkpoint(path, {'teacher': policy}, {'teacher': optimizer},
                {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
                dict(schema=SEQUENCE_SCHEMA, role='teacher', config=config, update=index,
                     model_sha256=runtime.manifest['model_sha256'], training_state=state,
                     render_objective=render_objective or curriculum.level))
            return str(path)
        if not args.resume_from:
            checkpoint()
            log.log(dict(update=0, elapsed_seconds=0., transitions=0, **{'curriculum/level':1}), step=0)
        (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(), status='running', phase='sequence-ppo')))
        (args.output/'run.json').write_text(json.dumps(dict(wandb_url=log.url, config=config), indent=2)+'\n')
        while time.monotonic()-started < args.train_seconds:
            tick = time.monotonic()
            rows, bootstrap, episodes = collector.collect(policy, gait, args.steps)
            metrics, steps_taken, reused = {}, 0, 0
            for epoch in range(3):
                result = update(policy, optimizer, rows, bootstrap, args.minibatch_worlds,
                    gamma=SEQUENCE_GAMMA, gae_lambda=args.gae_lambda, entropy_coef=args.entropy_coef,
                    check_replay=epoch==0, minibatch_updates=True, target_kl=.02)
                steps_taken += result['optimizer_steps']
                reused += result['optimized_transitions']
                metrics.update(result)
                if result['early_stop']:
                    break
            metrics.update(optimizer_steps=steps_taken, optimized_transitions=reused, optimizer_epochs=epoch+1)
            total_optimizer_steps += steps_taken
            metrics['ppo/total_optimizer_steps'] = total_optimizer_steps
            metrics['ppo/data_reuse'] = reused / (args.worlds * args.steps)
            for kind in ('task_reward', 'shaping_reward'):
                metrics['reward/'+kind] = float(torch.stack([r[kind] for r in rows]).mean())
            metrics['reward/task_mean'] = metrics['reward/task_reward']
            metrics['reward/shaping_mean'] = metrics['reward/shaping_reward']
            metrics['reward/completion_mean'] = float(torch.stack([r['completion_reward'] for r in rows]).float().mean())
            for kind in ('time_reward', 'failure_reward'):
                metrics['reward/'+kind] = float(torch.stack([r[kind] for r in rows]).mean())
            for name in CHECKPOINTS:
                metrics['stage_event/'+name] = float(torch.stack([r['milestone_rewards'][name] for r in rows]).sum())
                metrics['stage_live/'+name] = float(torch.stack([r['live_scores'][name] for r in rows]).mean())
                metrics['stage_reward/'+name] = float(torch.stack([r['stage_rewards'][name] for r in rows]).mean())
                metrics['stage_progress/'+name] = float(torch.stack([r['stage_scores'][name] for r in rows]).mean())
                metrics['stage_gain/'+name] = float(torch.stack([r['stage_gains'][name] for r in rows]).mean())
                metrics['stage_loss/'+name] = float(torch.stack([r['stage_losses'][name] for r in rows]).mean())
            for i, name in enumerate(('shoulder_0','shoulder_1','elbow_0','elbow_1','wrist_0','wrist_1','jaw')):
                actions = torch.stack([r['raw'][:, i].tanh() for r in rows])
                metrics['action_mean/'+name] = float(actions.mean())
                metrics['action_spread/'+name] = float(actions.std(unbiased=False))
                metrics['action_saturation/'+name] = float((actions.abs() > .95).float().mean())
                metrics['joint_speed/'+name] = float(torch.stack([r['joint_velocity'][:, i].abs() for r in rows]).mean())
            del rows, bootstrap
            index += 1
            transitions += args.worlds * args.steps
            total_episodes += len(episodes)
            metrics.update(update=index, transitions=transitions, completed_episodes=total_episodes,
                training_transitions_per_second=args.worlds*args.steps/(time.monotonic()-tick),
                **{'curriculum/level': curriculum.level, 'curriculum/promotion_streak': curriculum.streak})
            metrics.update({'episode/'+k: v for k, v in episode_metrics(episodes).items()})
            current = [e for e in episodes if e.get('objective') == curriculum.level]
            metrics.update({'current_objective/'+k: v for k, v in episode_metrics(current).items()})
            render_objective = curriculum.level
            if time.monotonic()-last_eval >= args.eval_every_seconds:
                eval_round += 1
                evaluations = [evaluate(target, policy, gait, args, curriculum.level,
                    args.seed + 10000 + 100*eval_round + i) for i, target in enumerate(runtimes[1:])]
                validation = [evaluations[0], evaluate(runtimes[1], policy, gait, args,
                    curriculum.level, args.seed + 10000 + 100*eval_round + 17)]
                training_result = {k: (sum(r[k] for r in validation) if k == 'episodes' or k.startswith('entries/')
                    else sum(r[k] for r in validation)/len(validation)) for k in validation[0]}
                evaluations[0] = training_result
                metrics['curriculum/validation_success'] = min(r['prefix_success'] for r in validation)
                for trial, result in enumerate(validation):
                    metrics[f'curriculum/validation_trial_{trial}'] = result['prefix_success']
                for label, result in zip(('training_evaluation/', 'evaluation/'), evaluations):
                    metrics.update({label+k: v for k, v in result.items()})
                metrics['evaluation/objective'] = curriculum.level
                # Full harvest remains distinct from current-prefix completion.
                candidate_rank = graph_rank(training_result) + graph_rank(evaluations[1])
                selected = best is None or candidate_rank > best
                if selected:
                    best = candidate_rank
                    best_path = str(args.output / f'checkpoint-{index:06d}.pt')
                    save_acceptance(args.output/'accepted.json', best_path,
                        dict(training_result, **{'heldout/'+k:v for k,v in evaluations[-1].items()}), {})
                metrics['acceptance/checkpoint'] = best_path
                metrics['acceptance/promoted'] = int(selected)
                metrics['acceptance/rollback'] = 0
                metrics['curriculum/evaluations'] = eval_round
                if curriculum.consider(validation):
                    # A batch boundary: discard unfinished old-objective episodes;
                    # retain the learned policy, critic and Adam states.
                    collector = HarvestCollector(runtime, role='teacher', reward_profile=SEQUENCE_PROFILE,
                        curriculum_stage=curriculum.level, stall_seconds=args.stall_seconds,
                        max_episode_seconds=args.max_episode_seconds, rehearse=True)
                    with (args.output/'curriculum-events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(update=index, level=curriculum.level,
                            elapsed_seconds=time.monotonic()-started, evaluations=evaluations))+'\n')
                metrics['curriculum/level'] = curriculum.level
                metrics['curriculum/promotion_streak'] = curriculum.streak
                last_eval = time.monotonic()
            metrics['elapsed_seconds'] = time.monotonic()-started
            checkpoint(render_objective)
            log.log(metrics, step=index)
            print(json.dumps(metrics), flush=True)
            if (args.output/'pause-request.json').exists():
                (args.output/'pause-request.json').unlink()
                (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(), status='paused')))
                log.finish()
                return
        report = dict(config=config, updates=index, transitions=transitions, completed_episodes=total_episodes,
            final_checkpoint=checkpoint(), best_checkpoint=best_path, wandb_url=log.url,
            curriculum=curriculum.state(), elapsed_seconds=time.monotonic()-started,
            numerical_failures=sum(sum(target.check()['flags']) for target in runtimes))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(), status='completed')))
    except BaseException as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc))))
        log.finish(success=False)
        raise
    log.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'eval-scene', 'gait-checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--wandb-run-id')
    p.add_argument('--worlds', type=int, default=1024)
    p.add_argument('--eval-worlds', type=int, default=32)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--minibatch-worlds', type=int, default=128)
    p.add_argument('--train-seconds', type=float, default=28800)
    p.add_argument('--eval-every-seconds', type=float, default=120)
    p.add_argument('--max-episode-seconds', type=float, default=15)
    p.add_argument('--stall-seconds', type=float, default=4)
    p.add_argument('--reset-jitter-rad', type=float, default=.04)
    p.add_argument('--arm-speed-rad-s', type=float, default=.5)
    p.add_argument('--initial-jaw-rad', type=float, default=-1.4)
    p.add_argument('--jaw-rate-rad-s', type=float, default=1.)
    p.add_argument('--fruit-damping', type=float, default=0.)
    p.add_argument('--fruit-jitter-m', type=float, nargs=3, default=(0., 0., 0.), metavar=('DX', 'DY', 'DZ'),
                   help='Half-extents of the per-world kiwi position randomisation (world frame); prefer --fruit-reach-fraction')
    p.add_argument('--fruit-reach-fraction', type=float, nargs=2, default=(0., 0.), metavar=('MIN', 'MAX'),
                   help='Randomise the kiwi horizontally (height fixed) so shoulder distance is within these fractions of full arm extension')
    p.add_argument('--fruit-sector-deg', type=float, default=70., help='Half-angle of the horizontal sector in front of the shoulder')
    p.add_argument('--max-level', type=int, default=3, help='Highest curriculum objective to train (carry and deposit are scripted)')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--gae-lambda', type=float, default=.99)
    p.add_argument('--entropy-coef', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb-mode', choices=('disabled','online','offline'), default='online')
    a = p.parse_args()
    if not (1 <= a.worlds <= 4096 and 32 <= a.eval_worlds <= 256 and 2 <= a.steps <= 256
            and 1 <= a.minibatch_worlds <= a.worlds and 1 <= a.train_seconds <= 28800
            and a.eval_every_seconds >= 1 and 0 <= a.reset_jitter_rad <= .5
            and 1 <= a.stall_seconds < a.max_episode_seconds <= 60 and 0 < a.arm_speed_rad_s <= 2.5 and -1.5708 <= a.initial_jaw_rad <= 0 and 0 < a.jaw_rate_rad_s <= 5 and 0 <= a.fruit_damping <= .05 and 1 <= a.max_level <= 5
            and 0 < a.learning_rate <= 1e-3 and 0 <= a.gae_lambda <= 1 and 0 <= a.entropy_coef <= .1):
        p.error('Invalid training configuration')
    if a.resume_from and a.wandb_mode == 'online' and not a.wandb_run_id and a.resume_from.resolve().parent == a.output.resolve():
        p.error('Online resume in place requires the existing W&B run ID')
    if a.wandb_run_id and not a.resume_from:
        p.error('A W&B run ID requires resume')
    import torch, warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        run(a)


if __name__ == '__main__':
    main()
