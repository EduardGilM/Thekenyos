"""Batched PPO for the TK-RL-003 task curriculum on rigid fruit + RGBD/R84."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_physical_smoke import build_policy as build_compact_policy
from treesim.kiwi_rl.training_log import TrainingLog, add_training_log_args
from treesim.kiwi_rl.training_monitor import LiveDashboard, add_monitor_args, spawn_progress_video
from treesim.kiwi_rl.curriculum import (
    evaluate_skills, next_stage, promotion_ready, sample_world_skills,
    stage_named, summarise_stage,
)


def build_policy():
    # 3 base commands (N3) + 7 arm/jaw (M3). Stages 01–03 zero the base in-env.
    return build_compact_policy(action_dim=10)


def _split_action(raw):
    applied = raw.tanh()
    return applied[:, :3], applied[:, 3:10]


def collect(runtime, policy, gait, steps, camera_every, *, deterministic=False, carry=None,
            reset_all=False):
    import torch
    from treesim.kiwi_rl.ppo import tanh_logprob
    if carry is None:
        carry = {}
    if reset_all or carry.get('memory') is None:
        runtime.reset()
        carry['memory'] = torch.zeros(runtime.worlds, 64, device='cuda:0')
        carry['reset'] = torch.ones(runtime.worlds, device='cuda:0', dtype=torch.bool)
        carry['rgbd'] = runtime.pixels().clone()
    memory, reset, rgbd = carry['memory'], carry['reset'], carry['rgbd']
    rows = []
    for index in range(steps):
        r84 = runtime.observe().clone()
        if index % camera_every == 0 or rgbd is None:
            rgbd = runtime.pixels().clone()
        with torch.no_grad():
            memory = memory * (~reset)[:, None]
            mean, logstd, value, memory = policy(rgbd, r84, memory)
            raw = mean if deterministic else mean + logstd.exp() * torch.randn_like(mean)
            logp = tanh_logprob(raw, mean, logstd)
            base, arm = _split_action(raw)
            runtime.set_base_commands(base.contiguous())
            runtime.set_gait_actions(gait(r84))
            _, reward, done, info = runtime.step(arm.contiguous())
            physical = info['success'].bool() | info['failed'].bool() | info['fallen']
            timeout = info['timed_out']
            rows.append(dict(rgbd=rgbd, r84=r84, raw=raw, logp=logp, value=value,
                reward=reward.clone(), terminated=physical.clone(), truncated=timeout.clone(),
                reset=reset.clone(), distance=info['distance_m'].clone(),
                success=info['success'].clone(), detached=info['detached'].clone(),
                grasped=info['grasped'].clone(), retained_detach=info['retained_detach'].clone()))
            reset = done.clone()
            if bool(reset.any()):
                runtime.reset(reset)
                rgbd = runtime.pixels().clone()
    with torch.no_grad():
        _, _, bootstrap, _ = policy(runtime.pixels(), runtime.observe(), memory * (~reset)[:, None])
    runtime.check()
    carry['memory'], carry['reset'], carry['rgbd'] = memory, reset, rgbd
    return rows, bootstrap, carry


def update(policy, optimizer, rows, bootstrap, minibatch_worlds=512, entropy_coef=0.005,
           gamma=0.9996, epochs=1):
    import torch
    from treesim.kiwi_rl.ppo import compute_gae_torch, tanh_logprob
    rewards = torch.stack([r['reward'] for r in rows])
    values = torch.stack([r['value'] for r in rows])
    ended = torch.stack([r['terminated'] for r in rows])
    if 'truncated' in rows[0]:
        truncated = torch.stack([r['truncated'] for r in rows])
    else:
        truncated = torch.zeros_like(ended)
    truncated[-1] = True
    advantages = compute_gae_torch(rewards, values, torch.cat((values[1:], bootstrap[None])),
                                   ended, truncated, gamma, .95)
    returns = (advantages + values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = min(minibatch_worlds, rewards.shape[1])
    metrics = []
    if not isinstance(epochs, int) or not 1 <= epochs <= 8:
        raise ValueError('epochs must be an integer in [1, 8]')
    if not np_finite(entropy_coef) or entropy_coef < 0:
        raise ValueError('entropy_coef must be finite and >= 0')
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, rewards.shape[1], batch_size):
            sl = slice(start, start + batch_size)
            memory = torch.zeros_like(rows[0]['r84'][sl, :64])
            logps, predictions, entropies = [], [], []
            for row in rows:
                memory = memory * (~row['reset'][sl])[:, None]
                mean, logstd, value, memory = policy(row['rgbd'][sl], row['r84'][sl], memory)
                new_logp = tanh_logprob(row['raw'][sl], mean, logstd)
                logps.append(new_logp)
                predictions.append(value)
                entropies.append(-new_logp)
            logratio = torch.stack(logps) - torch.stack([r['logp'][sl] for r in rows])
            ratio = logratio.exp()
            kl = ((ratio - 1.) - logratio).mean()
            if not torch.isfinite(kl):
                raise RuntimeError('Nonfinite PPO divergence')
            if float(kl.detach()) > .03:
                raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
            actor = -torch.minimum(ratio * advantages[:, sl], ratio.clamp(.8, 1.2) * advantages[:, sl]).mean()
            critic = .5 * (torch.stack(predictions) - returns[:, sl]).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = actor + .5 * critic - entropy_coef * entropy
            (loss * (rows[0]['reward'][sl].numel() / rewards.shape[1])).backward()
            metrics.append((float(loss.detach()), float(kl.detach()), float(entropy.detach())))
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
    return dict(loss=sum(m[0] for m in metrics)/len(metrics), kl=max(m[1] for m in metrics),
                entropy=sum(m[2] for m in metrics)/len(metrics),
                logstd_mean=float(policy.logstd.detach().mean()),
                grad_norm=float(grad), minibatches=len(metrics), optimized_transitions=int(rewards.numel()*epochs),
                reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)))


def np_finite(value):
    import math
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate(runtime, policy, gait, steps, camera_every, *, stage=None):
    import torch
    if stage is not None:
        runtime.configure_skills(evaluate_skills(stage, runtime.worlds))
    packed = collect(runtime, policy, gait, steps, camera_every, deterministic=True, reset_all=True)
    rows = packed[0]
    distances = torch.stack([r['distance'] for r in rows])
    successes = torch.stack([r['success'] for r in rows])
    grasped = torch.stack([r['grasped'] for r in rows]) if 'grasped' in rows[0] else successes
    detached = torch.stack([r['detached'] for r in rows]) if 'detached' in rows[0] else successes
    worlds = int(successes.shape[1])
    world_success = (successes.max(dim=0).values > 0).float()
    return {
        'evaluation/final_distance_m': float(distances[-1].mean()),
        'evaluation/closest_distance_m': float(distances.min()),
        'evaluation/mean_closest_distance_m': float(distances.min(dim=0).values.mean()),
        'evaluation/terminal_transitions': int(torch.stack([r['terminated'] for r in rows]).sum()),
        'evaluation/harvest_successes': int(successes.sum()),
        'evaluation/success_rate': float(world_success.mean()),
        'evaluation/detach_rate': float((detached.max(dim=0).values > 0).float().mean()),
        'evaluation/grasp_rate': float((grasped.max(dim=0).values > 0).float().mean()),
        'evaluation/worlds': worlds,
    }


def apply_stage(runtime, stage, rng, *, evaluate_only=False):
    skills = evaluate_skills(stage, runtime.worlds) if evaluate_only else sample_world_skills(stage, runtime.worlds, rng)
    runtime.configure_skills(skills)
    return skills


def run(args):
    import numpy as np
    import torch
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import load_checkpoint, save_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    numpy_rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    stage = stage_named(args.stage)
    manifest = json.loads((args.scene / 'manifest.json').read_text())
    config = dict(vars(args), approximations=manifest['approximation'],
                  scope='TK-RL-003 task curriculum on the rigid fast runtime; not field harvest',
                  curriculum=summarise_stage(stage),
                  training_ready=False,
                  rates={'physics_hz': manifest['numerical_profile']['frequency_hz'],
                         'policy_hz':50, 'camera_hz':50/args.camera_every})
    config = {k: str(v) if isinstance(v, Path) else v for k,v in config.items()}
    log = TrainingLog(args.output, config, wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name, upload_checkpoints=args.upload_checkpoints)
    dashboard = LiveDashboard(args.output, hub=args.monitor_hub)
    try:
        runtime = FastRuntime(args.scene, worlds=args.worlds, camera='hand_color_sensor',
                              nconmax=args.nconmax, njmax=args.njmax)
        gait = load_gait_artifact(args.gait_checkpoint, precision_profile='cuda-fp32').to('cuda:0').eval()
        policy = build_policy().to('cuda:0')
        if args.initialize_from:
            load_checkpoint(args.initialize_from, {'student':policy}, None,
                            expected_meta={'camera':'hand_color_sensor'})
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        apply_stage(runtime, stage, numpy_rng)
        collect(runtime, policy, gait, 4, args.camera_every, reset_all=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        reports = []
        initial_checkpoint = args.output / 'checkpoint-0000.pt'
        save_checkpoint(initial_checkpoint, {'student':policy}, {'student':optimizer},
            {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            dict(schema='fast-curriculum-rgbd-r84/v1', camera='hand_color_sensor',
                 model_sha256=manifest['model_sha256'], config=config, completed_updates=0,
                 curriculum_stage=stage.name))
        apply_stage(runtime, stage, numpy_rng, evaluate_only=True)
        baseline = evaluate(runtime, policy, gait, args.steps, args.camera_every, stage=stage)
        baseline.update(curriculum_stage=stage.name, curriculum_index=stage.index)
        log.log(baseline, step=0)
        dashboard.refresh()
        if args.video_every:
            spawn_progress_video(initial_checkpoint, dashboard.video_path(0),
                                 steps=args.video_steps, camera_every=args.camera_every)
        evaluations = [dict(update=0, **baseline)]
        eval_success_rates = []
        best_distance = baseline['evaluation/mean_closest_distance_m']
        best_checkpoint = str(initial_checkpoint)
        carry = {}
        apply_stage(runtime, stage, numpy_rng)
        for iteration in range(args.updates):
            began = time.monotonic()
            rows, bootstrap, carry = collect(runtime, policy, gait, args.steps, args.camera_every,
                                             carry=carry, reset_all=False)
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - began
            metrics = update(policy, optimizer, rows, bootstrap, args.minibatch_worlds,
                             entropy_coef=args.entropy_coef, gamma=args.gamma, epochs=args.ppo_epochs)
            torch.cuda.synchronize()
            duration = time.monotonic() - began
            metrics.update(update=iteration+1, transitions=(iteration+1)*args.steps*args.worlds,
                rollout_transitions_per_second=args.steps*args.worlds/rollout_seconds,
                training_transitions_per_second=args.steps*args.worlds/duration,
                rollout_seconds=rollout_seconds, update_seconds=duration-rollout_seconds,
                distance_final_m=float(rows[-1]['distance'].mean()),
                distance_closest_m=float(torch.stack([r['distance'] for r in rows]).min()),
                distance_mean_closest_m=float(torch.stack([r['distance'] for r in rows]).min(dim=0).values.mean()),
                terminal_transitions=int(torch.stack([r['terminated'] for r in rows]).sum()),
                harvest_successes=int(torch.stack([r['success'] for r in rows]).sum()),
                grasp_events=int((torch.stack([r['grasped'] for r in rows]).max(dim=0).values > 0).sum()),
                detach_events=int((torch.stack([r['detached'] for r in rows]).max(dim=0).values > 0).sum()),
                curriculum_stage=stage.name, curriculum_index=stage.index,
                guidance_weight=stage.guidance_weight,
                torch_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            checkpoint = args.output / f'checkpoint-{iteration+1:04d}.pt'
            meta = dict(schema='fast-curriculum-rgbd-r84/v1', camera='hand_color_sensor',
                        model_sha256=manifest['model_sha256'], config=config, completed_updates=iteration+1,
                        curriculum_stage=stage.name)
            save_checkpoint(checkpoint, {'student':policy}, {'student':optimizer},
                {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}, meta)
            del rows, bootstrap
            if (iteration+1) % args.eval_every == 0 or iteration+1 == args.updates:
                apply_stage(runtime, stage, numpy_rng, evaluate_only=True)
                eval_metrics = evaluate(runtime, policy, gait, args.steps, args.camera_every, stage=stage)
                evaluations.append(dict(update=iteration+1, **eval_metrics))
                metrics.update(eval_metrics)
                eval_success_rates.append(eval_metrics['evaluation/success_rate'])
                distance = eval_metrics['evaluation/mean_closest_distance_m']
                if distance < best_distance:
                    best_distance, best_checkpoint = distance, str(checkpoint)
                if promotion_ready(eval_success_rates, stage):
                    nxt = next_stage(stage)
                    metrics['curriculum_promoted'] = True
                    if nxt is not None:
                        stage = nxt
                        config['curriculum'] = summarise_stage(stage)
                        eval_success_rates = []
                        carry = {}
                        apply_stage(runtime, stage, numpy_rng)
                        metrics['curriculum_stage'] = stage.name
                        metrics['curriculum_index'] = stage.index
                else:
                    apply_stage(runtime, stage, numpy_rng)
                    carry = {}
            log.log(metrics, step=iteration+1)
            dashboard.refresh()
            if args.video_every and (iteration + 1) % args.video_every == 0:
                spawn_progress_video(checkpoint, dashboard.video_path(iteration + 1),
                                     steps=args.video_steps, camera_every=args.camera_every)
            reports.append(metrics)
            print(json.dumps(metrics), flush=True)
        log.log_checkpoint(checkpoint)
        report = dict(config=config, updates=reports, evaluation=evaluations[-1],
                      baseline=baseline, evaluations=evaluations, best_reach_checkpoint=best_checkpoint,
                      best_mean_closest_distance_m=best_distance, curriculum=summarise_stage(stage),
                      training_ready=False,
                      elapsed_seconds=time.monotonic()-start, wandb_url=log.url, numerical=runtime.check())
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        dashboard.refresh()
    except BaseException:
        dashboard.refresh()
        log.finish(success=False)
        raise
    log.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--gait-checkpoint', type=Path, required=True)
    p.add_argument('--initialize-from', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worlds', type=int, default=4096)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--minibatch-worlds', type=int, default=512)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--camera-every', type=int, default=2)
    p.add_argument('--nconmax', type=int, default=128)
    p.add_argument('--njmax', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage', default='deposit_pixels',
                   help='TK-RL-003 curriculum stage to start from')
    p.add_argument('--entropy-coef', type=float, default=0.005)
    p.add_argument('--gamma', type=float, default=0.9996)
    p.add_argument('--ppo-epochs', type=int, default=2)
    add_training_log_args(p)
    add_monitor_args(p)
    a = p.parse_args()
    if not 1 <= a.eval_every <= 10000 or not 1 <= a.minibatch_worlds <= 1024 or not 2 <= a.steps <= 256 or not 1 <= a.updates <= 10000 or not 1 <= a.camera_every <= 5:
        p.error('Invalid steps, updates or camera interval')
    if not 0 <= a.video_every <= 10000 or not 8 <= a.video_steps <= 512:
        p.error('Invalid video-every or video-steps')
    if a.stage not in {s.name for s in __import__('treesim.kiwi_rl.curriculum', fromlist=['STAGES']).STAGES}:
        p.error(f'Unknown curriculum stage {a.stage}')
    if not 0 <= a.entropy_coef <= 0.1 or not 0.9 <= a.gamma <= 1.0 or not 1 <= a.ppo_epochs <= 8:
        p.error('Invalid PPO entropy, gamma or epochs')
    run(a)


def run(args):
    import torch
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import load_checkpoint, save_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.scene / 'manifest.json').read_text())
    config = dict(vars(args), approximations=manifest['approximation'],
                  scope='batched reaching; full harvest success not established',
                  rates={'physics_hz': manifest['numerical_profile']['frequency_hz'],
                         'policy_hz':50, 'camera_hz':50/args.camera_every})
    config = {k: str(v) if isinstance(v, Path) else v for k,v in config.items()}
    log = TrainingLog(args.output, config, wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name, upload_checkpoints=args.upload_checkpoints)
    dashboard = LiveDashboard(args.output, hub=args.monitor_hub)
    try:
        runtime = FastRuntime(args.scene, worlds=args.worlds, camera='hand_color_sensor',
                              nconmax=args.nconmax, njmax=args.njmax)
        gait = load_gait_artifact(args.gait_checkpoint, precision_profile='cuda-fp32').to('cuda:0').eval()
        policy = build_policy().to('cuda:0')
        if args.initialize_from:
            load_checkpoint(args.initialize_from, {'student':policy}, None,
                            expected_meta={'camera':'hand_color_sensor'})
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # Compilation and warmup are outside measured rollout throughput.
        collect(runtime, policy, gait, 4, args.camera_every)
        torch.cuda.synchronize()
        start = time.monotonic()
        reports = []
        initial_checkpoint = args.output / 'checkpoint-0000.pt'
        save_checkpoint(initial_checkpoint, {'student':policy}, {'student':optimizer},
            {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            dict(schema='fast-reach-rgbd-r84/v1', camera='hand_color_sensor',
                 model_sha256=manifest['model_sha256'], config=config, completed_updates=0))
        baseline = evaluate(runtime, policy, gait, args.steps, args.camera_every)
        log.log(baseline, step=0)
        dashboard.refresh()
        if args.video_every:
            spawn_progress_video(initial_checkpoint, dashboard.video_path(0),
                                 steps=args.video_steps, camera_every=args.camera_every)
        evaluations = [dict(update=0, **baseline)]
        best_distance = baseline['evaluation/mean_closest_distance_m']
        best_checkpoint = str(initial_checkpoint)
        for iteration in range(args.updates):
            began = time.monotonic()
            rows, bootstrap = collect(runtime, policy, gait, args.steps, args.camera_every)
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - began
            metrics = update(policy, optimizer, rows, bootstrap, args.minibatch_worlds)
            torch.cuda.synchronize()
            duration = time.monotonic() - began
            metrics.update(update=iteration+1, transitions=(iteration+1)*args.steps*args.worlds,
                rollout_transitions_per_second=args.steps*args.worlds/rollout_seconds,
                training_transitions_per_second=args.steps*args.worlds/duration,
                rollout_seconds=rollout_seconds, update_seconds=duration-rollout_seconds,
                distance_final_m=float(rows[-1]['distance'].mean()),
                distance_closest_m=float(torch.stack([r['distance'] for r in rows]).min()),
                distance_mean_closest_m=float(torch.stack([r['distance'] for r in rows]).min(dim=0).values.mean()),
                terminal_transitions=int(torch.stack([r['terminated'] for r in rows]).sum()),
                harvest_successes=int(torch.stack([r['success'] for r in rows]).sum()),
                torch_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            checkpoint = args.output / f'checkpoint-{iteration+1:04d}.pt'
            meta = dict(schema='fast-reach-rgbd-r84/v1', camera='hand_color_sensor',
                        model_sha256=manifest['model_sha256'], config=config, completed_updates=iteration+1)
            save_checkpoint(checkpoint, {'student':policy}, {'student':optimizer},
                {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}, meta)
            del rows, bootstrap
            if (iteration+1) % args.eval_every == 0 or iteration+1 == args.updates:
                eval_metrics = evaluate(runtime, policy, gait, args.steps, args.camera_every)
                evaluations.append(dict(update=iteration+1, **eval_metrics))
                metrics.update(eval_metrics)
                distance = eval_metrics['evaluation/mean_closest_distance_m']
                if distance < best_distance:
                    best_distance, best_checkpoint = distance, str(checkpoint)
            log.log(metrics, step=iteration+1)
            dashboard.refresh()
            if args.video_every and (iteration + 1) % args.video_every == 0:
                spawn_progress_video(checkpoint, dashboard.video_path(iteration + 1),
                                     steps=args.video_steps, camera_every=args.camera_every)
            reports.append(metrics)
            print(json.dumps(metrics), flush=True)
        log.log_checkpoint(checkpoint)
        report = dict(config=config, updates=reports, evaluation=eval_metrics,
                      baseline=baseline, evaluations=evaluations, best_reach_checkpoint=best_checkpoint,
                      best_mean_closest_distance_m=best_distance,
                      elapsed_seconds=time.monotonic()-start, wandb_url=log.url, numerical=runtime.check())
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        dashboard.refresh()
    except BaseException:
        dashboard.refresh()
        log.finish(success=False)
        raise
    log.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--gait-checkpoint', type=Path, required=True)
    p.add_argument('--initialize-from', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worlds', type=int, default=4096)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--minibatch-worlds', type=int, default=512)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--camera-every', type=int, default=2)
    p.add_argument('--nconmax', type=int, default=128)
    p.add_argument('--njmax', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    add_training_log_args(p)
    add_monitor_args(p)
    a = p.parse_args()
    if not 1 <= a.eval_every <= 10000 or not 1 <= a.minibatch_worlds <= 1024 or not 2 <= a.steps <= 256 or not 1 <= a.updates <= 10000 or not 1 <= a.camera_every <= 5:
        p.error('Invalid steps, updates or camera interval')
    if not 0 <= a.video_every <= 10000 or not 8 <= a.video_steps <= 512:
        p.error('Invalid video-every or video-steps')
    run(a)


if __name__ == '__main__':
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        main()
