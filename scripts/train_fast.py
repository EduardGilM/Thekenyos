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
    EASY_PRESET, apply_easy_preset, apply_speedrun_preset, easy_start_far_frac,
    easy_teacher_mix,
    evaluate_skills, evaluation_horizon_steps, fruit_block_reason,
    idle_locomotion_mask, next_stage, promotion_ready, sample_world_skills,
    stage_named, summarise_stage,
)


def build_policy():
    # 3 base commands (N3) + 7 arm/jaw (M3). Stages 01–03 zero the base in-env.
    return build_compact_policy(action_dim=10)


def _split_action(raw):
    applied = raw.tanh()
    return applied[:, :3], applied[:, 3:10]


def policy_dim_mask(stage, device, enabled=True):
    """Zero N3 log-prob/entropy while the stage holds the chassis."""
    import torch
    if not enabled:
        return None
    mask = idle_locomotion_mask(10, stage.allow_locomotion)
    if mask is None:
        return None
    return torch.as_tensor(mask, device=device)


def should_persist_checkpoint(update_index, *, updates, eval_every, checkpoint_every,
                              video_every, promoted):
    """Write weights on eval, video, promotion, last update, or the checkpoint stride."""
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 1:
        raise ValueError('update_index must be a positive integer')
    for name, value in (('updates', updates), ('eval_every', eval_every),
                        ('checkpoint_every', checkpoint_every), ('video_every', video_every)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer')
    if updates < 1 or eval_every < 1 or checkpoint_every < 1:
        raise ValueError('updates, eval_every and checkpoint_every must be >= 1')
    if update_index == updates or promoted:
        return True
    if update_index % checkpoint_every == 0 or update_index % eval_every == 0:
        return True
    return bool(video_every) and update_index % video_every == 0


def apply_speedrun_cli(args, *, video_default=10):
    """Apply the wall-clock preset. Explicit --video-every, including 0, wins."""
    if not getattr(args, 'speedrun', False):
        return args
    preset = apply_speedrun_preset({})
    args.eval_every = preset['eval_every']
    args.checkpoint_every = preset['checkpoint_every']
    args.entropy_coef = preset['entropy_coef']
    args.eval_profile = preset['eval_profile']
    args.mask_idle_locomotion = preset['mask_idle_locomotion']
    if args.video_every == video_default:
        args.video_every = preset['video_every']
    return args


def apply_easy_cli(args):
    """Privileged deposit facilitation. Explicit --teacher-mix, including 0, wins."""
    if not getattr(args, 'easy', False):
        if getattr(args, 'teacher_mix', None) is None:
            args.teacher_mix = 0.0
        if getattr(args, 'shaping_coef', None) is None:
            args.shaping_coef = EASY_PRESET['default_shaping_coef']
        return args
    preset = apply_easy_preset({})
    if getattr(args, 'teacher_mix', None) is None:
        args.teacher_mix = preset['teacher_mix']
    if getattr(args, 'shaping_coef', None) is None:
        args.shaping_coef = preset['shaping_coef']
    return args


def mix_privileged_actions(raw, teacher_applied, mix_mask):
    """Replace pre-tanh arm samples with atanh(teacher) where mix_mask is true.

    The compact actor is 10-D (N3 + arm). The deposit teacher is 7-D arm/jaw
    only and must not overwrite locomotion commands.
    """
    import torch
    if raw.ndim != 2 or teacher_applied.ndim != 2:
        raise ValueError('raw and teacher actions must be rank-2')
    if mix_mask.shape != raw.shape[:1] or teacher_applied.shape[0] != raw.shape[0]:
        raise ValueError('mix_mask and teacher actions must be one row per world')
    applied = teacher_applied.clamp(-0.999, 0.999)
    teacher_raw = torch.atanh(applied)
    if teacher_applied.shape[-1] == raw.shape[-1]:
        return torch.where(mix_mask[:, None], teacher_raw, raw)
    if teacher_applied.shape[-1] == 7 and raw.shape[-1] == 10:
        mixed = raw.clone()
        mixed[:, 3:] = torch.where(mix_mask[:, None], teacher_raw, raw[:, 3:])
        return mixed
    raise ValueError('teacher actions must be 7 (arm) or match the student sample shape')


def collect(runtime, policy, gait, steps, camera_every, *, deterministic=False, carry=None,
            reset_all=False, dim_mask=None, teacher_mix=0.0):
    import torch
    from treesim.kiwi_rl.ppo import tanh_logprob
    if carry is None:
        carry = {}
    mix_prob = float(teacher_mix)
    if not np_finite(mix_prob) or not 0.0 <= mix_prob <= 1.0:
        raise ValueError('teacher_mix must be finite in [0, 1]')
    if mix_prob > 0.0 and deterministic:
        raise ValueError('privileged teacher mix stays off during deterministic eval')
    if reset_all or carry.get('memory') is None:
        runtime.reset()
        carry['memory'] = torch.zeros(runtime.worlds, 64, device='cuda:0')
        carry['reset'] = torch.ones(runtime.worlds, device='cuda:0', dtype=torch.bool)
        carry['rgbd'] = runtime.pixels().clone()
        carry.pop('teacher_mask', None)
    memory, reset, rgbd = carry['memory'], carry['reset'], carry['rgbd']
    memory0 = memory.clone()
    recovered_total = 0
    overflow_total = 0
    nonfinite_total = 0
    teacher_used = 0
    rows = []
    for index in range(steps):
        r84 = runtime.observe().clone()
        if index % camera_every == 0 or rgbd is None:
            rgbd = runtime.pixels().clone()
        with torch.no_grad():
            memory = memory * (~reset)[:, None]
            mean, logstd, value, memory = policy(rgbd, r84, memory)
            raw = mean if deterministic else mean + logstd.exp() * torch.randn_like(mean)
            if mix_prob > 0.0:
                # Persist the teacher per world across the chunk so a deposit
                # is not interrupted by per-step Bernoulli flicker. Resample
                # only on reset. mix=1.0 keeps every world on the teacher.
                mask = carry.get('teacher_mask')
                if mask is None:
                    mask = torch.ones(runtime.worlds, dtype=torch.bool, device=raw.device)
                if mix_prob >= 1.0:
                    mask = torch.ones(runtime.worlds, dtype=torch.bool, device=raw.device)
                else:
                    mask = torch.where(
                        reset, torch.rand(runtime.worlds, device=raw.device) < mix_prob, mask)
                carry['teacher_mask'] = mask
                raw = mix_privileged_actions(raw, runtime.privileged_deposit_action(), mask)
                teacher_used += int(mask.sum().item())
            logp = tanh_logprob(raw, mean, logstd, dim_mask=dim_mask)
            base, arm = _split_action(raw)
            runtime.set_base_commands(base.contiguous())
            runtime.set_gait_actions(gait(r84))
            _, reward, done, info = runtime.step(arm.contiguous())
            faults = runtime.drain_faults()
            recovered_total += faults['recovered_worlds']
            overflow_total += faults['overflow_worlds']
            nonfinite_total += int(faults.get('nonfinite_worlds', 0))
            if faults['recovered_worlds']:
                reward = reward.clone()
                reward[faults['mask']] = 0
                done = done | faults['mask']
            physical = info['success'].bool() | info['failed'].bool() | info['fallen']
            timeout = info['timed_out']
            if faults['recovered_worlds']:
                timeout = timeout | faults['mask']
            row = dict(rgbd=rgbd, r84=r84, raw=raw, logp=logp, value=value,
                reward=reward.clone(), terminated=physical.clone(), truncated=timeout.clone(),
                reset=reset.clone(), distance=info['distance_m'].clone(),
                success=info['success'].clone(), detached=info['detached'].clone(),
                grasped=info['grasped'].clone(), retained_detach=info['retained_detach'].clone(),
                harvested=info['harvested'].clone() if 'harvested' in info else info['success'].clone())
            if 'basket_distance_m' in info:
                row['basket_distance'] = info['basket_distance_m'].clone()
            if 'basket_xy_m' in info:
                row['basket_xy'] = info['basket_xy_m'].clone()
            if 'fallen' in info:
                row['fallen'] = info['fallen'].clone()
            if 'failed' in info:
                row['failed'] = info['failed'].clone()
            if 'ground_contact' in info:
                row['ground_contact'] = info['ground_contact'].clone()
            if 'hand_load_N' in info:
                row['hand_load_N'] = info['hand_load_N'].clone()
            rows.append(row)
            if index == 0:
                rows[0]['memory0'] = memory0
            reset = done.clone()
            if bool(reset.any()):
                runtime.reset(reset)
                rgbd = runtime.pixels().clone()
    with torch.no_grad():
        _, _, bootstrap, _ = policy(runtime.pixels(), runtime.observe(), memory * (~reset)[:, None])
    carry['memory'], carry['reset'], carry['rgbd'] = memory, reset, rgbd
    carry['recovered_worlds'] = recovered_total
    carry['overflow_worlds'] = overflow_total
    carry['nonfinite_worlds'] = nonfinite_total
    carry['teacher_actions'] = teacher_used
    return rows, bootstrap, carry


def update(policy, optimizer, rows, bootstrap, minibatch_worlds=512, entropy_coef=0.005,
           gamma=0.9996, epochs=1, dim_mask=None):
    import torch
    from treesim.kiwi_rl.ppo import (
        compute_gae_torch, gaussian_entropy, tanh_gaussian_entropy, tanh_logprob,
    )
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
    if 'memory0' in rows[0]:
        memory_root = rows[0]['memory0']
    else:
        memory_root = torch.zeros(rows[0]['r84'].shape[0], 64, device=rows[0]['r84'].device,
                                  dtype=rows[0]['r84'].dtype)
    completed_epochs = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        stop_extra = False
        for start in range(0, rewards.shape[1], batch_size):
            sl = slice(start, start + batch_size)
            memory = memory_root[sl].clone()
            logps, predictions, entropies, gaussians = [], [], [], []
            for row in rows:
                memory = memory * (~row['reset'][sl])[:, None]
                mean, logstd, value, memory = policy(row['rgbd'][sl], row['r84'][sl], memory)
                new_logp = tanh_logprob(row['raw'][sl], mean, logstd, dim_mask=dim_mask)
                logps.append(new_logp)
                predictions.append(value)
                draw = mean + logstd.exp() * torch.randn_like(mean)
                entropies.append(tanh_gaussian_entropy(logstd, raw=draw, mu=mean, dim_mask=dim_mask))
                gaussians.append(gaussian_entropy(logstd, dim_mask=dim_mask))
            logratio = torch.stack(logps) - torch.stack([r['logp'][sl] for r in rows])
            ratio = logratio.exp()
            kl = ((ratio - 1.) - logratio).mean()
            if not torch.isfinite(kl):
                continue
            mismatch = float(kl.detach())
            if epoch == 0 and mismatch > .03:
                raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
            if epoch > 0 and mismatch > .03:
                optimizer.zero_grad(set_to_none=True)
                stop_extra = True
                break
            actor = -torch.minimum(ratio * advantages[:, sl], ratio.clamp(.8, 1.2) * advantages[:, sl]).mean()
            critic = .5 * (torch.stack(predictions) - returns[:, sl]).square().mean()
            entropy = torch.stack(entropies).mean()
            entropy_g = torch.stack(gaussians).mean()
            loss = actor + .5 * critic - entropy_coef * entropy
            (loss * (rows[0]['reward'][sl].numel() / rewards.shape[1])).backward()
            if dim_mask is None:
                action_dim = int(logstd.shape[-1])
            else:
                action_dim = int(dim_mask.to(dtype=logstd.dtype).sum().clamp(min=1).item())
            metrics.append((float(loss.detach()), float(kl.detach()), float(entropy.detach()),
                            float(entropy_g.detach()), action_dim))
        if stop_extra:
            break
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        completed_epochs += 1
    if not metrics:
        raise RuntimeError('PPO produced no minibatches')
    entropy = sum(m[2] for m in metrics) / len(metrics)
    entropy_g = sum(m[3] for m in metrics) / len(metrics)
    action_dim = max(m[4] for m in metrics)
    return dict(loss=sum(m[0] for m in metrics)/len(metrics), kl=max(m[1] for m in metrics),
                entropy=entropy, entropy_gaussian=entropy_g,
                entropy_per_dim=entropy / action_dim,
                entropy_kind='tanh_gaussian_differential_nats',
                logstd_mean=float(policy.logstd.detach().mean()),
                grad_norm=float(grad), minibatches=len(metrics),
                optimized_transitions=int(rewards.numel() * completed_epochs),
                ppo_epochs_completed=completed_epochs,
                reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)))


def np_finite(value):
    import math
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_mission(runtime, policy, gait, camera_every, *, stage, control_dt=0.02,
                     eval_profile='default'):
    """Deterministic eval over the stage horizon without stacking RGBD history."""
    import torch
    runtime.configure_skills(evaluate_skills(stage, runtime.worlds))
    horizon = evaluation_horizon_steps(stage, control_dt, profile=eval_profile)
    runtime.reset()
    worlds = runtime.worlds
    memory = torch.zeros(worlds, 64, device='cuda:0')
    reset = torch.ones(worlds, device='cuda:0', dtype=torch.bool)
    rgbd = runtime.pixels().clone()
    closest = torch.full((worlds,), float('inf'), device='cuda:0')
    closest_basket = torch.full((worlds,), float('inf'), device='cuda:0')
    final_distance = torch.zeros(worlds, device='cuda:0')
    final_basket = torch.zeros(worlds, device='cuda:0')
    success_any = torch.zeros(worlds, dtype=torch.bool, device='cuda:0')
    grasp_any = torch.zeros_like(success_any)
    detach_any = torch.zeros_like(success_any)
    harvested_peak = torch.zeros(worlds, device='cuda:0')
    required = torch.ones(worlds, device='cuda:0')
    terminals = 0
    recovered_total = 0
    overflow_total = 0
    nonfinite_total = 0
    for index in range(horizon):
        r84 = runtime.observe()
        if index % camera_every == 0:
            rgbd = runtime.pixels().clone()
        with torch.no_grad():
            memory = memory * (~reset)[:, None]
            mean, logstd, value, memory = policy(rgbd, r84, memory)
            base, arm = _split_action(mean)
            runtime.set_base_commands(base.contiguous())
            runtime.set_gait_actions(gait(r84))
            _, reward, done, info = runtime.step(arm.contiguous())
            faults = runtime.drain_faults()
            recovered_total += faults['recovered_worlds']
            overflow_total += faults['overflow_worlds']
            nonfinite_total += int(faults.get('nonfinite_worlds', 0))
            if faults['recovered_worlds']:
                done = done | faults['mask']
            dist = info['distance_m']
            closest = torch.minimum(closest, dist)
            final_distance = dist.clone()
            if 'basket_distance_m' in info:
                closest_basket = torch.minimum(closest_basket, info['basket_distance_m'])
                final_basket = info['basket_distance_m'].clone()
            success_any |= info['success'].bool()
            grasp_any |= info['grasped'].bool()
            detach_any |= info['detached'].bool()
            if 'harvested' in info:
                harvested_peak = torch.maximum(harvested_peak, info['harvested'].float())
                required = info['required_harvests'].float().clamp(min=1.0)
            physical = info['success'].bool() | info['failed'].bool() | info['fallen']
            timeout = info['timed_out']
            terminals += int((physical | timeout).sum())
            reset = done.clone()
            if bool(reset.any()):
                runtime.reset(reset)
                rgbd = runtime.pixels().clone()
    harvest_fraction = harvested_peak / required
    if not bool(torch.isfinite(closest_basket).all()):
        closest_basket = closest
        final_basket = final_distance
    return {
        'evaluation/horizon_s': float(horizon * control_dt),
        'evaluation/horizon_steps': int(horizon),
        'evaluation/final_distance_m': float(final_distance.mean()),
        'evaluation/closest_distance_m': float(closest.min()),
        'evaluation/mean_closest_distance_m': float(closest.mean()),
        'evaluation/final_basket_distance_m': float(final_basket.mean()),
        'evaluation/closest_basket_distance_m': float(closest_basket.min()),
        'evaluation/mean_closest_basket_distance_m': float(closest_basket.mean()),
        'evaluation/terminal_transitions': terminals,
        'evaluation/harvest_successes': int(success_any.sum()),
        'evaluation/success_rate': float(success_any.float().mean()),
        'evaluation/detach_rate': float(detach_any.float().mean()),
        'evaluation/grasp_rate': float(grasp_any.float().mean()),
        'evaluation/harvest_fraction': float(harvest_fraction.mean()),
        'evaluation/mean_harvested': float(harvested_peak.mean()),
        'evaluation/recovered_worlds': recovered_total,
        'evaluation/overflow_worlds': overflow_total,
        'evaluation/nonfinite_worlds': nonfinite_total,
        'evaluation/eval_profile': eval_profile,
        'evaluation/worlds': worlds,
    }


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
    n_fruits = len(manifest.get('fruits') or [])
    if n_fruits < 1:
        raise ValueError('fast scene has no fruit bodies')
    if n_fruits < stage.fruit_count:
        raise ValueError(
            f'stage {stage.name} needs {stage.fruit_count} fruit bodies; scene has {n_fruits}. '
            f'Re-export with --fruit-count {stage.fruit_count}.')
    eval_profile = getattr(args, 'eval_profile', 'default')
    mask_idle = bool(getattr(args, 'mask_idle_locomotion', True))
    checkpoint_every = int(getattr(args, 'checkpoint_every', 1))
    easy = bool(getattr(args, 'easy', False))
    teacher_mix = float(getattr(args, 'teacher_mix', 0.0))
    shaping_coef = float(getattr(args, 'shaping_coef', EASY_PRESET['default_shaping_coef']))
    blocked_reason = fruit_block_reason(stage, n_fruits)
    config = dict(vars(args), approximations=manifest['approximation'],
                  scope='TK-RL-003 task curriculum on the rigid fast runtime; not field harvest',
                  curriculum=summarise_stage(stage, profile=eval_profile),
                  training_ready=False,
                  speedrun=bool(getattr(args, 'speedrun', False)),
                  easy=easy,
                  teacher_mix=teacher_mix,
                  teacher_horizon_updates=int(EASY_PRESET['teacher_horizon_updates']) if easy else 0,
                  teacher_anneal_after=-1,
                  shaping_coef=shaping_coef,
                  weld=False,
                  eval_profile=eval_profile,
                  mask_idle_locomotion=mask_idle,
                  entropy_kind='tanh_gaussian_differential_nats',
                  scene_fruit_count=n_fruits,
                  curriculum_blocked_reason=blocked_reason,
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
        if easy:
            easy_info = runtime.enable_easy(True, shaping_coef=shaping_coef)
            runtime.set_easy_progress(0.0, numpy_rng)
            config['hover_error_m'] = easy_info['hover_error_m']
            config['easy_start_error_m'] = easy_info['easy_start_error_m']
            config['easy_far_frac'] = 0.0
            config['easy_scope'] = easy_info['scope']
            config['hold_close_frac'] = easy_info.get('hold_close_frac')
            config['shaping_length_m'] = easy_info.get('shaping_length_m')
            config['deposit_reward'] = easy_info.get('deposit_reward')
            config['start_clearance_m'] = EASY_PRESET['start_clearance_m']
            config['start_side_y_m'] = EASY_PRESET['start_side_y_m']
            config['hold_sweep_slip_m'] = easy_info.get('hold_sweep_slip_m')
            config['hold_sweep_load_N'] = easy_info.get('hold_sweep_load_N')
            config['hold_sweep_rows'] = easy_info.get('hold_sweep_rows')
            config['grasp_local_m'] = easy_info.get('grasp_local_m')
            config['weld'] = False
            (args.output / 'config.json').write_text(
                json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
                           indent=2, default=str) + '\n')
        gait = load_gait_artifact(args.gait_checkpoint, precision_profile='cuda-fp32').to('cuda:0').eval()
        policy = build_policy().to('cuda:0')
        if args.initialize_from:
            load_checkpoint(args.initialize_from, {'student':policy}, None,
                            expected_meta={'camera':'hand_color_sensor'})
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        dim_mask = policy_dim_mask(stage, 'cuda:0', mask_idle)
        apply_stage(runtime, stage, numpy_rng)
        warmup_mix = easy_teacher_mix(0, start_mix=teacher_mix) if easy else teacher_mix
        collect(runtime, policy, gait, 4, args.camera_every, reset_all=True, dim_mask=dim_mask,
                teacher_mix=warmup_mix)
        torch.cuda.synchronize()
        start = time.monotonic()
        reports = []
        last_checkpoint = args.output / 'checkpoint-0000.pt'
        save_checkpoint(last_checkpoint, {'student':policy}, {'student':optimizer},
            {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            dict(schema='fast-curriculum-rgbd-r84/v1', camera='hand_color_sensor',
                 model_sha256=manifest['model_sha256'], config=config, completed_updates=0,
                 curriculum_stage=stage.name))
        apply_stage(runtime, stage, numpy_rng, evaluate_only=True)
        baseline = evaluate_mission(runtime, policy, gait, args.camera_every, stage=stage,
                                    control_dt=runtime.control_dt, eval_profile=eval_profile)
        baseline.update(curriculum_stage=stage.name, curriculum_index=stage.index)
        log.log(baseline, step=0)
        dashboard.refresh()
        if args.video_every:
            spawn_progress_video(last_checkpoint, dashboard.video_path(0),
                                 steps=args.video_steps, camera_every=args.camera_every)
        evaluations = [dict(update=0, **baseline)]
        eval_success_rates = []
        eval_episodes = 0
        best_distance = baseline['evaluation/mean_closest_distance_m']
        best_checkpoint = str(last_checkpoint)
        carry = {}
        teacher_anneal_after = None
        apply_stage(runtime, stage, numpy_rng)
        for iteration in range(args.updates):
            began = time.monotonic()
            if easy:
                far_frac = easy_start_far_frac(iteration)
                start_info = runtime.set_easy_progress(far_frac, numpy_rng)
                config['easy_far_frac'] = start_info['easy_far_frac']
                if teacher_anneal_after is None:
                    mix = float(teacher_mix)
                else:
                    mix = easy_teacher_mix(
                        iteration, start_mix=teacher_mix, anneal_after=teacher_anneal_after)
            else:
                start_info = {}
                mix = teacher_mix
            rows, bootstrap, carry = collect(runtime, policy, gait, args.steps, args.camera_every,
                                             carry=carry, reset_all=False, dim_mask=dim_mask,
                                             teacher_mix=mix)
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - began
            metrics = update(policy, optimizer, rows, bootstrap, args.minibatch_worlds,
                             entropy_coef=args.entropy_coef, gamma=args.gamma, epochs=args.ppo_epochs,
                             dim_mask=dim_mask)
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
                harvested_mean=float(torch.stack([r['harvested'] for r in rows]).max(dim=0).values.float().mean()),
                recovered_worlds=int(carry.get('recovered_worlds', 0)),
                overflow_worlds=int(carry.get('overflow_worlds', 0)),
                nonfinite_worlds=int(carry.get('nonfinite_worlds', 0)),
                curriculum_stage=stage.name, curriculum_index=stage.index,
                guidance_weight=stage.guidance_weight,
                eval_profile=eval_profile,
                idle_locomotion_masked=int(dim_mask is not None),
                easy=int(easy),
                teacher_mix=mix,
                teacher_actions=int(carry.get('teacher_actions', 0)),
                shaping_coef=shaping_coef,
                hover_error_m=float(runtime.hover_error_m),
                easy_far_frac=float(start_info.get('easy_far_frac', 0.0)),
                easy_start_index_mean=float(start_info.get('easy_start_index_mean', 0.0)),
                easy_start_index_max=int(start_info.get('easy_start_index_max', 0)),
                easy_hold_close_mean=float(start_info.get('easy_hold_close_mean', 0.0)),
                easy_hold_index_mean=float(start_info.get('easy_hold_index_mean', 0.0)),
                torch_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            if 'basket_distance' in rows[0]:
                basket = torch.stack([r['basket_distance'] for r in rows])
                metrics['basket_distance_mean_m'] = float(basket.mean())
                metrics['basket_distance_closest_m'] = float(basket.min(dim=0).values.mean())
            if 'basket_xy' in rows[0]:
                metrics['basket_xy_mean_m'] = float(torch.stack([r['basket_xy'] for r in rows]).mean())
            if 'fallen' in rows[0]:
                metrics['fallen_worlds'] = int(
                    (torch.stack([r['fallen'] for r in rows]).max(dim=0).values > 0).sum())
            if 'failed' in rows[0]:
                metrics['failed_worlds'] = int(
                    (torch.stack([r['failed'] for r in rows]).max(dim=0).values > 0).sum())
            if 'ground_contact' in rows[0]:
                metrics['ground_contact_worlds'] = int(
                    (torch.stack([r['ground_contact'] for r in rows]).max(dim=0).values > 0).sum())
            if 'hand_load_N' in rows[0]:
                load = torch.stack([r['hand_load_N'] for r in rows])
                metrics['hand_load_mean_N'] = float(load.mean())
                metrics['hand_load_max_N'] = float(load.max())
            if easy and teacher_anneal_after is None and int(metrics['harvest_successes']) >= 8:
                teacher_anneal_after = iteration + 1
                config['teacher_anneal_after'] = teacher_anneal_after
            metrics['teacher_anneal_after'] = (
                -1 if teacher_anneal_after is None else int(teacher_anneal_after))
            del rows, bootstrap
            promoted = False
            if (iteration+1) % args.eval_every == 0 or iteration+1 == args.updates:
                apply_stage(runtime, stage, numpy_rng, evaluate_only=True)
                eval_metrics = evaluate_mission(runtime, policy, gait, args.camera_every, stage=stage,
                                                control_dt=runtime.control_dt, eval_profile=eval_profile)
                evaluations.append(dict(update=iteration+1, **eval_metrics))
                metrics.update(eval_metrics)
                eval_success_rates.append(eval_metrics['evaluation/success_rate'])
                eval_episodes += int(eval_metrics['evaluation/worlds'])
                distance = eval_metrics['evaluation/mean_closest_distance_m']
                if promotion_ready(eval_success_rates, stage, episodes_seen=eval_episodes):
                    nxt = next_stage(stage)
                    if nxt is not None and nxt.fruit_count > n_fruits:
                        metrics['curriculum_promoted'] = False
                        metrics['curriculum_blocked'] = 1
                        metrics['curriculum_blocked_reason'] = fruit_block_reason(nxt, n_fruits)
                        apply_stage(runtime, stage, numpy_rng)
                        carry = {}
                    else:
                        metrics['curriculum_promoted'] = True
                        metrics['curriculum_blocked'] = 0
                        promoted = nxt is not None
                        if nxt is not None:
                            stage = nxt
                            dim_mask = policy_dim_mask(stage, 'cuda:0', mask_idle)
                            config['curriculum'] = summarise_stage(stage, profile=eval_profile)
                            config['stage'] = stage.name
                            config['curriculum_blocked_reason'] = fruit_block_reason(stage, n_fruits)
                            (args.output / 'config.json').write_text(
                                json.dumps(config, indent=2, default=str) + '\n')
                            eval_success_rates = []
                            eval_episodes = 0
                            carry = {}
                            apply_stage(runtime, stage, numpy_rng)
                            metrics['curriculum_stage'] = stage.name
                            metrics['curriculum_index'] = stage.index
                            metrics['idle_locomotion_masked'] = int(dim_mask is not None)
                else:
                    apply_stage(runtime, stage, numpy_rng)
                    carry = {}
            persist = should_persist_checkpoint(
                iteration + 1, updates=args.updates, eval_every=args.eval_every,
                checkpoint_every=checkpoint_every, video_every=args.video_every,
                promoted=promoted)
            checkpoint = last_checkpoint
            ckpt_models = {'student': policy}
            ckpt_opt = {'student': optimizer}
            ckpt_rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}
            meta = dict(schema='fast-curriculum-rgbd-r84/v1', camera='hand_color_sensor',
                        model_sha256=manifest['model_sha256'], config=config,
                        completed_updates=iteration+1, curriculum_stage=stage.name)
            save_checkpoint(args.output / 'latest.pt', ckpt_models, ckpt_opt, ckpt_rng, meta,
                            replace=True)
            if persist:
                checkpoint = args.output / f'checkpoint-{iteration+1:04d}.pt'
                save_checkpoint(checkpoint, ckpt_models, ckpt_opt, ckpt_rng, meta)
                last_checkpoint = checkpoint
                if 'evaluation/mean_closest_distance_m' in metrics:
                    distance = metrics['evaluation/mean_closest_distance_m']
                    if distance < best_distance:
                        best_distance, best_checkpoint = distance, str(checkpoint)
            metrics['checkpoint_written'] = int(persist)
            log.log(metrics, step=iteration+1)
            dashboard.refresh()
            if persist and args.video_every and (iteration + 1) % args.video_every == 0:
                spawn_progress_video(checkpoint, dashboard.video_path(iteration + 1),
                                     steps=args.video_steps, camera_every=args.camera_every)
            reports.append(metrics)
            print(json.dumps(metrics), flush=True)
        log.log_checkpoint(last_checkpoint)
        report = dict(config=config, updates=reports, evaluation=evaluations[-1],
                      baseline=baseline, evaluations=evaluations, best_reach_checkpoint=best_checkpoint,
                      best_mean_closest_distance_m=best_distance,
                      curriculum=summarise_stage(stage, profile=eval_profile),
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
    p.add_argument('--checkpoint-every', type=int, default=1,
                   help='Write student weights every N updates (always on eval/video/last)')
    p.add_argument('--camera-every', type=int, default=2)
    p.add_argument('--nconmax', type=int, default=128)
    p.add_argument('--njmax', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage', default='deposit_pixels',
                   help='TK-RL-003 curriculum stage to start from')
    p.add_argument('--speedrun', action='store_true',
                   help='Shorter eval, fewer checkpoints, mask idle locomotion; not field harvest')
    p.add_argument('--easy', action='store_true',
                   help='Kiwi starts in the jaws; scripted hold/open; RL deposits; not a weld')
    p.add_argument('--teacher-mix', type=float, default=None,
                   help='Fraction of training actions replaced by the privileged deposit teacher')
    p.add_argument('--shaping-coef', type=float, default=None,
                   help='Potential-shaping scale; --easy defaults to 5, otherwise 2')
    p.add_argument('--eval-profile', choices=['default', 'speedrun'], default='default')
    p.add_argument('--mask-idle-locomotion', action=argparse.BooleanOptionalAction, default=True,
                   help='Drop N3 from PPO log-prob/entropy while the stage holds the chassis')
    p.add_argument('--entropy-coef', type=float, default=0.005)
    p.add_argument('--gamma', type=float, default=0.9996)
    p.add_argument('--ppo-epochs', type=int, default=2)
    add_training_log_args(p)
    add_monitor_args(p)
    a = p.parse_args()
    apply_speedrun_cli(a)
    apply_easy_cli(a)
    if not 1 <= a.eval_every <= 10000 or not 1 <= a.minibatch_worlds <= 1024 or not 2 <= a.steps <= 256 or not 1 <= a.updates <= 10000 or not 1 <= a.camera_every <= 5:
        p.error('Invalid steps, updates or camera interval')
    if not 1 <= a.checkpoint_every <= 10000:
        p.error('Invalid checkpoint-every')
    if not 0 <= a.video_every <= 10000 or not 8 <= a.video_steps <= 512:
        p.error('Invalid video-every or video-steps')
    if a.stage not in {s.name for s in __import__('treesim.kiwi_rl.curriculum', fromlist=['STAGES']).STAGES}:
        p.error(f'Unknown curriculum stage {a.stage}')
    if not 0 <= a.entropy_coef <= 0.1 or not 0.9 <= a.gamma <= 1.0 or not 1 <= a.ppo_epochs <= 8:
        p.error('Invalid PPO entropy, gamma or epochs')
    if not 0.0 <= a.teacher_mix <= 1.0:
        p.error('Invalid teacher-mix')
    if not 0.0 <= a.shaping_coef <= 20.0:
        p.error('Invalid shaping-coef')
    run(a)


if __name__ == '__main__':
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        main()
