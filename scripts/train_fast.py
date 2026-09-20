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
    EASY_PRESET, IK_DEMO_PRESET, apply_easy_preset, apply_ik_demo_preset,
    apply_speedrun_preset, easy_start_far_frac,
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


def policy_dim_mask(stage, device, enabled=True, scripted_jaw=False):
    """Mask idle base dimensions and a jaw action overwritten by the runtime."""
    import torch
    if not enabled and not scripted_jaw:
        return None
    mask = idle_locomotion_mask(10, stage.allow_locomotion) if enabled else None
    if mask is None:
        mask = np.ones(10, dtype=np.float32)
    if scripted_jaw:
        mask[-1] = 0.0
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


def apply_ik_demo_cli(args):
    """Random-start IK demos, then a short RL fine-tune. Implies --easy."""
    if not getattr(args, 'ik_demo', False):
        return args
    args.easy = True
    preset = apply_ik_demo_preset({})
    if getattr(args, 'teacher_mix', None) is None:
        args.teacher_mix = float(preset['teacher_mix'])
    if getattr(args, 'shaping_coef', None) is None:
        args.shaping_coef = float(preset['shaping_coef'])
    args.updates = int(preset['updates'])
    args.entropy_coef = float(preset['entropy_coef'])
    args.ppo_epochs = int(preset['ppo_epochs'])
    args.eval_every = int(preset['eval_every'])
    args.checkpoint_every = int(preset['checkpoint_every'])
    return args


def apply_easy_cli(args):
    """Privileged deposit facilitation. Explicit --teacher-mix, including 0, wins."""
    if getattr(args, 'ik_demo', False):
        return args
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
    args.updates = int(preset['updates'])
    args.entropy_coef = float(preset['entropy_coef'])
    args.ppo_epochs = int(preset['ppo_epochs'])
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
            teacher_applied = None
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
                teacher_applied = runtime.privileged_deposit_action()
                raw = mix_privileged_actions(raw, teacher_applied, mask)
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
            if 'release_fired' in info:
                row['release_fired'] = info['release_fired'].clone()
            if teacher_applied is not None:
                row['teacher_applied'] = teacher_applied.clone()
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


def normalize_advantages(advantages, std_cap=None):
    """Center advantages. Optionally cap the std so a rare jackpot stays large.

    Full whitening makes +500 and +10000 look the same after a sparse
    deposit. ``std_cap`` is an engineering lever, not a measured scale.
    """
    import torch
    centered = advantages - advantages.mean()
    std = advantages.std(unbiased=False)
    if std_cap is None:
        return centered / (std + 1e-8)
    cap = float(std_cap)
    if not np_finite(cap) or cap <= 0:
        raise ValueError('adv_std_cap must be finite and > 0')
    return centered / (torch.clamp(std, max=cap) + 1e-8)


def success_world_order(success_any, repeat=1):
    """Keep every world, then append extra copies of worlds that harvested."""
    import torch
    if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= 64:
        raise ValueError('success_repeat must be an integer in [1, 64]')
    flag = success_any.to(dtype=torch.bool).reshape(-1)
    order = torch.arange(flag.numel(), device=flag.device)
    if repeat == 1 or not bool(flag.any()):
        return order
    extra = flag.nonzero(as_tuple=False).reshape(-1)
    return torch.cat([order, extra.repeat(repeat - 1)])


def causal_success_mask(rows):
    """Select only the episode prefix ending in each observed success.

    A success world can contain failed episodes and a post-success hover reset
    in the same 64-step chunk. Self-imitation must not clone those unrelated
    actions.
    """
    import torch
    if not rows or 'success' not in rows[0]:
        raise ValueError('causal_success_mask requires nonempty success rows')
    success = torch.stack([row['success'] for row in rows]).bool()
    resets = torch.stack([row['reset'] for row in rows]).bool()
    mask = torch.zeros_like(success)
    active = torch.zeros_like(success[0])
    for step in range(len(rows) - 1, -1, -1):
        active = active | success[step]
        mask[step] = active
        active = active & (~resets[step])
    return mask


def ppo_actor_surrogate(ratio, advantages, clip, unclip_positive=False):
    """Clipped PPO surrogate. Positive advantages may skip the ratio cap.

    Clipping both sides keeps a rare deposit from moving π. Leaving
    A>0 unclipped is an engineering pull, not a proven harvest method.
    """
    import torch
    if not np_finite(clip) or not 0.05 <= clip <= 2.0:
        raise ValueError('clip must be finite in [0.05, 2]')
    surr = ratio * advantages
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
    if unclip_positive:
        chosen = torch.where(advantages > 0, surr, torch.minimum(surr, clipped))
    else:
        chosen = torch.minimum(surr, clipped)
    return -chosen.mean()


def update(policy, optimizer, rows, bootstrap, minibatch_worlds=512, entropy_coef=0.005,
           gamma=0.9996, epochs=1, dim_mask=None, clip=0.2, grad_clip=0.5,
           adv_std_cap=None, value_coef=0.5, target_kl=0.03, unclip_positive=False,
           success_repeat=1, imitation_coef=0.0):
    import torch
    from treesim.kiwi_rl.ppo import (
        compute_gae_torch, gaussian_entropy, tanh_gaussian_entropy, tanh_logprob,
    )
    if not np_finite(clip) or not 0.05 <= clip <= 2.0:
        raise ValueError('clip must be finite in [0.05, 2]')
    if not np_finite(grad_clip) or not 0.1 <= grad_clip <= 20.0:
        raise ValueError('grad_clip must be finite in [0.1, 20]')
    if not np_finite(value_coef) or not 0.0 <= value_coef <= 2.0:
        raise ValueError('value_coef must be finite in [0, 2]')
    if not np_finite(target_kl) or not 0.01 <= target_kl <= 1.0:
        raise ValueError('target_kl must be finite in [0.01, 1]')
    if not np_finite(imitation_coef) or not 0.0 <= imitation_coef <= 20.0:
        raise ValueError('imitation_coef must be finite in [0, 20]')
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
    advantages = normalize_advantages(advantages, std_cap=adv_std_cap)
    if 'success' in rows[0]:
        success_causal = causal_success_mask(rows)
        success_any = success_causal.any(dim=0)
    else:
        success_causal = torch.zeros_like(rewards, dtype=torch.bool)
        success_any = torch.zeros(rewards.shape[1], dtype=torch.bool, device=rewards.device)
    order = success_world_order(success_any, success_repeat)
    n_order = int(order.numel())
    batch_size = min(minibatch_worlds, n_order)
    metrics = []
    if not isinstance(epochs, int) or not 1 <= epochs <= 32:
        raise ValueError('epochs must be an integer in [1, 32]')
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
        for start in range(0, n_order, batch_size):
            idx = order[start:start + batch_size]
            memory = memory_root[idx].clone()
            logps, predictions, entropies, gaussians = [], [], [], []
            for row in rows:
                memory = memory * (~row['reset'][idx])[:, None]
                mean, logstd, value, memory = policy(row['rgbd'][idx], row['r84'][idx], memory)
                new_logp = tanh_logprob(row['raw'][idx], mean, logstd, dim_mask=dim_mask)
                logps.append(new_logp)
                predictions.append(value)
                draw = mean + logstd.exp() * torch.randn_like(mean)
                entropies.append(tanh_gaussian_entropy(logstd, raw=draw, mu=mean, dim_mask=dim_mask))
                gaussians.append(gaussian_entropy(logstd, dim_mask=dim_mask))
            logratio = torch.stack(logps) - torch.stack([r['logp'][idx] for r in rows])
            ratio = logratio.exp()
            kl = ((ratio - 1.) - logratio).mean()
            if not torch.isfinite(kl):
                continue
            mismatch = float(kl.detach())
            if epoch == 0 and mismatch > .03:
                raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
            if epoch > 0 and mismatch > target_kl:
                optimizer.zero_grad(set_to_none=True)
                stop_extra = True
                break
            actor = ppo_actor_surrogate(ratio, advantages[:, idx], clip, unclip_positive)
            critic = .5 * (torch.stack(predictions) - returns[:, idx]).square().mean()
            entropy = torch.stack(entropies).mean()
            entropy_g = torch.stack(gaussians).mean()
            sil = rewards.new_zeros(())
            causal = success_causal[:, idx]
            if imitation_coef > 0.0 and bool(causal.any()):
                sil = -torch.stack(logps)[causal].mean()
            loss = actor + value_coef * critic - entropy_coef * entropy + imitation_coef * sil
            (loss * (idx.numel() / float(n_order))).backward()
            if dim_mask is None:
                action_dim = int(logstd.shape[-1])
            else:
                action_dim = int(dim_mask.to(dtype=logstd.dtype).sum().clamp(min=1).item())
            metrics.append((float(loss.detach()), float(kl.detach()), float(entropy.detach()),
                            float(entropy_g.detach()), action_dim, float(actor.detach()),
                            float(critic.detach()), float(sil.detach())))
        if stop_extra:
            break
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        completed_epochs += 1
    if not metrics:
        raise RuntimeError('PPO produced no minibatches')
    entropy = sum(m[2] for m in metrics) / len(metrics)
    entropy_g = sum(m[3] for m in metrics) / len(metrics)
    action_dim = max(m[4] for m in metrics)
    actor_loss = sum(m[5] for m in metrics) / len(metrics)
    value_loss = sum(m[6] for m in metrics) / len(metrics)
    sil_loss = sum(m[7] for m in metrics) / len(metrics)
    window = rewards.sum(dim=0)
    n_success = int(success_any.sum().item())
    transition_mean = float(rewards.mean())
    window_mean = float(window.mean())
    if 'success' in rows[0]:
        success_steps = torch.stack([r['success'] for r in rows]).to(dtype=rewards.dtype)
        deposit_mass = float((rewards * success_steps).sum())
        success_window = float(window[success_any].mean()) if n_success else 0.0
        fail_mass = float((rewards * ended.to(dtype=rewards.dtype) * (1.0 - success_steps)).sum())
    else:
        deposit_mass = 0.0
        success_window = 0.0
        fail_mass = 0.0
    # A 64-step sum of the time cost looks ~64× worse than the old per-step
    # mean and still hides +10000 among 3072 worlds. Headline the harvest
    # return when a world deposited; otherwise keep the per-step scale.
    reward_mean = success_window if n_success else transition_mean
    return dict(loss=sum(m[0] for m in metrics)/len(metrics), kl=max(m[1] for m in metrics),
                actor_loss=actor_loss, value_loss=value_loss, sil_loss=sil_loss,
                entropy=entropy, entropy_gaussian=entropy_g,
                entropy_per_dim=entropy / action_dim,
                entropy_kind='tanh_gaussian_differential_nats',
                logstd_mean=float(policy.logstd.detach().mean()),
                grad_norm=float(grad), minibatches=len(metrics),
                optimized_transitions=int(rewards.shape[0] * n_order * completed_epochs),
                ppo_epochs_completed=completed_epochs,
                ppo_clip=float(clip), ppo_grad_clip=float(grad_clip),
                ppo_value_coef=float(value_coef), ppo_target_kl=float(target_kl),
                ppo_adv_std_cap=None if adv_std_cap is None else float(adv_std_cap),
                ppo_unclip_positive=int(bool(unclip_positive)),
                ppo_success_repeat=int(success_repeat),
                ppo_imitation_coef=float(imitation_coef),
                ppo_success_worlds=n_success,
                ppo_causal_success_transitions=int(success_causal.sum().item()),
                reward_mean=reward_mean,
                reward_window_mean=window_mean,
                reward_transition_mean=transition_mean,
                reward_std=float(window.std(unbiased=False)),
                success_window_return_mean=success_window,
                deposit_return_sum=deposit_mass,
                deposit_return_mean=deposit_mass / float(rewards.shape[1]),
                fail_return_sum=fail_mass)


def imitation_update(policy, optimizer, rows, epochs=6, dim_mask=None, grad_clip=0.5,
                    minibatch_worlds=64):
    """Behavior-clone the privileged IK teacher. Fruit stays a free body.

    Unrolls time inside a world minibatch so the backward graph is not one
    full-batch 64-step RGB-D tape. Recurrent memory stays per-world.
    """
    import torch
    if rows and rows[0]['r84'].is_cuda:
        torch.cuda.empty_cache()
    if not rows or 'teacher_applied' not in rows[0]:
        raise ValueError('imitation_update requires teacher_applied labels')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 32:
        raise ValueError('epochs must be an integer in [1, 32]')
    if not np_finite(grad_clip) or not 0.1 <= grad_clip <= 20.0:
        raise ValueError('grad_clip must be finite in [0.1, 20]')
    if not isinstance(minibatch_worlds, int) or isinstance(minibatch_worlds, bool):
        raise ValueError('minibatch_worlds must be a positive integer')
    if not 1 <= minibatch_worlds <= 1024:
        raise ValueError('minibatch_worlds must be an integer in [1, 1024]')
    if 'memory0' in rows[0]:
        memory_root = rows[0]['memory0']
    else:
        memory_root = torch.zeros(rows[0]['r84'].shape[0], 64, device=rows[0]['r84'].device,
                                  dtype=rows[0]['r84'].dtype)
    worlds = int(rows[0]['r84'].shape[0])
    batch_size = min(int(minibatch_worlds), worlds)
    if dim_mask is None:
        arm_mask = None
    else:
        arm_mask = dim_mask.to(dtype=rows[0]['r84'].dtype)
        if arm_mask.shape[-1] == 10:
            arm_mask = arm_mask[3:10]
    losses = []
    n_minibatches = 0
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    for _ in range(epochs):
        order = torch.randperm(worlds, device=rows[0]['r84'].device)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for start in range(0, worlds, batch_size):
            idx = order[start:start + batch_size]
            memory = memory_root[idx].clone()
            pred, label, weights = [], [], []
            for row in rows:
                memory = memory * (~row['reset'][idx])[:, None]
                mean, _, _, memory = policy(row['rgbd'][idx], row['r84'][idx], memory)
                applied = mean.tanh()[:, 3:10] if mean.shape[-1] == 10 else mean.tanh()
                target = row['teacher_applied'][idx]
                if applied.shape != target.shape:
                    raise ValueError('teacher labels must match the arm action width')
                pred.append(applied)
                label.append(target)
                n_batch = int(idx.numel())
                weights.append(torch.ones(n_batch, device=applied.device)
                               if arm_mask is None else arm_mask.reshape(1, -1).expand(n_batch, -1))
            stacked_w = torch.stack(weights)
            err = (torch.stack(pred) - torch.stack(label)) * stacked_w
            loss = err.square().sum() / stacked_w.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite imitation loss')
            (loss * (idx.numel() / float(worlds))).backward()
            epoch_loss += float(loss.detach()) * (idx.numel() / float(worlds))
            n_minibatches += 1
            del pred, label, weights, stacked_w, err, loss
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(epoch_loss))
        if rows[0]['r84'].is_cuda:
            torch.cuda.empty_cache()
    changed = [name for name, p in policy.named_parameters() if not torch.equal(p, before[name])]
    if not any(name.startswith('vision.') for name in changed) or 'mean.weight' not in changed:
        raise RuntimeError('Imitation update did not change both vision and action weights')
    return dict(
        loss=float(losses[-1]), bc_loss=float(losses[-1]), bc_loss_initial=float(losses[0]),
        bc_epochs_completed=int(epochs), grad_norm=float(grad),
        kl=0.0, actor_loss=0.0, value_loss=0.0, sil_loss=0.0,
        entropy=0.0, entropy_gaussian=0.0, entropy_per_dim=0.0,
        entropy_kind='tanh_gaussian_differential_nats',
        logstd_mean=float(policy.logstd.detach().mean()),
        minibatches=int(n_minibatches), optimized_transitions=int(len(rows) * worlds * epochs),
        ppo_epochs_completed=0, ppo_clip=0.0, ppo_grad_clip=float(grad_clip),
        ppo_value_coef=0.0, ppo_target_kl=0.0, ppo_adv_std_cap=None,
        ppo_unclip_positive=0, ppo_success_repeat=1, ppo_imitation_coef=0.0,
        ppo_success_worlds=int(torch.stack([r['success'] for r in rows]).any(dim=0).sum())
        if 'success' in rows[0] else 0,
        ppo_causal_success_transitions=0,
        reward_mean=float(torch.stack([r['reward'] for r in rows]).mean()),
        reward_window_mean=float(torch.stack([r['reward'] for r in rows]).sum(dim=0).mean()),
        reward_transition_mean=float(torch.stack([r['reward'] for r in rows]).mean()),
        reward_std=float(torch.stack([r['reward'] for r in rows]).std(unbiased=False)),
        success_window_return_mean=0.0, deposit_return_sum=0.0,
        deposit_return_mean=0.0, fail_return_sum=0.0,
        demo_phase=1, vision_changed=True, action_changed=True,
    )


def np_finite(value):
    import math
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_mission(runtime, policy, gait, camera_every, *, stage, control_dt=0.02,
                     eval_profile='default'):
    """Deterministic eval over the stage horizon without stacking RGBD history."""
    import numpy as np
    import torch
    runtime.configure_skills(evaluate_skills(stage, runtime.worlds))
    horizon = evaluation_horizon_steps(stage, control_dt, profile=eval_profile)
    saved_hover = None
    if getattr(runtime, '_ik_demo', False) and hasattr(runtime, 'set_carry_progress'):
        saved_hover = runtime.snapshot_easy_hover()
        runtime.set_carry_progress(np.random.default_rng(0))
    elif getattr(runtime, '_easy', False) and hasattr(runtime, 'clear_easy_hover_starts'):
        saved_hover = runtime.snapshot_easy_hover()
        runtime.clear_easy_hover_starts(np.random.default_rng(0))
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
    result = {
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
    if saved_hover is not None:
        runtime.restore_easy_hover(saved_hover)
    return result


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
    ik_demo = bool(getattr(args, 'ik_demo', False))
    teacher_mix = float(getattr(args, 'teacher_mix', 0.0))
    shaping_coef = float(getattr(args, 'shaping_coef', EASY_PRESET['default_shaping_coef']))
    demo_updates = int(IK_DEMO_PRESET['demo_updates']) if ik_demo else 0
    knobs = IK_DEMO_PRESET if ik_demo else EASY_PRESET
    blocked_reason = fruit_block_reason(stage, n_fruits)
    config = dict(vars(args), approximations=manifest['approximation'],
                  scope='TK-RL-003 task curriculum on the rigid fast runtime; not field harvest',
                  curriculum=summarise_stage(stage, profile=eval_profile),
                  training_ready=False,
                  speedrun=bool(getattr(args, 'speedrun', False)),
                  easy=easy,
                  ik_demo=ik_demo,
                  demo_updates=demo_updates,
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
            easy_info = runtime.enable_easy(True, shaping_coef=shaping_coef, ik_demo=ik_demo)
            if ik_demo:
                runtime.set_carry_progress(numpy_rng)
            else:
                runtime.set_easy_progress(0.0, numpy_rng)
            config['hover_error_m'] = easy_info['hover_error_m']
            config['easy_start_error_m'] = easy_info['easy_start_error_m']
            config['easy_far_frac'] = 0.0
            config['easy_scope'] = easy_info['scope']
            config['hold_close_frac'] = easy_info.get('hold_close_frac')
            config['shaping_length_m'] = easy_info.get('shaping_length_m')
            config['deposit_reward'] = easy_info.get('deposit_reward')
            config['fail_reward'] = easy_info.get('fail_reward')
            config['ppo_lr'] = float(knobs['ppo_lr'])
            config['ppo_epochs'] = int(knobs['ppo_epochs'])
            config['ppo_clip'] = float(knobs['ppo_clip'])
            config['ppo_grad_clip'] = float(knobs['ppo_grad_clip'])
            config['ppo_adv_std_cap'] = knobs['ppo_adv_std_cap']
            config['ppo_value_coef'] = float(knobs['ppo_value_coef'])
            config['ppo_target_kl'] = float(knobs['ppo_target_kl'])
            config['ppo_unclip_positive'] = bool(knobs['ppo_unclip_positive'])
            config['ppo_success_repeat'] = int(knobs['ppo_success_repeat'])
            config['ppo_imitation_coef'] = float(knobs['ppo_imitation_coef'])
            config['ppo_success_epochs'] = int(knobs['ppo_success_epochs'])
            config['n_waypoints'] = easy_info.get('n_waypoints')
            config['n_hard_starts'] = easy_info.get('n_hard_starts')
            config['ik_demo'] = bool(ik_demo)
            config['start_over_opening'] = EASY_PRESET['start_over_opening']
            config['start_open_radius_m'] = EASY_PRESET['start_open_radius_m']
            config['start_inset_x_m'] = EASY_PRESET['start_inset_x_m']
            config['start_clearance_m'] = EASY_PRESET['start_clearance_m']
            config['start_side_y_m'] = EASY_PRESET['start_side_y_m']
            config['shape_hand_and_fruit'] = easy_info.get('shape_hand_and_fruit')
            config['release_at_center'] = easy_info.get('release_at_center')
            config['release_over_opening'] = easy_info.get('release_over_opening')
            config['release_opening_inset_m'] = easy_info.get('release_opening_inset_m')
            config['release_max_above_rim_m'] = easy_info.get('release_max_above_rim_m')
            config['release_target_clearance_m'] = knobs['release_target_clearance_m']
            config['release_target_inset_x_m'] = knobs['release_target_inset_x_m']
            config['safe_hover_arm_q'] = [float(q) for q in EASY_PRESET['safe_hover_arm_q']]
            config['far_horizon_updates'] = easy_info.get('far_horizon_updates')
            config['far_frac_cap'] = easy_info.get('far_frac_cap')
            config['hover_clearance_m'] = easy_info.get('hover_clearance_m')
            config['shape_to_hover'] = easy_info.get('shape_to_hover')
            config['hold_sweep_slip_m'] = easy_info.get('hold_sweep_slip_m')
            config['hold_sweep_load_N'] = easy_info.get('hold_sweep_load_N')
            config['hold_sweep_rows'] = easy_info.get('hold_sweep_rows')
            config['grasp_local_m'] = easy_info.get('grasp_local_m')
            config['hover_start_index'] = easy_info.get('hover_start_index')
            config['n_start_poses'] = easy_info.get('n_start_poses')
            config['weld'] = False
            (args.output / 'config.json').write_text(
                json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
                           indent=2, default=str) + '\n')
        gait = load_gait_artifact(args.gait_checkpoint, precision_profile='cuda-fp32').to('cuda:0').eval()
        policy = build_policy().to('cuda:0')
        if args.initialize_from:
            load_checkpoint(args.initialize_from, {'student':policy}, None,
                            expected_meta={'camera':'hand_color_sensor'})
        ppo_lr = float(knobs['ppo_lr']) if easy else 3e-4
        if not np_finite(ppo_lr) or not 1e-5 <= ppo_lr <= 1e-2:
            raise ValueError('ppo_lr must be finite in [1e-5, 1e-2]')
        optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_lr)
        dim_mask = policy_dim_mask(stage, 'cuda:0', mask_idle, scripted_jaw=easy)
        apply_stage(runtime, stage, numpy_rng)
        warmup_mix = easy_teacher_mix(0, start_mix=teacher_mix) if easy else teacher_mix
        collect(runtime, policy, gait, 4, args.camera_every, reset_all=True, dim_mask=dim_mask,
                teacher_mix=warmup_mix)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
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
            demo_phase = bool(ik_demo and iteration < demo_updates)
            if ik_demo:
                start_info = runtime.set_carry_progress(numpy_rng)
                config['easy_far_frac'] = start_info['easy_far_frac']
                mix = 1.0 if demo_phase else 0.0
            elif easy:
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
            torch.cuda.empty_cache()
            rollout_seconds = time.monotonic() - began
            if ik_demo:
                live_wp = np.asarray(runtime._waypoint_index.numpy(), dtype=np.int32).reshape(-1)
                start_info = dict(start_info)
                start_info['carry_waypoint_mean'] = float(live_wp.mean()) if live_wp.size else 0.0
                start_info['carry_waypoint_max'] = int(live_wp.max()) if live_wp.size else 0
            if demo_phase:
                for row in rows:
                    for key in ('raw', 'logp', 'value'):
                        row.pop(key, None)
                del bootstrap
                bootstrap = None
                bc_batch = min(int(args.minibatch_worlds), int(IK_DEMO_PRESET['bc_minibatch_worlds']))
                metrics = imitation_update(
                    policy, optimizer, rows, epochs=int(IK_DEMO_PRESET['bc_epochs']),
                    dim_mask=dim_mask, grad_clip=float(knobs['ppo_grad_clip']),
                    minibatch_worlds=bc_batch)
            else:
                if easy:
                    ppo_clip = float(knobs['ppo_clip'])
                    ppo_grad = float(knobs['ppo_grad_clip'])
                    raw_std_cap = knobs['ppo_adv_std_cap']
                    ppo_std_cap = None if raw_std_cap is None else float(raw_std_cap)
                    ppo_value = float(knobs['ppo_value_coef'])
                    ppo_kl = float(knobs['ppo_target_kl'])
                    ppo_unclip = bool(knobs['ppo_unclip_positive'])
                    ppo_repeat = int(knobs['ppo_success_repeat'])
                    ppo_sil = float(knobs['ppo_imitation_coef'])
                    ppo_epochs = int(knobs['ppo_epochs'])
                    if int(torch.stack([r['success'] for r in rows]).any(dim=0).sum()) > 0:
                        ppo_epochs = max(ppo_epochs, int(knobs['ppo_success_epochs']))
                else:
                    ppo_clip, ppo_grad, ppo_std_cap, ppo_value, ppo_kl = 0.2, 0.5, None, 0.5, 0.03
                    ppo_unclip, ppo_repeat, ppo_sil, ppo_epochs = False, 1, 0.0, int(args.ppo_epochs)
                metrics = update(policy, optimizer, rows, bootstrap, args.minibatch_worlds,
                                 entropy_coef=args.entropy_coef, gamma=args.gamma, epochs=ppo_epochs,
                                 dim_mask=dim_mask, clip=ppo_clip, grad_clip=ppo_grad,
                                 adv_std_cap=ppo_std_cap, value_coef=ppo_value, target_kl=ppo_kl,
                                 unclip_positive=ppo_unclip, success_repeat=ppo_repeat,
                                 imitation_coef=ppo_sil)
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
                harvest_jackpot_sum=float(int(metrics['ppo_success_worlds']) * (
                    float(knobs['deposit_reward']) if easy else 20.0)),
                demo_phase=int(demo_phase),
                ik_demo=int(ik_demo),
                carry_easy_start_worlds=int(start_info.get('carry_easy_start_worlds', 0)),
                carry_hard_start_worlds=int(start_info.get('carry_hard_start_worlds', 0)),
                carry_waypoint_mean=float(start_info.get('carry_waypoint_mean', 0.0)),
                carry_waypoint_max=int(start_info.get('carry_waypoint_max', 0)),
                grasp_offset_mean_m=float(getattr(runtime, '_grasp_offset_mean_m', 0.0)),
                grasp_offset_std_m=float(getattr(runtime, '_grasp_offset_std_m', 0.0)),
                grasp_offset_max_m=float(getattr(runtime, '_grasp_offset_max_m', 0.0)),
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
                easy_hover_cohort_worlds=int(start_info.get('easy_hover_cohort_worlds', 0)),
                easy_hover_start_worlds=int(start_info.get('easy_hover_start_worlds', 0)),
                easy_outside_start_worlds=int(start_info.get('easy_outside_start_worlds', 0)),
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
            if 'release_fired' in rows[0]:
                metrics['release_fired_worlds'] = int(
                    torch.stack([r['release_fired'] for r in rows]).any(dim=0).sum())
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
                            dim_mask = policy_dim_mask(
                                stage, 'cuda:0', mask_idle, scripted_jaw=easy)
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
    p.add_argument('--ik-demo', action='store_true',
                   help='Random easy/hard starts; IK demos to the basket centre; then short RL')
    p.add_argument('--teacher-mix', type=float, default=None,
                   help='Fraction of training actions replaced by the privileged deposit teacher')
    p.add_argument('--shaping-coef', type=float, default=None,
                   help='Potential-shaping scale; --easy defaults to 25, otherwise 2')
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
    apply_ik_demo_cli(a)
    apply_easy_cli(a)
    if not 1 <= a.eval_every <= 10000 or not 1 <= a.minibatch_worlds <= 1024 or not 2 <= a.steps <= 256 or not 1 <= a.updates <= 10000 or not 1 <= a.camera_every <= 5:
        p.error('Invalid steps, updates or camera interval')
    if not 1 <= a.checkpoint_every <= 10000:
        p.error('Invalid checkpoint-every')
    if not 0 <= a.video_every <= 10000 or not 8 <= a.video_steps <= 512:
        p.error('Invalid video-every or video-steps')
    if a.stage not in {s.name for s in __import__('treesim.kiwi_rl.curriculum', fromlist=['STAGES']).STAGES}:
        p.error(f'Unknown curriculum stage {a.stage}')
    if not 0 <= a.entropy_coef <= 0.1 or not 0.9 <= a.gamma <= 1.0 or not 1 <= a.ppo_epochs <= 32:
        p.error('Invalid PPO entropy, gamma or epochs')
    if not 0.0 <= a.teacher_mix <= 1.0:
        p.error('Invalid teacher-mix')
    if not 0.0 <= a.shaping_coef <= 50.0:
        p.error('Invalid shaping-coef')
    run(a)


if __name__ == '__main__':
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        main()
