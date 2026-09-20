"""Batched hackathon reaching PPO with rigid fruit and real RGBD/R84 inputs."""
from pathlib import Path
import argparse
import json
import math
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_physical_smoke import build_policy
from treesim.kiwi_rl.training_log import TrainingLog, add_training_log_args


def collect(runtime, policy, gait, steps, camera_every, *, deterministic=False):
    import torch
    from treesim.kiwi_rl.ppo import tanh_logprob
    runtime.reset()
    memory = torch.zeros(runtime.worlds, 64, device='cuda:0')
    rows = []
    reset = torch.ones(runtime.worlds, device='cuda:0', dtype=torch.bool)
    for index in range(steps):
        r84 = runtime.observe().clone()
        if index % camera_every == 0:
            rgbd = runtime.pixels().clone()
        with torch.no_grad():
            memory = memory * (~reset)[:, None]
            mean, logstd, value, memory = policy(rgbd, r84, memory)
            raw = mean if deterministic else mean + logstd.exp() * torch.randn_like(mean)
            logp = tanh_logprob(raw, mean, logstd)
            runtime.set_gait_actions(gait(r84))
            _, reward, done, info = runtime.step(raw.tanh())
            rows.append(dict(rgbd=rgbd, r84=r84, raw=raw, logp=logp, value=value,
                reward=reward.clone(), terminated=done.clone(), reset=reset.clone(),
                distance=info['distance_m'].clone(),
                success=info['success'].clone(), detached=info['detached'].clone()))
            reset = done.clone()
            # Preserve other worlds, including recurrent state. Reset on GPU;
            # refresh the camera after a reset so it cannot show the old episode.
            if bool(reset.any()):
                runtime.reset(reset)
                rgbd = runtime.pixels().clone()
    with torch.no_grad():
        _, _, bootstrap, _ = policy(runtime.pixels(), runtime.observe(), memory * (~reset)[:, None])
    runtime.check()
    return rows, bootstrap


def update(policy, optimizer, rows, bootstrap, minibatch_worlds=512, *, gamma=.99, check_replay=True,
           teacher_coef=0., gae_lambda=.95, entropy_coef=0., minibatch_updates=False, target_kl=.03):
    import torch
    from treesim.kiwi_rl.ppo import compute_gae_torch, tanh_logprob
    if any(row.get('kind', 'factual_on_policy') != 'factual_on_policy' for row in rows):
        raise ValueError('PPO update accepts factual on-policy rows only')
    if teacher_coef and any('teacher_action' not in row for row in rows):
        raise ValueError('teacher_coef requires teacher actions on every rollout row')
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError('gae_lambda must be finite and in [0, 1]')
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('entropy_coef must be finite and nonnegative')
    rewards = torch.stack([r['reward'] for r in rows])
    if not math.isfinite(teacher_coef) or teacher_coef < 0:
        raise ValueError('teacher_coef must be finite and nonnegative')
    values = torch.stack([r['value'] for r in rows])
    ended = torch.stack([r['terminated'] for r in rows])
    truncated = torch.stack([r.get('truncated', torch.zeros_like(r['terminated'])) for r in rows])
    next_values = torch.cat((values[1:], bootstrap[None]))
    for t, row in enumerate(rows):
        if 'timeout_value' in row:
            next_values[t] = torch.where(truncated[t], row['timeout_value'], next_values[t])
    advantages = compute_gae_torch(rewards, values, next_values, ended, truncated, gamma, gae_lambda)
    returns = (advantages + values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    # Sequence minibatches bound camera activation memory as worlds scale.
    batch_size = min(minibatch_worlds, rewards.shape[1])
    role = rows[0].get('role', 'student')
    if any(row.get('role', role) != role for row in rows):
        raise ValueError('PPO update cannot mix teacher and student rows')
    observation_key = 'privileged' if role == 'teacher' else 'rgbd'
    if any(observation_key not in row for row in rows):
        raise ValueError(f'{role} PPO rows are missing {observation_key} observations')
    return_variance = returns.var(unbiased=False)
    explained_variance = (1 - (returns - values).var(unbiased=False) / return_variance
                          if float(return_variance) > 1e-12 else returns.new_zeros(()))
    metrics = []
    optimizer_steps = 0
    early_stop = False
    order = torch.randperm(rewards.shape[1], device=rewards.device) if minibatch_updates else None
    # Verify the entire factual batch before any mutation. After the first
    # minibatch step, divergence is expected and is controlled by the KL guard.
    if minibatch_updates and check_replay:
        with torch.no_grad():
            for start in range(0, rewards.shape[1], batch_size):
                sl = slice(start, start + batch_size)
                memory = rows[0].get('initial_memory', torch.zeros_like(rows[0]['r84'][:, :64]))[sl].detach()
                for row in rows:
                    memory = memory * (~row['reset'][sl])[:, None]
                    mean, logstd, _, memory = policy(row[observation_key][sl], row['r84'][sl], memory)
                    error = tanh_logprob(row['raw'][sl], mean, logstd) - row['logp'][sl]
                    if not torch.isfinite(error).all() or float(error.abs().max()) > 2.e-4:
                        raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, rewards.shape[1], batch_size):
        sl = order[start:start + batch_size] if minibatch_updates else slice(start, start + batch_size)
        if minibatch_updates:
            optimizer.zero_grad(set_to_none=True)
        memory = rows[0].get('initial_memory', torch.zeros_like(rows[0]['r84'][:, :64]))[sl].detach()
        logps, predictions, bounded_actions, entropies, action_stds = [], [], [], [], []
        for row in rows:
            memory = memory * (~row['reset'][sl])[:, None]
            mean, logstd, value, memory = policy(row[observation_key][sl], row['r84'][sl], memory)
            logps.append(tanh_logprob(row['raw'][sl], mean, logstd))
            predictions.append(value)
            bounded_actions.append(mean.tanh())
            action_stds.append(logstd.exp().detach().expand_as(mean).mean(dim=0))
            if entropy_coef:
                # Reparameterized entropy includes tanh's Jacobian. Unsquashed
                # Gaussian entropy alone can drive the controls into saturation.
                draw = mean + logstd.exp() * torch.randn_like(mean)
                entropies.append(-tanh_logprob(draw, mean, logstd))
        logratio = torch.stack(logps) - torch.stack([r['logp'][sl] for r in rows])
        ratio = logratio.exp()
        kl = ((ratio - 1.) - logratio).mean()
        if not torch.isfinite(kl):
            raise RuntimeError('Nonfinite PPO divergence')
        if float(kl.detach()) > target_kl:
            if check_replay and not minibatch_updates:
                raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
            optimizer.zero_grad(set_to_none=True)
            if not minibatch_updates:
                return dict(kl=float(kl.detach()), early_stop=1, optimizer_steps=0, optimized_transitions=0)
            early_stop = True
            break
        actor = -torch.minimum(ratio * advantages[:, sl], ratio.clamp(.8, 1.2) * advantages[:, sl]).mean()
        critic = .5 * (torch.stack(predictions) - returns[:, sl]).square().mean()
        entropy = torch.stack(entropies).mean() if entropies else rewards.new_zeros(())
        loss = actor + .5 * critic - entropy_coef * entropy
        teacher_loss = rewards.new_zeros(())
        if teacher_coef and 'teacher_action' in rows[0]:
            teacher = torch.stack([r['teacher_action'][sl] for r in rows])
            if not torch.isfinite(teacher).all():
                raise ValueError('teacher actions must be finite')
            distill = (torch.stack(bounded_actions) - teacher).square().mean()
            teacher_loss = distill
            loss = loss + teacher_coef * distill
        # Accumulate across every world before stepping: a KL stop after the
        # first tiny minibatch would waste nearly all collected experience.
        weight = 1. if minibatch_updates else rows[0]['reward'][sl].numel() / rewards.shape[1]
        (loss * weight).backward()
        if minibatch_updates:
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
        metrics.append((float(loss.detach()), float(kl.detach()), float(teacher_loss.detach()),
                        float(actor.detach()), float(critic.detach()), float(entropy.detach()),
                        torch.stack(action_stds).mean(dim=0), rewards[:, sl].numel()))
    if not metrics:
        return dict(kl=float(kl.detach()), early_stop=1, optimizer_steps=0, optimized_transitions=0)
    if not minibatch_updates:
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        optimizer_steps = 1
    used = sum(m[7] for m in metrics)
    return dict(loss=sum(m[0] for m in metrics)/len(metrics), kl=max(m[1] for m in metrics),
                optimizer_steps=optimizer_steps, early_stop=int(early_stop),
                teacher_mse=sum(m[2] for m in metrics)/len(metrics),
                grad_norm=float(grad), minibatches=len(metrics), optimized_transitions=used,
                reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                actor_loss=sum(m[3]*m[7] for m in metrics)/used,
                critic_loss=sum(m[4]*m[7] for m in metrics)/used,
                entropy=sum(m[5]*m[7] for m in metrics)/used,
                explained_variance=float(explained_variance), return_variance=float(return_variance),
                gae_lambda=gae_lambda, entropy_coef=entropy_coef,
                **{f'action_std/{i}': float(sum(m[6][i]*m[7] for m in metrics)/used)
                   for i in range(metrics[0][6].numel())})


def evaluate(runtime, policy, gait, steps, camera_every):
    import torch
    rows, _ = collect(runtime, policy, gait, steps, camera_every, deterministic=True)
    distances = torch.stack([r['distance'] for r in rows])
    return {
        'evaluation/final_distance_m': float(distances[-1].mean()),
        'evaluation/closest_distance_m': float(distances.min()),
        'evaluation/mean_closest_distance_m': float(distances.min(dim=0).values.mean()),
        'evaluation/terminal_transitions': int(torch.stack([r['terminated'] for r in rows]).sum()),
        'evaluation/harvest_successes': int(torch.stack([r['success'] for r in rows]).sum()),
    }


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
    try:
        runtime = FastRuntime(args.scene, worlds=args.worlds, camera='hand_camera',
                              nconmax=args.nconmax, njmax=args.njmax)
        gait = load_gait_artifact(args.gait_checkpoint).to('cuda:0').eval()
        policy = build_policy().to('cuda:0')
        if args.initialize_from:
            load_checkpoint(args.initialize_from, {'student':policy}, None,
                            expected_meta={'camera':'hand_camera', 'camera_profile':manifest['cameras']})
        # Persist the actual starting observation so a run is visually auditable.
        from PIL import Image
        rgb = runtime.pixels()[0, :3].permute(1, 2, 0).detach().cpu().numpy()
        Image.fromarray((rgb.clip(0, 1) * 255).astype('uint8')).save(args.output / 'policy-camera-start.png')
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # Compilation and warmup are outside measured rollout throughput.
        collect(runtime, policy, gait, 4, args.camera_every)
        torch.cuda.synchronize()
        start = time.monotonic()
        reports = []
        initial_checkpoint = args.output / 'checkpoint-0000.pt'
        save_checkpoint(initial_checkpoint, {'student':policy}, {'student':optimizer},
            {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
            dict(schema='fast-reach-rgbd-r84/v1', camera='hand_camera', camera_profile=manifest['cameras'],
                 model_sha256=manifest['model_sha256'], config=config, completed_updates=0))
        baseline = evaluate(runtime, policy, gait, args.steps, args.camera_every)
        log.log(baseline, step=0)
        print(json.dumps(dict(update=0, **baseline)), flush=True)
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
            meta = dict(schema='fast-reach-rgbd-r84/v1', camera='hand_camera', camera_profile=manifest['cameras'],
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
            reports.append(metrics)
            print(json.dumps(metrics), flush=True)
        log.log_checkpoint(checkpoint)
        report = dict(config=config, updates=reports, evaluation=eval_metrics,
                      baseline=baseline, evaluations=evaluations, best_reach_checkpoint=best_checkpoint,
                      best_mean_closest_distance_m=best_distance,
                      elapsed_seconds=time.monotonic()-start, wandb_url=log.url, numerical=runtime.check())
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    except BaseException:
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
    a = p.parse_args()
    if not 1 <= a.eval_every <= 10000 or not 1 <= a.minibatch_worlds <= 1024 or not 2 <= a.steps <= 256 or not 1 <= a.updates <= 10000 or not 1 <= a.camera_every <= 5:
        p.error('Invalid steps, updates or camera interval')
    run(a)


if __name__ == '__main__':
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        main()
