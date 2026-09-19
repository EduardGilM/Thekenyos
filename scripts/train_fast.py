"""Batched hackathon reaching PPO with rigid fruit and real RGBD/R84 inputs."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_physical_smoke import build_policy
from treesim.kiwi_rl.training_log import TrainingLog, add_training_log_args
from treesim.kiwi_rl.training_monitor import LiveDashboard, add_monitor_args, spawn_progress_video


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


def update(policy, optimizer, rows, bootstrap, minibatch_worlds=512):
    import torch
    from treesim.kiwi_rl.ppo import compute_gae_torch, tanh_logprob
    rewards = torch.stack([r['reward'] for r in rows])
    values = torch.stack([r['value'] for r in rows])
    ended = torch.stack([r['terminated'] for r in rows])
    truncated = torch.zeros_like(ended)
    truncated[-1] = True
    advantages = compute_gae_torch(rewards, values, torch.cat((values[1:], bootstrap[None])), ended, truncated, .99, .95)
    returns = (advantages + values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    # Sequence minibatches bound camera activation memory as worlds scale.
    batch_size = min(minibatch_worlds, rewards.shape[1])
    metrics = []
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, rewards.shape[1], batch_size):
        sl = slice(start, start + batch_size)
        memory = torch.zeros_like(rows[0]['r84'][sl, :64])
        logps, predictions = [], []
        for row in rows:
            memory = memory * (~row['reset'][sl])[:, None]
            mean, logstd, value, memory = policy(row['rgbd'][sl], row['r84'][sl], memory)
            logps.append(tanh_logprob(row['raw'][sl], mean, logstd))
            predictions.append(value)
        logratio = torch.stack(logps) - torch.stack([r['logp'][sl] for r in rows])
        ratio = logratio.exp()
        kl = ((ratio - 1.) - logratio).mean()
        if not torch.isfinite(kl):
            raise RuntimeError('Nonfinite PPO divergence')
        if float(kl.detach()) > .03:
            raise RuntimeError('Rollout/replay policy mismatch before optimizer step')
        actor = -torch.minimum(ratio * advantages[:, sl], ratio.clamp(.8, 1.2) * advantages[:, sl]).mean()
        critic = .5 * (torch.stack(predictions) - returns[:, sl]).square().mean()
        loss = actor + .5 * critic
        # Accumulate across every world before stepping: a KL stop after the
        # first tiny minibatch would waste nearly all collected experience.
        (loss * (rows[0]['reward'][sl].numel() / rewards.shape[1])).backward()
        metrics.append((float(loss.detach()), float(kl.detach())))
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=sum(m[0] for m in metrics)/len(metrics), kl=max(m[1] for m in metrics),
                grad_norm=float(grad), minibatches=len(metrics), optimized_transitions=int(rewards.numel()),
                reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)))


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
