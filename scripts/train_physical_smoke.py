"""GPU deformable-fruit reaching: real sensor policy updates and checkpoint resume.

This is a first physical skill, not a complete grasp/deposit or orchard policy.
The actor gets camera pixels and R84 robot measurements. Fruit geometry is used
by the reward evaluator and optional scripted action teacher. No assistance, fruit teleportation or grasp weld.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from treesim.kiwi_rl.training_log import TrainingLog, add_training_log_args

SCHEMA = 'physical-reach-rgbd-r84/v1'


def build_policy():
    import torch
    nn = torch.nn

    class ReachPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Sequential(nn.Conv2d(5, 16, 5, 2, 2), nn.SiLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
            self.belief = nn.GRUCell(128 + 84, 64)
            self.mean = nn.Linear(64, 7)
            self.value = nn.Linear(64, 1)
            self.logstd = nn.Parameter(torch.full((7,), -1.6))

        def forward(self, rgbd, r84, memory):
            memory = self.belief(torch.cat((self.vision(rgbd), r84), dim=-1), memory)
            return self.mean(memory), self.logstd.clamp(-5., 1.), self.value(memory).squeeze(-1), memory

    return ReachPolicy()


def tensors(observation, device):
    import torch
    return (torch.as_tensor(observation.rgbd.copy(), device=device),
            torch.as_tensor(observation.r84.copy(), device=device))


def collect(adapter, policy, steps, gait, *, deterministic=False, control_dt=.04, teacher=None):
    import torch
    from treesim.kiwi_rl.ppo import tanh_logprob
    device = next(policy.parameters()).device
    observation = adapter.reset()
    memory = torch.zeros(adapter.runtime.worlds, 64, device=device)
    rows = []
    for index in range(steps):
        rgbd, r84 = tensors(observation, device)
        prior_targets = adapter.runtime.control.targets.numpy()[:, 12:19].copy()
        teacher_result = teacher.step() if teacher is not None else None
        distance_before = adapter.reward.distance() if teacher is not None else None
        with torch.no_grad():
            mean, logstd, value, next_memory = policy(rgbd, r84, memory)
            raw = mean if deterministic else mean + logstd.exp() * torch.randn_like(mean)
            logp = tanh_logprob(raw, mean, logstd)
            delta = .10 * raw.tanh().cpu().numpy()
        targets = teacher_result['targets'] if teacher_result is not None else prior_targets + delta
        observation, applied, reward, info = adapter.step(targets, control_dt, gait)
        chassis = adapter.runtime.control.chassis
        heights = adapter.runtime.data.xpos.numpy()[:, chassis, 2]
        rotations = adapter.runtime.data.xmat.numpy()[:, chassis]
        fallen = (heights < .3) | (rotations[:, 2, 2] < np.cos(.8))
        if reward is None or not np.isfinite(reward).all():
            raise RuntimeError('Physical reach reward is missing or nonfinite')
        reward = np.asarray(reward) - fallen.astype(np.float32)
        row = dict(rgbd=rgbd, r84=r84, raw=raw.detach(), logp=logp.detach(), value=value.detach(),
                         reward=torch.as_tensor(reward, device=device), applied=applied,
                         terminated=torch.as_tensor(fallen, device=device),
                         distance=info['tcp_fruit_distance_m'].copy())
        if teacher_result is not None:
            row.update(teacher_label=torch.as_tensor((applied - prior_targets) / .10, device=device),
                      teacher_distance_before=distance_before.copy(),
                      teacher_distance_after=np.asarray(info['tcp_fruit_distance_m']).copy())
        if deterministic:
            row['qpos'] = adapter.runtime.data.qpos.numpy().copy()
        rows.append(row)
        memory = next_memory.detach()
        if fallen.any():
            break
        print(json.dumps(dict(event='physical_step', step=index + 1,
            distance_m=np.asarray(info['tcp_fruit_distance_m']).tolist(), reward=np.asarray(reward).tolist())), flush=True)
    with torch.no_grad():
        _, _, bootstrap, _ = policy(*tensors(observation, device), memory)
    return rows, bootstrap


def imitation_update(policy, optimizer, rows):
    import torch
    memory = torch.zeros(len(rows[0]['reward']), 64, device=rows[0]['rgbd'].device)
    predictions, labels = [], []
    for row in rows:
        mean, _, _, memory = policy(row['rgbd'], row['r84'], memory)
        predictions.append(mean.tanh())
        labels.append(row['teacher_label'])
    loss = (torch.stack(predictions) - torch.stack(labels)).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite imitation loss')
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    changed = [name for name, p in policy.named_parameters() if not torch.equal(p, before[name])]
    if not any(name.startswith('vision.') for name in changed) or 'mean.weight' not in changed:
        raise RuntimeError('Imitation update did not change both vision and action weights')
    before_distance = np.concatenate([row['teacher_distance_before'].ravel() for row in rows])
    after_distance = np.concatenate([row['teacher_distance_after'].ravel() for row in rows])
    return dict(loss=float(loss.detach()), grad_norm=float(grad), vision_changed=True,
                action_changed=True, teacher_distance_change=float(np.mean(after_distance - before_distance)),
                teacher_distance_initial=float(np.mean(rows[0]['teacher_distance_before'])),
                teacher_distance_final=float(np.mean(rows[-1]['teacher_distance_after'])),
                teacher_improved=bool(np.all(rows[-1]['teacher_distance_after'] < rows[0]['teacher_distance_before'])
                                      and not any(bool(row['terminated'].any()) for row in rows)))


def update(policy, optimizer, rows, bootstrap):
    import torch
    from treesim.kiwi_rl.ppo import compute_gae_torch, tanh_logprob
    rewards = torch.stack([row['reward'] for row in rows])
    if float(rewards.std(unbiased=False)) < 1e-10:
        raise RuntimeError('Measured task rewards do not vary; refuse an entropy-only learning claim')
    values = torch.stack([row['value'] for row in rows])
    next_values = torch.cat((values[1:], bootstrap[None]))
    ended = torch.stack([row.get('terminated', torch.zeros_like(row['reward'], dtype=torch.bool)) for row in rows])
    truncated = ended.clone()
    truncated[-1] = True
    advantages = compute_gae_torch(rewards, values, next_values, ended, truncated, .99, .95)
    returns = (advantages + values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    memory = torch.zeros(len(bootstrap), 64, device=bootstrap.device)
    new_logp, predictions = [], []
    for row in rows:
        mean, logstd, value, memory = policy(row['rgbd'], row['r84'], memory)
        new_logp.append(tanh_logprob(row['raw'], mean, logstd))
        predictions.append(value)
    logratio = torch.stack(new_logp) - torch.stack([row['logp'] for row in rows])
    ratio = logratio.exp()
    kl = ((ratio - 1.) - logratio).mean().detach()
    if not torch.isfinite(kl) or float(kl) > .02:
        raise RuntimeError(f'Rollout/replay policy mismatch: KL={float(kl)}')
    actor_loss = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
    value_loss = .5 * (torch.stack(predictions) - returns).square().mean()
    loss = actor_loss + .5 * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    changed = [name for name, p in policy.named_parameters() if not torch.equal(p, before[name])]
    if not any(name.startswith('vision.') for name in changed) or 'mean.weight' not in changed:
        raise RuntimeError('Task update did not change both vision and action weights')
    return dict(loss=float(loss.detach()), grad_norm=float(grad), kl=float(kl),
                reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                vision_changed=True, action_changed=True)


def _run(args, training_log):
    import torch
    from treesim.kiwi_rl.runtime import BatchedDeformableRuntime
    from treesim.kiwi_rl.physical_rollout import BatchedPhysicalRollout, RuntimeReachReward
    from treesim.kiwi_rl.control import load_gait_artifact, gait_cpu_inference
    from treesim.kiwi_rl.contact_acceptance import read_contact_gate
    from treesim.kiwi_rl.reach_teacher import PrivilegedReachTeacher
    from treesim.kiwi_rl.ppo import save_checkpoint, load_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.scene / 'manifest.json').read_text())
    gate = read_contact_gate(args.contact_gate, manifest, max_penetration_m=.002)
    if not gate['accepted']:
        raise RuntimeError(f'Contact evidence failed the hackathon screen: {gate}')
    started = time.monotonic()
    def fresh_episode():
        # Reusing MJWarp state after reset produced nonfinite values in live
        # repeated-episode tests. Rebuild the runtime until cache reset is fixed.
        runtime = BatchedDeformableRuntime(args.scene, worlds=args.worlds, resolution=(64, 64))
        return BatchedPhysicalRollout(runtime, RuntimeReachReward(runtime), camera=args.camera)
    gait_actor = load_gait_artifact(args.gait_checkpoint)
    gait = lambda obs: gait_cpu_inference(gait_actor, obs)
    policy = build_policy().to('cuda:0')
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    meta = dict(schema=SCHEMA, model_sha256=manifest['model_sha256'], seed=args.seed, camera=args.camera,
                action='seven bounded joint-target increments, maximum 0.10 rad per control step',
                algorithm=args.algorithm, imitation_epochs=args.imitation_epochs if args.algorithm == 'imitation' else 0,
                control_dt_s=args.control_dt, gait_sha256=hashlib.sha256(args.gait_checkpoint.read_bytes()).hexdigest(),
                contact_acceptance=gate, scope='physical reaching only; not grasping, deposit or full orchard readiness',
                reset_strategy='fresh simulation per episode; cached reset remains under investigation',
                source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                    for name in ('scripts/train_physical_smoke.py', 'treesim/kiwi_rl/physical_rollout.py',
                                 'treesim/kiwi_rl/runtime.py', 'treesim/kiwi_rl/control_warp.py', 'treesim/kiwi_rl/reach_teacher.py')})
    completed = 0
    initialized_from = None
    initialized_source_model_sha256 = None
    if args.resume:
        saved = load_checkpoint(args.resume, {'student': policy}, {'student': optimizer},
            expected_meta={key: meta[key] for key in ('schema', 'model_sha256', 'gait_sha256', 'control_dt_s', 'camera')})
        saved_algorithm = saved['meta'].get('algorithm', 'ppo')
        if saved_algorithm != args.algorithm and not args.allow_algorithm_change:
            raise RuntimeError('Checkpoint algorithm differs; pass --allow-algorithm-change explicitly')
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        completed = saved['meta']['completed_updates']
    elif args.initialize_from:
        saved = load_checkpoint(args.initialize_from, {'student': policy}, None,
            expected_meta={key: meta[key] for key in ('schema', 'gait_sha256', 'control_dt_s', 'camera')})
        initialized_from = str(args.initialize_from)
        initialized_source_model_sha256 = saved['meta'].get('model_sha256')
        meta['initialized_from'] = initialized_from
        meta['initialized_source_model_sha256'] = initialized_source_model_sha256
    baseline = None
    if args.algorithm == 'imitation':
        adapter = fresh_episode()
        baseline_rows, _ = collect(adapter, policy.eval(), args.steps, gait, deterministic=True, control_dt=args.control_dt)
        baseline = dict(distance_m=[row['distance'].tolist() for row in baseline_rows],
                        fallen=baseline_rows[-1]['terminated'].cpu().tolist(), sensor_only=True)
        del adapter
        policy.train()
    reports = []
    transitions = 0
    for iteration in range(args.updates):
        update_started = time.monotonic()
        adapter = fresh_episode()
        teacher = PrivilegedReachTeacher(adapter.runtime) if args.algorithm == 'imitation' else None
        rows, bootstrap = collect(adapter, policy, args.steps, gait, control_dt=args.control_dt,
                                  teacher=teacher)
        np.savez_compressed(args.output / f'rollout-{completed + 1:04d}.npz',
            joint_targets_rad=np.stack([row['applied'] for row in rows]),
            rewards=np.stack([row['reward'].cpu().numpy() for row in rows]),
            distance_m=np.stack([row['distance'] for row in rows]), control_dt_s=args.control_dt)
        if args.algorithm == 'imitation':
            losses = []
            for _ in range(args.imitation_epochs):
                metrics = imitation_update(policy, optimizer, rows)
                losses.append(metrics['loss'])
            metrics.update(imitation_epochs=args.imitation_epochs, initial_loss=losses[0], final_loss=losses[-1])
        else:
            metrics = update(policy, optimizer, rows, bootstrap)
        completed += 1
        meta['completed_updates'] = completed
        checkpoint = args.output / f'checkpoint-{completed:04d}.pt'
        digest = save_checkpoint(checkpoint, {'student': policy}, {'student': optimizer},
            {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, meta)
        clone = build_policy().to('cuda:0')
        clone_optimizer = torch.optim.Adam(clone.parameters(), lr=3e-4)
        saved = load_checkpoint(checkpoint, {'student': clone}, {'student': clone_optimizer},
            expected_meta={'schema': SCHEMA, 'model_sha256': manifest['model_sha256']})
        for name, parameter in policy.state_dict().items():
            torch.testing.assert_close(parameter, clone.state_dict()[name], rtol=0, atol=0)
        zero_memory = torch.zeros(args.worlds, 64, device='cuda:0')
        with torch.no_grad():
            expected = policy(rows[0]['rgbd'], rows[0]['r84'], zero_memory)
            actual = clone(rows[0]['rgbd'], rows[0]['r84'], zero_memory)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        transitions += len(rows) * args.worlds
        metrics.update(transitions=transitions,
            transitions_per_second=len(rows) * args.worlds / (time.monotonic() - update_started),
            elapsed_seconds=time.monotonic() - started)
        metrics.update(event='optimizer_update', update=completed, checkpoint=str(checkpoint), sha16=digest)
        reports.append(metrics)
        training_log.log(metrics, step=completed)
        training_log.log_checkpoint(checkpoint, step=completed)
        print(json.dumps(metrics), flush=True)
        del adapter
    adapter = fresh_episode()
    evaluation, _ = collect(adapter, clone.eval(), args.steps, gait, deterministic=True, control_dt=args.control_dt)
    from PIL import Image
    frame = evaluation[-1]['rgbd'][0, :3].detach().cpu().numpy().transpose(1, 2, 0)
    Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(args.output / 'policy-camera.png')
    report = dict(scope=meta['scope'], schema=SCHEMA, algorithm=args.algorithm,
        real_physical_updates=len(reports),
        resumed_from=str(args.resume) if args.resume else None, completed_updates=completed,
        initialized_from=initialized_from,
        initialized_source_model_sha256=initialized_source_model_sha256,
        checkpoint_reload_exact=True, sensor_only_actor=True, training_ready=False,
        contact_acceptance=gate, updates=reports, numerical=adapter.runtime.monitor.check(),
        reset_strategy=meta['reset_strategy'], baseline_evaluation=baseline,
        evaluation=dict(distance_m=[row['distance'].tolist() for row in evaluation],
                        reward=[row['reward'].cpu().tolist() for row in evaluation],
                        fallen=evaluation[-1]['terminated'].cpu().tolist(),
                        sensor_only=True, note='short deterministic reach evaluation; no harvesting success claim'),
        elapsed_seconds=time.monotonic() - started)
    report['wandb_url'] = training_log.url
    training_log.log({
        'evaluation/final_distance_m': float(np.mean(evaluation[-1]['distance'])),
        'evaluation/closest_distance_m': float(np.min([row['distance'] for row in evaluation])),
        'evaluation/fall_fraction': float(evaluation[-1]['terminated'].float().mean()),
        'evaluation/reward_mean': float(torch.stack([row['reward'] for row in evaluation]).mean()),
    }, step=completed + 1)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve existing training outputs')
    args.output.mkdir(parents=True)
    training_log = TrainingLog(args.output, {
        'approximations': {'policy_input': 'rgbd+r84', 'fruit': 'deformable'},
        'exptseed': args.seed,
        'worlds': args.worlds,
        'rates': {'control_dt_s': args.control_dt},
    }, wandb_mode=args.wandb_mode, wandb_project=args.wandb_project,
       wandb_entity=args.wandb_entity, wandb_name=args.wandb_name,
       upload_checkpoints=args.upload_checkpoints)
    try:
        report = _run(args, training_log)
    except Exception:
        training_log.finish(success=False)
        raise
    training_log.finish(success=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--contact-gate', type=Path, required=True)
    parser.add_argument('--gait-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--resume', type=Path)
    modes.add_argument('--initialize-from', type=Path,
                       help='Load student weights only; start a fresh optimizer and RNG')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--worlds', type=int, default=1)
    parser.add_argument('--control-dt', type=float, default=.04)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', choices=('ppo', 'imitation'), default='ppo')
    parser.add_argument('--allow-algorithm-change', action='store_true')
    parser.add_argument('--imitation-epochs', type=int, default=32)
    parser.add_argument('--camera', choices=('hand_camera', 'body_camera'), default='body_camera')
    add_training_log_args(parser)
    args = parser.parse_args()
    if not 1 <= args.imitation_epochs <= 1000:
        parser.error('Imitation epochs must be within 1 to 1000')
    if not 2 <= args.steps <= 512 or not 1 <= args.updates <= 10000 or not 1 <= args.worlds <= 64:
        parser.error('Invalid rollout dimensions')
    if not np.isfinite(args.control_dt) or not .02 <= args.control_dt <= .1:
        parser.error('Control period must be within .02 to .1 seconds')
    run(args)


if __name__ == '__main__':
    main()
