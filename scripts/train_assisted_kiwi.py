import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np


def workspace_error(target_xy):
    return float(np.linalg.norm(np.maximum([.45-target_xy[0], target_xy[0]-.75, abs(target_xy[1])-.15], 0.)))


class WorkspaceApproach(gym.Wrapper):
    def __init__(self, env, weight=0.):
        super().__init__(env)
        if not np.isfinite(weight) or not 0 <= weight <= 4:
            raise ValueError('Workspace shaping weight must be in [0, 4]')
        self.weight = weight

    @property
    def stage(self):
        return self.unwrapped.stage

    def set_stage(self, stage):
        self.unwrapped.set_stage(stage)

    def _error(self):
        env = self.unwrapped
        forward = env.data.xmat[env.chassis].reshape(3, 3)[:2, 0]
        forward = forward/max(np.linalg.norm(forward), 1e-8)
        delta = env.target[:2]-env.data.xpos[env.chassis, :2]
        return workspace_error([forward@delta, np.array([-forward[1], forward[0]])@delta])

    def _metrics(self, info):
        return dict(info, workspace_error_m=self.error, best_workspace_error_m=self.best_error,
                    workspace_reached=self.reached, workspace_settled=self.settled,
                    first_workspace_s=self.first_workspace_s, base_path_m=self.path_m,
                    forward_displacement_m=float(self.unwrapped.data.xpos[self.unwrapped.chassis, 0]-self.initial_xy[0]),
                    combined_success=bool(info['success'] and self.settled))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.initial_xy = self.unwrapped.data.xpos[self.unwrapped.chassis, :2].copy()
        self.previous_xy = self.initial_xy.copy()
        self.error = self.best_error = self._error()
        self.reached = self.error <= 1e-6
        self.settled = False
        self.settle_steps = 0
        self.first_workspace_s = 0. if self.reached else None
        self.path_m = 0.
        return obs, self._metrics(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        error = self._error()
        xy = self.unwrapped.data.xpos[self.unwrapped.chassis, :2].copy()
        self.path_m += float(np.linalg.norm(xy-self.previous_xy))
        self.previous_xy = xy
        reached = error <= 1e-6
        if reached and not self.reached:
            self.first_workspace_s = info['elapsed_s']
        self.reached = self.reached or reached
        self.settle_steps = self.settle_steps+1 if reached and np.linalg.norm(self.unwrapped._base_velocity()[:2]) < .3 else 0
        settled = self.settle_steps >= 3
        shaping = self.weight*(16*(self.error-error)-.12*min(error, 2.)+float(settled and not self.settled)*2.)
        self.settled = self.settled or settled
        self.error, self.best_error = error, min(self.best_error, error)
        info = self._metrics(info)
        info['reward_terms'] = dict(info['reward_terms'], workspace=shaping)
        return obs, reward+shaping, term, trunc, info

    def _info(self):
        return self._metrics(self.unwrapped._info())


ENV_SOURCES = ('treesim/assisted_kiwi_env.py', 'treesim/spot.py')


def emit(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'records'}, allow_nan=False), flush=True)


def checkpoint_compatible(saved, manifest, mode):
    if mode not in ('resume', 'warm_start'):
        raise ValueError('Unknown checkpoint loading mode')
    expected = manifest['source_sha256']
    names = tuple(expected) if mode == 'resume' else ENV_SOURCES
    if any(name not in saved['source_sha256'] or saved['source_sha256'][name] != expected[name] for name in names):
        raise ValueError('Checkpoint source mismatch; environment and physics compatibility are required')
    old_task, new_task = dict(saved['task']), dict(manifest['task'])
    if mode == 'warm_start':
        old_task.pop('stage', None)
        new_task.pop('stage', None)
    if old_task != new_task:
        raise ValueError('Checkpoint task mismatch; capture/retention/physics settings must agree')
    if mode == 'resume' and saved.get('training_config') != manifest['training_config']:
        raise ValueError('Resume requires matching training configuration; use explicit weight-only warm start')


@dataclass
class Curriculum:
    stage: int
    passing_evaluations: int = 0
    require_workspace: bool = False

    def update(self, report, fixed=False):
        successes = report['combined_successes'] if self.require_workspace else report['successes']
        if report['stage'] != self.stage or not 0 <= successes <= report['successes'] <= report['episodes'] or report['episodes'] <= 0:
            raise ValueError('Invalid curriculum evaluation')
        self.passing_evaluations = self.passing_evaluations+1 if successes/report['episodes'] >= .75 else 0
        promoted = not fixed and self.passing_evaluations >= 2 and self.stage < 2
        if promoted:
            self.stage += 1
            self.passing_evaluations = 0
        return promoted


def score(report):
    return (report['successes']/report['episodes'], report.get('combined_successes', 0)/report['episodes'],
            -report.get('mean_best_workspace_error_m', 0.), -report['mean_best_distance_m'])


def make_env(relic, task, workspace_weight=0.):
    from stable_baselines3.common.monitor import Monitor
    from treesim.assisted_kiwi_env import AssistedKiwiEnv, AssistedTask
    return Monitor(WorkspaceApproach(AssistedKiwiEnv(relic, task=AssistedTask(**task)), workspace_weight))


def frame(env, label):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(env.render())
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    draw.rectangle((0, 0, image.width, 95), fill=(18, 24, 32))
    draw.text((15, 8), 'ASSISTED RL | pretrained gait | artificial grip | NOT contact-only grasp', fill='white', font=font)
    draw.text((15, 36), label, fill='white', font=font)
    info = env._info()
    draw.text((15, 64), f"target {info['target_index']} | gap {info['distance_m']:.3f} m | body {info['base_travel_m']:.2f} m | "
              f"workspace {info.get('workspace_error_m', 0.):.2f} m | {info['outcome']}", fill='white', font=font)
    return image


def snapshot(env, path, label):
    frame(env, label).save(path)


def evaluate(model, env, seeds, out, label, images, record_video=False):
    records = []
    saved_success = False
    for index, seed in enumerate(seeds):
        encoder = None
        obs, info = env.reset(seed=seed)
        if images and index == 0:
            snapshot(env, out/f'{label}-start.png', f'{label} | seed {seed} | initial state')
        total = 0.
        try:
            if record_video and index == 0:
                encoder = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '1280x720', '-r', '10', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(out/f'{label}.mp4')], stdin=subprocess.PIPE)
                encoder.stdin.write(np.asarray(frame(env, f'{label} | first fixed seed {seed}')).tobytes())
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                if encoder:
                    encoder.stdin.write(np.asarray(frame(env, f'{label} | seed {seed} | t={info["elapsed_s"]:.1f}s')).tobytes())
                if term or trunc:
                    break
        finally:
            if encoder:
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError('Evaluation video encoding failed')
        records.append(dict(seed=seed, return_sum=total, **info))
        if images and index == 0:
            snapshot(env, out/f'{label}-final.png', f'{label} | seed {seed} | fixed-seed evaluation')
        if images and info['success'] and not saved_success:
            snapshot(env, out/f'{label}-example-success.png', f'{label} | selected success example | seed {seed}')
            saved_success = True
    report = dict(label=label, stage=env.stage, episodes=len(records),
                  successes=sum(int(r['success']) for r in records),
                  mean_best_distance_m=float(np.mean([r['best_distance_m'] for r in records])),
                  mean_base_travel_m=float(np.mean([r['base_travel_m'] for r in records])),
                  workspace_reached=sum(int(r.get('workspace_reached', False)) for r in records),
                  workspace_settled=sum(int(r.get('workspace_settled', False)) for r in records),
                  combined_successes=sum(int(r.get('combined_success', False)) for r in records),
                  mean_best_workspace_error_m=float(np.mean([r.get('best_workspace_error_m', 0.) for r in records])),
                  mean_base_path_m=float(np.mean([r.get('base_path_m', 0.) for r in records])),
                  targets=sorted({r['target_index'] for r in records}),
                  failures={name: sum(r['outcome'] == name for r in records)
                            for name in sorted({r['outcome'] for r in records})}, records=records)
    with (out/f'{label}.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description='PPO for explicitly assisted Spot kiwi approach and capture')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--stage', type=int, choices=range(3), default=0)
    parser.add_argument('--fixed-stage', action='store_true')
    parser.add_argument('--workspace-weight', type=float, default=0.)
    parser.add_argument('--eval-all-stages', action='store_true')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--entropy', type=float, default=.002)
    parser.add_argument('--exploration-std', type=float, default=.45)
    parser.add_argument('--gripper-std', type=float, default=.25)
    parser.add_argument('--eval-every', type=int, default=4096)
    parser.add_argument('--eval-episodes', type=int, default=8)
    parser.add_argument('--heldout-seed', type=int, default=30000)
    parser.add_argument('--episode-seconds', type=float, default=15.)
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--policy-device', default='cpu')
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument('--resume', type=Path)
    loading.add_argument('--warm-start', type=Path)
    parser.add_argument('--no-images', action='store_true')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.eval_episodes, args.num_envs, args.rollout_steps, args.batch_size) <= 0 or args.seed < 0:
        parser.error('Counts must be positive; seed nonnegative')
    rollout_size = args.num_envs*args.rollout_steps
    if not args.smoke and (args.steps % rollout_size or rollout_size < args.batch_size or rollout_size % args.batch_size):
        parser.error('Training steps must be a multiple of rollout size; batch size must divide rollout size')
    for value, low, high in ((args.learning_rate, 1e-7, .01), (args.entropy, 0., .1),
                             (args.exploration_std, .05, 1.), (args.gripper_std, .05, 1.), (args.workspace_weight, 0., 4.)):
        if not np.isfinite(value) or not low <= value <= high:
            parser.error('Learning settings must be finite and in supported ranges')
    if args.heldout_seed < 10000+args.eval_episodes or (args.no_images and args.record_video):
        parser.error('Held-out seeds must follow monitoring seeds; video requires rendering')
    from treesim.assisted_kiwi_env import AssistedKiwiEnv, AssistedTask, SCOPE
    task = AssistedTask(time_limit_s=args.episode_seconds, physics_hz=args.physics_hz, stage=args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *(source/name for name in ENV_SOURCES)]
    training_config = dict(num_envs=args.num_envs, rollout_steps=args.rollout_steps, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, entropy=args.entropy,
                           exploration_std=args.exploration_std, gripper_std=args.gripper_std,
                           n_epochs=5, gamma=.99, gae_lambda=.95, clip_range=.15, target_kl=.02,
                           workspace_weight=args.workspace_weight)
    manifest = dict(scope=SCOPE, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    task=asdict(task), training_config=training_config,
                    physics='Native MuJoCo CPU; floating Spot; fixed simplified pergola; rigid kiwi',
                    assistance='Unchanged 12 cm capture radius, commanded closure, artificial stem release/grip weld, .3s hold',
                    observations='Unchanged ideal simulator target position and proprioception, not camera perception',
                    actions='Unchanged 10 Hz body velocity x/y/yaw, Cartesian hand velocity x/y/z, jaw close/open',
                    workspace='Training-only reward shaping; target .45-.75m forward, within .15m lateral in yaw frame; '
                              'settled for .3s below .3m/s; fixed pre-pick target; evaluation shaping disabled; no action overrides',
                    frozen_gait_sha256=hashlib.sha256((args.relic/'source/relic/relic/assets/spot/pretrained/policy.onnx').read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    source_revision=subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                    relic_revision=subprocess.run(['git', '-C', str(args.relic.resolve()), 'rev-parse', 'HEAD'],
                                                  capture_output=True, text=True).stdout.strip(),
                    versions={n: version(n) for n in ('newton', 'warp-lang', 'mujoco', 'gymnasium', 'onnxruntime', 'numpy', 'scipy')})
    eval_env = WorkspaceApproach(AssistedKiwiEnv(args.relic, task=replace(task),
                                                render_mode=None if args.no_images else 'rgb_array'))
    train_env = None
    try:
        if args.smoke:
            obs, info = eval_env.reset(seed=args.seed)
            if not args.no_images:
                snapshot(eval_env, args.output/'smoke-start.png', 'Scripted smoke test, not learned behavior')
            for _ in range(min(args.steps, int(args.episode_seconds*10))):
                action = np.zeros(7)
                action[3:6] = np.clip(obs[:3]*5., -1., 1.)
                action[6] = 1. if np.linalg.norm(obs[:3]) < task.capture_radius_m*.9 else -1.
                obs, _, term, trunc, info = eval_env.step(action)
                if term or trunc:
                    break
            if not args.no_images:
                snapshot(eval_env, args.output/'smoke-final.png', 'Scripted smoke test, not learned behavior')
            emit(args.output/'progress.jsonl', dict(event='smoke', **info))
            if not info['success']:
                raise RuntimeError(f"Scripted assisted smoke did not succeed: {info['outcome']}")
            return
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.logger import configure
        torch.set_num_threads(1)
        if args.policy_device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA policy device requested but unavailable; no silent fallback')
        manifest['versions'].update({n: version(n) for n in ('torch', 'stable-baselines3')})
        parent = args.resume or args.warm_start
        saved = None
        if parent:
            saved = json.loads(parent.with_suffix('.json').read_text())
            checkpoint_compatible(saved, manifest, 'resume' if args.resume else 'warm_start')
            manifest['parent_checkpoint'] = dict(path=str(parent.resolve()), sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                                                  parent_steps=saved['steps'], mode='resume' if args.resume else 'weights_only')
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        makers = [partial(make_env, str(args.relic.resolve()), asdict(task), args.workspace_weight) for _ in range(args.num_envs)]
        train_env = DummyVecEnv(makers) if args.num_envs == 1 else SubprocVecEnv(makers, start_method='spawn')
        train_env.seed(args.seed)
        model = (PPO.load(parent, env=train_env, device=args.policy_device) if args.resume else
                 PPO('MlpPolicy', train_env, device=args.policy_device, seed=args.seed,
                     n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=5, learning_rate=args.learning_rate,
                     gamma=.99, gae_lambda=.95, ent_coef=args.entropy, clip_range=.15, target_kl=.02,
                     policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])), verbose=0))
        if args.warm_start:
            previous = PPO.load(parent, device=args.policy_device)
            if previous.observation_space != model.observation_space or previous.action_space != model.action_space:
                raise ValueError('Checkpoint observation/action spaces differ')
            model.policy.load_state_dict(previous.policy.state_dict(), strict=True)
            del previous
        if not args.resume:
            with torch.no_grad():
                model.policy.log_std.fill_(np.log(args.exploration_std))
                model.policy.log_std[-1] = np.log(args.gripper_std)
        curriculum = Curriculum(saved['stage'] if args.resume else args.stage, require_workspace=args.workspace_weight > 0)
        train_env.env_method('set_stage', curriculum.stage)
        eval_env.set_stage(curriculum.stage)
        model.set_logger(configure(str(args.output), ['csv']))
        model.save(args.output/'initial-policy.zip')
        initial = torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()]).clone()
        seeds = list(range(10000, 10000+args.eval_episodes))
        initial_report = evaluate(model, eval_env, seeds, args.output, 'initial', not args.no_images)
        emit(args.output/'progress.jsonl', dict(event='baseline', **initial_report))
        (args.output/'initial-policy.json').write_text(json.dumps(dict(source_sha256=manifest['source_sha256'],
            task=manifest['task'], training_config=training_config, stage=curriculum.stage,
            steps=model.num_timesteps, schema='assisted-training/v2'), indent=2)+'\n')
        started = time.monotonic()
        best = {curriculum.stage: dict(checkpoint='initial-policy.zip', score=score(initial_report), evaluation='initial')}
        (args.output/'best.json').write_text(json.dumps(best, indent=2)+'\n')

        class Progress(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last_eval = self.last_text = self.last_report = 0
                self.start_steps = model.num_timesteps
                self.episodes = self.successes = 0

            def _on_step(self):
                for done, info in zip(self.locals['dones'], self.locals['infos']):
                    if done:
                        self.episodes += 1
                        self.successes += int(info['success'])
                if self.num_timesteps-self.last_text >= 1024:
                    elapsed = time.monotonic()-started
                    emit(args.output/'progress.jsonl', dict(event='progress', steps=self.num_timesteps,
                         steps_this_run=self.num_timesteps-self.start_steps, wall_seconds=elapsed,
                         samples_per_second=(self.num_timesteps-self.start_steps)/max(elapsed, 1e-6),
                         stage=curriculum.stage, episodes=self.episodes, exploratory_successes=self.successes))
                    self.last_text = self.num_timesteps
                return True

            def _on_rollout_start(self):
                if self.num_timesteps-self.start_steps-self.last_eval >= args.eval_every:
                    self.last_eval = self.num_timesteps-self.start_steps
                    self.report()

            def report(self):
                if self.last_report == self.num_timesteps:
                    return
                self.last_report = self.num_timesteps
                eval_env.set_stage(curriculum.stage)
                label = f'eval-{self.num_timesteps:08d}-stage-{curriculum.stage}'
                report = evaluate(model, eval_env, seeds, args.output, label, not args.no_images)
                emit(args.output/'progress.jsonl', dict(event='evaluation', steps=self.num_timesteps, **report))
                checkpoint = args.output/f'policy-{self.num_timesteps:08d}.zip'
                if checkpoint.exists():
                    raise FileExistsError(checkpoint)
                model.save(checkpoint)
                checkpoint.with_suffix('.json').write_text(json.dumps(dict(source_sha256=manifest['source_sha256'],
                    task=manifest['task'], training_config=training_config, stage=curriculum.stage,
                    steps=self.num_timesteps, schema='assisted-training/v2'), indent=2)+'\n')
                if curriculum.stage not in best or score(report) > tuple(best[curriculum.stage]['score']):
                    best[curriculum.stage] = dict(checkpoint=checkpoint.name, score=score(report), evaluation=label)
                    (args.output/'best.json').write_text(json.dumps(best, indent=2)+'\n')
                if curriculum.update(report, fixed=args.fixed_stage):
                    train_env.env_method('set_stage', curriculum.stage)
                    emit(args.output/'progress.jsonl', dict(event='promotion_scheduled', stage=curriculum.stage,
                                                          scope='New stage applies at each worker reset; not generalization evidence'))

        callback = Progress()
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        callback.report()
        change = float(torch.linalg.vector_norm(torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()])-initial))
        if not np.isfinite(change) or change <= 0:
            raise RuntimeError('Policy parameters did not update')
        initial_model = PPO.load(args.output/'initial-policy.zip', device=args.policy_device)
        heldout_seeds = list(range(args.heldout_seed, args.heldout_seed+args.eval_episodes))
        heldout = []
        for stage in sorted(best):
            eval_env.set_stage(stage)
            baseline = evaluate(initial_model, eval_env, heldout_seeds, args.output, f'heldout-initial-stage-{stage}', False)
            selected = PPO.load(args.output/best[stage]['checkpoint'], device=args.policy_device)
            result = evaluate(selected, eval_env, heldout_seeds, args.output, f'heldout-best-stage-{stage}',
                              not args.no_images, record_video=args.record_video)
            comparison = dict(stage=stage, initial_successes=baseline['successes'], trained_successes=result['successes'],
                              episodes=result['episodes'], checkpoint=best[stage]['checkpoint'])
            heldout.append(comparison)
            emit(args.output/'progress.jsonl', dict(event='heldout_comparison', **comparison))
        cross_stage = []
        selected_stage = max((stage for stage in best if best[stage]['score'][0] > 0), default=max(best))
        recommended = best[selected_stage]['checkpoint']
        if args.eval_all_stages:
            selected = PPO.load(args.output/recommended, device=args.policy_device)
            for stage in range(3):
                eval_env.set_stage(stage)
                baseline = evaluate(initial_model, eval_env, heldout_seeds, args.output, f'cross-initial-stage-{stage}', False)
                result = evaluate(selected, eval_env, heldout_seeds, args.output, f'cross-trained-stage-{stage}',
                                  not args.no_images, record_video=args.record_video)
                comparison = dict(stage=stage, initial_successes=baseline['successes'], trained_successes=result['successes'],
                                  initial_workspace=baseline['workspace_settled'], trained_workspace=result['workspace_settled'],
                                  initial_combined=baseline['combined_successes'], trained_combined=result['combined_successes'],
                                  episodes=result['episodes'], checkpoint=recommended)
                cross_stage.append(comparison)
                emit(args.output/'progress.jsonl', dict(event='cross_stage_comparison', **comparison))
        if hashlib.sha256((args.relic/'source/relic/relic/assets/spot/pretrained/policy.onnx').read_bytes()).hexdigest() != manifest['frozen_gait_sha256']:
            raise RuntimeError('Frozen RELIC weights changed during training')
        summary = dict(event='finished', steps=model.num_timesteps, steps_this_run=args.steps, parameter_change_l2=change,
                       recommended_checkpoint=recommended, cross_stage=cross_stage, frozen_gait_unchanged=True,
                       wall_seconds=time.monotonic()-started, final_stage=curriculum.stage,
                       evaluated_stages=sorted(best), best=best, heldout=heldout,
                       scope=SCOPE, generalization_validated=False)
        emit(args.output/'progress.jsonl', summary)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    except Exception as exc:
        emit(args.output/'progress.jsonl', dict(event='error', error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        if train_env is not None:
            train_env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
