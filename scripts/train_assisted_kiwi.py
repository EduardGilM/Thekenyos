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

    def set_ablation(self, mode):
        self.env.set_ablation(mode)

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
        return self._metrics(self.env._info())


ENV_SOURCES = ('treesim/assisted_kiwi_env.py', 'treesim/spot.py')


def emit(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'records'}, allow_nan=False), flush=True)


def checkpoint_compatible(saved, manifest, mode):
    if mode not in ('resume', 'warm_start', 'encoder'):
        raise ValueError('Unknown checkpoint loading mode')
    old_vision, new_vision = json.loads(json.dumps(saved.get('vision') or {})), json.loads(json.dumps(manifest.get('vision') or {}))
    if mode == 'encoder':
        if not old_vision or not new_vision:
            raise ValueError('Encoder transfer requires a camera checkpoint')
        old_vision.pop('lesson', None)
        new_vision.pop('lesson', None)
        if old_vision != new_vision:
            raise ValueError('Checkpoint sensor contract mismatch')
        encoder = 'treesim/visual_kiwi_policy.py'
        if saved.get('source_sha256', {}).get(encoder) != manifest['source_sha256'].get(encoder):
            raise ValueError('Visual encoder source mismatch')
        return
    if bool(saved.get('stationary', False)) != bool(manifest.get('stationary', False)):
        raise ValueError('Checkpoint stationary/action contract mismatch')
    if mode == 'resume' and bool(saved.get('estimated_target')) != bool(manifest.get('estimated_target')):
        raise ValueError('Checkpoint estimated-target contract mismatch')
    expected = manifest['source_sha256']
    names = tuple(name for name in expected if mode == 'resume' or name != 'scripts/train_assisted_kiwi.py')
    if mode == 'warm_start' and manifest.get('estimated_target'):
        required = ['treesim/assisted_kiwi_env.py', 'treesim/spot.py']
        if 'treesim/basket_kiwi_env.py' in saved.get('source_sha256', {}) and 'treesim/basket_kiwi_env.py' in expected:
            required.extend(('treesim/basket_kiwi_env.py', 'treesim/basket.py'))
        names = tuple(name for name in names if name in required)
    if any(name not in saved['source_sha256'] or saved['source_sha256'][name] != expected[name] for name in names):
        raise ValueError('Checkpoint source mismatch; environment and physics compatibility are required')
    if mode == 'warm_start' and old_vision and new_vision:
        old_vision.pop('lesson', None)
        new_vision.pop('lesson', None)
    if old_vision != new_vision:
        raise ValueError('Checkpoint sensor contract mismatch')
    old_task, new_task = dict(saved['task']), dict(manifest['task'])
    if mode == 'warm_start':
        curriculum_fields = ('stage', 'start_phase', 'picks')
        if (old_vision and new_vision) or manifest.get('estimated_target'):
            curriculum_fields += ('time_limit_s', 'settle_time_s')
        for name in curriculum_fields:
            old_task.pop(name, None)
            new_task.pop(name, None)
    if old_task != new_task:
        raise ValueError('Checkpoint task mismatch; capture/retention/physics settings must agree')
    if mode == 'resume' and saved.get('training_config') != manifest['training_config']:
        raise ValueError('Resume requires matching training configuration; use explicit weight-only warm start')


def transfer_visual_encoder(model, previous):
    source = getattr(previous.policy, 'pi_features_extractor', None) or previous.policy.features_extractor
    dest = getattr(model.policy, 'pi_features_extractor', None) or model.policy.features_extractor
    weights, target = source.state_dict(), dest.state_dict()
    if weights.keys() != target.keys() or any(weights[key].shape != target[key].shape for key in weights):
        raise ValueError('Visual encoder architecture mismatch')
    dest.load_state_dict(weights, strict=True)


def initialize_walking_policy(policy, exploration_std, gripper_std, *, encoder_transfer):
    import torch
    with torch.no_grad():
        if encoder_transfer:
            policy.action_net.weight.zero_()
            policy.action_net.bias.zero_()
        policy.log_std.fill_(float(np.log(exploration_std)))
        policy.log_std[-1] = float(np.log(gripper_std))
        if encoder_transfer:
            policy.log_std[:-1] = float(np.log(min(exploration_std, .15)))


@dataclass
class Curriculum:
    stage: int
    passing_evaluations: int = 0
    require_workspace: bool = False
    require_stationary: bool = False

    def update(self, report, fixed=False):
        successes = report['combined_successes'] if self.require_workspace else report['successes']
        if self.require_stationary:
            successes = report['stationary_successes']
        if report['stage'] != self.stage or not 0 <= successes <= report['successes'] <= report['episodes'] or report['episodes'] <= 0:
            raise ValueError('Invalid curriculum evaluation')
        s0v = self.require_stationary and self.stage == 0
        threshold, needed, last = (.8, 1, 3) if s0v else (.75, 2, 3 if self.require_stationary else 2)
        self.passing_evaluations = self.passing_evaluations+1 if successes/report['episodes'] >= threshold else 0
        promoted = not fixed and self.passing_evaluations >= needed and self.stage < last
        if promoted:
            self.stage += 1
            self.passing_evaluations = 0
        return promoted


def score(report):
    result = (report['successes']/report['episodes'], report.get('mean_retained_count', 0.),
              report.get('combined_successes', 0)/report['episodes'],
              -report.get('mean_best_workspace_error_m', 0.), -report['mean_best_distance_m'])
    return (report['stationary_successes']/report['episodes'], *result) if 'stationary_successes' in report else result


def make_env(relic, task, workspace_weight=0., basket=False, vision=None, stationary=False,
             estimate=False, estimate_vision=None, guidance_weight=1., view_weight=0.):
    from stable_baselines3.common.monitor import Monitor
    from treesim.assisted_kiwi_env import AssistedKiwiEnv, AssistedTask
    if vision is not None:
        from treesim.basket_kiwi_env import BasketTask
        from treesim.visual_kiwi_env import VisualKiwiEnv, VisionConfig
        env_class = VisualKiwiEnv
        if stationary:
            from treesim.stationary_kiwi_env import StationaryKiwiEnv
            env_class = StationaryKiwiEnv
        return Monitor(env_class(relic, task=BasketTask(**task), vision=VisionConfig(**vision),
                                 guidance_weight=guidance_weight, view_weight=view_weight))
    if basket:
        from treesim.basket_kiwi_env import BasketKiwiEnv, BasketTask
        env = BasketKiwiEnv(relic, task=BasketTask(**task))
        if estimate:
            from treesim.estimated_kiwi_env import EstimatedTarget
            from treesim.visual_kiwi_env import VisionConfig
            env = EstimatedTarget(env, vision=VisionConfig(**(estimate_vision or {})))
        return Monitor(env)
    env = AssistedKiwiEnv(relic, task=AssistedTask(**task))
    if estimate:
        from treesim.estimated_kiwi_env import EstimatedTarget
        from treesim.visual_kiwi_env import VisionConfig
        env = EstimatedTarget(env, vision=VisionConfig(**(estimate_vision or {})))
    return Monitor(WorkspaceApproach(env, workspace_weight))


def frame(env, label):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(env.render())
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    draw.rectangle((0, 0, image.width, 95), fill=(18, 24, 32))
    draw.text((15, 8), 'ASSISTED RL | pretrained gait | artificial grip | NOT contact-only grasp', fill='white', font=font)
    draw.text((15, 36), label, fill='white', font=font)
    info = env._info()
    detail = (f"{info['phase']} | deposited {info['deposited_count']} | start {info['start_phase']}" if 'phase' in info else
              f"workspace {info.get('workspace_error_m', 0.):.2f} m")
    draw.text((15, 64), f"target {info['target_index']} | gap {info['distance_m']:.3f} m | body {info['base_travel_m']:.2f} m | "
              f"{detail} | {info['outcome']}", fill='white', font=font)
    if hasattr(env, 'sensor_image'):
        image.paste(env.sensor_image().resize((640, 235)), (640, 485))
        lesson = (f'ARM ONLY | level {env.stage} | body commands OFF' if info.get('stationary_lesson') else
                  f'{env.vision.lesson} lesson | provisional sensors')
        draw.text((15, 104), f'CAMERA POLICY | {lesson}', fill='white', font=font)
    return image


def snapshot(env, path, label):
    frame(env, label).save(path)
    if hasattr(env, 'sensor_image'):
        env.sensor_image().save(path.with_name(path.stem+'-sensors.png'))


def evaluate(model, env, seeds, out, label, images, record_video=False, video_limit=1, video_fps=10):
    records = []
    saved_success = False
    if video_limit < 1 or video_fps not in range(5, 13):
        raise ValueError('video_limit must be at least 1; video_fps must be 5-12')
    for index, seed in enumerate(seeds):
        encoder = None
        obs, info = env.reset(seed=seed)
        if images and index == 0:
            snapshot(env, out/f'{label}-start.png', f'{label} | seed {seed} | initial state')
        total, events = 0., []
        try:
            if record_video and index < video_limit:
                # One RGB frame per 10 Hz control step; default video_fps=10 is realtime.
                name = f'{label}.mp4' if index == 0 else f'{label}-seed-{seed}.mp4'
                encoder = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '1280x720', '-r', str(video_fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(out/name)], stdin=subprocess.PIPE)
                encoder.stdin.write(np.asarray(frame(env, f'{label} | seed {seed} | initial state')).tobytes())
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                events.extend(info.get('events', []))
                if encoder:
                    encoder.stdin.write(np.asarray(frame(env, f'{label} | seed {seed} | t={info["elapsed_s"]:.1f}s')).tobytes())
                if term or trunc:
                    break
        finally:
            if encoder:
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError('Evaluation video encoding failed')
        records.append(dict(info, seed=seed, return_sum=total, events=events))
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
                  mean_deposited_count=float(np.mean([r.get('deposited_count', 0) for r in records])),
                  mean_retained_count=float(np.mean([r.get('retained_count', 0) for r in records])),
                  spilled_count=sum(len(r.get('spilled_ids', [])) for r in records),
                  targets=sorted({r['target_index'] for r in records}),
                  failures={name: sum(r['outcome'] == name for r in records)
                            for name in sorted({r['outcome'] for r in records})}, records=records)
    if 'estimate_error_m' in records[0]:
        report.update(mean_estimate_error_m=float(np.mean([r['estimate_error_m'] for r in records])),
                      estimate_valid_episodes=sum(int(r.get('estimate_valid', False)) for r in records))
    if records[0].get('stationary_lesson'):
        report.update(stationary_successes=sum(int(r['stationary_success']) for r in records),
                      mean_arm_motion_m=float(np.mean([r['arm_motion_m'] for r in records])),
                      max_base_travel_m=max(r['base_travel_m'] for r in records),
                      peak_fruit_contact_force_N=max(r['peak_fruit_contact_force_N'] for r in records))
    with (out/f'{label}.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description='PPO for explicitly assisted Spot kiwi approach and capture')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--basket', action='store_true')
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--estimate', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--visual-lesson', choices=('grab', 'collect'), default='grab')
    parser.add_argument('--camera-size', type=int, choices=(64, 84, 96), default=64)
    parser.add_argument('--picks', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--start-phase', choices=('pick', 'carry', 'release'), default='pick')
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
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--policy-device', default='cpu')
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument('--resume', type=Path)
    loading.add_argument('--warm-start', type=Path)
    loading.add_argument('--encoder-warm-start', type=Path)
    parser.add_argument('--no-images', action='store_true')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    vision = None
    estimate_vision = None
    if args.stationary:
        args.vision = True
        if args.visual_lesson != 'grab' or args.start_phase != 'pick':
            parser.error('Stationary training is an arm-only grab lesson')
        if args.stage not in range(4):
            parser.error('Stationary stage must be 0-3 (S0v, then 24/32/44 cm)')
    elif args.stage not in range(3):
        parser.error('Stage must be 0-2')
    if args.vision:
        from treesim.visual_kiwi_env import VisionConfig
        args.basket = True
        if args.visual_lesson == 'grab':
            args.picks = 1
        vision = asdict(VisionConfig(size=args.camera_size, lesson=args.visual_lesson))
        if args.visual_lesson == 'grab' and args.start_phase != 'pick':
            parser.error('Visual grab lessons require --start-phase pick')
    if args.estimate and (args.vision or args.smoke or args.encoder_warm_start):
        parser.error('Estimated-target training is not a visual CNN; do not combine with --vision, --smoke or --encoder-warm-start')
    if args.estimate:
        from treesim.visual_kiwi_env import VisionConfig
        estimate_vision = asdict(VisionConfig(size=args.camera_size, lesson='grab'))
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
    if args.basket and (args.workspace_weight or args.smoke):
        parser.error('Basket training uses its own progress rewards; use check_basket_kiwi.py for scripted checks')
    if not args.basket and args.start_phase != 'pick':
        parser.error('Carry/release curriculum requires --basket')
    from treesim.assisted_kiwi_env import AssistedKiwiEnv, AssistedTask, SCOPE
    default_seconds = 120. if args.basket and not (args.vision and args.visual_lesson == 'grab') else 30.
    if args.stationary:
        default_seconds = 6.
    args.episode_seconds = args.episode_seconds if args.episode_seconds is not None else default_seconds
    task = AssistedTask(time_limit_s=args.episode_seconds, physics_hz=args.physics_hz,
                        stage=0 if args.stationary else args.stage)
    if args.basket:
        from treesim.basket_kiwi_env import BasketKiwiEnv, BasketTask, SCOPE
        task = BasketTask(**asdict(task), picks=args.picks, start_phase=args.start_phase)
        if args.start_phase != 'pick':
            args.fixed_stage = True
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    names = ENV_SOURCES+('treesim/basket_kiwi_env.py', 'treesim/basket.py') if args.basket else ENV_SOURCES
    if args.vision:
        names += ('treesim/visual_kiwi_env.py', 'treesim/visual_kiwi_policy.py', 'treesim/visual_servo.py')
    if args.stationary:
        names += ('treesim/stationary_kiwi_env.py',)
    if args.estimate:
        names += ('treesim/estimated_kiwi_env.py', 'treesim/visual_kiwi_env.py', 'treesim/visual_servo.py')
    paths = [Path(__file__).resolve(), *(source/name for name in names)]
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
    if args.basket:
        manifest.update(assistance='12 cm capture and artificial stem/grip weld; actual opening removes grip; free settling deposit',
                        observations='99 ideal target/proprioception, basket displacement, phase, deposited and remaining-fruit values',
                        basket='Existing 1.2kg chassis-mounted geometry; independent fruit; drops/spills fail; no basket attachment',
                        curriculum_start=args.start_phase, evaluation_guidance_weight=0.)
    eval_env = (BasketKiwiEnv(args.relic, task=replace(task), guidance_weight=0.,
                              render_mode=None if args.no_images else 'rgb_array') if args.basket else
                WorkspaceApproach(AssistedKiwiEnv(args.relic, task=replace(task),
                                                 render_mode=None if args.no_images else 'rgb_array')))
    if args.vision:
        from treesim.visual_kiwi_env import VisualKiwiEnv, VisionConfig, SCOPE
        eval_env.close()
        env_class = VisualKiwiEnv
        if args.stationary:
            from treesim.stationary_kiwi_env import StationaryKiwiEnv, GAPS_M, OFF_AXIS_M, SCOPE
            env_class = StationaryKiwiEnv
        eval_env = env_class(args.relic, task=replace(task), vision=VisionConfig(**vision), guidance_weight=0.,
                                 view_weight=0., render_mode=None if args.no_images else 'rgb_array')
        manifest.update(scope=SCOPE, vision=vision,
                        observations='Actor: stacked wrist RGB-D, 22-D proprioception, phase, grasp/place flags and ages. '
                                     'Critic: privileged 99-D basket state plus proprioception. No fruit XYZ, IDs or segmentation on the actor',
                        actions='10Hz body velocity x/y/yaw, Cartesian hand velocity XYZ, nullspace wrist angular XYZ, jaw; '
                                'bounded IK and unchanged joint torque/collision limits',
                        sensing='Named MuJoCo ee_cam/ee_depth on the wrist; Boston Dynamics gripper vertical FOV 46.4deg RGB and 44deg depth; '
                                'no chassis head camera; 10Hz, one-frame delay, four-frame history; 5% frame loss, 3% depth dropout, '
                                '0.15-3m optical-axis ToF; pose/light/color/gain DR; cameras on at train and eval',
                        scene_randomization='Independent fruit XYZ reset offsets after arm initialization; any unpicked fruit may be captured',
                        workspace='No workspace wrapper or action teacher; oracle-only distance shaping disabled at evaluation',
                        evaluation_view_weight=0.)
    if args.stationary:
        for key in ('exploration_std', 'gripper_std'):
            training_config.pop(key)
        training_config.update(distribution='MultiCategorical', share_features_extractor=False,
                               guidance_weight=0., view_weight=1., policy='AsymmetricVisualPolicy')
        manifest.update(stationary=True, scope=SCOPE, training_guidance_weight=0., training_view_weight=1.,
                        actions='Categorical Cartesian XYZ (-.1575,0,.1575 m/s) and binary open/close; body and wrist angular commands zero',
                        stationary_curriculum=dict(s0v_range_m=(.20, .40), tcp_gaps_m=GAPS_M, off_axis_m=OFF_AXIS_M,
                            promotion='S0v: one >=80% in-FOV pregrasp pass, cameras on; then two consecutive >=75% arm-only assisted holds at 24/32/44 cm'),
                        scene_randomization='Same robot spawn at every level; stage 0 places fruit in the wrist FOV at 20-40 cm; later levels offset in body YZ beyond the 12 cm weld; '
                                            'constant body-+X reach cannot capture later levels; low hanging practice fixture, not full canopy harvesting',
                        workspace='No workspace wrapper; privileged distance/capture shaping off; view in_view/centering/view_loss on in training, off at evaluation')
    if args.estimate:
        from treesim.estimated_kiwi_env import EstimatedTarget, SCOPE
        from treesim.visual_kiwi_env import VisionConfig
        eval_env.close()
        inner = (BasketKiwiEnv(args.relic, task=replace(task), guidance_weight=0.,
                               render_mode=None if args.no_images else 'rgb_array') if args.basket else
                 AssistedKiwiEnv(args.relic, task=replace(task),
                                 render_mode=None if args.no_images else 'rgb_array'))
        estimated = EstimatedTarget(inner, vision=VisionConfig(**estimate_vision))
        eval_env = estimated if args.basket else WorkspaceApproach(estimated, 0.)
        manifest.update(scope=SCOPE, estimated_target=True, estimate_cameras=estimate_vision,
                        observations='Camera-estimated fruit XYZ in TCP and chassis frames plus proprioception; '
                                     'no simulator fruit coordinates in actor inputs',
                        sensing='Provisional hand RGB and hand ToF; delayed packet; odometry hold on missed detections; '
                                'held fruit uses gripper TCP, not simulator pose; far dummy if never seen; '
                                'diagnostic estimate_error_m is info-only',
                        workspace='Training-only privileged workspace shaping on grab lessons; evaluation shaping disabled')
        if args.basket:
            manifest.update(observations='78-D estimated fruit XYZ plus proprioception; basket extras are not actor inputs; '
                                         'success is free rear-basket deposit and 0.5 s settling',
                            basket='Existing 1.2kg chassis-mounted geometry; independent fruit; drops/spills fail',
                            workspace='No workspace wrapper; basket progress rewards only; evaluation guidance disabled')
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
        parent = args.resume or args.warm_start or args.encoder_warm_start
        saved = None
        if parent:
            saved = json.loads(parent.with_suffix('.json').read_text())
            mode = 'resume' if args.resume else 'encoder' if args.encoder_warm_start else 'warm_start'
            checkpoint_compatible(saved, manifest, mode)
            manifest['parent_checkpoint'] = dict(path=str(parent.resolve()), sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                                                  parent_steps=saved['steps'],
                                                  mode='resume' if args.resume else 'encoder' if args.encoder_warm_start else 'weights_only')
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        makers = [partial(make_env, str(args.relic.resolve()), asdict(task), args.workspace_weight, args.basket, vision,
                          args.stationary, args.estimate, estimate_vision, 0. if args.stationary else 1.,
                          1. if args.vision else 0.)
                  for _ in range(args.num_envs)]
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        policy_class = 'MlpPolicy'
        if args.vision:
            from treesim.visual_kiwi_policy import ActorVisualExtractor, AsymmetricVisualPolicy
            policy_class = AsymmetricVisualPolicy
            policy_kwargs.update(features_extractor_class=ActorVisualExtractor, share_features_extractor=False,
                                 ortho_init=False)
        train_env = DummyVecEnv(makers) if args.num_envs == 1 else SubprocVecEnv(makers, start_method='spawn')
        train_env.seed(args.seed)
        model = (PPO.load(parent, env=train_env, device=args.policy_device) if args.resume else
                 PPO(policy_class, train_env, device=args.policy_device, seed=args.seed,
                     n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=5, learning_rate=args.learning_rate,
                     gamma=.99, gae_lambda=.95, ent_coef=args.entropy, clip_range=.15, target_kl=.02,
                     policy_kwargs=policy_kwargs, verbose=0))
        if args.warm_start:
            previous = PPO.load(parent, device=args.policy_device)
            if previous.observation_space != model.observation_space or previous.action_space != model.action_space:
                raise ValueError('Checkpoint observation/action spaces differ')
            model.policy.load_state_dict(previous.policy.state_dict(), strict=True)
            del previous
        if args.encoder_warm_start:
            previous = PPO.load(parent, device=args.policy_device)
            if previous.observation_space != model.observation_space:
                raise ValueError('Encoder transfer requires matching camera observations')
            transfer_visual_encoder(model, previous)
            del previous
        if not args.resume and not args.stationary:
            initialize_walking_policy(model.policy, args.exploration_std, args.gripper_std,
                                     encoder_transfer=bool(args.encoder_warm_start))
        curriculum = Curriculum(saved['stage'] if args.resume else args.stage, require_workspace=args.workspace_weight > 0,
                                require_stationary=args.stationary)
        train_env.env_method('set_stage', curriculum.stage)
        eval_env.set_stage(curriculum.stage)
        model.set_logger(configure(str(args.output), ['csv']))
        model.save(args.output/'initial-policy.zip')
        initial = torch.cat([p.detach().flatten().cpu() for p in model.policy.parameters()]).clone()
        seeds = list(range(10000, 10000+args.eval_episodes))
        initial_report = evaluate(model, eval_env, seeds, args.output, 'initial', not args.no_images)
        emit(args.output/'progress.jsonl', dict(event='baseline', **initial_report))
        (args.output/'initial-policy.json').write_text(json.dumps(dict(source_sha256=manifest['source_sha256'],
            task=manifest['task'], vision=vision, stationary=args.stationary, estimated_target=args.estimate,
            training_config=training_config, stage=curriculum.stage,
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
                    task=manifest['task'], vision=vision, stationary=args.stationary, estimated_target=args.estimate,
                    training_config=training_config, stage=curriculum.stage,
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
                              not args.no_images, record_video=args.record_video, video_limit=4)
            comparison = dict(stage=stage, initial_successes=baseline['successes'], trained_successes=result['successes'],
                              episodes=result['episodes'], checkpoint=best[stage]['checkpoint'])
            if args.stationary:
                comparison.update(initial_stationary_successes=baseline['stationary_successes'],
                                  trained_stationary_successes=result['stationary_successes'])
            heldout.append(comparison)
            emit(args.output/'progress.jsonl', dict(event='heldout_comparison', **comparison))
        cross_stage = []
        selected_stage = max((stage for stage in best if best[stage]['score'][0] > 0), default=max(best))
        recommended = best[selected_stage]['checkpoint']
        if args.eval_all_stages:
            selected = PPO.load(args.output/recommended, device=args.policy_device)
            for stage in range(4 if args.stationary else 3):
                eval_env.set_stage(stage)
                baseline = evaluate(initial_model, eval_env, heldout_seeds, args.output, f'cross-initial-stage-{stage}', False)
                result = evaluate(selected, eval_env, heldout_seeds, args.output, f'cross-trained-stage-{stage}',
                                  not args.no_images, record_video=args.record_video, video_limit=4)
                comparison = dict(stage=stage, initial_successes=baseline['successes'], trained_successes=result['successes'],
                                  initial_workspace=baseline['workspace_settled'], trained_workspace=result['workspace_settled'],
                                  initial_combined=baseline['combined_successes'], trained_combined=result['combined_successes'],
                                  episodes=result['episodes'], checkpoint=recommended)
                if args.stationary:
                    comparison.update(initial_stationary_successes=baseline['stationary_successes'],
                                      trained_stationary_successes=result['stationary_successes'])
                cross_stage.append(comparison)
                emit(args.output/'progress.jsonl', dict(event='cross_stage_comparison', **comparison))
        ablations = []
        if (args.vision or args.estimate) and not args.stationary:
            selected = PPO.load(args.output/recommended, device=args.policy_device)
            eval_env.set_stage(selected_stage)
            for mode in ('none', 'all', 'hand', 'tof'):
                eval_env.set_ablation(mode)
                report = evaluate(selected, eval_env, heldout_seeds, args.output, f'camera-ablation-{mode}',
                                  not args.no_images and mode in ('none', 'all'),
                                  record_video=args.record_video and mode in ('none', 'all'), video_limit=2)
                comparison = dict(mode=mode, episodes=report['episodes'], successes=report['successes'],
                                  mean_best_distance_m=report['mean_best_distance_m'], failures=report['failures'])
                ablations.append(comparison)
                emit(args.output/'progress.jsonl', dict(event='camera_ablation', **comparison))
            eval_env.set_ablation('none')
        if hashlib.sha256((args.relic/'source/relic/relic/assets/spot/pretrained/policy.onnx').read_bytes()).hexdigest() != manifest['frozen_gait_sha256']:
            raise RuntimeError('Frozen RELIC weights changed during training')
        summary = dict(event='finished', steps=model.num_timesteps, steps_this_run=args.steps, parameter_change_l2=change,
                       recommended_checkpoint=recommended, cross_stage=cross_stage, frozen_gait_unchanged=True,
                       wall_seconds=time.monotonic()-started, final_stage=curriculum.stage,
                       evaluated_stages=sorted(best), best=best, heldout=heldout,
                       scope=SCOPE, camera_ablations=ablations, generalization_validated=False)
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
