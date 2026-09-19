import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_assisted_kiwi import frame
from treesim.basket_kiwi_env import BasketTask
from treesim.visual_kiwi_env import VisualKiwiEnv, VisionConfig
from treesim.visual_servo import BodyFirstServo


def main():
    parser = argparse.ArgumentParser(description='Camera geometry and body-first visual-servo experiment; no oracle action controller')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('fixed', 'body_first', 'simultaneous', 'learned'), default='body_first')
    parser.add_argument('--standoff', type=float, default=.5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 12, 13, 14])
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--size', type=int, choices=(64, 96, 192), default=192)
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--stage', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ablation', choices=('none', 'all', 'hand', 'tof'), default='none')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    if args.mode == 'learned' and args.checkpoint is None:
        parser.error('Learned replay requires --checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    vision = VisionConfig(size=args.size)
    task = BasketTask(picks=1, physics_hz=args.physics_hz, time_limit_s=args.seconds, stage=args.stage)
    env = VisualKiwiEnv(args.relic, task=task, vision=vision, guidance_weight=0., render_mode='rgb_array')
    controller = BodyFirstServo(args.standoff, 'fixed' if args.mode == 'learned' else args.mode)
    policy = None
    if args.mode == 'learned':
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        policy = PPO.load(args.checkpoint, device='cuda')
    source = Path(__file__).resolve().parents[1]
    names = ('treesim/visual_kiwi_env.py', 'treesim/visual_servo.py', 'treesim/basket_kiwi_env.py',
             'treesim/assisted_kiwi_env.py', 'treesim/spot.py', 'treesim/basket.py',
             'treesim/visual_kiwi_policy.py', 'scripts/train_assisted_kiwi.py', 'scripts/check_visual_approach.py')
    for name in names:
        path = args.output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source/name).read_bytes())
    gait = args.relic/'source/relic/relic/assets/spot/pretrained/policy.onnx'
    gait_hash = hashlib.sha256(gait.read_bytes()).hexdigest()
    manifest = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    task=asdict(task), vision=asdict(vision), source_sha256={n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},
                    gait_sha256=gait_hash, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                    scope='Frozen learned replay' if policy else 'SCRIPTED camera-only baseline, not PPO learning; assisted grasp only',
                    sensing='Registered wrist RGB/ToF brown-mask components and 3cm fruit radius prior; calibrated robot kinematics/odometry surrogate; no fruit ground truth in actions; no head camera')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    reports = []
    env.set_ablation(args.ablation)
    try:
        for index, seed in enumerate(args.seeds):
            obs, info = env.reset(seed=seed)
            controller.reset()
            trace, encoder = [], None
            def image():
                from PIL import ImageDraw
                canvas = frame(env, '')
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 1280, 125), fill=(18, 24, 32))
                draw.text((15, 10), f"{manifest['scope']} | {args.mode} | standoff {args.standoff:.2f}m | seed {seed}", fill='white')
                draw.text((15, 40), f"t={info['elapsed_s']:.1f}s | {controller.last} ", fill='white')
                draw.text((15, 70), f"{info['outcome']} | peak fruit contact {info['peak_fruit_contact_force_N']:.2f}N | "
                          f"max penetration {1000*info['max_fruit_penetration_m']:.2f}mm | ToF invalid is NOT clear space", fill='white')
                return canvas
            try:
                if args.video and index == 0:
                    encoder = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '1280x720', '-r', '10', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(args.output/'replay.mp4')], stdin=subprocess.PIPE)
                    encoder.stdin.write(np.asarray(image()).tobytes())
                while True:
                    packet = env.sensor_packet()
                    action = policy.predict(obs, deterministic=True)[0] if policy else controller.action(packet)
                    diagnostic = dict(time_s=info['elapsed_s'], **controller.last)
                    if 'estimate_base_m' in controller.last:
                        e = env.env
                        rotation = e.data.xmat[e.chassis].reshape(3, 3)
                        truth = (e.data.xpos[e.fruit_bodies]-e.data.xpos[e.chassis])@rotation
                        diagnostic['nearest_truth_error_m'] = float(np.min(np.linalg.norm(truth-controller.last['estimate_base_m'], axis=1)))
                    trace.append(diagnostic)
                    obs, _, term, trunc, info = env.step(action)
                    if encoder:
                        encoder.stdin.write(np.asarray(image()).tobytes())
                    if term or trunc:
                        break
                if index == 0:
                    image().save(args.output/'final.png')
                    env.sensor_image().save(args.output/'final-sensors.png')
                record = dict(info, seed=seed, trace=trace)
                reports.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != 'trace'}), flush=True)
            finally:
                if encoder:
                    encoder.stdin.close()
                    if encoder.wait():
                        raise RuntimeError('Video encoder failed')
        summary = dict(episodes=len(reports), successes=sum(r['success'] for r in reports),
                       mean_best_distance_m=float(np.mean([r['best_distance_m'] for r in reports])),
                       peak_fruit_contact_force_N=max(r['peak_fruit_contact_force_N'] for r in reports),
                       max_fruit_penetration_m=max(r['max_fruit_penetration_m'] for r in reports),
                       failures={name: sum(r['outcome'] == name for r in reports) for name in sorted({r['outcome'] for r in reports})},
                       scope=manifest['scope'], frozen_gait_unchanged=hashlib.sha256(gait.read_bytes()).hexdigest() == gait_hash)
        (args.output/'result.json').write_text(json.dumps(dict(summary=summary, records=reports), indent=2)+'\n')
        print(json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
