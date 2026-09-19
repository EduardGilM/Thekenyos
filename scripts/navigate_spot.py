import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Physical Spot A-to-B navigation: scripted goal tracking, pretrained RELIC gait')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start', nargs=2, type=float, default=(-2., 0.), metavar=('X', 'Y'))
    parser.add_argument('--goal', nargs=2, type=float, default=(2., 0.), metavar=('X', 'Y'))
    parser.add_argument('--terrain-amplitude', type=float, default=.05)
    parser.add_argument('--terrain-wavelength', type=float, default=1.8)
    parser.add_argument('--terrain-extent', type=float, default=5.)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--output', type=Path, default=Path('output/spot-navigation'))
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--stand', action='store_true', help='Matched-seed zero-command baseline; success is not required')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if not args.no_video:
        from treesim.gl_backend import nvidia_gpu_present
        if not nvidia_gpu_present():
            parser.error('Video requires NVIDIA EGL; use --no-video for CPU-only hosts')
        os.environ['MUJOCO_GL'] = 'egl'
    from treesim.spot_navigation import NavigationTask, SpotNavigationEnv, goal_action
    try:
        task = NavigationTask(start_xy_m=tuple(args.start), goal_xy_m=tuple(args.goal),
                              terrain_amplitude_m=args.terrain_amplitude,
                              terrain_wavelength_m=args.terrain_wavelength,
                              terrain_extent_m=args.terrain_extent,
                              time_limit_s=args.seconds, physics_hz=args.physics_hz)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    env = SpotNavigationEnv(args.relic, task=task, device=args.device,
                            render_mode=None if args.no_video else 'rgb_array')
    encoder = None
    records = []
    started = time.monotonic()
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('DejaVuSans.ttf', 22) if not args.no_video else None
    info = {'outcome': 'initialization_failed', 'success': False}
    error = None
    try:
        obs, info = env.reset(seed=args.seed)
        initial_position = info['position_m']
        for step in range(int(np.ceil(args.seconds*50))):
            action = np.zeros(3, np.float32) if args.stand or step < 50 else goal_action(obs)
            previous_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            records.append(dict(**info, action=action.tolist(), reward=reward,
                                terminated=terminated, truncated=truncated,
                                observation={k: v.tolist() for k, v in previous_obs.items()},
                                next_observation={k: v.tolist() for k, v in obs.items()}))
            if step % 50 == 0 or terminated or truncated:
                print(f"[navigate] t={info['simulation_time_s']:.2f}s position={np.round(info['position_m'], 3)} "
                      f"distance={info['distance_to_goal_m']:.3f}m tilt={np.degrees(info['tilt_rad']):.1f}deg "
                      f"outcome={info['outcome']} graph={info['cuda_graph']}", flush=True)
            if not args.no_video and (step % 2 == 0 or terminated or truncated):
                image = Image.fromarray(env.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 100), fill=(18, 24, 32))
                draw.text((20, 10), 'SPOT A-to-B | scripted navigation + pretrained RELIC gait | no new training', font=font, fill='white')
                draw.text((20, 40), f"A {args.start}  ->  B {args.goal} m | terrain seed {env.terrain_seed} | relief {args.terrain_amplitude:.2f} m", font=font, fill='white')
                draw.text((20, 70), f"t={info['simulation_time_s']:.2f}s | remaining {info['distance_to_goal_m']:.2f} m | {info['outcome']} | {info['physics_device']} physics + EGL", font=font, fill='white')
                if encoder is None:
                    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                        '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', '25',
                        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output/'navigation.mp4')],
                        stdin=subprocess.PIPE)
                encoder.stdin.write(np.asarray(image).tobytes())
                if step == 0:
                    image.save(args.output/'start.png')
                if terminated or truncated:
                    image.save(args.output/'final.png')
            if terminated or truncated:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        info.update(outcome=env.outcome if env.sim is not None and env.done else 'error', success=False)
        raise
    finally:
        encoder_failed = False
        if encoder:
            encoder.stdin.close()
            encoder_failed = encoder.wait() != 0
        poses = np.array([r['position_m'] for r in records])
        source = Path(__file__).resolve().parents[1]
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__).resolve(), *(source/'treesim'/name for name in
                            ('spot_navigation.py', 'spot.py', 'builder.py', 'config.py', 'sim.py')))}
        revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True).stdout.strip()
        relic_revision = subprocess.run(['git', '-C', str(args.relic.expanduser()), 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True).stdout.strip()
        result = dict(**info, task=asdict(task), episode_seed=args.seed, records=len(records),
                      wall_seconds=time.monotonic()-started, error=error, encoder_failed=encoder_failed,
                      controller='zero command' if args.stand else 'scripted goal tracker',
                      gait='external RELIC pretrained ONNX; CPU inference; no retraining',
                      physics='Newton / MuJoCo-Warp' if env.sim and env.sim.model.device.is_cuda else 'native MuJoCo CPU',
                      terrain=env.sim.tree.terrain_params if env.sim else None,
                      max_tilt_deg=float(max((np.degrees(r['tilt_rad']) for r in records), default=0)),
                      path_length_m=float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum()) if len(poses) > 1 else 0.,
                      source_sha256=hashes, source_revision=revision, relic_revision=relic_revision,
                      versions={name: version(name) for name in
                                ('newton', 'warp-lang', 'mujoco', 'mujoco-warp', 'gymnasium', 'onnxruntime')})
        result['success'] = bool(info['success'] and error is None and not encoder_failed)
        if records:
            result['initial_position_m'] = initial_position
        (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (args.output/'trajectory.json').write_text(json.dumps(records, allow_nan=False)+'\n')
        env.close()
        print(json.dumps(result, indent=2), flush=True)
        if encoder_failed:
            raise RuntimeError('ffmpeg failed')
    if not info['success'] and not (args.stand and info['outcome'] == 'timeout'):
        raise SystemExit(f"Navigation did not succeed: {info['outcome']}; diagnostics preserved in {args.output}")


if __name__ == '__main__':
    main()
