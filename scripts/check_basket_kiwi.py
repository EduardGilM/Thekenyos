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

import mujoco
import numpy as np
from treesim.basket import CENTER, SIZE
from treesim.basket_kiwi_env import BasketKiwiEnv, BasketTask, SCOPE


class ScriptedBasketController:
    def __init__(self):
        self.previous_phase = None
        self.waypoint = 0
        self.release_requested = False
        self.returning = False

    def predict(self, env):
        rotation = env.data.xmat[env.chassis].reshape(3, 3)
        base = env.data.xpos[env.chassis]
        fruit = rotation.T@(env.data.xpos[env.target_body]-base)
        tcp = rotation.T@(env.data.site_xpos[env.tcp]-base)
        action = np.zeros(7)
        action[6] = -1.
        if env.phase != self.previous_phase:
            self.waypoint = 0
            self.release_requested = False
            self.previous_phase = env.phase
            self.retract_goal = tcp+[0., 0., .15]
            self.returning = env.phase == 'approach' and bool(env.deposited.any())
        if env.phase == 'approach':
            ready = np.array([.48, 0., .76])
            if self.returning:
                goals = [np.array([.0, .45, .8]), np.array([.48, .45, .8]), ready]
                if np.linalg.norm(tcp-goals[self.waypoint]) < .06:
                    self.waypoint += 1
                if self.waypoint < len(goals):
                    action[3:6] = np.clip((goals[self.waypoint]-tcp)*4., -1., 1.)
                    return action
                self.returning = False
                self.waypoint = 0
            bearing = np.arctan2(fruit[1], fruit[0])
            action[0] = np.clip((fruit[0]-.55)*2.5, 0., .8)*max(0., 1-abs(bearing))
            action[1] = np.clip(fruit[1]*1.5, -.5, .5)
            action[2] = np.clip(bearing*1.5, -.7, .7)
            reachable = .4 < fruit[0] < .72 and abs(fruit[1]) < .12
            goal = fruit if reachable else ready
            action[3:6] = np.clip((goal-tcp)*4., -1., 1.)
            action[6] = 1. if reachable and env._distance() < .105 else -1.
        elif env.phase == 'carry':
            goals = [np.array([.45, .30, .90]), np.array([-.10, .30, .80]), CENTER+[0., 0., SIZE[2]+.12]]
            if env.task.start_phase == 'release' and env.deposited.sum() == 0:
                self.waypoint = 2
            goal = goals[self.waypoint]
            if np.linalg.norm(fruit-goal) < .045 and self.waypoint < 2:
                self.waypoint += 1
                goal = goals[self.waypoint]
            action[3:6] = np.clip((goal-fruit)*4., -1., 1.)
            over_mouth = np.all(np.abs(fruit[:2]-CENTER[:2]) < [.15, .08])
            above_rim = .05 < fruit[2]-CENTER[2]-SIZE[2] < .22
            if self.waypoint == 2 and over_mouth and above_rim and env.hold_ticks*.02 >= env.task.hold_time_s:
                self.release_requested = True
            action[6] = -1. if self.release_requested else 1.
        else:
            action[3:6] = np.clip((self.retract_goal-tcp)*4., -1., 1.)
        return action


def frame(env, label):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(env.render())
    rotation = env.data.xmat[env.chassis].reshape(3, 3)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = env.data.xpos[env.chassis]+rotation@(CENTER+[0., 0., .12])
    camera.distance, camera.elevation = .9, -50.
    camera.azimuth = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))+120.
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    env.renderer.update_scene(env.data, camera, scene_option=option)
    image.paste(Image.fromarray(env.renderer.render()).resize((384, 216)), (880, 488))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    draw.rectangle((0, 0, image.width, 95), fill=(18, 24, 32))
    draw.text((15, 8), 'SCRIPTED MULTI-KIWI CHECK | frozen RELIC | assisted attachment | free basket release', fill='white', font=font)
    info = env._info()
    draw.text((15, 36), f"{label} | t={info['elapsed_s']:.1f}s | target {info['target_index']} | {info['phase']}", fill='white', font=font)
    draw.text((15, 64), f"deposited {info['deposited_count']}/{env.task.picks} | {info['outcome']} | NOT learned deposit", fill='white', font=font)
    draw.rectangle((880, 488, 1264, 516), fill=(18, 24, 32))
    draw.text((890, 491), 'Basket: released fruit stays free', fill='white', font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description='Scripted physical check for assisted pick, free basket deposit and next target')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--picks', type=int, default=2)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--start-phase', choices=('pick', 'carry', 'release'), default='pick')
    parser.add_argument('--physics-hz', type=int, choices=(1000, 2000), default=1000)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--episode-seconds', type=float, default=120.)
    args = parser.parse_args()
    task = BasketTask(picks=args.picks, stage=args.stage, start_phase=args.start_phase, physics_hz=args.physics_hz,
                      time_limit_s=args.episode_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    names = ('scripts/check_basket_kiwi.py', 'treesim/basket_kiwi_env.py', 'treesim/assisted_kiwi_env.py', 'treesim/spot.py', 'treesim/basket.py')
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    gait = args.relic/'source/relic/relic/assets/spot/pretrained/policy.onnx'
    gait_hash = hashlib.sha256(gait.read_bytes()).hexdigest()
    env = BasketKiwiEnv(args.relic, task=task, render_mode='rgb_array' if args.video else None)
    encoder, rows, events = None, [], []
    try:
        _, info = env.reset(seed=args.seed)
        controller = ScriptedBasketController()
        if args.video:
            frame(env, 'initial state').save(args.output/'start.png')
            encoder = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                        '-s', '1280x720', '-r', '10', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                        '-movflags', '+faststart', str(args.output/'sequence.mp4')], stdin=subprocess.PIPE)
        for step in range(int(task.time_limit_s*10)):
            action = controller.predict(env)
            _, reward, term, trunc, info = env.step(action)
            events.extend(info['events'])
            rows.append(dict(step=step, action=action.tolist(), reward=reward, **info))
            if step % 50 == 0 or info['events'] or term or trunc:
                print(json.dumps({k: v for k, v in info.items() if k != 'reward_terms'}), flush=True)
            if encoder:
                encoder.stdin.write(np.asarray(frame(env, f'seed {args.seed}')).tobytes())
            if term or trunc:
                break
        if args.video:
            frame(env, 'final state').save(args.output/'final.png')
        frozen = hashlib.sha256(gait.read_bytes()).hexdigest() == gait_hash
        report = dict(scope=SCOPE, controller='scripted, not learned', task=asdict(task), seed=args.seed, events=events,
                      source_sha256=hashes, frozen_gait_sha256=gait_hash, frozen_gait_unchanged=frozen, final=info, samples=rows)
        np.savez(args.output/'final-state.npz', qpos=env.data.qpos, qvel=env.data.qvel, eq_active=env.data.eq_active,
                 site_pos=env.model.site_pos, site_quat=env.model.site_quat, target_index=env.target_index, deposited=env.deposited)
        (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not frozen:
            raise RuntimeError('Frozen RELIC weights changed')
        if not info['success']:
            raise RuntimeError(f"Scripted basket sequence failed: {info['outcome']}")
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('Video encoding failed')
        env.close()


if __name__ == '__main__':
    main()
