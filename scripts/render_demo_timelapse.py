"""Cinematic timelapse of a recorded orchard demo: chase camera that follows the robot, HUD with the
harvest count, fixed output length. Replays recorded joint states; no physics is re-run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True, help='Scene directory (scene.xml, manifest.json); may be a visual-only variant')
    p.add_argument('--replay', type=Path, required=True, help='Demo output directory with states.npz and report.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=15.)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--distance', type=float, default=3.4)
    p.add_argument('--elevation', type=float, default=-18.)
    p.add_argument('--azimuth-offset', type=float, default=145., help='Camera azimuth relative to the robot heading (degrees); 180 = directly behind')
    p.add_argument('--title', default='Autonomous kiwi harvest')
    a = p.parse_args()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    states = np.load(a.replay / 'states.npz')['qpos']
    report = json.loads((a.replay / 'report.json').read_text())
    record_dt = report['simulated_seconds'] / max(1, len(states) - 1)
    manifest = json.loads((a.scene / 'manifest.json').read_text())
    robot = manifest['robot']
    model = mujoco.MjModel.from_xml_path(str(a.scene / 'scene.xml'))
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    model.vis.map.znear = .01
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=1080, width=1920)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    chassis = model.body(robot['chassis']).id
    total_frames = int(a.seconds * a.fps)
    indices = np.linspace(0, len(states) - 1, total_frames).round().astype(int)
    # Harvest count over time from the recorded events.
    harvests = sorted(e['t'] for e in report.get('events', []) if e.get('event') == 'harvest' and e.get('outcome') == 'in_basket')
    attempts = sorted(e['t'] for e in report.get('events', []) if e.get('event') == 'harvest')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', str(a.fps),
           '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.output / 'timelapse.mp4')]
    a.output.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    look = None
    azimuth = None
    try:
        for frame, index in enumerate(indices):
            data.qpos[:] = states[index]
            mujoco.mj_forward(model, data)
            pos = data.xpos[chassis].copy()
            rot = data.xmat[chassis].reshape(3, 3)
            yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
            target = pos + np.array([0., 0., .45])
            desired_az = yaw + a.azimuth_offset
            if look is None:
                look, azimuth = target, desired_az
            else:
                look = .85 * look + .15 * target
                delta = (desired_az - azimuth + 180.) % 360. - 180.
                azimuth += .08 * delta
            camera.lookat[:] = look
            camera.distance, camera.azimuth, camera.elevation = a.distance, azimuth, a.elevation
            renderer.update_scene(data, camera=camera, scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image, 'RGBA')
            sim_t = index * record_dt
            done = sum(1 for h in harvests if h <= sim_t)
            tried = sum(1 for h in attempts if h <= sim_t)
            draw.rectangle([0, 0, 1920, 92], fill=(10, 12, 16, 190))
            draw.text((36, 22), a.title, font=font, fill=(255, 255, 255))
            draw.text((1220, 30), f'kiwis in basket  {done} / {report.get("fruits", len(attempts))}', font=font, fill=(255, 214, 90))
            draw.text((36, 1080 - 56), f'sim {sim_t:6.1f} s   |   {report["simulated_seconds"] / a.seconds:.0f}x timelapse   |   learned gait + RL arm policy + scripted deposit',
                      font=small, fill=(230, 230, 230))
            encoder.stdin.write(image.tobytes())
            if frame == total_frames // 3:
                image.save(a.output / 'preview.png')
    finally:
        encoder.stdin.close()
        encoder.wait()
    print(json.dumps(dict(frames=total_frames, seconds=a.seconds, harvested=len(harvests), output=str(a.output / 'timelapse.mp4'))))


if __name__ == '__main__':
    main()
