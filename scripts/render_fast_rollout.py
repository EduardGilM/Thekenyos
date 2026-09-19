"""Record a policy rollout and render synchronized world and arm-camera views."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def main():
    import mujoco
    import torch
    import warp as wp
    from PIL import Image, ImageDraw, ImageFont
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import load_checkpoint
    from train_physical_smoke import build_policy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gait-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--role', choices=('student', 'teacher'), default='student',
                   help='Teacher uses privileged state; video is labelled accordingly')
    p.add_argument('--replay', type=Path, help='Re-render recorded states without another physics rollout')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.replay:
        states = np.load(a.replay/'states.npz')['qpos']
        metadata = json.loads((a.replay/'report.json').read_text())
        metadata.pop('front_camera', None)
        robot = json.loads((a.scene/'manifest.json').read_text())['robot']
    else:
        torch.set_num_threads(1)
        wp.init()
        stream = torch.cuda.Stream()
        states = []
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            if a.role == 'teacher':
                from treesim.kiwi_rl.fast_teacher import build_privileged_policy, privileged_observation
                policy = build_privileged_policy().cuda().eval()
            else:
                policy = build_policy().cuda().eval()
            scene_manifest = json.loads((a.scene/'manifest.json').read_text())
            expected = ({'role': 'teacher', 'model_sha256': scene_manifest['model_sha256']}
                        if a.role == 'teacher' else
                        {'camera': 'hand_camera', 'camera_profile': scene_manifest['cameras']})
            saved = load_checkpoint(a.checkpoint, {a.role: policy}, expected_meta=expected)
            arm_speed = saved['meta'].get('config', {}).get('arm_speed_rad_s', 2.5)
            solver_iterations = saved['meta'].get('config', {}).get('solver_iterations', 20)
            jaw_cap = saved['meta'].get('config', {}).get('jaw_cap_Nm', .3)
            reward_profile=saved['meta'].get('config',{}).get('reward_profile','potential-harvest/v1')
            graph_profile=reward_profile=='graph-harvest/v1'
            rt = FastRuntime(a.scene, worlds=1, camera='hand_camera',arm_speed_rad_s=arm_speed,
                             solver_iterations=solver_iterations, jaw_cap_Nm=jaw_cap,
                             task_profile=reward_profile if graph_profile else None)
            gait = load_gait_artifact(a.gait_checkpoint).cuda().eval()
            from treesim.kiwi_rl.harvest_training import EpisodeProgress, signals
            progress=EpisodeProgress(signals(rt), reward_profile=reward_profile, guidance=0.,
                stall_steps=round(saved['meta'].get('config',{}).get('stall_seconds',4.)/.02),
                max_steps=round(saved['meta'].get('config',{}).get('max_episode_seconds',30.)/.02))
            memory = torch.zeros(1,64,device='cuda')
            with torch.no_grad():
                for i in range(a.steps):
                    if i % 2 == 0:
                        rgbd = rt.pixels()
                        states.append(rt.data.qpos.numpy()[0].copy())
                    obs = rt.observe()
                    inputs = privileged_observation(rt) if a.role == 'teacher' else rgbd
                    mean, _, _, memory = policy(inputs, obs, memory)
                    rt.set_gait_actions(gait(obs))
                    _, _, done, info = rt.step(mean.tanh())
                    _,terminated,truncated,_=progress.step(signals(rt),done)
                    done=terminated|truncated
                    if bool(done.any()):
                        states.append(rt.data.qpos.numpy()[0].copy())
                        break
            numerical = rt.check()
            metadata = dict(checkpoint=str(a.checkpoint), role=a.role, scene=str(a.scene), frames=len(states), fps=25,
                            simulated_seconds=(i+1)*.02, terminated=bool(done.any()),
                            success=bool(info['success'][0]), final_distance_m=float(info['distance_m'][0]),
                            numerical=numerical,reward_profile=reward_profile,
                            arm_camera='same sensor pose and FOV; rendered at higher resolution than policy input',
                            arm_speed_rad_s=arm_speed,solver_iterations=solver_iterations)
            if graph_profile:
                metadata.update(graph_score=float(progress.graph_score[0]),
                    graph_stage=int(progress.graph_stage[0]),ground_drop=bool(progress.graph_ground_drop[0]))
            np.savez_compressed(a.output/'states.npz', qpos=np.array(states))
            robot = rt.manifest['robot']
    model = mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'))
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.map.znear = .0001
    data = mujoco.MjData(model)
    main_render = mujoco.Renderer(model, height=720, width=1280)
    inset_render = mujoco.Renderer(model, height=240, width=320)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    world = mujoco.MjvCamera()
    world.type = mujoco.mjtCamera.mjCAMERA_FREE
    data.qpos[:] = states[0]; mujoco.mj_forward(model,data)
    world.lookat[:] = data.xpos[model.body(robot['chassis']).id] + np.array([.25,0,.35])
    world.distance, world.azimuth, world.elevation = 2.8, 135, -5
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720',
           '-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'rollout.mp4')]
    encoder = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for index,qpos in enumerate(states):
            data.qpos[:] = qpos; mujoco.mj_forward(model,data)
            model.vis.map.znear = .01
            main_render.update_scene(data,camera=world,scene_option=option)
            # Imported visual shells include faces visible from both sides.
            main_render.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = False
            image = Image.fromarray(main_render.render())
            for name,y,label in [('hand_camera',50,'ARM CAMERA')]:
                model.vis.map.znear = .0001
                inset_render.update_scene(data,camera=name,scene_option=option)
                inset = Image.fromarray(inset_render.render())
                image.paste(inset,(940,y+27))
                draw = ImageDraw.Draw(image)
                draw.rectangle((936,y,1264,y+271),outline='white',width=2)
                draw.rectangle((938,y+2,1262,y+26),fill='#17202b')
                draw.text((944,y+3),label,font=font,fill='white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,1280,35),fill='#17202b')
            label = 'PRIVILEGED TEACHER' if metadata.get('role', a.role) == 'teacher' else 'SENSOR-ONLY POLICY'
            draw.text((16,7),f'{label} ROLLOUT  |  {a.checkpoint.stem}  |  t = {index/25:.2f} s',font=font,fill='white')
            if index == 0: image.save(a.output/'preview.png')
            encoder.stdin.write(image.tobytes())
    finally:
        encoder.stdin.close(); code=encoder.wait()
        main_render.close(); inset_render.close()
    if code: raise RuntimeError('Video encoding failed')
    (a.output/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__ == '__main__': main()
