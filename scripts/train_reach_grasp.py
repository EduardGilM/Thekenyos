"""Short PPO diagnostic, with matched held-out evaluation and honest video labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from treesim.reach_grasp_env import ReachGraspEnv


def evaluate(policy, relic, seeds, video=None):
    env=ReachGraspEnv(relic, guidance_weight=0., render_mode='rgb_array' if video else None)
    records=[]; encoder=None
    try:
        for episode,seed in enumerate(seeds):
            obs,_=env.reset(seed=seed); done=False; total=0.; steps=0; peak=np.zeros(2)
            if video and episode==0:
                import warp as wp
                env.render()  # Create render buffers before setting the close-up camera.
                fruit=env.sim.body_q_np()[env.observer.body,:3]
                env.viewer.set_camera(pos=wp.vec3(*(fruit+[1.1,1.1,.4])), yaw=-135., pitch=-14.)
            while not done:
                action=np.zeros(7) if policy is None else policy.predict(obs,deterministic=True)[0]
                obs,reward,done,_,info=env.step(action); total+=reward; steps+=1
                peak=np.maximum(peak,info['physical']['jaw_forces_N'])
                if video and episode==0:
                    from PIL import Image,ImageDraw,ImageFont
                    image=Image.fromarray(env.render()); w,h=image.size
                    draw=ImageDraw.Draw(image); font=ImageFont.truetype('DejaVuSans.ttf',22)
                    draw.rectangle((0,0,w,115),fill=(18,24,32))
                    draw.text((15,10),'PPO PILOT - learned actions | fixed base | rigid fruit surrogate',font=font,fill='white')
                    draw.text((15,43),f'seed {seed} | t={env.sim.sim_time:.2f}s | target distance {env.physical.tcp_distance_m*100:.1f} cm',font=font,fill='white')
                    draw.text((15,77),f'2x slow motion | guidance OFF | outcome: {info["outcome"] or "running"}',font=font,fill='white')
                    if encoder is None:
                        video.parent.mkdir(parents=True,exist_ok=True)
                        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i','-','-vf','scale=1280:-2','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
                    encoder.stdin.write(np.asarray(image).tobytes())
            records.append(dict(seed=seed,success=info['success'],outcome=info['outcome'],
                                best_distance_m=info['best_distance_m'],final_distance_m=env.physical.tcp_distance_m,
                                return_without_guidance=total,steps=steps,peak_sampled_jaw_force_N=peak.tolist(),
                                damage=env.physical.damage,actuator_work_J=env.work))
            print('EVAL',json.dumps(records[-1]),flush=True)
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait(): raise RuntimeError('Video encoding failed')
        env.close()
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('output/reach-grasp-pilot'))
    p.add_argument('--evaluate-only', action='store_true', help='Evaluate saved initial/final checkpoints without retraining')
    p.add_argument('--steps',type=int,default=8192)
    p.add_argument('--eval-episodes',type=int,default=8)
    p.add_argument('--video',type=Path)
    p.add_argument('--contact-report',type=Path,required=True)
    a=p.parse_args()
    if a.steps <= 0 or a.eval_episodes <= 0: p.error('Steps and evaluation episodes must be positive')
    contact=json.loads(a.contact_report.read_text())
    if any(x.get('warnings',0) or x.get('execution_returncode',0) for x in contact['cases'].values() if x['rigid']):
        raise RuntimeError('Resolve rigid contact numerical failures before the rigid pilot')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    env=ReachGraspEnv(a.relic)
    model=PPO('MultiInputPolicy',env,device='cpu',seed=123,verbose=1,
              n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,
              gamma=env.task.gamma_per_second**.02,ent_coef=.005,
              policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])))
    if not a.evaluate_only:
        model.set_logger(configure(str(a.output),['stdout','csv']))
    seeds=list(range(1000,1000+a.eval_episodes))
    if a.evaluate_only:
        baseline=json.loads((a.output/'baseline.json').read_text())
        initial_model=PPO.load(a.output/'initial-policy',device='cpu')
        initial=torch.cat([p.detach().flatten() for p in initial_model.policy.parameters()])
        model=PPO.load(a.output/'policy',device='cpu')
        elapsed=json.loads((a.output/'training.json').read_text())['training_seconds'] if (a.output/'training.json').exists() else None
    else:
        baseline=evaluate(model,a.relic,seeds)
        (a.output/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        model.save(a.output/'initial-policy')
        initial=torch.cat([p.detach().flatten() for p in model.policy.parameters()]).clone()
        started=time.monotonic()
        model.learn(total_timesteps=a.steps)
        elapsed=time.monotonic()-started
        model.save(a.output/'policy')
        (a.output/'training.json').write_text(json.dumps(dict(training_seconds=elapsed, training_steps=model.num_timesteps, ppo_updates=model._n_updates),indent=2)+'\n')
    final=torch.cat([p.detach().flatten() for p in model.policy.parameters()])
    change=float(torch.linalg.vector_norm(final-initial))
    if not np.isfinite(change) or change<=0: raise RuntimeError('Policy did not update')
    trained=evaluate(model,a.relic,seeds,a.video)
    summary=dict(training_steps=model.num_timesteps,training_seconds=elapsed,ppo_updates=model._n_updates,
                 parameter_change_l2=change,training_seed=123,evaluation_seeds=seeds,
                 contact_transfer_accepted=contact['transfer_accepted'],
                 scope='Local reach/grasp pilot; rigid fruit, IK reset only, no demonstrations, no collection policy',
                 baseline=baseline,trained=trained)
    for name,records in [('baseline',baseline),('trained',trained)]:
        summary[name+'_summary']=dict(successes=sum(r['success'] for r in records),episodes=len(records),
                mean_best_distance_m=float(np.mean([r['best_distance_m'] for r in records])),
                within_3cm=sum(r['best_distance_m']<.03 for r in records),
                failures={outcome:sum(r['outcome']==outcome for r in records) for outcome in sorted({r['outcome'] for r in records})})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PILOT_RESULT',json.dumps(summary),flush=True)
    env.close()


if __name__=='__main__': main()
