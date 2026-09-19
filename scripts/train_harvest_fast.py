"""Wall-time bounded harvesting training with persistent episodes and full evaluations."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_fast import update
from train_physical_smoke import build_policy


def evaluate(runtime, policy, gait, args):
    from treesim.kiwi_rl.harvest_training import HarvestCollector,episode_metrics
    collector = HarvestCollector(runtime,stall_seconds=args.stall_seconds,
        max_episode_seconds=args.max_episode_seconds,guidance=0.)
    completed = {}
    for _ in range(int(args.max_episode_seconds / runtime.control_dt) // args.steps + 2):
        _,_,episodes = collector.collect(policy,gait,args.steps,deterministic=True,store=False)
        for episode in episodes:
            completed.setdefault(episode['world'],episode)
        if len(completed)==runtime.worlds:
            break
    if len(completed)!=runtime.worlds:
        raise RuntimeError('Full-episode evaluation did not complete every world')
    return episode_metrics(list(completed.values()))


def run(args):
    import torch
    import warp as wp
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.harvest_training import HarvestCollector,episode_metrics
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import save_checkpoint,load_checkpoint
    from treesim.kiwi_rl.training_log import TrainingLog
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(scope='full physical episodes; event-guided grasp-to-deposit curriculum',
        actor_inputs='gripper RGB-D and R84 only',
        approximations='rigid fruit, 200 Hz; uncalibrated 8 N stem and 15 N damage thresholds')
    log=TrainingLog(args.output,config,wandb_mode=args.wandb_mode,wandb_project='Thekenyos',
                    wandb_entity='juampab',wandb_name=args.output.name)
    try:
        runtime=FastRuntime(args.scene,worlds=args.worlds,camera='hand_camera')
        evaluation_runtime=FastRuntime(args.eval_scene or args.scene,worlds=args.eval_worlds,camera='hand_camera')
        gait=load_gait_artifact(args.gait_checkpoint).cuda().eval()
        policy=build_policy().cuda()
        if args.initialize_from:
            load_checkpoint(args.initialize_from,{'student':policy},expected_meta={
                'camera':'hand_camera','camera_profile':runtime.manifest['cameras']})
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        meta=dict(schema='fast-harvest-rgbd-r84/v2',camera='hand_camera',
            camera_profile=runtime.manifest['cameras'],model_sha256=runtime.manifest['model_sha256'],config=config)
        def checkpoint(index):
            path=args.output/f'checkpoint-{index:06d}.pt'
            if path.exists():
                return str(path)
            save_checkpoint(path,{'student':policy},{'student':optimizer},
                {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},dict(meta,update=index))
            return str(path)
        initial=checkpoint(0)
        baseline=evaluate(evaluation_runtime,policy,gait,args)
        print(json.dumps(dict(update=0,evaluation=baseline)),flush=True)
        log.log({f'evaluation/{k}':v for k,v in baseline.items()},step=0)
        collector=HarvestCollector(runtime,stall_seconds=args.stall_seconds,
            max_episode_seconds=args.max_episode_seconds,guidance=1.)
        from PIL import Image
        rgb=runtime.pixels()[0,:3].permute(1,2,0).cpu().numpy()
        Image.fromarray((rgb.clip(0,1)*255).astype('uint8')).save(args.output/'policy-camera-start.png')
        started=time.monotonic(); last_eval=started
        index=0; total_episodes=0; evaluations=[dict(update=0,**baseline)]
        best=(baseline['success'],baseline['detached'],baseline['grasp'],-baseline['closest_distance_m'])
        best_checkpoint=initial
        while time.monotonic()-started < args.train_seconds:
            tick=time.monotonic()
            rows,bootstrap,episodes=collector.collect(policy,gait,args.steps)
            metrics=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=.999)
            epochs=1
            for _ in range(2):
                replay=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=.999,check_replay=False)
                if replay.get('early_stop'):
                    break
                metrics.update(replay); epochs+=1
            metrics['optimizer_epochs']=epochs
            del rows,bootstrap
            torch.cuda.synchronize(); index+=1
            total_episodes+=len(episodes)
            metrics.update(update=index,transitions=index*args.worlds*args.steps,
                training_transitions_per_second=args.worlds*args.steps/(time.monotonic()-tick),
                elapsed_seconds=time.monotonic()-started,completed_episodes=total_episodes,
                max_live_episode_seconds=float(collector.progress.age.max())*runtime.control_dt)
            metrics.update({f'episode/{k}':v for k,v in episode_metrics(episodes).items()})
            if time.monotonic()-last_eval >= args.eval_every_seconds:
                path=checkpoint(index)
                result=evaluate(evaluation_runtime,policy,gait,args)
                evaluations.append(dict(update=index,**result))
                metrics.update({f'evaluation/{k}':v for k,v in result.items()})
                score=(result['success'],result['detached'],result['grasp'],-result['closest_distance_m'])
                if score>best:best,best_checkpoint=score,path
                # Remove guidance only after unguided full-episode success is established.
                if result['success']>=.5:collector.progress.guidance=.25
                if result['success']>=.8:collector.progress.guidance=0.
                last_eval=time.monotonic()
            log.log(metrics,step=index)
            print(json.dumps(metrics),flush=True)
        final=checkpoint(index)
        result=evaluate(evaluation_runtime,policy,gait,args)
        evaluations.append(dict(update=index,**result))
        score=(result['success'],result['detached'],result['grasp'],-result['closest_distance_m'])
        if score>best:best,best_checkpoint=score,final
        report=dict(config=config,baseline=baseline,evaluation=result,evaluations=evaluations,
            updates=index,transitions=index*args.worlds*args.steps,completed_episodes=total_episodes,
            elapsed_seconds=time.monotonic()-started,best_checkpoint=best_checkpoint,
            final_checkpoint=final,wandb_url=log.url,numerical_failures=sum(runtime.check()['flags']))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        log.log({f'final/{k}':v for k,v in result.items()},step=index+1)
        print(json.dumps(report),flush=True)
    except BaseException:
        log.finish(success=False)
        raise
    log.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','gait-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initialize-from',type=Path)
    p.add_argument('--eval-scene',type=Path)
    p.add_argument('--worlds',type=int,default=4096)
    p.add_argument('--eval-worlds',type=int,default=32)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--minibatch-worlds',type=int,default=512)
    p.add_argument('--train-seconds',type=float,default=3600)
    p.add_argument('--eval-every-seconds',type=float,default=120)
    p.add_argument('--stall-seconds',type=float,default=4.)
    p.add_argument('--max-episode-seconds',type=float,default=30.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--wandb-mode',choices=('disabled','online','offline'),default='online')
    a=p.parse_args()
    if not (1<=a.worlds<=4096 and 1<=a.eval_worlds<=256 and 2<=a.steps<=256 and
            1<=a.minibatch_worlds<=1024 and 1<=a.train_seconds<=7200 and
            1<=a.stall_seconds<a.max_episode_seconds<=60 and a.eval_every_seconds>=1):
        p.error('Invalid training size or duration')
    import torch,warp as wp
    wp.init(); stream=torch.cuda.Stream()
    with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
        run(a)


if __name__=='__main__':main()
