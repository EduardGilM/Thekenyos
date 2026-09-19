"""Wall-time bounded harvesting training with persistent episodes and full evaluations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_fast import update
from train_physical_smoke import build_policy
from treesim.kiwi_rl.fast_teacher import TEACHER_SCHEMA, build_privileged_policy, validate_teacher_report


def evaluation_score(result):
    # A destructive detachment must not beat an intact, unsuccessful reach.
    return (result['success'], -result['physical_failure'], result['grasp'],
            -result['closest_distance_m'])


def evaluate(runtime, policy, gait, args, role='student'):
    from treesim.kiwi_rl.harvest_training import HarvestCollector,episode_metrics
    collector = HarvestCollector(runtime,stall_seconds=args.stall_seconds,
        max_episode_seconds=args.max_episode_seconds,guidance=0.,role=role)
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


def resume_history(output, checkpoint, config):
    """Recover logged counters only when the saved optimizer matches that boundary."""
    if (output/'report.json').exists():
        raise ValueError('Completed runs must not be resumed in place')
    manifest=json.loads(Path(str(checkpoint)+'.json').read_text())
    saved=manifest['config']
    for key in ('role','scene','eval_scene','gait_checkpoint','worlds','eval_worlds','steps',
                'stall_seconds','max_episode_seconds','arm_speed_rad_s','reward_profile',
                'reward_gamma','solver_iterations','jaw_cap_Nm','seed'):
        if saved.get(key)!=config.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')
    history=[json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()]
    latest=next(row for row in reversed(history) if 'update' in row and 'elapsed_seconds' in row)
    if manifest['update']!=latest['update']:
        raise ValueError('Resume checkpoint must match the latest logged update; refusing silent rollback')
    evaluations=[dict(update=row.get('update',row['step']),
        **{k.removeprefix('evaluation/'):v for k,v in row.items() if k.startswith('evaluation/')})
        for row in history if 'evaluation/success' in row]
    if not evaluations: raise ValueError('Resume history has no physical evaluations')
    return dict(latest=latest,index=max(row['step'] for row in history),evaluations=evaluations)


def run(args):
    import torch
    import warp as wp
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.harvest_training import HarvestCollector,episode_metrics,HARVEST_GAMMA,REWARD_PROFILE
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.ppo import save_checkpoint,load_checkpoint
    from treesim.kiwi_rl.training_log import TrainingLog
    role=args.role
    if args.cti and role != 'teacher':
        raise ValueError('Selective CTI currently supports the privileged teacher only')
    if role == 'teacher' and (args.eval_scene is None or args.eval_worlds < args.teacher_min_eval_episodes):
        raise ValueError('Teacher runs need a distinct --eval-scene and enough eval worlds for the episode gate')
    if role == 'teacher':
        train_sha=json.loads((args.scene/'manifest.json').read_text())['model_sha256']
        eval_sha=json.loads((args.eval_scene/'manifest.json').read_text())['model_sha256']
        if train_sha == eval_sha:
            raise ValueError('Teacher readiness evidence requires a distinct held-out scene model hash')
    if args.teacher_checkpoint and role != 'student':
        raise ValueError('--teacher-checkpoint is valid only for --role student')
    if args.teacher_checkpoint and args.teacher_report is None:
        raise ValueError('Student DAgger requires a teacher checkpoint and its validated report')
    if args.teacher_report and not args.teacher_checkpoint:
        raise ValueError('--teacher-report requires --teacher-checkpoint')
    if role == 'student' and args.teacher_checkpoint:
        if args.eval_scene is None:
            raise ValueError('Teacher distillation requires an explicit held-out --eval-scene')
        training_manifest=json.loads((args.scene/'manifest.json').read_text())
        evaluation_manifest=json.loads((args.eval_scene/'manifest.json').read_text())
        validate_teacher_report(args.teacher_report,args.teacher_checkpoint,
            training_model_sha256=training_manifest['model_sha256'],
            evaluation_model_sha256=evaluation_manifest['model_sha256'],
            minimum_episodes=args.teacher_min_eval_episodes)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=bool(args.resume_from))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(role=role,scope='full physical episodes; bounded potential guidance',
        reward_profile=REWARD_PROFILE,reward_gamma=HARVEST_GAMMA,
        actor_inputs=('privileged simulator state and R84' if role=='teacher' else 'gripper RGB-D and R84'),
        checkpoint_roles={'student':'sensor-only PPO actor','teacher':'privileged PPO actor'},
        solver_iterations=100,
        jaw_cap_Nm=1.0,
        force_limit_scope='per_jaw_and_nonpad_group',
        optimizer_resume=('full_optimizer_checkpoint' if args.resume_from else 'fresh_optimizer'),
        cti_version=2 if args.cti else 0,
        approximations='rigid fruit, 200 Hz; uncalibrated 8 N stem and 15 N damage thresholds')
    resumed=resume_history(args.output,args.resume_from,config) if args.resume_from else None
    if resumed and resumed['latest']['elapsed_seconds']>=args.train_seconds:
        raise ValueError('The total training budget has already been used')
    log=TrainingLog(args.output,config,wandb_run_id=args.wandb_run_id,wandb_mode=args.wandb_mode,wandb_project='Thekenyos',
                    wandb_entity='juampab',wandb_name=args.output.name)
    try:
        camera='hand_camera' if role=='student' else None
        runtime=FastRuntime(args.scene,worlds=args.worlds,camera=camera,arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],jaw_cap_Nm=config['jaw_cap_Nm'])
        evaluation_runtime=FastRuntime(args.eval_scene or args.scene,worlds=args.eval_worlds,camera=camera,arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],jaw_cap_Nm=config['jaw_cap_Nm'])
        cti = None
        if args.cti:
            from treesim.kiwi_rl.selective_cti import SelectiveCTI, update_cti
            cti_runtime=FastRuntime(args.scene,worlds=args.cti_worlds,camera=None,
                arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],
                jaw_cap_Nm=config['jaw_cap_Nm'])
        config['epa_horizon_capacity']=runtime.epa_horizon_capacity
        gait=load_gait_artifact(args.gait_checkpoint).cuda().eval()
        if args.cti:
            cti=SelectiveCTI(cti_runtime,gait,stall_seconds=args.stall_seconds,
                             max_episode_seconds=args.max_episode_seconds)
        policy=(build_privileged_policy() if role=='teacher' else build_policy()).cuda()
        teacher=None
        if args.teacher_checkpoint:
            teacher=build_privileged_policy().cuda().eval()
            for parameter in teacher.parameters(): parameter.requires_grad_(False)
            load_checkpoint(args.teacher_checkpoint,{'teacher':teacher},expected_meta={
                'role':'teacher','schema':TEACHER_SCHEMA,
                'model_sha256':runtime.manifest['model_sha256']})
        if args.initialize_from:
            if role != 'student': raise ValueError('--initialize-from is supported only for a student')
            load_checkpoint(args.initialize_from,{'student':policy},expected_meta={
                'camera':'hand_camera','camera_profile':runtime.manifest['cameras']})
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        schema=TEACHER_SCHEMA if role=='teacher' else 'fast-harvest-rgbd-r84/v2'
        meta=dict(schema=schema,role=role,model_sha256=runtime.manifest['model_sha256'],config=config)
        if role=='student':
            meta.update(camera='hand_camera',camera_profile=runtime.manifest['cameras'])
        role_state={role:policy}
        restored=None
        if args.resume_from:
            restored=load_checkpoint(args.resume_from,role_state,{role:optimizer},expected_meta={
                'role':role,'schema':schema,'model_sha256':runtime.manifest['model_sha256']})

        def checkpoint(index):
            path=args.output/f'checkpoint-{index:06d}.pt'
            if path.exists():
                return str(path)
            save_checkpoint(path,role_state,{role:optimizer},
                {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},dict(meta,update=index))
            return str(path)
        if resumed:
            initial=str(args.output/'checkpoint-000000.pt')
            baseline={k:v for k,v in resumed['evaluations'][0].items() if k!='update'}
        else:
            initial=checkpoint(0)
            baseline=evaluate(evaluation_runtime,policy,gait,args,role=role)
            print(json.dumps(dict(update=0,evaluation=baseline)),flush=True)
            log.log({f'evaluation/{k}':v for k,v in baseline.items()},step=0)
        collector=HarvestCollector(runtime,stall_seconds=args.stall_seconds,
            max_episode_seconds=args.max_episode_seconds,guidance=1.,role=role,teacher_policy=teacher)
        if role=='student':
            from PIL import Image
            rgb=runtime.pixels()[0,:3].permute(1,2,0).cpu().numpy()
            Image.fromarray((rgb.clip(0,1)*255).astype('uint8')).save(args.output/'policy-camera-start.png')
        offset=resumed['latest']['elapsed_seconds'] if resumed else 0.
        started=time.monotonic()-offset; last_eval=time.monotonic()
        last_cti=time.monotonic()-args.cti_every_seconds
        previous=resumed['latest'] if resumed else {}
        cti_seconds=previous.get('cti/total_seconds',0.)
        cti_transitions=previous.get('cti/total_transitions',0)
        cti_selected=previous.get('cti/total_selected_worlds',0)
        cti_rounds=previous.get('cti/rounds',0)
        if cti is not None: cti.rounds=cti_rounds
        index=resumed['index'] if resumed else 0
        transitions=previous.get('transitions',0)
        total_episodes=previous.get('completed_episodes',0)
        evaluations=resumed['evaluations'] if resumed else [dict(update=0,**baseline)]
        best_eval=max(evaluations,key=evaluation_score)
        best=evaluation_score(best_eval)
        best_checkpoint=str(args.output/f"checkpoint-{best_eval['update']:06d}.pt")
        best_result={k:v for k,v in best_eval.items() if k!='update'}
        phase='cti-v2' if args.cti else role
        if restored:
            torch.set_rng_state(restored['rng']['torch'])
            torch.cuda.set_rng_state_all(restored['rng']['cuda'])
            event=dict(event='resume',phase=phase,version=2 if args.cti else 0,at=time.time(),
                elapsed_seconds=offset,update=index,source_checkpoint=str(args.resume_from),
                episodes_reset=True,remaining_seconds=args.train_seconds-offset)
            with (args.output/'phase-events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='running',phase=phase)))
        paused=False
        while time.monotonic()-started < args.train_seconds:
            tick=time.monotonic()
            rows,bootstrap,episodes=collector.collect(policy,gait,args.steps)
            teacher_coef=args.teacher_distill_weight if teacher is not None else 0.
            metrics=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=collector.progress.gamma,
                           teacher_coef=teacher_coef)
            epochs=1
            for _ in range(2):
                replay=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=collector.progress.gamma,
                              check_replay=False,teacher_coef=teacher_coef)
                if replay.get('early_stop'):
                    break
                metrics.update(replay); epochs+=1
            # Counterfactual targets use a separate loss after factual PPO.
            # Target 10% amortized wall time; one in-progress round may overshoot.
            elapsed=time.monotonic()-started
            if (cti is not None and time.monotonic()-last_cti>=args.cti_every_seconds
                    and cti_seconds <= .10*max(elapsed,1.)
                    and (cti_rounds==0 or args.train_seconds-elapsed>60)):
                cti_started=time.monotonic()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    examples,cti_metrics=cti.run(policy)
                    records=cti_metrics.pop('branch_records',[])
                    if records:
                        with (args.output/'cti-branches.jsonl').open('a') as f:
                            f.write(json.dumps(dict(update=index+1,elapsed_seconds=time.monotonic()-started,
                                                   version=2,records=records))+'\n')
                    # A short factual sequence checks drift outside the selected branches.
                    cti_metrics.update(update_cti(policy,optimizer,examples,coef=args.cti_coef,
                                                  anchor_rows=rows[:4]))
                cti_seconds+=time.monotonic()-cti_started
                cti_transitions+=cti_metrics.get('transitions',0)
                cti_selected+=cti_metrics.get('selected_worlds',0)
                cti_rounds+=1;last_cti=time.monotonic()
                metrics.update({f'cti/{k}':v for k,v in cti_metrics.items()})
                del examples
            if cti is not None:
                metrics.update({'cti/version':2,'cti/total_seconds':cti_seconds,'cti/total_transitions':cti_transitions,
                                'cti/total_selected_worlds':cti_selected,'cti/rounds':cti_rounds})
            metrics['optimizer_epochs']=epochs
            metrics['reward/task_mean']=float(torch.stack([r['task_reward'] for r in rows]).mean())
            metrics['reward/shaping_mean']=float(torch.stack([r['shaping_reward'] for r in rows]).mean())
            del rows,bootstrap
            torch.cuda.synchronize(); index+=1
            transitions+=args.worlds*args.steps
            total_episodes+=len(episodes)
            metrics.update(update=index,transitions=transitions,
                training_transitions_per_second=args.worlds*args.steps/(time.monotonic()-tick),
                elapsed_seconds=time.monotonic()-started,completed_episodes=total_episodes,
                max_live_episode_seconds=float(collector.progress.age.max())*runtime.control_dt)
            metrics.update({f'episode/{k}':v for k,v in episode_metrics(episodes).items()})
            if time.monotonic()-last_eval >= args.eval_every_seconds:
                path=checkpoint(index)
                result=evaluate(evaluation_runtime,policy,gait,args,role=role)
                evaluations.append(dict(update=index,**result))
                metrics.update({f'evaluation/{k}':v for k,v in result.items()})
                score=evaluation_score(result)
                if score>best:best,best_checkpoint,best_result=score,path,result
                last_eval=time.monotonic()
            metrics['elapsed_seconds']=time.monotonic()-started
            log.log(metrics,step=index)
            print(json.dumps(metrics),flush=True)
            if (args.output/'pause-request.json').exists():
                saved_path=checkpoint(index)
                (args.output/'pause-request.json').unlink()
                (args.output/'active-process.json').write_text(json.dumps(dict(
                    pid=os.getpid(),status='paused',phase=phase,checkpoint=saved_path)))
                paused=True
                break
        if paused:
            log.finish()
            return
        final=checkpoint(index)
        result=evaluate(evaluation_runtime,policy,gait,args,role=role)
        evaluations.append(dict(update=index,**result))
        score=evaluation_score(result)
        if score>best:best,best_checkpoint,best_result=score,final,result
        teacher_sha256=hashlib.sha256(Path(best_checkpoint).read_bytes()).hexdigest()
        report=dict(config=config,baseline=baseline,evaluation=result,evaluations=evaluations,
            updates=index,transitions=transitions,completed_episodes=total_episodes,
            elapsed_seconds=time.monotonic()-started,best_checkpoint=best_checkpoint,
            final_checkpoint=final,wandb_url=log.url,numerical_failures=sum(runtime.check()['flags']))
        if cti is not None:
            report['cti']=dict(rounds=cti_rounds,seconds=cti_seconds,transitions=cti_transitions,
                               selected_worlds=cti_selected)
        if role=='teacher':
            report.update(role='teacher',schema=TEACHER_SCHEMA,
                teacher_checkpoint=best_checkpoint,teacher_checkpoint_sha256=teacher_sha256,
                training_model_sha256=runtime.manifest['model_sha256'],
                evaluation_model_sha256=evaluation_runtime.manifest['model_sha256'],
                guidance_enabled=False,evaluation=best_result)
        (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='completed',phase=phase)))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        log.log({f'final/{k}':v for k,v in result.items()},step=index+1)
        print(json.dumps(report),flush=True)
    except BaseException as exc:
        # Preserve bounded diagnostic evidence before a later process/reset can
        # discard it. This is the state at detection, not the first bad substep.
        import numpy as np
        for name in ('runtime', 'evaluation_runtime', 'cti_runtime'):
            failed_runtime = locals().get(name)
            if failed_runtime is None:
                continue
            flags = failed_runtime._flags.numpy()
            ids = np.flatnonzero(flags)[:8]
            if len(ids):
                np.savez_compressed(args.output/f'{name}-failure-state.npz',
                    worlds=ids, flags=flags[ids],
                    qpos=wp.to_torch(failed_runtime.data.qpos)[ids.tolist()].cpu().numpy(),
                    qvel=wp.to_torch(failed_runtime.data.qvel)[ids.tolist()].cpu().numpy(),
                    targets=wp.to_torch(failed_runtime.control.targets)[ids.tolist()].cpu().numpy())
        (args.output/'failure.json').write_text(json.dumps(dict(
            error_type=type(exc).__name__,error=str(exc),config=config),indent=2)+'\n')
        log.finish(success=False)
        raise
    log.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','gait-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initialize-from',type=Path)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--wandb-run-id')
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
    p.add_argument('--arm-speed-rad-s',type=float,default=.5)
    p.add_argument('--role',choices=('teacher','student'),default='student')
    p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--teacher-report',type=Path)
    p.add_argument('--teacher-min-eval-episodes',type=int,default=32)
    p.add_argument('--teacher-distill-weight',type=float,default=1.)
    p.add_argument('--cti',action='store_true',help='Selective counterfactual action repair for teacher training')
    p.add_argument('--cti-worlds',type=int,default=16)
    p.add_argument('--cti-every-seconds',type=float,default=120.)
    p.add_argument('--cti-coef',type=float,default=.1)
    p.add_argument('--wandb-mode',choices=('disabled','online','offline'),default='online')
    a=p.parse_args()
    if not (1<=a.worlds<=4096 and 1<=a.eval_worlds<=256 and 2<=a.steps<=256 and
            1<=a.minibatch_worlds<=1024 and 1<=a.train_seconds<=7200 and
            1<=a.stall_seconds<a.max_episode_seconds<=60 and a.eval_every_seconds>=1 and
            0<a.arm_speed_rad_s<=2.5 and 1<=a.teacher_min_eval_episodes<=256 and
            (a.role!='teacher' or a.teacher_min_eval_episodes<=a.eval_worlds) and
            0<=a.teacher_distill_weight<=10 and 1<=a.cti_worlds<=64 and
            a.cti_every_seconds>=1 and 0<a.cti_coef<=1 and (not a.cti or a.role=='teacher')):
        p.error('Invalid training size or duration')
    if a.resume_from and (a.initialize_from or (a.wandb_mode=='online' and not a.wandb_run_id)):
        p.error('Resume requires an explicit W&B run ID online and cannot initialize fresh weights')
    if a.wandb_run_id and not a.resume_from:
        p.error('A W&B run ID requires --resume-from')
    import torch,warp as wp
    wp.init(); stream=torch.cuda.Stream()
    with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
        run(a)


if __name__=='__main__':main()
