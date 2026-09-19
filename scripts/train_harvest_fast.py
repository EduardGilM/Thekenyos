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
from treesim.kiwi_rl.reward_graph import SOFT_GRAPH_PROFILE as GRAPH_PROFILE
from treesim.kiwi_rl.graph_training import graph_rank, save_acceptance
from treesim.kiwi_rl.fast_teacher import TEACHER_SCHEMA, build_privileged_policy, validate_teacher_report


def evaluation_score(result):
    if 'completed/position' in result:
        return graph_rank(result)
    # A destructive detachment must not beat an intact, unsuccessful reach.
    if 'graph_score' in result:
        return (result['success'], -result['physical_failure'], result['graph_score'],
                result.get('held_detach',0.), result['grasp'], -result['closest_distance_m'])
    return (result['success'], -result['physical_failure'], result['grasp'],
            -result['closest_distance_m'])


def evaluate(runtime, policy, gait, args, role='student'):
    from treesim.kiwi_rl.harvest_training import HarvestCollector,episode_metrics
    import torch
    episodes=[]
    # Same scene, nominal plus two seeded stochastic trials. This tests action
    # robustness, not scene generalization beyond the separate held-out scene.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for trial in range(3 if args.reward_graph else 1):
            torch.manual_seed(args.seed+10000+trial)
            collector = HarvestCollector(runtime,stall_seconds=args.stall_seconds,
                max_episode_seconds=args.max_episode_seconds,guidance=0.,role=role,
                reward_profile=(GRAPH_PROFILE if args.reward_graph else 'potential-harvest/v1'))
            completed = {}
            for _ in range(int(args.max_episode_seconds / runtime.control_dt) // args.steps + 2):
                _,_,batch = collector.collect(policy,gait,args.steps,deterministic=trial==0,store=False)
                for episode in batch: completed.setdefault(episode['world'],episode)
                if len(completed)==runtime.worlds: break
            if len(completed)!=runtime.worlds:
                raise RuntimeError('Full-episode evaluation did not complete every world')
            episodes.extend(completed.values())
    return episode_metrics(episodes)


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
    for key in ('gae_lambda','entropy_coef','cti_optimizer','curriculum','reward_graph','cti_version','cti_alternatives','cti_search_iterations'):
        if saved.get(key)!=config.get(key):
            raise ValueError(f'Resume learning configuration mismatch: {key}')
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
    if args.reward_graph and args.cti:
        raise ValueError('Graph v4 is a PPO-only baseline; omit --cti')
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
    config.update(role=role,scope=('temporal physical reward graph' if args.reward_graph else 'full physical episodes; bounded guidance'),
        reward_profile=(GRAPH_PROFILE if args.reward_graph else 'milestone-harvest/v1' if args.curriculum else REWARD_PROFILE),reward_gamma=HARVEST_GAMMA,
        actor_inputs=('privileged simulator state and R84' if role=='teacher' else 'gripper RGB-D and R84'),
        checkpoint_roles={'student':'sensor-only PPO actor','teacher':'privileged PPO actor'},
        source_dir=str(Path(__file__).resolve().parents[1]),
        solver_iterations=100,
        jaw_cap_Nm=1.0,
        force_limit_scope='per_jaw_and_nonpad_group',
        optimizer_resume=('full_optimizer_checkpoint' if args.resume_from else 'fresh_optimizer'),
        cti_version=(8 if args.reward_graph else 5) if args.cti else 0,
        initialization=('checkpoint' if args.initialize_from or args.resume_from else 'random'),
        cti_optimizer=('independent_adam_vtrace' if args.reward_graph else 'independent_adam') if args.cti else None,
        approximations='rigid fruit, 200 Hz; uncalibrated 8 N stem and 15 N damage thresholds')
    if args.reward_graph:
        config.update(stage_practice_fraction=0., evaluation_trials=3, evaluation_scenes=2,
            checkpoint_rollback=False, regression_loss_multiplier=2., reward_accounting='episode_peak_gain_minus_weighted_loss')
        from treesim.kiwi_rl.reward_graph import GRAPH_CONSTANTS, REGRESSION_WEIGHTS
        config['regression_weights']=REGRESSION_WEIGHTS
        config['reward_graph_constants']={k:v for k,v in GRAPH_CONSTANTS.items()
            if k not in ('approach_radius_m','approach_outer_m','insertion_range_m')}
        config['reward_graph_constants'].update(position_scale_m=.15, reset_settle_seconds=4., cti_minimum_remaining_seconds=6.)
    resumed=resume_history(args.output,args.resume_from,config) if args.resume_from else None
    if resumed and resumed['latest']['elapsed_seconds']>=args.train_seconds:
        raise ValueError('The total training budget has already been used')
    log=TrainingLog(args.output,config,wandb_run_id=args.wandb_run_id,wandb_mode=args.wandb_mode,wandb_project='Thekenyos',
                    wandb_entity='juampab',wandb_name=args.output.name)
    try:
        camera='hand_camera' if role=='student' else None
        runtime=FastRuntime(args.scene,worlds=args.worlds,camera=camera,arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],jaw_cap_Nm=config['jaw_cap_Nm'],
                task_profile=GRAPH_PROFILE if args.reward_graph else None)
        evaluation_runtime=FastRuntime(args.eval_scene or args.scene,worlds=args.eval_worlds,camera=camera,arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],jaw_cap_Nm=config['jaw_cap_Nm'],
                task_profile=GRAPH_PROFILE if args.reward_graph else None)
        training_evaluation_runtime = (FastRuntime(args.scene,worlds=args.eval_worlds,camera=camera,
            arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],
            jaw_cap_Nm=config['jaw_cap_Nm'],task_profile=GRAPH_PROFILE) if args.reward_graph else None)
        cti = None
        if args.cti:
            from treesim.kiwi_rl.selective_cti import SelectiveCTI, update_cti
            cti_runtime=FastRuntime(args.scene,worlds=args.cti_worlds*(args.cti_alternatives+1) if args.reward_graph else args.cti_worlds,camera=None,
                arm_speed_rad_s=args.arm_speed_rad_s,solver_iterations=config['solver_iterations'],
                jaw_cap_Nm=config['jaw_cap_Nm'],
                task_profile=GRAPH_PROFILE if args.reward_graph else None)
        config['epa_horizon_capacity']=runtime.epa_horizon_capacity
        gait=load_gait_artifact(args.gait_checkpoint).cuda().eval()
        if args.reward_graph:
            for target in (runtime, evaluation_runtime, training_evaluation_runtime, *([cti_runtime] if args.cti else [])):
                target.prepare_settled_reset(gait)
        if args.cti:
            if args.reward_graph:
                from treesim.kiwi_rl.branch_cti import BranchCTI
                from treesim.kiwi_rl.cti_learning import update_branch_cti
                cti=BranchCTI(cti_runtime,gait,roots=args.cti_worlds,
                    alternatives=args.cti_alternatives,search_iterations=args.cti_search_iterations)
            else:
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
            expected = ({'role':'teacher','schema':TEACHER_SCHEMA,
                         'model_sha256':runtime.manifest['model_sha256']} if role=='teacher' else
                        {'camera':'hand_camera','camera_profile':runtime.manifest['cameras']})
            load_checkpoint(args.initialize_from,{role:policy},expected_meta=expected)
            if args.curriculum or args.reward_graph:
                # Keep the grasping actor, relearn values for the changed rewards.
                policy.value.reset_parameters()
        optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
        cti_optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate) if args.cti else None
        optimizer_states={role:optimizer}
        if cti_optimizer is not None: optimizer_states['cti']=cti_optimizer
        schema=TEACHER_SCHEMA if role=='teacher' else 'fast-harvest-rgbd-r84/v2'
        meta=dict(schema=schema,role=role,model_sha256=runtime.manifest['model_sha256'],config=config)
        if role=='student':
            meta.update(camera='hand_camera',camera_profile=runtime.manifest['cameras'])
        role_state={role:policy}
        restored=None
        if args.resume_from:
            restored=load_checkpoint(args.resume_from,role_state,optimizer_states,expected_meta={
                'role':role,'schema':schema,'model_sha256':runtime.manifest['model_sha256']})

        def evaluate_candidate():
            result=evaluate(evaluation_runtime,policy,gait,args,role=role)
            if training_evaluation_runtime is not None:
                result.update({'training_scene/'+key:value for key,value in
                    evaluate(training_evaluation_runtime,policy,gait,args,role=role).items()})
            return result

        def checkpoint(index, continuation=False):
            prefix="continuation" if continuation else "checkpoint"
            path=args.output/f'{prefix}-{index:06d}.pt'
            if path.exists():
                return str(path)
            save_checkpoint(path,role_state,optimizer_states,
                {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},dict(meta,update=index))
            return str(path)
        if resumed:
            initial=str(args.output/'checkpoint-000000.pt')
            baseline={k:v for k,v in resumed['evaluations'][0].items() if k!='update'}
        else:
            initial=checkpoint(0)
            baseline=evaluate_candidate()
            print(json.dumps(dict(update=0,evaluation=baseline)),flush=True)
            log.log({f'evaluation/{k}':v for k,v in baseline.items()},step=0)
        collector=HarvestCollector(runtime,stall_seconds=args.stall_seconds,
            max_episode_seconds=args.max_episode_seconds,guidance=1.,role=role,teacher_policy=teacher,
            reward_profile=config['reward_profile'],curriculum_stage=(1 if args.curriculum and
                baseline['grasp']>=.75 and baseline['physical_failure']<=.1 else 0))
        cti_queue=None
        if cti is not None:
            from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
            cti_queue=PPODecisionQueue(worlds=args.cti_worlds,segment_steps=args.steps,
                minimum_remaining_seconds=6. if args.reward_graph else 0.)
        if role=='student':
            from PIL import Image
            rgb=runtime.pixels()[0,:3].permute(1,2,0).cpu().numpy()
            Image.fromarray((rgb.clip(0,1)*255).astype('uint8')).save(args.output/'policy-camera-start.png')
        offset=resumed['latest']['elapsed_seconds'] if resumed else 0.
        started=time.monotonic()-offset; last_eval=time.monotonic()
        last_cti=time.monotonic()-args.cti_every_seconds
        previous=resumed['latest'] if resumed else {}
        if args.curriculum:
            collector.progress.curriculum_stage=previous.get('curriculum/stage',collector.progress.curriculum_stage)
        if cti_queue is not None and previous.get('cti/version',0)>=3:
            for key in ('total_collected_roots','total_dropped_roots','total_expired_roots','total_popped_roots'):
                setattr(cti_queue,key,previous.get('cti/'+key,0))
            cti_queue.total_dropped_roots+=previous.get('cti/queue_roots',0)
        cti_seconds=previous.get('cti/total_seconds',0.)
        cti_transitions=previous.get('cti/total_transitions',0)
        cti_selected=previous.get('cti/total_selected_worlds',0)
        cti_learned=previous.get('cti/total_learned_transitions',0)
        cti_optimizer_steps=previous.get('cti/total_optimizer_steps',0)
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
        accepted_result=baseline
        skill_floor=dict(baseline)
        accepted_checkpoint=initial
        gate_path=args.output/'accepted.json'
        if resumed and args.reward_graph:
            gate=json.loads(gate_path.read_text())
            accepted_result=gate['evaluation'];accepted_checkpoint=gate['checkpoint']
            skill_floor=gate.get('skill_floor',accepted_result)
        if args.reward_graph:
            save_acceptance(gate_path,accepted_checkpoint,accepted_result,skill_floor)
        phase=f'cti-v{config["cti_version"]}' if args.cti else role
        if restored:
            torch.set_rng_state(restored['rng']['torch'])
            torch.cuda.set_rng_state_all(restored['rng']['cuda'])
            event=dict(event='resume',phase=phase,version=config['cti_version'],at=time.time(),
                elapsed_seconds=offset,update=index,source_checkpoint=str(args.resume_from),
                episodes_reset=True,remaining_seconds=args.train_seconds-offset)
            with (args.output/'phase-events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        (args.output/'active-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='running',phase=phase)))
        paused=False
        policy_reverted=False
        while time.monotonic()-started < args.train_seconds:
            policy_reverted=False
            tick=time.monotonic()
            rows,bootstrap,episodes=collector.collect(policy,gait,args.steps,
                cti_queue=cti_queue,policy_version=index)
            teacher_coef=args.teacher_distill_weight if teacher is not None else 0.
            metrics=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=collector.progress.gamma,
                           teacher_coef=teacher_coef,gae_lambda=args.gae_lambda,entropy_coef=args.entropy_coef)
            epochs=1
            for _ in range(2):
                replay=update(policy,optimizer,rows,bootstrap,args.minibatch_worlds,gamma=collector.progress.gamma,
                              check_replay=False,teacher_coef=teacher_coef,gae_lambda=args.gae_lambda,entropy_coef=args.entropy_coef)
                if replay.get('early_stop'):
                    break
                metrics.update(replay); epochs+=1
            # Counterfactual targets use a separate loss after factual PPO.
            # Target the configured amortized wall-time share; a round may overshoot.
            elapsed=time.monotonic()-started
            if (cti is not None and len(cti_queue) and time.monotonic()-last_cti>=args.cti_every_seconds
                    and cti_seconds <= args.cti_time_fraction*max(elapsed,1.)
                    and (cti_rounds==0 or args.train_seconds-elapsed>60)):
                cti_started=time.monotonic()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    batch=cti_queue.pop(index)
                    examples,cti_metrics=(cti.run(policy,batch) if batch is not None else
                        ([],dict(version=config['cti_version'],source='ppo',queue_empty=True)))
                    records=cti_metrics.pop('branch_records',[])
                    if records:
                        with (args.output/'cti-branches.jsonl').open('a') as f:
                            f.write(json.dumps(dict(update=index+1,elapsed_seconds=time.monotonic()-started,
                                                   version=config['cti_version'],records=records))+'\n')
                    # A short factual sequence checks drift outside the selected branches.
                    if args.reward_graph and batch is not None:
                        branch_rows,branch_bootstrap=examples
                        cti_metrics.update(update_branch_cti(policy,cti_optimizer,branch_rows,
                            branch_bootstrap,coef=args.cti_coef,anchor_rows=rows[:4]))
                        if cti_metrics.get('updated'):
                            cti_learned+=cti_metrics.get('valid_transitions',0)
                        cti_optimizer_steps+=cti_metrics.get('optimizer_steps',0)
                        del branch_rows,branch_bootstrap
                    elif not args.reward_graph:
                        cti_metrics.update(update_cti(policy,cti_optimizer,examples,coef=args.cti_coef,
                                                      anchor_rows=rows[:4]))
                cti_seconds+=time.monotonic()-cti_started
                cti_transitions+=cti_metrics.get('transitions',0)
                cti_selected+=cti_metrics.get('selected_worlds',0)
                cti_rounds+=int(batch is not None);last_cti=time.monotonic()
                metrics.update({f'cti/{k}':v for k,v in cti_metrics.items()})
                del examples
            if cti is not None:
                metrics.update({f'cti/{k}':v for k,v in cti_queue.metrics().items()})
                metrics.update({'cti/version':config['cti_version'],'cti/total_learned_transitions':cti_learned,
                                'cti/total_optimizer_steps':cti_optimizer_steps,'cti/total_seconds':cti_seconds,'cti/total_transitions':cti_transitions,
                                'cti/total_selected_worlds':cti_selected,'cti/rounds':cti_rounds})
            metrics['optimizer_epochs']=epochs
            metrics['reward/task_mean']=float(torch.stack([r['task_reward'] for r in rows]).mean())
            metrics['reward/shaping_mean']=float(torch.stack([r['shaping_reward'] for r in rows]).mean())
            if args.reward_graph:
                metrics['reward/completion_mean']=float(torch.stack([r['completion_reward'] for r in rows]).mean())
                for name in rows[0]['stage_rewards']:
                    metrics[f'stage_reward/{name}'] = float(torch.stack([r['stage_rewards'][name] for r in rows]).mean())
                    metrics[f'stage_gain/{name}'] = float(torch.stack([r['stage_gains'][name] for r in rows]).mean())
                    metrics[f'stage_loss/{name}'] = float(torch.stack([r['stage_losses'][name] for r in rows]).mean())
                    metrics[f'stage_progress/{name}'] = float(torch.stack([r['stage_scores'][name] for r in rows]).mean())
                for i, name in enumerate(('shoulder_0','shoulder_1','elbow_0','elbow_1','wrist_0','wrist_1','jaw')):
                    actions = torch.stack([r['raw'][:,i].tanh() for r in rows])
                    metrics[f'action_mean/{name}'] = float(actions.mean())
                    metrics[f'action_spread/{name}'] = float(actions.std(unbiased=False))
                    metrics[f'action_saturation/{name}'] = float((actions.abs()>.95).float().mean())
                    metrics[f'joint_speed/{name}'] = float(torch.stack([r['joint_velocity'][:,i].abs() for r in rows]).mean())
                    metrics[f'target_error/{name}'] = float(torch.stack([r['target_error'][:,i].abs() for r in rows]).mean())
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
                result=evaluate_candidate()
                evaluations.append(dict(update=index,**result))
                if args.curriculum and result['physical_failure']<=.1:
                    stage=collector.progress.curriculum_stage
                    if result['grasp']>=.75: stage=max(stage,1)
                    if result['grasp']>=.5 and result.get('held_detach',0)>=.5: stage=2
                    collector.progress.curriculum_stage=stage
                metrics.update({f'evaluation/{k}':v for k,v in result.items()})
                score=evaluation_score(result)
                if args.reward_graph:
                    # Evaluation selects a display checkpoint; it never changes
                    # the active actor, critic, optimizer or physical episodes.
                    promoted=score>best
                    if promoted:
                        best,best_checkpoint,best_result=score,path,result
                        accepted_checkpoint,accepted_result=path,result
                        save_acceptance(gate_path,path,result,{})
                    metrics.update({'acceptance/promoted':int(promoted),'acceptance/rollback':0,
                        'acceptance/checkpoint':accepted_checkpoint})
                    for key,value in accepted_result.items(): metrics['accepted/'+key]=value
                elif score>best:best,best_checkpoint,best_result=score,path,result
                last_eval=time.monotonic()
            if args.reward_graph:
                metrics['practice/starts']=0
                metrics['practice/boundaries']=0
                metrics['graph/score_mean']=float(collector.progress.graph_score.mean())
                metrics['graph/enclosure_fraction']=float(collector.progress.graph_enclosed.float().mean())
                metrics['graph/secure_grip_fraction']=float(collector.progress.graph_grip.float().mean())
                metrics['graph/slip_m_s']=float(collector.progress.graph_slip.mean())
                for stage in range(7):
                    metrics[f'graph/stage_{stage}_fraction']=float((collector.progress.graph_stage==stage).float().mean())
            if args.curriculum:
                metrics['curriculum/stage']=collector.progress.curriculum_stage
                metrics['curriculum/live_credit_mean']=float(collector.progress.curriculum_credit.mean())
            metrics['elapsed_seconds']=time.monotonic()-started
            log.log(metrics,step=index)
            print(json.dumps(metrics),flush=True)
            if (args.output/'pause-request.json').exists():
                saved_path=checkpoint(index,continuation=policy_reverted)
                (args.output/'pause-request.json').unlink()
                (args.output/'active-process.json').write_text(json.dumps(dict(
                    pid=os.getpid(),status='paused',phase=phase,checkpoint=saved_path)))
                paused=True
                break
        if paused:
            log.finish()
            return
        final=checkpoint(index,continuation=policy_reverted)
        result=evaluate_candidate()
        evaluations.append(dict(update=index,**result))
        score=evaluation_score(result)
        if args.reward_graph:
            if score>best:
                best,best_checkpoint,best_result=score,final,result
                save_acceptance(gate_path,final,result,{})
        elif score>best:best,best_checkpoint,best_result=score,final,result
        teacher_sha256=hashlib.sha256(Path(best_checkpoint).read_bytes()).hexdigest()
        report=dict(config=config,baseline=baseline,evaluation=result,evaluations=evaluations,
            updates=index,transitions=transitions,completed_episodes=total_episodes,
            elapsed_seconds=time.monotonic()-started,best_checkpoint=best_checkpoint,
            final_checkpoint=final,wandb_url=log.url,numerical_failures=sum(runtime.check()['flags']))
        if cti is not None:
            report['cti']=dict(rounds=cti_rounds,seconds=cti_seconds,transitions=cti_transitions,
                               selected_worlds=cti_selected,learned_transitions=cti_learned,optimizer_steps=cti_optimizer_steps)
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
        for name in ('runtime', 'evaluation_runtime', 'training_evaluation_runtime', 'cti_runtime'):
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
    profiles=p.add_mutually_exclusive_group()
    profiles.add_argument('--curriculum',action='store_true')
    profiles.add_argument('--reward-graph',action='store_true')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--entropy-coef',type=float,default=0.)
    p.add_argument('--cti-time-fraction',type=float,default=.1)
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
    p.add_argument('--cti-worlds',type=int,default=16,help='Distinct factual PPO roots per search')
    p.add_argument('--cti-alternatives',type=int,default=12)
    p.add_argument('--cti-search-iterations',type=int,default=3)
    p.add_argument('--cti-every-seconds',type=float,default=0.)
    p.add_argument('--cti-coef',type=float,default=.1)
    p.add_argument('--wandb-mode',choices=('disabled','online','offline'),default='online')
    a=p.parse_args()
    if not (1<=a.worlds<=4096 and 1<=a.eval_worlds<=256 and 2<=a.steps<=256 and
            1<=a.minibatch_worlds<=1024 and 1<=a.train_seconds<=7200 and
            1<=a.stall_seconds<a.max_episode_seconds<=60 and a.eval_every_seconds>=1 and
            0<a.arm_speed_rad_s<=2.5 and 1<=a.teacher_min_eval_episodes<=256 and
            (a.role!='teacher' or a.teacher_min_eval_episodes<=a.eval_worlds) and
            0<=a.teacher_distill_weight<=10 and 1<=a.cti_worlds<=64 and
            2<=a.cti_alternatives<=32 and 1<=a.cti_search_iterations<=5 and
            a.cti_every_seconds>=0 and 0<a.cti_coef<=1 and (not a.cti or (a.role=='teacher' and a.cti_worlds<=a.worlds))):
        p.error('Invalid training size or duration')
    if not (0<a.learning_rate<=1e-3 and 0<=a.gae_lambda<=1 and 0<=a.entropy_coef<=.1 and 0<a.cti_time_fraction<=.5):
        p.error('Invalid learning settings')
    if (a.curriculum or a.reward_graph) and a.role!='teacher':
        p.error('Curriculum currently requires privileged teacher training')
    if a.resume_from and (a.initialize_from or (a.wandb_mode=='online' and not a.wandb_run_id)):
        p.error('Resume requires an explicit W&B run ID online and cannot initialize fresh weights')
    if a.wandb_run_id and not a.resume_from:
        p.error('A W&B run ID requires --resume-from')
    import torch,warp as wp
    wp.init(); stream=torch.cuda.Stream()
    with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
        run(a)


if __name__=='__main__':main()
