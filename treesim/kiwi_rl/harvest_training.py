"""Persistent harvesting episodes, physical-event guidance, and stall detection."""
import torch
import warp as wp
from treesim.basket import CENTER
from .reward_graph import GRAPH_PROFILE, GRAPH_PROFILES, STAGE_NAMES, CollisionGeometry, graph_step

HARVEST_GAMMA = .999
REWARD_PROFILE = 'potential-harvest/v1'
CURRICULUM_PROFILE = 'milestone-harvest/v1'


def signals(runtime):
    pos = wp.to_torch(runtime.data.xpos)
    rotation = wp.to_torch(runtime.data.xmat)[:, runtime.chassis]
    center = torch.as_tensor(CENTER, dtype=pos.dtype, device=pos.device)
    basket = pos[:, runtime.chassis] + (rotation @ center.expand(runtime.worlds, 3)[..., None]).squeeze(-1)
    task = runtime.task
    result = dict(distance=wp.to_torch(runtime._distance).clone(),
                basket_distance=(pos[:, runtime.fruit_body] - basket).norm(dim=-1),
                grasp=wp.to_torch(task.ever_grasped).bool().clone(),
                holding=wp.to_torch(task.stable_grasp).bool().clone(),
                detached=wp.to_torch(task.detached).bool().clone(),
                touching=wp.to_torch(task.hand_contact).bool().clone(),
                stem_force=wp.to_torch(task.stem_force).clone(),
                success=wp.to_torch(task.success).bool().clone(),
                failed=wp.to_torch(task.failed).bool().clone(),
                settle_time=(wp.to_torch(task.settle_time).clone() if hasattr(task, 'settle_time')
                             else torch.zeros_like(wp.to_torch(runtime._distance))))
    if getattr(runtime, 'task_profile', None) in GRAPH_PROFILES:
        if not hasattr(runtime, '_graph_geometry'):
            runtime._graph_geometry = CollisionGeometry(runtime)
        result.update(runtime._graph_geometry.signals(runtime))
        result.update(bilateral=wp.to_torch(task.bilateral_contact).bool().clone(),
            max_load=torch.stack([wp.to_torch(x) for x in (task.finger_load, task.jaw_load, task.palm_load)]).amax(0),
            damage=wp.to_torch(task.damage_proxy).clone(),
            ground_contact=wp.to_torch(task.ground_contact).bool().clone())
    return result


class EpisodeProgress:
    """Track meaningful best-so-far progress, never motion or repeat events."""
    def __init__(self, initial, *, stall_steps=200, max_steps=1500, guidance=1., gamma=HARVEST_GAMMA,
                 reward_profile=REWARD_PROFILE, curriculum_stage=0, control_dt=.02):
        if reward_profile not in (REWARD_PROFILE, CURRICULUM_PROFILE, *GRAPH_PROFILES):
            raise ValueError('Unknown harvesting reward profile')
        if curriculum_stage not in (0, 1, 2):
            raise ValueError('Curriculum stage must be 0, 1, or 2')
        self.reward_profile, self.curriculum_stage = reward_profile, curriculum_stage
        self.stall_steps, self.max_steps, self.guidance = stall_steps, max_steps, guidance
        self.gamma = gamma
        self.control_dt = control_dt
        self.age = torch.zeros_like(initial['distance'], dtype=torch.long)
        self.stale = self.age.clone()
        self.closest = initial['distance'].clone()
        self.best_basket = initial['basket_distance'].clone()
        self.best_force = torch.zeros_like(self.closest)
        self.ever_grasp = torch.zeros_like(self.closest, dtype=torch.bool)
        self.ever_detached = self.ever_grasp.clone()
        self.ever_held_detach = self.ever_grasp.clone()
        self.previous = {k:v.clone() for k,v in initial.items()}
        self.curriculum_best_reach = (1 - initial['distance'] / .25).clamp(0, 1)
        self.curriculum_best_carry = (1 - initial['basket_distance']).clamp(0, 1)
        self.curriculum_best_settle = torch.zeros_like(self.closest)
        self.curriculum_grasp_paid = torch.zeros_like(self.ever_grasp)
        self.curriculum_detach_paid = torch.zeros_like(self.ever_grasp)
        self.curriculum_stage_ids = torch.full_like(self.age, curriculum_stage)
        self.curriculum_credit = torch.zeros_like(self.closest)
        self.graph_dwell = torch.zeros_like(self.closest)
        self.graph_grip = torch.zeros_like(self.ever_grasp)
        self.graph_detached = torch.zeros_like(self.ever_grasp)
        self.graph_enclosed = torch.zeros_like(self.ever_grasp)
        self.graph_ground_drop = torch.zeros_like(self.ever_grasp)
        self.graph_slip = torch.zeros_like(self.closest)
        self.graph_score = torch.zeros_like(self.closest)
        self.graph_stage = torch.zeros_like(self.age)
        if reward_profile in GRAPH_PROFILES:
            graph_step(self, initial)
            self.graph_dwell.zero_()
        self.graph_best = self.graph_score.clone()
        self.graph_reference = self.graph_score.clone()
        self.graph_progress_reward = {name: torch.zeros_like(self.closest) for name in STAGE_NAMES}
        self.graph_time = {name: torch.zeros_like(self.closest) for name in STAGE_NAMES}
        self.graph_reached = {name: torch.zeros_like(self.ever_grasp) for name in STAGE_NAMES}
        self.graph_forward = torch.zeros_like(self.age)
        self.graph_backward = torch.zeros_like(self.age)

    @staticmethod
    def potential(state):
        # Bounded state credit, not permanent event bonuses. No angle or
        # prescribed grasp sequence is part of the physical success criterion.
        reach = (1 - state['distance'] / .25).clamp(0, 1)
        carry = (1 - state['basket_distance'] / 1.).clamp(0, 1)
        holding = state['holding'].float()
        attached = (~state['detached']).float()
        return attached * reach + 2 * holding + state['detached'].float() * holding * (4 + 2 * carry)

    def reset(self, mask, initial):
        self.age[mask] = 0; self.stale[mask] = 0
        self.closest[mask] = initial['distance'][mask]
        self.best_basket[mask] = initial['basket_distance'][mask]
        self.best_force[mask] = 0
        self.ever_grasp[mask] = False; self.ever_detached[mask] = False
        self.ever_held_detach[mask] = False
        if self.curriculum_stage not in (0, 1, 2):
            raise ValueError('Curriculum stage must be 0, 1, or 2')
        self.curriculum_best_reach[mask] = (1 - initial['distance'][mask] / .25).clamp(0, 1)
        self.curriculum_best_carry[mask] = (1 - initial['basket_distance'][mask]).clamp(0, 1)
        self.curriculum_best_settle[mask] = 0
        self.curriculum_grasp_paid[mask] = False
        self.curriculum_detach_paid[mask] = False
        self.curriculum_stage_ids[mask] = self.curriculum_stage
        self.curriculum_credit[mask] = 0
        if self.reward_profile in GRAPH_PROFILES:
            fresh = EpisodeProgress(initial, reward_profile=self.reward_profile, control_dt=self.control_dt,
                                    gamma=self.gamma)
            for name, value in vars(self).items():
                if name.startswith('graph_') and isinstance(value, torch.Tensor):
                    value[mask] = getattr(fresh, name)[mask]
                elif name.startswith('graph_') and isinstance(value, dict):
                    for key in value:
                        value[key][mask] = getattr(fresh, name)[key][mask]
        for key,value in initial.items():
            self.previous[key][mask] = value[mask]

    def _milestone_bonus(self, now, physical_done):
        # This curriculum is deliberately non-potential guidance. Bounded,
        # irreversible credit survives stalls; it never declares task success.
        safe = ~now['failed'] & ~(physical_done & ~now['success'])
        held = now['holding'] & safe
        new_grasp = held & ~self.curriculum_grasp_paid
        new_detach = held & now['detached'] & ~self.curriculum_detach_paid
        reach = (1 - now['distance'] / .25).clamp(0, 1)
        carry = (1 - now['basket_distance']).clamp(0, 1)
        # The task timer advances only when released, contained, contacting the
        # basket, detached and below its physical settling-speed threshold.
        settle = (now.get('settle_time', torch.zeros_like(reach)) / .5).clamp(0, 1)
        dr = torch.where(safe & ~now['detached'], (reach - self.curriculum_best_reach).clamp_min(0), 0.)
        dc = torch.where(held & now['detached'], (carry - self.curriculum_best_carry).clamp_min(0), 0.)
        ds = torch.where(safe & now['detached'], (settle - self.curriculum_best_settle).clamp_min(0), 0.)
        # All stages retain every skill. Stage changes take effect at world reset
        # so switching weights cannot repay an earlier milestone.
        weights = reach.new_tensor(((.5, 1.5, .75, .5, .75),
                                    (.25, .75, 1.5, .75, .75),
                                    (.25, .5, .75, 1.25, 1.25)))[self.curriculum_stage_ids]
        earned = torch.stack((dr, new_grasp.float(), new_detach.float(), dc, ds), dim=-1)
        bonus = (earned * weights).sum(dim=-1).clamp_min(0)
        bonus = torch.minimum(bonus, (4. - self.curriculum_credit).clamp_min(0))
        self.curriculum_best_reach += dr
        self.curriculum_best_carry += dc
        self.curriculum_best_settle += ds
        self.curriculum_grasp_paid |= new_grasp
        self.curriculum_detach_paid |= new_detach
        self.curriculum_credit += bonus
        return self.guidance * bonus

    def step(self, now, physical_done):
        if self.reward_profile in GRAPH_PROFILES:
            return self._graph_step(now, physical_done)
        self.age += 1; self.stale += 1
        new_grasp = now['grasp'] & ~self.ever_grasp
        new_detach = now['detached'] & ~self.ever_detached
        reach_progress = ~now['detached'] & (now['distance'] < self.closest - .005)
        carry_progress = now['detached'] & now['holding'] & (now['basket_distance'] < self.best_basket - .005)
        pull_progress = now['holding'] & ~now['detached'] & (now['stem_force'] > self.best_force + .5)
        progress = reach_progress | carry_progress | pull_progress | new_grasp | (new_detach & now['holding'])
        self.stale[progress] = 0
        self.closest[reach_progress] = now['distance'][reach_progress]
        # Start the carry progress tracker at actual detachment, not an old approach pose.
        self.best_basket[new_detach | carry_progress] = now['basket_distance'][new_detach | carry_progress]
        self.best_force[pull_progress] = now['stem_force'][pull_progress]
        self.ever_grasp |= now['grasp']; self.ever_detached |= now['detached']
        self.ever_held_detach |= (now['detached'] & now['holding'] & ~now['failed'] &
                                 ~(physical_done & ~now['success']))
        stalled = (self.stale >= self.stall_steps) & ~physical_done
        terminated = physical_done | stalled
        truncated = (self.age >= self.max_steps) & ~terminated
        # Discount-consistent shaping telescopes over an episode. True terminals
        # clear all credit; hard timeouts retain it for the critic bootstrap.
        next_potential = torch.where(terminated, 0., self.potential(now))
        if self.reward_profile == CURRICULUM_PROFILE:
            self.shaping_reward = self._milestone_bonus(now, physical_done)
        else:
            self.shaping_reward = self.guidance * (self.gamma * next_potential - self.potential(self.previous))
        failure = physical_done & ~now['success']
        self.task_reward = -.001 + 20 * now['success'] - 5 * failure - .5 * stalled
        reward = self.task_reward + self.shaping_reward
        self.previous = {k:v.clone() for k,v in now.items()}
        return reward, terminated, truncated, stalled


    def _graph_step(self, now, physical_done):
        self.age += 1; self.stale += 1
        previous_score = self.graph_score.clone()
        previous_components = self.graph_components
        previous_stage = self.graph_stage.clone()
        score = graph_step(self, now)
        if self.reward_profile == GRAPH_PROFILE:
            # A moving reference recognizes accumulated small gains and recovery.
            # Regression re-arms it, but cannot itself reset the inactivity timer.
            improved = score > self.graph_reference + .005
            self.graph_reference = torch.where(improved | (score < self.graph_reference), score, self.graph_reference)
        else:
            improved = score > self.graph_best + .005
        self.stale[improved] = 0
        self.graph_best = torch.maximum(self.graph_best, score)
        self.graph_forward += (self.graph_stage > previous_stage).long()
        self.graph_backward += (self.graph_stage < previous_stage).long()
        detached = self.graph_detached
        settling = detached & ~now['touching'] & (now['settle_time'] > 0)
        activity = dict(position=~now['enclosed'] & ~detached,
            grip=now['enclosed'] & ~self.graph_grip & ~detached,
            extract=(self.graph_grip & ~detached) | (detached & ~self.graph_grip & ~settling),
            carry=detached & self.graph_grip,
            deposit=settling & ~now['success'], complete=now['success'])
        for name in STAGE_NAMES:
            self.graph_progress_reward[name] = self.gamma*self.graph_components[name] - previous_components[name]
            active = activity[name]
            self.graph_time[name] += active.float()*self.control_dt
            self.graph_reached[name] |= active

        self.closest = torch.minimum(self.closest, now['distance'])
        self.ever_grasp |= self.graph_grip
        self.ever_detached |= self.graph_detached
        self.ever_held_detach |= self.graph_detached & self.graph_grip
        ending = physical_done | self.graph_ground_drop
        stalled = (self.stale >= self.stall_steps) & ~ending
        terminated = ending | stalled
        truncated = (self.age >= self.max_steps) & ~terminated
        failure = now['failed'] | (physical_done & ~now['success'])
        self.task_reward = -.001 + 20*now['success'] - 5*failure - .5*stalled
        self.shaping_reward = self.gamma*score - previous_score
        self.previous = {k:v.clone() for k,v in now.items()}
        return self.task_reward+self.shaping_reward, terminated, truncated, stalled


class HarvestCollector:
    """Keep simulation state and GRU memory across optimizer batch boundaries."""
    def __init__(self, runtime, *, stall_seconds=4., max_episode_seconds=30., guidance=1.,
                 role='student', teacher_policy=None, reward_profile=REWARD_PROFILE, curriculum_stage=0):
        if role not in ('teacher', 'student'):
            raise ValueError("role must be 'teacher' or 'student'")
        if role == 'teacher' and teacher_policy is not None:
            raise ValueError('A teacher run cannot also load a teacher checkpoint')
        if reward_profile in GRAPH_PROFILES and getattr(runtime, 'task_profile', None) != reward_profile:
            raise ValueError('Graph reward requires matching physical runtime profile')
        self.runtime = runtime
        self.role, self.teacher_policy = role, teacher_policy
        runtime.reset()
        self.memory = torch.zeros(runtime.worlds,64,device=runtime.device_name)
        self.teacher_memory = (torch.zeros_like(self.memory) if teacher_policy is not None else None)
        self.reset_mask = torch.ones(runtime.worlds,dtype=torch.bool,device=runtime.device_name)
        self.episode_ids = torch.zeros(runtime.worlds,dtype=torch.long,device=runtime.device_name)
        self.progress = EpisodeProgress(signals(runtime),
            stall_steps=round(stall_seconds/runtime.control_dt),
            max_steps=round(max_episode_seconds/runtime.control_dt), guidance=guidance,
            reward_profile=reward_profile, curriculum_stage=curriculum_stage, control_dt=runtime.control_dt)
        self.tick = 0
        self.rgbd = None

    def collect(self, policy, gait, steps, *, deterministic=False, store=True,
                cti_queue=None, policy_version=0):
        from .ppo import tanh_logprob
        rt = self.runtime
        if cti_queue is not None and steps != cti_queue.segment_steps:
            raise ValueError(f'CTI factual segments require exactly {cti_queue.segment_steps} PPO steps')
        rows, episodes = [], []
        cti_batch = None
        cti_active = torch.ones(rt.worlds,dtype=torch.bool,device=rt.device_name)
        from .fast_teacher import privileged_observation
        with torch.no_grad():
            initial_memory = self.memory.clone()
            for _ in range(steps):
                obs = rt.observe().clone()
                privileged = privileged_observation(rt, obs) if self.role == 'teacher' or self.teacher_policy is not None else None
                if self.role == 'student' and (self.rgbd is None or self.tick % 2 == 0):
                    self.rgbd = rt.pixels().clone()
                self.memory *= (~self.reset_mask)[:,None]
                if cti_queue is not None and cti_batch is None:
                    cti_batch = cti_queue.begin(rt,self,policy,policy_version)
                    if cti_batch is not None: cti_active.fill_(True)
                if self.role == 'teacher':
                    policy_input = privileged
                else:
                    policy_input = self.rgbd
                mean,logstd,value,self.memory = policy(policy_input,obs,self.memory)
                raw = mean if deterministic else mean + logstd.exp()*torch.randn_like(mean)
                teacher_action = None
                if self.teacher_policy is not None:
                    self.teacher_memory *= (~self.reset_mask)[:,None]
                    teacher_mean,_,_,self.teacher_memory = self.teacher_policy(
                        privileged,obs,self.teacher_memory)
                    teacher_action = teacher_mean.tanh()
                gait_action = gait(obs)
                rt.set_gait_actions(gait_action)
                next_obs,_,done,_ = rt.step(raw.tanh())
                now = signals(rt)
                reward,terminated,truncated,stalled = self.progress.step(now,done)
                ending = terminated | truncated
                if cti_batch is not None:
                    cti_queue.record(cti_batch,obs=obs,raw_action=raw,action=raw.tanh(),
                        noise=(raw-mean)/logstd.exp(),
                        gait_action=gait_action,next_obs=next_obs,runtime=rt,signals=now,
                        reward=reward,task_reward=self.progress.task_reward,
                        shaping_reward=self.progress.shaping_reward,terminated=terminated,
                        truncated=truncated,stalled=stalled,episode_ids=self.episode_ids,
                        active_mask=cti_active,
                        graph_state=({name:value for name,value in vars(self.progress).items()
                                      if name.startswith('graph_')} if self.progress.reward_profile in GRAPH_PROFILES else None))
                    cti_active[ending] = False
                timeout_value = torch.zeros_like(value)
                if bool(truncated.any()):
                    final_obs = rt.observe()
                    final_input = (privileged_observation(rt,final_obs) if self.role == 'teacher'
                                   else (rt.pixels() if (self.tick + 1) % 2 == 0 else self.rgbd))
                    _,_,timeout_value,_ = policy(final_input,final_obs,self.memory)
                if store:
                    row = dict(r84=obs,raw=raw,
                        logp=tanh_logprob(raw,mean,logstd),value=value,
                        reward=reward,task_reward=self.progress.task_reward,
                        shaping_reward=self.progress.shaping_reward,terminated=terminated,truncated=truncated,
                        timeout_value=timeout_value,reset=self.reset_mask.clone(),role=self.role,
                        kind='factual_on_policy')
                    if self.role == 'teacher':
                        row['privileged'] = privileged
                    else:
                        row['rgbd'] = self.rgbd
                    if teacher_action is not None:
                        row['teacher_action'] = teacher_action
                    if self.progress.reward_profile in GRAPH_PROFILES:
                        row['stage_rewards'] = {k:v.clone() for k,v in self.progress.graph_progress_reward.items()}
                        row['stage_scores'] = {k:v.clone() for k,v in self.progress.graph_components.items()}
                        row['stage_rewards']['complete'] = 20*now['success'].float()
                        row['joint_velocity'] = wp.to_torch(rt.data.qvel)[:, self.runtime.control.contract.dofs[12:]].clone()
                        row['target_error'] = (wp.to_torch(rt.control.targets)[:,12:] - wp.to_torch(rt.data.qpos)[:,rt.control.contract.qids[12:]]).clone()
                    rows.append(row)
                if bool(ending.any()):
                    # Do not erase a numerically failed world's state in a reset.
                    rt.check()
                    # Summaries are episode outcomes; each world contributes once.
                    ids = ending.nonzero(as_tuple=False).flatten()
                    packed = torch.stack((ids, now['success'][ids],
                        self.progress.ever_grasp[ids],self.progress.ever_detached[ids],self.progress.ever_held_detach[ids],
                        (done & ~now['success'])[ids],stalled[ids],truncated[ids],
                        self.progress.age[ids]*rt.control_dt,self.progress.closest[ids]),dim=1).cpu().tolist()
                    for world,success,grasp,detached,held_detach,failure,stall,timeout,duration,closest in packed:
                        episodes.append(dict(world=int(world),success=bool(success),grasp=bool(grasp),
                            detached=bool(detached),held_detach=bool(held_detach),physical_failure=bool(failure),stalled=bool(stall),
                            timeout=bool(timeout),duration_s=duration,closest_distance_m=closest))
                    if self.progress.reward_profile in GRAPH_PROFILES:
                        for entry, world in zip(episodes[-len(ids):], ids.tolist()):
                            entry.update(graph_score=float(self.progress.graph_score[world]),
                                graph_stage=int(self.progress.graph_stage[world]),
                                enclosure=bool(self.progress.graph_enclosed[world]),
                                secure_grip=bool(self.progress.graph_grip[world]),
                                slip_m_s=float(self.progress.graph_slip[world]),
                                ground_drop=bool(self.progress.graph_ground_drop[world]),
                                forward_transitions=int(self.progress.graph_forward[world]),
                                backward_transitions=int(self.progress.graph_backward[world]),
                                **{f'reached/{k}':bool(v[world]) for k,v in self.progress.graph_reached.items()},
                                **{f'time/{k}_seconds':float(v[world]) for k,v in self.progress.graph_time.items()})
                    rt.reset(ending)
                    self.episode_ids[ending] += 1
                    self.memory[ending] = 0
                    if self.teacher_memory is not None:
                        self.teacher_memory[ending] = 0
                    self.progress.reset(ending,signals(rt))
                    if self.role == 'student':
                        # Only reset worlds receive a fresh initial frame. Do
                        # not change another world's 25 Hz sensor delivery.
                        self.rgbd = torch.where(ending[:,None,None,None],
                                                rt.pixels(), self.rgbd)
                self.reset_mask = ending
                self.tick += 1
            final_obs = rt.observe()
            final_input = (privileged_observation(rt,final_obs) if self.role == 'teacher'
                           else (rt.pixels() if self.tick % 2 == 0 else self.rgbd))
            _,_,bootstrap,_ = policy(final_input,final_obs,self.memory)
        if cti_batch is not None:
            cti_queue.finish(cti_batch)
        if rows:
            rows[0]['initial_memory'] = initial_memory
        rt.check()
        return rows,bootstrap,episodes


def episode_metrics(episodes):
    if not episodes:
        return {'episodes':0}
    keys = ('success','grasp','detached','physical_failure','stalled','timeout','duration_s','closest_distance_m')
    return {'episodes':len(episodes),
            'held_detach':sum(r.get('held_detach',False) for r in episodes)/len(episodes),
            **{k:sum(r[k] for r in episodes)/len(episodes) for k in keys},
            **{k:sum(r.get(k,0) for r in episodes)/len(episodes) for k in
               ('graph_score','graph_stage','enclosure','secure_grip','slip_m_s','ground_drop','forward_transitions','backward_transitions',
                *[f'reached/{k}' for k in STAGE_NAMES], *[f'time/{k}_seconds' for k in STAGE_NAMES]) if any(k in r for r in episodes)}}
