"""Persistent harvesting episodes, physical-event guidance, and stall detection."""
import torch
import warp as wp
from treesim.basket import CENTER

HARVEST_GAMMA = .999
REWARD_PROFILE = 'potential-harvest/v1'


def signals(runtime):
    pos = wp.to_torch(runtime.data.xpos)
    rotation = wp.to_torch(runtime.data.xmat)[:, runtime.chassis]
    center = torch.as_tensor(CENTER, dtype=pos.dtype, device=pos.device)
    basket = pos[:, runtime.chassis] + (rotation @ center.expand(runtime.worlds, 3)[..., None]).squeeze(-1)
    task = runtime.task
    return dict(distance=wp.to_torch(runtime._distance).clone(),
                basket_distance=(pos[:, runtime.fruit_body] - basket).norm(dim=-1),
                grasp=wp.to_torch(task.ever_grasped).bool().clone(),
                holding=wp.to_torch(task.stable_grasp).bool().clone(),
                detached=wp.to_torch(task.detached).bool().clone(),
                touching=wp.to_torch(task.hand_contact).bool().clone(),
                stem_force=wp.to_torch(task.stem_force).clone(),
                success=wp.to_torch(task.success).bool().clone(),
                failed=wp.to_torch(task.failed).bool().clone())


class EpisodeProgress:
    """Track meaningful best-so-far progress, never motion or repeat events."""
    def __init__(self, initial, *, stall_steps=200, max_steps=1500, guidance=1., gamma=HARVEST_GAMMA):
        self.stall_steps, self.max_steps, self.guidance = stall_steps, max_steps, guidance
        self.gamma = gamma
        self.age = torch.zeros_like(initial['distance'], dtype=torch.long)
        self.stale = self.age.clone()
        self.closest = initial['distance'].clone()
        self.best_basket = initial['basket_distance'].clone()
        self.best_force = torch.zeros_like(self.closest)
        self.ever_grasp = torch.zeros_like(self.closest, dtype=torch.bool)
        self.ever_detached = self.ever_grasp.clone()
        self.previous = {k:v.clone() for k,v in initial.items()}

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
        for key,value in initial.items():
            self.previous[key][mask] = value[mask]

    def step(self, now, physical_done):
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
        stalled = (self.stale >= self.stall_steps) & ~physical_done
        terminated = physical_done | stalled
        truncated = (self.age >= self.max_steps) & ~terminated
        # Discount-consistent shaping telescopes over an episode. True terminals
        # clear all credit; hard timeouts retain it for the critic bootstrap.
        next_potential = torch.where(terminated, 0., self.potential(now))
        self.shaping_reward = self.guidance * (self.gamma * next_potential - self.potential(self.previous))
        failure = physical_done & ~now['success']
        self.task_reward = -.001 + 20 * now['success'] - 5 * failure - .5 * stalled
        reward = self.task_reward + self.shaping_reward
        self.previous = {k:v.clone() for k,v in now.items()}
        return reward, terminated, truncated, stalled


class HarvestCollector:
    """Keep simulation state and GRU memory across optimizer batch boundaries."""
    def __init__(self, runtime, *, stall_seconds=4., max_episode_seconds=30., guidance=1.,
                 role='student', teacher_policy=None):
        if role not in ('teacher', 'student'):
            raise ValueError("role must be 'teacher' or 'student'")
        if role == 'teacher' and teacher_policy is not None:
            raise ValueError('A teacher run cannot also load a teacher checkpoint')
        self.runtime = runtime
        self.role, self.teacher_policy = role, teacher_policy
        runtime.reset()
        self.memory = torch.zeros(runtime.worlds,64,device=runtime.device_name)
        self.teacher_memory = (torch.zeros_like(self.memory) if teacher_policy is not None else None)
        self.reset_mask = torch.ones(runtime.worlds,dtype=torch.bool,device=runtime.device_name)
        self.progress = EpisodeProgress(signals(runtime),
            stall_steps=round(stall_seconds/runtime.control_dt),
            max_steps=round(max_episode_seconds/runtime.control_dt), guidance=guidance)
        self.tick = 0
        self.rgbd = None

    def collect(self, policy, gait, steps, *, deterministic=False, store=True):
        from .ppo import tanh_logprob
        rt = self.runtime
        rows, episodes = [], []
        from .fast_teacher import privileged_observation
        with torch.no_grad():
            initial_memory = self.memory.clone()
            for _ in range(steps):
                obs = rt.observe().clone()
                privileged = privileged_observation(rt, obs) if self.role == 'teacher' or self.teacher_policy is not None else None
                if self.role == 'student' and (self.rgbd is None or self.tick % 2 == 0):
                    self.rgbd = rt.pixels().clone()
                self.memory *= (~self.reset_mask)[:,None]
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
                rt.set_gait_actions(gait(obs))
                _,_,done,_ = rt.step(raw.tanh())
                now = signals(rt)
                reward,terminated,truncated,stalled = self.progress.step(now,done)
                ending = terminated | truncated
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
                    rows.append(row)
                if bool(ending.any()):
                    # Do not erase a numerically failed world's state in a reset.
                    rt.check()
                    # Summaries are episode outcomes; each world contributes once.
                    ids = ending.nonzero(as_tuple=False).flatten()
                    packed = torch.stack((ids, now['success'][ids],
                        self.progress.ever_grasp[ids],self.progress.ever_detached[ids],
                        (done & ~now['success'])[ids],stalled[ids],truncated[ids],
                        self.progress.age[ids]*rt.control_dt,self.progress.closest[ids]),dim=1).cpu().tolist()
                    for world,success,grasp,detached,failure,stall,timeout,duration,closest in packed:
                        episodes.append(dict(world=int(world),success=bool(success),grasp=bool(grasp),
                            detached=bool(detached),physical_failure=bool(failure),stalled=bool(stall),
                            timeout=bool(timeout),duration_s=duration,closest_distance_m=closest))
                    rt.reset(ending)
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
        if rows:
            rows[0]['initial_memory'] = initial_memory
        rt.check()
        return rows,bootstrap,episodes


def episode_metrics(episodes):
    if not episodes:
        return {'episodes':0}
    keys = ('success','grasp','detached','physical_failure','stalled','timeout','duration_s','closest_distance_m')
    return {'episodes':len(episodes),**{k:sum(r[k] for r in episodes)/len(episodes) for k in keys}}
