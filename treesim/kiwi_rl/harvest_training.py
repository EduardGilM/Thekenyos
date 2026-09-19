"""Persistent harvesting episodes, physical-event guidance, and stall detection."""
import torch
import warp as wp
from treesim.basket import CENTER


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
    def __init__(self, initial, *, stall_steps=200, max_steps=1500, guidance=1.):
        self.stall_steps, self.max_steps, self.guidance = stall_steps, max_steps, guidance
        self.age = torch.zeros_like(initial['distance'], dtype=torch.long)
        self.stale = self.age.clone()
        self.closest = initial['distance'].clone()
        self.best_basket = initial['basket_distance'].clone()
        self.best_force = torch.zeros_like(self.closest)
        self.ever_grasp = torch.zeros_like(self.closest, dtype=torch.bool)
        self.ever_detached = self.ever_grasp.clone()
        self.previous = {k:v.clone() for k,v in initial.items()}

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
        carry_progress = now['detached'] & (now['basket_distance'] < self.best_basket - .005)
        pull_progress = now['holding'] & ~now['detached'] & (now['stem_force'] > self.best_force + .5)
        progress = reach_progress | carry_progress | pull_progress | new_grasp | new_detach
        self.stale[progress] = 0
        self.closest[reach_progress] = now['distance'][reach_progress]
        # Start the carry progress tracker at actual detachment, not an old approach pose.
        self.best_basket[new_detach | carry_progress] = now['basket_distance'][new_detach | carry_progress]
        self.best_force[pull_progress] = now['stem_force'][pull_progress]
        self.ever_grasp |= now['grasp']; self.ever_detached |= now['detached']
        stalled = (self.stale >= self.stall_steps) & ~physical_done
        terminated = physical_done | stalled
        truncated = (self.age >= self.max_steps) & ~terminated
        # Phase-specific progress differences avoid a reward jump at detachment.
        reach = 5 * (self.previous['distance'] - now['distance'])
        carry = 5 * (self.previous['basket_distance'] - now['basket_distance'])
        guidance = torch.where(self.previous['detached'], carry, reach)
        # A collision can detach fruit while also damaging it. Never pay a
        # retained-detachment bonus for that hit or for a past, lost grasp.
        guidance += 2 * new_grasp + 4 * (new_detach & now['holding'])
        failure = physical_done & ~now['success']
        guidance = torch.where(failure, torch.minimum(guidance, torch.zeros_like(guidance)), guidance)
        reward = self.guidance * guidance - .001
        reward += 20 * now['success'] - 5 * failure - .5 * stalled
        self.previous = {k:v.clone() for k,v in now.items()}
        return reward, terminated, truncated, stalled


class HarvestCollector:
    """Keep simulation state and GRU memory across optimizer batch boundaries."""
    def __init__(self, runtime, *, stall_seconds=4., max_episode_seconds=30., guidance=1.):
        self.runtime = runtime
        runtime.reset()
        self.memory = torch.zeros(runtime.worlds,64,device=runtime.device_name)
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
        with torch.no_grad():
            initial_memory = self.memory.clone()
            for _ in range(steps):
                obs = rt.observe().clone()
                if self.rgbd is None or self.tick % 2 == 0:
                    self.rgbd = rt.pixels().clone()
                self.memory *= (~self.reset_mask)[:,None]
                mean,logstd,value,self.memory = policy(self.rgbd,obs,self.memory)
                raw = mean if deterministic else mean + logstd.exp()*torch.randn_like(mean)
                rt.set_gait_actions(gait(obs))
                _,_,done,_ = rt.step(raw.tanh())
                now = signals(rt)
                reward,terminated,truncated,stalled = self.progress.step(now,done)
                ending = terminated | truncated
                timeout_value = torch.zeros_like(value)
                if bool(truncated.any()):
                    _,_,timeout_value,_ = policy(rt.pixels(),rt.observe(),self.memory)
                if store:
                    rows.append(dict(rgbd=self.rgbd,r84=obs,raw=raw,
                        logp=tanh_logprob(raw,mean,logstd),value=value,
                        reward=reward,terminated=terminated,truncated=truncated,
                        timeout_value=timeout_value,reset=self.reset_mask.clone()))
                if bool(ending.any()):
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
                    self.progress.reset(ending,signals(rt))
                    self.rgbd = rt.pixels().clone()
                self.reset_mask = ending
                self.tick += 1
            _,_,bootstrap,_ = policy(rt.pixels(),rt.observe(),self.memory)
        if rows:
            rows[0]['initial_memory'] = initial_memory
        rt.check()
        return rows,bootstrap,episodes


def episode_metrics(episodes):
    if not episodes:
        return {'episodes':0}
    keys = ('success','grasp','detached','physical_failure','stalled','timeout','duration_s','closest_distance_m')
    return {'episodes':len(episodes),**{k:sum(r[k] for r in episodes)/len(episodes) for k in keys}}
