"""Local reach/grasp curriculum on the existing rigid-fruit Spot simulator.

IK initializes an open hand near the fruit; it never supplies policy actions.
This diagnostic pilot is not evidence of rigid-to-deformable grasp transfer.
"""
import numpy as np
from scipy.optimize import least_squares
import newton
from .harvest_env import SpotHarvestEnv
from .harvest_task import TaskDefinition, rotation


class ReachGraspEnv(SpotHarvestEnv):
    def __init__(self, relic, *, render_mode=None, guidance_weight=1., device='cpu'):
        super().__init__(relic, render_mode=render_mode, device=device,
                         task=TaskDefinition(time_limit_s=4., guidance_weight=0.))
        self.guidance_weight = guidance_weight

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        sim = self.sim
        initial = sim.state_0.joint_q.numpy().copy()
        scratch = sim.model.state()
        fruit = sim.body_q_np()[self.observer.body,:3].copy()
        # Easy local curriculum: 8 cm horizontal / 6 cm vertical clearance.
        # The hand is OPEN and there is no contact constraint or demonstration.
        target = fruit + [-.08, self.np_random.uniform(-.015,.015), -.06]
        home = initial[self.arm_q[:6]].copy()
        def residual(angles):
            q = initial.copy(); q[self.arm_q[:6]] = angles
            scratch.joint_q.assign(q)
            newton.eval_fk(sim.model, scratch.joint_q, scratch.joint_qd, scratch)
            pose = scratch.body_q.numpy()[self.observer.tcp_body]
            tcp = pose[:3]+rotation(pose[3:])@self.observer.offset
            return np.r_[tcp-target, .001*(angles-home)]
        result = least_squares(residual, home.astype(float), bounds=(self.lower[:6],self.upper[:6]),
                               diff_step=.001, max_nfev=120, gtol=1e-7)
        if np.linalg.norm(residual(result.x)[:3])>.01:
            raise RuntimeError('Near-target initialization is unreachable')
        self.controller.targets[12:18] = result.x
        self.controller.target.assign(self.controller.targets)
        for state in (sim.state_0,sim.state_1):
            q=initial.copy(); q[self.arm_q[:6]]=result.x
            state.joint_q.assign(q); state.joint_qd.zero_()
            newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
        sim._host_step += 1
        self.physical=self.observer.observe(0.)
        self.oracle.reset(self.physical)
        self.previous_distance=self.physical.tcp_distance_m
        self.best_distance=self.previous_distance
        return self._observation(), self._info()

    def step(self, action):
        obs,reward,done,truncated,info=super().step(action)
        distance=self.physical.tcp_distance_m
        self.best_distance=min(self.best_distance,distance)
        shaping=self.guidance_weight*20*(self.previous_distance-distance)
        self.previous_distance=distance
        reward += shaping
        info['reward_terms']['local_reach_guidance']=shaping
        # Evaluate the same per-substep stable-contact event as the task oracle.
        # A detached target or an oracle failure cannot become a grasp success.
        success=bool(self.oracle.grasped and self.physical.attached and not done
                     and min(self.physical.jaw_forces_N) >= self.task.contact_min_N
                     and self.physical.slip_speed_m_s < self.task.settle_speed_m_s)
        if success:
            reward += 5.
            info['reward_terms']['reach_grasp_success']=5.
            done=self.done=True
            info['outcome']='stable_grasp'
        info.update(success=success, best_distance_m=self.best_distance,
                    stage='reach_grasp', collection_success=False)
        return obs,float(reward),done,truncated,info
