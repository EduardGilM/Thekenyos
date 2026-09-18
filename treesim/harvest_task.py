"""Single-fruit RL task and privileged evaluator; never issues robot commands.

Policy action: seven normalized joint velocity commands (six arm + one jaw).
The actuator adapter must enforce URDF joint limits and measured effort limits.
The first curriculum fixes the chassis and holds the legs; this module does
not create that fixture or implement an RL training loop.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaskDefinition:
    guidance_weight: float = 1.  # Optional grasp/reach shaping; anneal to 0 BETWEEN episodes.
    time_limit_s: float = 30.
    max_observation_gap_s: float = .1
    grasp_dwell_s: float = .12
    settle_dwell_s: float = .5
    contact_min_N: float = .2
    settle_speed_m_s: float = .05
    damage_limit: float = .05  # Proxy score, NOT a measured bruise threshold.
    jaw_force_limit_N: float = 15.  # Engineering stop, NOT a validated safe force.
    gamma_per_second: float = .99
    success_reward: float = 10.
    failure_penalty: float = 10.
    damage_weight: float = 20.
    time_cost_per_s: float = .02
    effort_weight: float = .001  # Absolute actuator work in joules.

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError('Task parameters must be finite')
        if self.guidance_weight < 0 or any(v <= 0 for k,v in vars(self).items() if k != "guidance_weight") or self.gamma_per_second > 1:
            raise ValueError('Task parameters must be positive; discount must be <= 1')

    def action_velocity(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Expected seven finite actions in [-1, 1]')
        # Initial curriculum limits, not Spot hardware maxima.
        return action*np.array([.5]*6+[.5])


@dataclass(frozen=True)
class HarvestObservation:
    time_s: float
    fruit_id: int
    attached: bool
    tcp_distance_m: float
    jaw_forces_N: tuple
    slip_speed_m_s: float
    in_basket: bool       # Entire ellipsoid within the basket's interior.
    basket_contact: bool
    basket_relative_speed_m_s: float
    ground_contact: bool
    damage: float
    fsa_deg: float        # Diagnostic only: no reward for posing at 60 degrees.
    stem_tension_N: float
    stem_threshold_N: float
    actuator_work_J: float  # Cumulative absolute work since environment reset.
    collateral_losses: int = 0  # Other detached canopy fruit or spilled payload.
    forbidden_collision: bool = False

    def validate(self):
        values = [self.time_s, self.tcp_distance_m, *self.jaw_forces_N,
                  self.slip_speed_m_s, self.basket_relative_speed_m_s, self.damage,
                  self.fsa_deg, self.stem_tension_N, self.stem_threshold_N, self.actuator_work_J]
        if len(self.jaw_forces_N) != 2 or not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Invalid physical observation')
        if self.fsa_deg > 180 or self.damage > 1 or self.collateral_losses < 0:
            raise ValueError('Observation outside physical ranges')


class HarvestOracle:
    """Reward/outcome evaluator. Reset with the initial physical observation.

    No rewards for crossing a commanded angle or merely opening the jaw.
    Dense shaping uses a bounded reach potential with terminal correction.
    Event rewards cannot repeat, and damage cannot recover for reward purposes.
    """
    def __init__(self, task=None):
        self.task = task or TaskDefinition()
        self.initial = None

    @staticmethod
    def potential(obs):
        return float(np.exp(-obs.tcp_distance_m/.15))

    def reset(self, obs):
        obs.validate()
        if not obs.attached or obs.ground_contact or obs.in_basket or obs.damage >= self.task.damage_limit:
            raise ValueError('Episode must start with attached, undamaged target fruit')
        self.initial = self.previous = obs
        self.grasp_time = self.settle_time = 0.
        self.previous_stable = self.previous_settled = False
        self.grasped = self.detached_with_grasp = False
        self.max_damage = obs.damage
        self.max_losses = obs.collateral_losses
        self.done = False
        self.outcome = None
        self.total_reward = 0.
        self.previous_potential = self.potential(obs)

    def update(self, obs):
        if self.initial is None:
            raise RuntimeError('Call reset before update')
        obs.validate()
        if obs.fruit_id != self.initial.fruit_id:
            raise ValueError('Cannot switch target during an episode')
        if self.done:
            return dict(reward=0., terminated=True, success=self.outcome == 'success',
                        outcome=self.outcome, terms={}, events=[])
        dt = obs.time_s-self.previous.time_s
        if dt <= 0 or dt > self.task.max_observation_gap_s+1e-9 or obs.actuator_work_J < self.previous.actuator_work_J:
            raise ValueError('Time must advance within the observation gap limit; work cannot decrease')
        if not self.previous.attached and obs.attached:
            raise ValueError('Detached fruit cannot reattach within an episode')
        task = self.task
        both = min(obs.jaw_forces_N) >= task.contact_min_N
        stable = both and obs.slip_speed_m_s <= .02
        self.grasp_time = self.grasp_time+dt if stable and self.previous_stable else 0.
        events = []
        terms = {'time': -task.time_cost_per_s*dt,
                 'effort': -task.effort_weight*(obs.actuator_work_J-self.previous.actuator_work_J),
                 'damage': -task.damage_weight*max(0., obs.damage-self.max_damage),
                 'collateral': -task.failure_penalty*max(0, obs.collateral_losses-self.max_losses)}
        # A grasp must already be physically established while attached.
        if not self.grasped and obs.attached and self.grasp_time >= task.grasp_dwell_s:
            self.grasped = True
            terms['grasp'] = task.guidance_weight
            events.append('stable_grasp')
        detached_now = self.previous.attached and not obs.attached
        if detached_now and self.grasped and stable and self.grasp_time >= task.grasp_dwell_s:
            self.detached_with_grasp = True
            terms['detach'] = 2.*task.guidance_weight
            events.append('retained_detachment')
        if detached_now and not self.detached_with_grasp:
            events.append('detached_without_confirmed_grasp')
        self.max_damage = max(self.max_damage, obs.damage)
        new_losses = obs.collateral_losses > self.max_losses
        self.max_losses = max(self.max_losses, obs.collateral_losses)
        released = max(obs.jaw_forces_N) < task.contact_min_N
        settled = (not obs.attached and released and obs.in_basket
                   and obs.basket_contact and obs.basket_relative_speed_m_s <= task.settle_speed_m_s)
        self.settle_time = self.settle_time+dt if settled and self.previous_settled else 0.
        reason = None
        if obs.ground_contact:
            reason = 'dropped'
        elif self.max_damage >= task.damage_limit:
            reason = 'damage_limit'
        elif max(obs.jaw_forces_N) > task.jaw_force_limit_N:
            reason = 'jaw_overload'
        elif obs.forbidden_collision:
            reason = 'forbidden_collision'
        elif new_losses:
            reason = 'collateral_loss'
        elif obs.time_s-self.initial.time_s > task.time_limit_s:
            reason = 'timeout'
        elif self.settle_time >= task.settle_dwell_s:
            reason = 'success'
        if reason:
            self.done, self.outcome = True, reason
            terms['terminal'] = task.success_reward if reason == 'success' else -task.failure_penalty
            events.append(reason)
        phi = 0. if self.done else self.potential(obs)
        terms['reach_potential'] = task.guidance_weight*(task.gamma_per_second**dt*phi-self.previous_potential)
        self.previous_potential = phi
        self.previous = obs
        self.previous_stable, self.previous_settled = stable, settled
        reward = float(sum(terms.values()))
        self.total_reward += reward
        return dict(reward=reward, terminated=self.done, success=self.outcome == 'success',
                    outcome=self.outcome, terms=terms, events=events)


def rotation(q):
    x,y,z,w = np.asarray(q, float)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class SimHarvestObserver:
    """Read-only native-contact adapter for one target and two actual jaw bodies.

    The caller identifies the gripper's TCP and pad bodies from its asset. No
    synthetic grasp flag, contact, force, detachment or containment is supplied.
    Actuator work is passed from the control loop's substep energy integral.
    """
    def __init__(self, sim, fruit_index, tcp_body, tcp_offset, pad_bodies):
        if sim.kiwi_damage is None or not sim.tree.robot_data or not sim.tree.robot_data.get('basket'):
            raise ValueError('Observer requires kiwi contacts and a robot basket')
        if len(set(pad_bodies)) != 2 or not 0 <= fruit_index < len(sim.tree.apple_bodies):
            raise ValueError('Need two distinct jaw bodies and a valid target')
        self.sim, self.index = sim, fruit_index
        self.body = sim.tree.apple_bodies[fruit_index]
        self.tcp_body, self.offset = tcp_body, np.asarray(tcp_offset, float)
        self.pads = tuple(pad_bodies)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('TCP offset must be a finite 3-vector')
        for body in (tcp_body, *pad_bodies):
            if not 0 <= body < sim.model.body_count or body == self.body:
                raise ValueError('Invalid gripper body')
        self.chassis = sim.tree.robot_data['chassis']
        self.basket = sim.tree.robot_data['basket']
        self.shape_body = sim.model.shape_body.numpy()
        shape = np.flatnonzero(self.shape_body == self.body)[0]
        self.radii = sim.model.shape_scale.numpy()[shape]
        from .basket import SpillTracker
        self.spills = SpillTracker(self.basket)
        self.other_initial = sim.apples.detached.copy()

    def observe(self, actuator_work_J):
        from .basket import CENTER, SIZE, WALL
        sim = self.sim
        q, v = sim.body_q_np(), sim.state_0.body_qd.numpy()
        r = rotation(q[self.chassis,3:])
        local = r.T@(q[self.body,:3]-q[self.chassis,:3])
        relative_rotation = r.T@rotation(q[self.body,3:])
        extent = np.sqrt((relative_rotation**2)@(self.radii**2))
        inside = bool(np.all(np.abs(local[:2]-CENTER[:2])+extent[:2] < SIZE[:2]/2-WALL)
                      and local[2]-extent[2] >= CENTER[2]+WALL/2-.002
                      and local[2]+extent[2] < CENTER[2]+SIZE[2])
        contacts = sim.contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        s0, s1 = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
        normals = contacts.rigid_contact_normal.numpy()[:count]
        force = contacts.force.numpy()[:count,:3]
        jaw = np.zeros(2)
        ground, basket_contact, forbidden = False, False, False
        for a,b,f,n in zip(s0,s1,force,normals):
            load = abs(float(np.dot(f,n)))
            if load < .01:
                continue
            ba, bb = self.shape_body[a], self.shape_body[b]
            if self.body not in (ba,bb):
                continue
            other, other_shape = (bb,b) if ba == self.body else (ba,a)
            if other in self.pads:
                jaw[self.pads.index(other)] += load
            elif other < 0:
                ground = True
            elif self.basket['first_shape'] <= other_shape < self.basket['shape_end']:
                basket_contact = True
            elif other >= sim.tree.robot_data['body_start'] and other not in self.basket['fruit_bodies']:
                forbidden |= load > 2.
            elif other in self.basket['fruit_bodies']:
                basket_contact = True
        tcp_r = rotation(q[self.tcp_body,3:])@self.offset
        tcp = q[self.tcp_body,:3]+tcp_r
        com = sim.model.body_com.numpy()
        tcp_v = v[self.tcp_body,:3]+np.cross(v[self.tcp_body,3:], tcp_r-rotation(q[self.tcp_body,3:])@com[self.tcp_body])
        chassis_com = q[self.chassis,:3]+r@com[self.chassis]
        basket_v = v[self.chassis,:3]+np.cross(v[self.chassis,3:], q[self.body,:3]-chassis_com)
        self.spills.update(q,self.chassis)
        others = sim.apples.detached & ~self.other_initial
        others[self.index] = False
        return HarvestObservation(sim.sim_time, self.body, not bool(sim.apples.detached[self.index]),
            float(np.linalg.norm(tcp-q[self.body,:3])), tuple(jaw),
            float(np.linalg.norm(v[self.body,:3]-tcp_v)), inside, basket_contact,
            float(np.linalg.norm(v[self.body,:3]-basket_v)), ground,
            float(sim.kiwi_damage.damage.numpy()[self.body]), float(sim.apples.fsa.numpy()[self.index]),
            float(sim.apples._tension.numpy()[self.index]), float(sim.apples.threshold.numpy()[self.index]),
            actuator_work_J, int(others.sum()+self.spills.spilled.sum()), forbidden)
