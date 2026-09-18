"""Single-instance Gymnasium integration environment; rigid fruit, not calibrated RL.

One native-contact physics step per oracle sample avoids losing brief failures.
Reset rebuilds solver state for correctness; this is intentionally not a fast
batched training implementation. The deformable contact bench remains separate.
"""
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from . import builder
from .config import TreeConfig
from .lsystem import generate
from .pergola import place_fruit
from .sim import Sim
from .spot import ARM, SpotController
from .harvest_task import TaskDefinition, HarvestOracle, SimHarvestObserver, rotation
from .basket import CENTER, SIZE


class SpotHarvestEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, relic, *, task=None, render_mode=None, physics_hz=1000):
        self.relic = Path(relic).resolve()
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Use render_mode=None or rgb_array')
        if physics_hz not in (1000, 2000):
            raise ValueError('Supported physics rates: 1000 or 2000 Hz')
        self.render_mode, self.physics_hz = render_mode, physics_hz
        self.task = task or TaskDefinition()
        self.action_space = spaces.Box(-1., 1., (7,), dtype=np.float32)
        sizes = dict(joint_position=7, joint_velocity=7, joint_target=7,
                     fruit_relative=3, fruit_quaternion=4, fruit_velocity=3, fruit_radii=3,
                     tcp_quaternion=4, basket_relative=3, jaw_forces=2,
                     previous_action=7, time_remaining=1)
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf,np.inf,(n,),np.float32)
                                             for k,n in sizes.items()})
        self.sim = self.viewer = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Reset options are not supported')
        self.close()
        cfg = TreeConfig.compliant('pergola')
        cfg.seed = int(self.np_random.integers(0, 2**30))
        cfg.lsystem.pergola_rows = cfg.lsystem.pergola_columns = 2
        cfg.fruit.enabled, cfg.fruit.max_count = True, 1
        cfg.fruit.colors = ((.39,.27,.12),(.48,.34,.17))
        cfg.robot.enabled, cfg.robot.kind = True, 'spot'
        cfg.robot.fixed_base, cfg.robot.basket = True, True
        cfg.robot.payload_mass, cfg.robot.yaw = 0., 0.
        cfg.robot.relic_path = str(self.relic)
        skeleton = generate(cfg.lsystem, seed=cfg.seed)
        fruit = place_fruit(skeleton, cfg.fruit, seed=cfg.seed)
        # Reachable target region; geometry and relative lateral position vary.
        cfg.robot.position = tuple(fruit[0].attach[:2]-[.55, self.np_random.uniform(-.04,.04)])
        self.sim = Sim(builder.build(cfg, skeleton, apple_placements=fruit),
                       fps=self.physics_hz, substeps=1, collisions=True)
        sim = self.sim
        constants = sim.tree.robot_data['constants']
        constants['ARM_EFFORT_LIMIT'] = list(constants['ARM_EFFORT_LIMIT'])
        constants['ARM_EFFORT_LIMIT'][-1] = .3  # Bench setting, not a hardware-safe limit.
        self.controller = SpotController(sim, locomotion=False)
        self.arm_q = self.controller.qids[12:]
        self.arm_dof = self.controller.dofs[12:]
        urdf = ET.parse(self.relic/'source/relic/relic/assets/spot/spot_with_arm.urdf').getroot()
        limits = [urdf.find(f"joint[@name='{n}']/limit") for n in ARM]
        self.lower = np.array([float(x.get('lower')) for x in limits])
        self.upper = np.array([float(x.get('upper')) for x in limits])
        self.controller.targets[-1] = -1.  # Open jaw initially.
        # Initial joint pose must agree with the controller target.
        import newton
        for state in (sim.state_0, sim.state_1):
            q = state.joint_q.numpy(); q[self.arm_q] = self.controller.targets[12:]
            state.joint_q.assign(q)
            newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
        self.controller.target.assign(self.controller.targets)
        labels = {name.rsplit('/',1)[-1]:i for i,name in enumerate(sim.model.body_label)}
        self.observer = SimHarvestObserver(sim, 0, labels['arm_link_wr1'], (.195,0,.005),
                                          (labels['arm_link_fngr'], labels['arm_link_jaw']),
                                          arm_bodies=[i for name,i in labels.items() if name.startswith('arm_link_')])
        self.oracle = HarvestOracle(self.task)
        self.work = 0.; self.previous_action = np.zeros(7, np.float32)
        self.physical = self.observer.observe(self.work)
        self.oracle.reset(self.physical)
        self.done = False
        self.base_pose = sim.body_q_np()[sim.tree.robot_data['chassis']].copy()
        # No CUDA capture: a one-substep graph would alias swapped state buffers.
        return self._observation(), self._info()

    def _observation(self):
        sim = self.sim; poses = sim.body_q_np()
        tcp_pose = poses[self.observer.tcp_body]
        tcp = tcp_pose[:3]+rotation(tcp_pose[3:])@self.observer.offset
        fruit = poses[self.observer.body]
        base = poses[self.observer.chassis]
        basket = base[:3]+rotation(base[3:])@(CENTER+[0,0,SIZE[2]/2])
        values = dict(joint_position=sim.state_0.joint_q.numpy()[self.arm_q],
                      joint_velocity=sim.state_0.joint_qd.numpy()[self.arm_dof],
                      joint_target=self.controller.targets[12:], fruit_relative=fruit[:3]-tcp,
                      fruit_radii=self.observer.radii, fruit_quaternion=fruit[3:], fruit_velocity=sim.state_0.body_qd.numpy()[self.observer.body,:3],
                      tcp_quaternion=tcp_pose[3:], basket_relative=basket-tcp,
                      jaw_forces=self.physical.jaw_forces_N, previous_action=self.previous_action,
                      time_remaining=[max(0.,self.task.time_limit_s-sim.sim_time)])
        obs = {k:np.asarray(v,np.float32).copy() for k,v in values.items()}
        if not self.observation_space.contains(obs) or not all(np.isfinite(v).all() for v in obs.values()):
            raise RuntimeError('Nonfinite or malformed policy observation')
        return obs

    def _info(self):
        return dict(physical=asdict(self.physical), fruit_model='rigid, uncalibrated contact/damage proxy',
                    simulation_time_s=self.sim.sim_time, outcome=self.oracle.outcome,
                    success=self.oracle.outcome=='success')

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a new or finished episode')
        velocity = self.task.action_velocity(action)  # Validate before any mutation.
        self.previous_action = np.asarray(action,np.float32).copy()
        reward = 0.; terms = {}; events = []; start = self.sim.sim_time
        for _ in range(self.physics_hz//50):
            self.controller.targets[12:] = np.clip(self.controller.targets[12:]+velocity/self.physics_hz,
                                                    self.lower,self.upper)
            self.controller.target.assign(self.controller.targets)
            before = self.sim.state_0.joint_qd.numpy()[self.controller.dofs]
            self.sim.step()
            after = self.sim.state_0.joint_qd.numpy()[self.controller.dofs]
            effort = self.sim.control.joint_f.numpy()[self.controller.dofs]
            self.work += float(np.sum(np.abs(effort*(before+after)/2)))/self.physics_hz
            self.physical = self.observer.observe(self.work)
            result = self.oracle.update(self.physical)
            discount = self.task.gamma_per_second**(self.sim.sim_time-start-1/self.physics_hz)
            reward += discount*result['reward']
            for key,value in result['terms'].items(): terms[key]=terms.get(key,0.)+discount*value
            events += result['events']
            self.done = result['terminated']
            if not np.isfinite(self.sim.state_0.joint_q.numpy()).all():
                raise RuntimeError('Nonfinite simulator state')
            if self.done: break
        info = self._info(); info.update(reward_terms=terms, events=events,
                                        elapsed_s=self.sim.sim_time-start)
        # Timeout is an explicit finite-horizon task failure, not a rollout cutoff.
        return self._observation(), float(reward), self.done, False, info

    def render(self):
        if self.render_mode != 'rgb_array' or self.sim is None:
            raise RuntimeError('Reset an rgb_array environment before rendering')
        if self.viewer is None:
            import newton.viewer as V
            import warp as wp
            self.viewer = V.ViewerGL(headless=True); self.viewer.set_model(self.sim.model)
            base = self.base_pose[:3]
            self.viewer.set_camera(pos=wp.vec3(*(base+[3.2,3.2,2.])), yaw=-135., pitch=-24.)
        self.viewer.begin_frame(self.sim.sim_time)
        self.viewer.log_state(self.sim.state_0)
        self.sim.apples.render(self.viewer, self.sim.state_0)
        self.viewer.end_frame()
        return self.viewer.get_frame().numpy().copy()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        if self.sim is not None:
            self.sim.robot_controller = None
        self.controller = self.observer = self.oracle = None
        self.sim = None
        self.done = True


# GPU reductions can differ in the last few bits even for identical seeds.
# The integration check separately verifies reset identity and bounded step drift.
gym.register('Thekenyos/SpotHarvest-v0', entry_point='treesim.harvest_env:SpotHarvestEnv',
             nondeterministic=True)
