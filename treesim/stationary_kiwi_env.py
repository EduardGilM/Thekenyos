import gymnasium as gym
import mujoco
import numpy as np

from .basket_kiwi_env import BasketTask
from .visual_kiwi_env import DEPTH_FOVY_DEG, S0V_RANGE_M, VisualKiwiEnv, VisualTaskPhysics


SCOPE = 'Arm-only near-to-far camera RL; S0v in-FOV pregrasp then off-axis grab; frozen standing gait; assisted grasp, not physical grasp or collection'
# Stage 0 is S0v (20-40 cm in camera FOV). Stages 1-3 keep fruit outside the 12 cm weld.
GAPS_M = (.24, .32, .44)
OFF_AXIS_M = ((.18, .20), (.19, .22), (.20, .24))
STAGES = 4


def fruit_offset_m(level, rng):
    if level not in range(len(GAPS_M)):
        raise ValueError('Invalid off-axis stationary stage')
    gap = GAPS_M[level]+rng.uniform(-.005, .005)
    lo, hi = OFF_AXIS_M[level]
    radius = rng.uniform(lo, hi)
    angle = rng.uniform(0., 2*np.pi)
    return np.array([gap, radius*np.cos(angle), radius*np.sin(angle)]), float(radius)


def s0v_fruit_position(physics, rng):
    cam = physics.ee_cam
    origin = physics.data.cam_xpos[cam]
    rotation = physics.data.cam_xmat[cam].reshape(3, 3)
    forward, right, up = -rotation[:, 2], rotation[:, 0], rotation[:, 1]
    dist = rng.uniform(*S0V_RANGE_M)
    radius = dist*np.tan(np.deg2rad(DEPTH_FOVY_DEG/2))*.8*np.sqrt(rng.uniform(0., 1.))
    angle = rng.uniform(0., 2*np.pi)
    position = origin+forward*dist+right*(radius*np.cos(angle))+up*(radius*np.sin(angle))
    return position, float(dist), float(radius)


def arm_action(action):
    action = np.asarray(action)
    if (action.shape != (4,) or not np.isfinite(action).all() or not np.equal(action, np.floor(action)).all()
            or np.any(action < 0) or np.any(action >= [3, 3, 3, 2])):
        raise ValueError('Expected three categorical XYZ commands and a binary jaw command')
    result = np.zeros(10)
    result[3:6] = (action[:3]-1.)*.45
    result[-1] = 1. if action[-1] else -1.
    return result


def scripted_reach(physics):
    """Privileged test controller: body-frame steps toward the fruit. Not a learned policy."""
    if physics.captured:
        return [1, 1, 1, 1]
    rotation = physics.data.xmat[physics.chassis].reshape(3, 3)
    delta = rotation.T@(physics.data.xpos[physics.target_body]-physics.data.site_xpos[physics.tcp])
    axes = [2 if d > .02 else 0 if d < -.02 else 1 for d in delta]
    return axes+[int(np.linalg.norm(delta) <= physics.task.capture_radius_m)]


class StationaryTaskPhysics(VisualTaskPhysics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vision.lesson != 'grab' or self.task.start_phase != 'pick' or self.task.picks != 1:
            raise ValueError('Stationary lessons support a single initially unpicked fruit')
        self.stationary_active = False

    def set_stage(self, stage):
        if stage not in range(STAGES):
            raise ValueError('Invalid stationary stage')
        self.stage = int(stage)

    def reset(self, *, seed=None, options=None):
        self.stationary_active = False
        level = self.stage
        self.stage = 0
        try:
            super().reset(seed=seed, options=options)
        finally:
            self.stage = level
        self.episode_stage = level
        if level == 0:
            position, dist, radius = s0v_fruit_position(self, self.np_random)
            self.off_axis_m = radius
            self.s0v_range_m = dist
        else:
            rotation = self.data.xmat[self.chassis].reshape(3, 3)
            offset, radius = fruit_offset_m(level-1, self.np_random)
            self.off_axis_m = radius
            self.s0v_range_m = 0.
            position = self.data.site_xpos[self.tcp]+rotation@offset
        index = self.target_index
        joint = self.model.joint(f'assisted_kiwi_{index}').id
        q, v = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        self.data.qpos[q:q+3] = position
        self.data.qvel[v:v+6] = 0.
        self.model.site_pos[self.model.site(f'anchor_{index}').id] = position
        self.fruit_positions[index] = position
        self.target = position.copy()
        mujoco.mj_forward(self.model, self.data)
        self.start_distance = self.previous_distance = self.best_distance = self._distance()
        self.initial_tcp_base = self._tcp_base()
        self.stationary_active = True
        return self._observation(), self._info()

    def step(self, action):
        action = np.asarray(action)
        if action.shape == (10,) and np.any(action[[0, 1, 2, 6, 7, 8]] != 0.):
            raise ValueError('Walking and wrist-rotation commands are disabled in stationary lessons')
        return super().step(action)

    def _tcp_base(self):
        return self.data.xmat[self.chassis].reshape(3, 3).T@(self.data.site_xpos[self.tcp]-self.data.xpos[self.chassis])

    def _info(self):
        info = super()._info()
        info.update(scope=SCOPE, stationary_lesson=True, walking_commands_enabled=False)
        if self.stationary_active:
            motion = float(np.linalg.norm(self._tcp_base()-self.initial_tcp_base))
            if self.episode_stage == 0:
                arm_only = bool(info.get('pregrasp') and info['base_travel_m'] < .10 and motion > .015)
                gap = getattr(self, 's0v_range_m', float(np.mean(S0V_RANGE_M)))
            else:
                arm_only = bool(info['success'] and info['base_travel_m'] < .10 and motion > .015)
                gap = GAPS_M[self.episode_stage-1]
            info.update(arm_motion_m=motion, nominal_tcp_gap_m=gap,
                        off_axis_m=self.off_axis_m, stationary_success=arm_only, s0v=self.episode_stage == 0)
            info['success'] = arm_only
            if arm_only:
                info['outcome'] = 'stationary_pregrasp' if self.episode_stage == 0 else 'stationary_assisted_grasp'
        return info


class StationaryKiwiEnv(VisualKiwiEnv):
    physics_class = StationaryTaskPhysics

    def __init__(self, relic, *, task=None, **kwargs):
        super().__init__(relic, task=task or BasketTask(picks=1, time_limit_s=6.), **kwargs)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 2])

    def step(self, action):
        return super().step(arm_action(action))


gym.register('Thekenyos/StationaryKiwi-v0', entry_point='treesim.stationary_kiwi_env:StationaryKiwiEnv')
