from dataclasses import dataclass
from pathlib import Path
import runpy
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .spot import ARM, LEGS, OBS_JOINTS


PREFIX = 'spot_with_arm_'
SCOPE = 'Assisted reach/grab RL; artificial stem release and grip weld; no physical grasp or fruit-safety claim'
STAGES = ((0., .15), (.25, .65), (.75, 1.5))
TCP_OFFSET = np.array([.195, 0., .005])


@dataclass(frozen=True)
class AssistedTask:
    time_limit_s: float = 15.
    capture_radius_m: float = .12
    hold_time_s: float = .3
    physics_hz: int = 1000
    stage: int = 0

    def __post_init__(self):
        if self.stage not in range(len(STAGES)) or self.physics_hz not in (1000, 2000):
            raise ValueError('Invalid curriculum stage or physics rate')
        for name, lo, hi in (('time_limit_s', .1, 120.), ('capture_radius_m', .02, .15),
                             ('hold_time_s', .1, 2.)):
            value = getattr(self, name)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} must be finite in [{lo}, {hi}]')


def can_capture(distance, close_requested, jaw_angle, speed):
    return bool(close_requested and jaw_angle > -.65 and np.isfinite([distance, speed]).all()
                and distance >= 0 and speed >= 0 and speed < .6)


def build_scene(relic, physics_hz):
    import newton
    from .config import TreeConfig
    from .spot import build_robot

    asset = Path(relic).resolve()/'source/relic/relic/assets/spot'
    constants = runpy.run_path(str(asset/'constants.py'))
    cfg = TreeConfig.compliant('pergola')
    cfg.robot.relic_path = str(relic)
    cfg.robot.position, cfg.robot.yaw, cfg.robot.base_z = (0., 0.), 0., .6
    cfg.robot.fixed_base, cfg.robot.basket = False, False
    builder = newton.ModelBuilder()
    build_robot(builder, cfg.robot)
    builder.add_ground_plane()
    model = builder.finalize(device='cpu')
    with tempfile.TemporaryDirectory(prefix='assisted-kiwi-') as directory:
        path = Path(directory)/'spot.xml'
        newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                                   save_to_mjcf=str(path))
        root = ET.parse(path).getroot()
    option = root.find('option')
    option.set('timestep', str(1/physics_hz))
    option.set('solver', 'Newton')
    option.set('iterations', '50')
    option.set('integrator', 'implicitfast')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('midphase', 'disable')
    asset_element = root.find('asset')
    if asset_element is None:
        asset_element = ET.SubElement(root, 'asset')
    ET.SubElement(asset_element, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='.35 .5 .7', rgb2='.8 .85 .9', width='256', height='256')
    ET.SubElement(asset_element, 'texture', name='soil', type='2d', builtin='checker',
                  rgb1='.28 .34 .18', rgb2='.35 .4 .23', width='128', height='128')
    ET.SubElement(asset_element, 'material', name='soil', texture='soil', texrepeat='12 12')
    urdf = ET.parse(asset/'spot_with_arm.urdf').getroot()
    for link in urdf.findall('link'):
        name = link.get('name')
        body = root.find(f'.//body[@name="{PREFIX}{name}"]')
        if body is None:
            continue
        for geom in body.findall('geom'):
            geom.set('group', '3' if link.findall('visual') else '0')
            geom.set('rgba', '.2 .22 .25 1')
        for i, visual in enumerate(link.findall('visual')):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            mesh_name = f'visual_{name}_{i}'
            ET.SubElement(asset_element, 'mesh', name=mesh_name,
                          file=str((asset/mesh.get('filename')).resolve()), scale=mesh.get('scale', '1 1 1'))
            origin = visual.find('origin')
            attrs = dict(type='mesh', mesh=mesh_name, contype='0', conaffinity='0', group='2', mass='0',
                         rgba='.96 .73 .04 1' if name == 'body' or 'uleg' in name or name in
                         ('arm_link_sh0', 'arm_link_sh1') else '.18 .2 .23 1')
            if origin is not None:
                attrs['pos'] = origin.get('xyz', '0 0 0')
                quat = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_quat()
                attrs['quat'] = ' '.join(map(str, quat[[3, 0, 1, 2]]))
            ET.SubElement(body, 'geom', **attrs)
    wrist = root.find(f'.//body[@name="{PREFIX}arm_link_wr1"]')
    ET.SubElement(wrist, 'site', name='tcp', pos=' '.join(map(str, TCP_OFFSET)), size='.008', rgba='1 .4 .1 1')
    world = root.find('worldbody')
    for geom in world.findall('geom'):
        if geom.get('type') == 'plane':
            geom.set('material', 'soil')
    ET.SubElement(world, 'light', pos='-2 -3 6', dir='.3 .4 -1', directional='true')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.6 .6 .6')
    for x in (-1.7, 1.7):
        for y in (-1.6, 1.6):
            ET.SubElement(world, 'geom', type='capsule', fromto=f'{x} {y} 0 {x} {y} 1.65',
                          size='.055', rgba='.32 .21 .1 1')
    for y in (-1.6, -.65, .65, 1.6):
        ET.SubElement(world, 'geom', type='capsule', fromto=f'-1.7 {y} 1.65 1.7 {y} 1.65',
                      size='.022', rgba='.35 .24 .12 1')
    for x in (-1.7, 0., 1.7):
        ET.SubElement(world, 'geom', type='capsule', fromto=f'{x} -1.6 1.65 {x} 1.6 1.65',
                      size='.016', rgba='.36 .27 .13 1')
    rng = np.random.default_rng(7)
    for _ in range(130):
        p = [rng.uniform(-1.7, 1.7), rng.uniform(-1.6, 1.6), rng.uniform(1.68, 1.78)]
        ET.SubElement(world, 'geom', type='ellipsoid', pos=' '.join(map(str, p)), size='.13 .08 .008',
                      rgba=f'.16 {rng.uniform(.32, .5)} .08 1', contype='0', conaffinity='0', group='2')
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    positions = np.array([[x, y, 1.47] for y in (-.65, .65) for x in (-1., 0., 1.)])
    for i, position in enumerate(positions):
        name = f'assisted_kiwi_{i}'
        body = ET.SubElement(world, 'body', name=name, pos=' '.join(map(str, position)))
        ET.SubElement(body, 'freejoint', name=name)
        ET.SubElement(body, 'geom', name=name, type='ellipsoid', size='.03 .03 .04', mass='.11',
                      rgba='.44 .29 .12 1', friction='.44 .005 .0001', solref='.01 1')
        ET.SubElement(world, 'site', name=f'anchor_{i}', pos=' '.join(map(str, position)), size='.001', rgba='0 0 0 0')
        ET.SubElement(body, 'site', name=f'fruit_frame_{i}', size='.001', rgba='0 0 0 0')
        ET.SubElement(wrist, 'site', name=f'hand_frame_{i}', size='.001', rgba='0 0 0 0')
        ET.SubElement(equality, 'weld', name=f'stem_{i}', site1=f'anchor_{i}', site2=f'fruit_frame_{i}', solref='.01 1')
        ET.SubElement(equality, 'weld', name=f'grip_{i}', site1=f'hand_frame_{i}', site2=f'fruit_frame_{i}',
                      active='false', solref='.01 1')
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), constants, positions


class AssistedKiwiEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, relic, *, task=None, render_mode=None):
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported render mode')
        self.relic, self.task, self.render_mode = str(Path(relic).resolve()), task or AssistedTask(), render_mode
        self.stage = self.task.stage
        self.action_space = spaces.Box(-1., 1., (7,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (78,), np.float32)
        self.model = self.data = self.renderer = None
        self.done = True

    def _build(self):
        import onnxruntime as ort
        self.model, self.constants, self.fruit_positions = build_scene(self.relic, self.task.physics_hz)
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        m = self.model
        joints = [m.joint(PREFIX+n).id for n in LEGS+ARM]
        self.qids, self.dofs = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.joint_limits = m.jnt_range[joints]
        self.home = np.array([self.constants['SPOT_DEFAULT_JOINT_POS'][n] for n in LEGS+ARM])
        order = [list(LEGS+ARM).index(n) for n in OBS_JOINTS]
        self.obs_order = np.array(order)
        self.chassis, self.wrist, self.tcp = m.body(PREFIX+'body').id, m.body(PREFIX+'arm_link_wr1').id, m.site('tcp').id
        self.base_q = m.jnt_qposadr[m.joint(PREFIX+'floating_base').id]
        self.fruit_bodies = np.array([m.body(f'assisted_kiwi_{i}').id for i in range(6)])
        self.stems = np.array([m.equality(f'stem_{i}').id for i in range(6)])
        self.grips = np.array([m.equality(f'grip_{i}').id for i in range(6)])
        self.original_eq = m.eq_data.copy()
        self.original_site_pos, self.original_site_quat = m.site_pos.copy(), m.site_quat.copy()
        self.hand_sites = [m.site(f'hand_frame_{i}').id for i in range(6)]
        m.site_sameframe[self.hand_sites] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        self.kp = np.r_[np.full(12, 60.), self.constants['ARM_STIFFNESS']]
        self.kd = np.r_[np.full(12, 1.5), self.constants['ARM_DAMPING']]
        self.effort = np.r_[np.full(8, 45.), np.full(4, 113.24), self.constants['ARM_EFFORT_LIMIT']]
        self.knee_table = np.asarray(self.constants['JOINT_PARAMETER_LOOKUP_TABLE'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.gait = ort.InferenceSession(str(Path(self.relic)/'source/relic/relic/assets/spot/pretrained/policy.onnx'),
                                        sess_options=options, providers=['CPUExecutionProvider'])
        self.gait_input = self.gait.get_inputs()[0].name
        self.jacobian = np.zeros((3, m.nv))

    def _base_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.chassis, velocity, 1)
        return np.r_[velocity[3:], velocity[:3]]

    def _gait_tick(self, command):
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        velocity = self._base_velocity()
        obs = np.r_[velocity, rotation.T@[0., 0., -1.], command, self.targets[12:], np.zeros(12),
                    [0., 0., .55], (self.data.qpos[self.qids]-self.home)[self.obs_order],
                    self.data.qvel[self.dofs][self.obs_order], self.gait_action].astype(np.float32)
        if obs.shape != (84,) or not np.isfinite(obs).all():
            raise RuntimeError('Invalid RELIC observation')
        self.gait_action = self.gait.run(None, {self.gait_input: obs[None]})[0][0]
        if not np.isfinite(self.gait_action).all():
            raise RuntimeError('Nonfinite RELIC action')
        self.targets[:12] = self.home[:12]+.2*self.gait_action

    def _physics_step(self):
        d = self.data
        q, v = d.qpos[self.qids], d.qvel[self.dofs]
        torque = np.clip(self.kp*(self.targets-q)-self.kd*v, -self.effort, self.effort)
        knee_limits = np.interp(q[8:12], self.knee_table[:, 0], self.knee_table[:, 2])
        torque[8:12] = np.clip(torque[8:12], -knee_limits, knee_limits)
        torque[8:12] = np.clip(torque[8:12], -96.9972*np.clip(1+v[8:12]/15., 0, 1),
                              96.9972*np.clip(1-v[8:12]/14., 0, 1))
        d.qfrc_applied[self.dofs] = torque
        mujoco.mj_step(self.model, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):
            self.done = True
            raise RuntimeError('Invalid assisted-task physics state; transition rejected')
        mujoco.mj_kinematics(self.model, d)
        tilt = np.arccos(np.clip(d.xmat[self.chassis].reshape(3, 3)[2, 2], -1, 1))
        if d.xpos[self.chassis, 2] < .25 or tilt > 1.:
            self.failure = 'fall'
        if np.max(np.abs(d.xpos[self.chassis, :2])) > 4.:
            self.failure = 'out_of_bounds'

    def _initial_arm(self, position):
        self.scratch.qpos[:] = self.data.qpos
        home = self.home[12:18]
        def residual(q):
            self.scratch.qpos[self.qids[12:18]] = q
            mujoco.mj_kinematics(self.model, self.scratch)
            return np.r_[self.scratch.site_xpos[self.tcp]-position, .002*(q-home)]
        lower, upper = self.joint_limits[12:18].T
        result = least_squares(residual, np.clip(home, lower+1e-5, upper-1e-5), bounds=(lower, upper), max_nfev=120)
        if np.linalg.norm(residual(result.x)[:3]) > .03:
            raise RuntimeError('Initial arm curriculum pose is unreachable')
        self.targets[12:18] = result.x
        self.data.qpos[self.qids[12:18]] = result.x
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Use task/stage configuration rather than reset options')
        if self.model is None:
            self._build()
        mujoco.mj_resetData(self.model, self.data)
        self.model.eq_data[:] = self.original_eq
        self.model.site_pos[:] = self.original_site_pos
        self.model.site_quat[:] = self.original_site_quat
        self.target_index = int(self.np_random.integers(6))
        self.target_body = int(self.fruit_bodies[self.target_index])
        self.target = self.fruit_positions[self.target_index].copy()
        self.episode_stage = self.stage
        self.start_distance = float(self.np_random.uniform(*STAGES[self.episode_stage]))
        self.targets = self.home.copy()
        self.targets[-1] = -1.
        self.data.qpos[self.qids] = self.targets
        self.data.qpos[self.base_q:self.base_q+3] = [self.target[0]-.5-self.start_distance,
                                                     self.target[1]+self.np_random.uniform(-.04, .04), .6]
        self.data.qpos[self.base_q+3:self.base_q+7] = [1., 0., 0., 0.]
        self.gait_action = np.zeros(12)
        self.failure = None
        self.done = False
        self.captured = False
        self.hold_ticks = self.steps = 0
        self.previous_action = np.zeros(7)
        mujoco.mj_forward(self.model, self.data)
        self._initial_arm(self.target+[-.17-self.start_distance, 0., -.05])
        for _ in range(25):
            self._gait_tick(np.zeros(3))
            for _ in range(self.task.physics_hz//50):
                self._physics_step()
        if self.failure:
            raise RuntimeError(f'Invalid initial standing pose: {self.failure}')
        self.start_time = float(self.data.time)
        self.initial_base = self.data.xpos[self.chassis].copy()
        self.previous_distance = self._distance()
        self.best_distance = self.previous_distance
        return self._observation(), self._info()

    def set_stage(self, stage):
        if stage not in range(len(STAGES)):
            raise ValueError('Invalid stage')
        self.stage = int(stage)

    def _distance(self):
        return float(np.linalg.norm(self.data.xpos[self.target_body]-self.data.site_xpos[self.tcp]))

    def _observation(self):
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        tcp = self.data.site_xpos[self.tcp]
        obs = np.r_[rotation.T@(self.data.xpos[self.target_body]-tcp),
                    rotation.T@(self.data.xpos[self.target_body]-self.data.xpos[self.chassis]),
                    rotation.T@(tcp-self.data.xpos[self.chassis]), self._base_velocity(),
                    rotation.T@[0., 0., -1.], self.data.qpos[self.qids]-self.home,
                    self.data.qvel[self.dofs]*.1, self.previous_action, float(self.captured),
                    self.hold_ticks*.02/self.task.hold_time_s,
                    max(0., 1-(self.data.time-self.start_time)/self.task.time_limit_s),
                    self.gait_action].astype(np.float32)
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError(f'Invalid observation: {obs.shape}')
        return obs

    def _info(self):
        return dict(scope=SCOPE, success=self.captured and self.hold_ticks*.02 >= self.task.hold_time_s and not self.failure,
                    outcome=self.failure or ('assisted_grasp' if self.captured and self.hold_ticks*.02 >= self.task.hold_time_s
                                             else 'timeout' if self.steps >= np.ceil(self.task.time_limit_s*10) else 'running'),
                    target_index=self.target_index, stage=self.episode_stage, assisted=self.captured,
                    distance_m=self._distance(), best_distance_m=self.best_distance,
                    base_travel_m=float(np.linalg.norm(self.data.xpos[self.chassis, :2]-self.initial_base[:2])),
                    elapsed_s=float(self.data.time-self.start_time), start_distance_m=self.start_distance,
                    physical_grasp_validated=False)

    def _capture(self):
        rotation = self.data.xmat[self.wrist].reshape(3, 3)
        fruit_rotation = self.data.xmat[self.target_body].reshape(3, 3)
        eq = self.grips[self.target_index]
        site = self.hand_sites[self.target_index]
        self.model.site_pos[site] = rotation.T@(self.data.xpos[self.target_body]-self.data.xpos[self.wrist])
        quat = Rotation.from_matrix(rotation.T@fruit_rotation).as_quat()
        self.model.site_quat[site] = quat[[3, 0, 1, 2]]
        self.data.eq_active[self.stems[self.target_index]] = False
        self.data.eq_active[eq] = True
        self.captured = True

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a finished episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Action must be seven finite values in [-1, 1]')
        captured_before = self.captured
        for _ in range(5):
            rotation = self.data.xmat[self.chassis].reshape(3, 3)
            mujoco.mj_jacSite(self.model, self.data, self.jacobian, None, self.tcp)
            jac = self.jacobian[:, self.dofs[12:18]]
            qd = jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(3), rotation@(action[3:6]*.35))
            self.targets[12:18] = np.clip(self.targets[12:18]+.02*np.clip(qd, -.8, .8),
                                        self.joint_limits[12:18, 0], self.joint_limits[12:18, 1])
            self.targets[-1] = 0. if action[6] > .2 else -1.
            self._gait_tick(action[:3]*[.4, .25, .7])
            for _ in range(self.task.physics_hz//50):
                self._physics_step()
                if self.failure:
                    break
            distance = self._distance()
            if not self.captured and distance <= self.task.capture_radius_m and can_capture(
                    distance, action[6] > .2, self.data.qpos[self.qids[-1]], np.linalg.norm(self._base_velocity()[:3])):
                self._capture()
            if self.captured:
                if distance > self.task.capture_radius_m+.05:
                    self.failure = self.failure or 'retention_failed'
                if action[6] <= .2:
                    self.data.eq_active[self.grips[self.target_index]] = False
                    self.failure = self.failure or 'released'
                self.hold_ticks += 1
            if self.failure:
                break
        self.steps += 1
        distance = self._distance()
        self.best_distance = min(self.best_distance, distance)
        terms = dict(progress=8*(self.previous_distance-distance), proximity=.08*np.exp(-4*distance),
                     time=-.015, smooth=-.005*float(np.sum((action-self.previous_action)**2)),
                     capture=3. if self.captured and not captured_before and not self.failure else 0.)
        self.previous_distance, self.previous_action = distance, action.copy()
        info = self._info()
        terminated = bool(self.failure or info['success'])
        truncated = bool(not terminated and self.steps >= np.ceil(self.task.time_limit_s*10))
        self.done = terminated or truncated
        terms.update(success=10. if info['success'] else 0., failure=-5. if self.failure else 0.)
        info['reward_terms'] = terms
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def render(self):
        if self.render_mode != 'rgb_array' or self.model is None:
            raise RuntimeError('Reset an rgb_array environment first')
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [self.target[0]-.3, self.target[1], .95]
        camera.distance, camera.azimuth, camera.elevation = 3.8, 120., -18.
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        scene = self.renderer.scene
        for i, body in enumerate(self.fruit_bodies):
            if not self.data.eq_active[self.stems[i]]:
                continue
            stem = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(stem, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                               np.eye(3).ravel(), np.array([.3, .4, .12, 1.], np.float32))
            mujoco.mjv_connector(stem, mujoco.mjtGeom.mjGEOM_CAPSULE, .003,
                                self.data.xpos[body]+[0., 0., .04], self.fruit_positions[i]+[0., 0., .18])
            scene.ngeom += 1
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, self.task.capture_radius_m),
                           self.data.xpos[self.target_body], np.eye(3).ravel(), np.array([.2, 1., .3, .18], np.float32))
        scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.done = True


gym.register('Thekenyos/AssistedKiwi-v0', entry_point='treesim.assisted_kiwi_env:AssistedKiwiEnv')
