from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import newton
import numpy as np
import warp as wp

from .builder import TreeModel, _add_terrain, _qconj, _qrot
from .config import TreeConfig
from .sim import Sim
from .skeleton import TreeSkeleton
from .spot import SpotController, build_robot, finalize_maps


COMMAND_LIMITS = np.array([.4, .25, .7])


@dataclass(frozen=True)
class NavigationTask:
    start_xy_m: tuple = (-2., 0.)
    goal_xy_m: tuple = (2., 0.)
    terrain_amplitude_m: float = .05
    terrain_wavelength_m: float = 1.8
    terrain_extent_m: float = 5.
    goal_radius_m: float = .20
    settle_time_s: float = .5
    time_limit_s: float = 30.
    physics_hz: int = 1000

    def __post_init__(self):
        for point in (self.start_xy_m, self.goal_xy_m):
            if np.shape(point) != (2,) or not np.isfinite(point).all():
                raise ValueError('A and B must be finite XY pairs in metres')
        for name in ('terrain_wavelength_m', 'terrain_extent_m', 'goal_radius_m',
                     'settle_time_s', 'time_limit_s'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.terrain_amplitude_m) or not 0 <= self.terrain_amplitude_m <= .3:
            raise ValueError('terrain_amplitude_m must be in [0, .3]')
        if self.physics_hz not in (1000, 2000):
            raise ValueError('physics_hz must be 1000 or 2000')
        if np.max(np.abs([self.start_xy_m, self.goal_xy_m])) + 1. >= self.terrain_extent_m:
            raise ValueError('A and B need more than 1 m clearance from the terrain edge')
        if np.linalg.norm(np.subtract(self.goal_xy_m, self.start_xy_m)) <= 2*self.goal_radius_m:
            raise ValueError('A and B must be separated by more than two goal radii')


def goal_action(observation):
    delta = np.asarray(observation['goal_body_xy_m'])
    distance = float(np.linalg.norm(delta))
    if distance < .10:
        return np.zeros(3, np.float32)
    heading = math.atan2(delta[1], delta[0])
    velocity = min(.4, .8*distance) * max(0., math.cos(heading))
    return np.clip([velocity/.4, 0., 1.8*heading/.7], -1, 1).astype(np.float32)


@wp.kernel
def _check_state(poses: wp.array(dtype=wp.transform), velocities: wp.array(dtype=wp.spatial_vector),
                 heights: wp.array2d(dtype=float), extent: float, chassis: int,
                 failures: wp.array(dtype=int)):
    i = wp.tid()
    pose, velocity = poses[i], velocities[i]
    for k in range(7):
        if not wp.isfinite(pose[k]):
            wp.atomic_max(failures, 2, 1)
    for k in range(6):
        if not wp.isfinite(velocity[k]):
            wp.atomic_max(failures, 2, 1)
    if i == chassis:
        p, q = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
        if wp.isfinite(p[0]) and wp.isfinite(p[1]):
            n = heights.shape[0] - 1
            u = wp.clamp((p[0]/(2.*extent)+.5)*float(n), 0., float(n)-.0001)
            v = wp.clamp((p[1]/(2.*extent)+.5)*float(n), 0., float(n)-.0001)
            x, y = int(u), int(v)
            fu, fv = u-float(x), v-float(y)
            height = ((1.-fu)*heights[y,x]+fu*heights[y,x+1])*(1.-fv)
            height += ((1.-fu)*heights[y+1,x]+fu*heights[y+1,x+1])*fv
            if p[2]-height < .25 or 1.-2.*(q[0]*q[0]+q[1]*q[1]) < .5403023:
                wp.atomic_max(failures, 0, 1)
            if wp.max(wp.abs(p[0]), wp.abs(p[1])) >= extent-1.:
                wp.atomic_max(failures, 1, 1)


class NavigationController(SpotController):
    def __init__(self, sim, heights, extent):
        super().__init__(sim)
        self.heights = wp.array(heights, dtype=float, device=self.device)
        self.extent = extent
        self.failures = wp.zeros(3, dtype=int, device=self.device)

    def check(self, state):
        wp.launch(_check_state, dim=self.sim.model.body_count,
                  inputs=[state.body_q, state.body_qd, self.heights, self.extent,
                          self.data['chassis'], self.failures], device=self.device)

    def apply(self, state, control):
        self.check(state)
        super().apply(state, control)


class SpotNavigationEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, relic, *, task=None, device='cuda:0', render_mode=None):
        self.task = task or NavigationTask()
        self.relic = Path(relic).expanduser().resolve()
        self.device = device
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('render_mode must be None or rgb_array')
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1., 1., (3,), np.float32)
        sizes = dict(goal_body_xy_m=2, base_velocity_body=6, gravity_body=3,
                     joint_position_rad=19, joint_velocity_rad_s=19,
                     previous_action=3, gait_previous_action=12, goal_hold_time_s=1,
                     time_remaining_s=1)
        self.observation_space = spaces.Dict({key: spaces.Box(-np.inf, np.inf, (size,), np.float32)
                                             for key, size in sizes.items()})
        self.sim = self.controller = self.renderer = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Configure A/B and terrain through NavigationTask')
        self.close()
        task = self.task
        self.terrain_seed = int(self.np_random.integers(0, 2**31))
        cfg = TreeConfig.compliant('pergola')
        cfg.device, cfg.seed = self.device, self.terrain_seed
        cfg.physics.terrain = True
        cfg.physics.terrain_amplitude = task.terrain_amplitude_m
        cfg.physics.terrain_wavelength = task.terrain_wavelength_m
        cfg.physics.terrain_extent = task.terrain_extent_m
        cfg.physics.terrain_seed = self.terrain_seed
        cfg.robot.enabled, cfg.robot.kind = True, 'spot'
        cfg.robot.relic_path = str(self.relic)
        cfg.robot.position = tuple(task.start_xy_m)
        delta = np.subtract(task.goal_xy_m, task.start_xy_m)
        cfg.robot.yaw = math.atan2(delta[1], delta[0])
        b = newton.ModelBuilder()
        height = _add_terrain(b, cfg.physics, self.terrain_seed, kiwi=True)
        axis = np.linspace(-task.terrain_extent_m, task.terrain_extent_m,
                           int(np.clip(round(2*task.terrain_extent_m/.08), 48, 384))+1)
        heights = np.array([[height(x, y) for x in axis] for y in axis], dtype=np.float32)
        cfg.robot.base_z = .65 + max(height(task.start_xy_m[0]+dx, task.start_xy_m[1]+dy)
                                     for dx in np.linspace(-.7, .7, 9) for dy in np.linspace(-.7, .7, 9))
        maps = build_robot(b, cfg.robot)
        b.add_ground_plane()
        model = b.finalize(device=self.device)
        empty = np.empty(0, dtype=int)
        tree = TreeModel(model=model, config=cfg, skeleton=TreeSkeleton([]),
                         seg_to_body=empty, joint_ids=[], joint_seg=empty,
                         joint_kp=np.empty((0, 3)), joint_kd=np.empty((0, 3)),
                         joint_dof_start=empty, joint_ndof=3, bend_mask=np.ones(3, bool),
                         rupture=np.empty(0), n_bodies=model.body_count,
                         robot_data=finalize_maps(model, maps), terrain_height=height,
                         terrain_params=dict(seed=self.terrain_seed, amplitude_m=task.terrain_amplitude_m,
                                             wavelength_m=task.terrain_wavelength_m,
                                             half_extent_m=task.terrain_extent_m,
                                             min_z_m=float(heights.min()), max_z_m=float(heights.max()),
                                             kind='quintic_value_fbm', octaves=3))
        self.sim = Sim(tree, fps=50, substeps=task.physics_hz//50, collisions=True)
        self.controller = NavigationController(self.sim, heights, task.terrain_extent_m)
        self.chassis = maps['chassis']
        self.previous_action = np.zeros(3, np.float32)
        self.stable_steps = self.steps = 0
        self.outcome = 'running'
        self.done = False
        self.sim._capture()
        self.controller.failures.zero_()
        self._measure()
        self.previous_distance = self.distance
        return self._observation(), self._info()

    def _measure(self):
        self.pose = self.sim.body_q_np()[self.chassis].copy()
        self.velocity = self.sim.state_0.body_qd.numpy()[self.chassis].copy()
        self.distance = float(np.linalg.norm(np.subtract(self.task.goal_xy_m, self.pose[:2])))
        self.tilt = float(np.arccos(np.clip(1.-2.*(self.pose[3]**2+self.pose[4]**2), -1, 1)))

    def _observation(self):
        inv = _qconj(self.pose[3:])
        delta = np.r_[np.subtract(self.task.goal_xy_m, self.pose[:2]), 0.]
        values = dict(goal_body_xy_m=_qrot(inv, delta)[:2],
                      base_velocity_body=np.r_[_qrot(inv, self.velocity[:3]), _qrot(inv, self.velocity[3:])],
                      gravity_body=_qrot(inv, np.array([0., 0., -1.])),
                      joint_position_rad=self.sim.state_0.joint_q.numpy()[self.controller.qids],
                      joint_velocity_rad_s=self.sim.state_0.joint_qd.numpy()[self.controller.dofs],
                      previous_action=self.previous_action,
                      gait_previous_action=self.controller.last_action,
                      goal_hold_time_s=[self.stable_steps/50.],
                      time_remaining_s=[max(0., self.task.time_limit_s-self.sim.sim_time)])
        obs = {k: np.asarray(v, np.float32).copy() for k, v in values.items()}
        if not all(np.isfinite(v).all() for v in obs.values()):
            raise RuntimeError('Nonfinite navigation observation')
        return obs

    def _info(self):
        return dict(outcome=self.outcome, success=self.outcome == 'success',
                    distance_to_goal_m=self.distance, position_m=self.pose[:3].tolist(),
                    tilt_rad=self.tilt, speed_m_s=float(np.linalg.norm(self.velocity[:2])),
                    simulation_time_s=self.sim.sim_time, terrain_seed=self.terrain_seed,
                    stable_goal_time_s=self.stable_steps/50.,
                    physics_device=str(self.sim.model.device),
                    cuda_graph=self.sim._graph is not None)

    def validate_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
            raise ValueError('Action must be three finite values in [-1, 1]')
        return action*COMMAND_LIMITS

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a new or finished episode')
        command = self.validate_action(action)
        self.controller.update(command)
        self.sim.step()
        self.controller.check(self.sim.state_0)
        failures = self.controller.failures.numpy()
        if failures[2] or not all(np.isfinite(a.numpy()).all() for a in
                                 (self.sim.state_0.joint_q, self.sim.state_0.joint_qd)):
            self.done, self.outcome = True, 'nonfinite'
            raise RuntimeError('Nonfinite navigation physics state')
        self._measure()
        self.steps += 1
        stable = (self.distance <= self.task.goal_radius_m and np.linalg.norm(self.velocity[:2]) < .10
                  and abs(self.velocity[5]) < .15 and self.tilt < .35)
        self.stable_steps = self.stable_steps+1 if stable else 0
        self.outcome = ('fall' if failures[0] else 'out_of_bounds' if failures[1] else
                        'success' if self.stable_steps/50. >= self.task.settle_time_s else
                        'timeout' if self.steps >= math.ceil(self.task.time_limit_s*50) else 'running')
        terminated = self.outcome in ('fall', 'out_of_bounds', 'success')
        truncated = self.outcome == 'timeout'
        self.done = terminated or truncated
        action = np.asarray(action, np.float32)
        terms = dict(progress=self.previous_distance-self.distance, time=-.002,
                     action_smoothness=-.001*float(np.sum((action-self.previous_action)**2)),
                     success=10. if self.outcome == 'success' else 0.,
                     failure=-10. if self.outcome in ('fall', 'out_of_bounds') else 0.)
        self.previous_action, self.previous_distance = action.copy(), self.distance
        info = self._info()
        info['reward_terms'] = terms
        return self._observation(), float(sum(terms.values())), terminated, truncated, info

    def _render_model(self):
        import mujoco
        physical = self.sim.solver.mj_model
        root = ET.Element('mujoco')
        ET.SubElement(root, 'compiler', angle='radian')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
        ET.SubElement(visual, 'headlight', ambient='.3 .3 .3', diffuse='.6 .6 .6')
        asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'texture', type='skybox', builtin='gradient',
                      rgb1='.35 .50 .68', rgb2='.78 .85 .9', width='256', height='256')
        ET.SubElement(asset, 'texture', name='soil', type='2d', builtin='checker',
                      rgb1='.28 .34 .18', rgb2='.33 .39 .22', width='128', height='128')
        ET.SubElement(asset, 'material', name='soil', texture='soil', texrepeat='10 10', reflectance='0')
        ET.SubElement(asset, 'hfield', name='terrain', nrow=str(physical.hfield_nrow[0]),
                      ncol=str(physical.hfield_ncol[0]), size=' '.join(map(str, physical.hfield_size[0])))
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='-3 -4 7', dir='.3 .4 -1', directional='true',
                      diffuse='.7 .7 .65', castshadow='true')
        ground = int(np.flatnonzero(physical.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)[0])
        ET.SubElement(world, 'geom', type='hfield', hfield='terrain', material='soil',
                      pos=' '.join(map(str, physical.geom_pos[ground])),
                      quat=' '.join(map(str, physical.geom_quat[ground])), contype='0', conaffinity='0')
        asset_dir = Path(self.sim.tree.robot_data['asset'])
        links = {link.get('name'): link for link in ET.parse(asset_dir/'spot_with_arm.urdf').getroot().findall('link')}
        for name in self.sim.model.body_label:
            link_name = name.rsplit('/', 1)[-1]
            body = ET.SubElement(world, 'body', name=link_name, mocap='true')
            link = links[link_name]
            shapes = link.findall('visual') or link.findall('collision')
            yellow = link_name == 'body' or 'uleg' in link_name or link_name in ('arm_link_sh0', 'arm_link_sh1')
            for i, shape in enumerate(shapes):
                origin = shape.find('origin')
                attrs = dict(pos=origin.get('xyz', '0 0 0') if origin is not None else '0 0 0',
                             euler=origin.get('rpy', '0 0 0') if origin is not None else '0 0 0',
                             rgba='.96 .73 .04 1' if yellow else '.16 .18 .21 1',
                             contype='0', conaffinity='0')
                geometry = shape.find('geometry')[0]
                if geometry.tag == 'mesh':
                    mesh = f'{link_name}_{i}'
                    ET.SubElement(asset, 'mesh', name=mesh,
                                  file=str(asset_dir/geometry.get('filename')),
                                  scale=geometry.get('scale', '1 1 1'))
                    attrs.update(type='mesh', mesh=mesh)
                elif geometry.tag == 'sphere':
                    attrs.update(type='sphere', size=geometry.get('radius'))
                elif geometry.tag == 'box':
                    attrs.update(type='box', size=' '.join(str(float(v)/2) for v in geometry.get('size').split()))
                elif geometry.tag == 'cylinder':
                    attrs.update(type='cylinder', size=f"{geometry.get('radius')} {float(geometry.get('length'))/2}")
                else:
                    raise ValueError(f'Unsupported Spot visual geometry: {geometry.tag}')
                ET.SubElement(body, 'geom', **attrs)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        model.hfield_data[:] = physical.hfield_data
        self.render_body_ids = [model.body(name.rsplit('/', 1)[-1]).id for name in self.sim.model.body_label]
        self.render_mocap_ids = model.body_mocapid[self.render_body_ids]
        return model

    def render(self):
        if self.render_mode != 'rgb_array' or self.sim is None:
            raise RuntimeError('Reset an rgb_array environment before rendering')
        import mujoco
        if self.renderer is None:
            self.render_model = self._render_model()
            self.render_data = mujoco.MjData(self.render_model)
            self.renderer = mujoco.Renderer(self.render_model, height=720, width=1280)
            self.camera = mujoco.MjvCamera()
            self.camera.azimuth, self.camera.elevation = 65., -28.
            self.camera.distance = max(5., np.linalg.norm(np.subtract(self.task.goal_xy_m, self.task.start_xy_m))*1.4)
            self.camera.lookat[:] = [*np.mean([self.task.start_xy_m, self.task.goal_xy_m], axis=0), .35]
        poses = self.sim.body_q_np()
        self.render_data.mocap_pos[self.render_mocap_ids] = poses[:, :3]
        self.render_data.mocap_quat[self.render_mocap_ids] = poses[:, [6, 3, 4, 5]]
        mujoco.mj_forward(self.render_model, self.render_data)
        if not np.allclose(self.render_data.xpos[self.render_body_ids], poses[:, :3], atol=1e-6, rtol=0):
            raise RuntimeError('Render/physics body poses differ')
        self.renderer.update_scene(self.render_data, camera=self.camera)
        for point, color, label in ((self.task.start_xy_m, [0., .4, 1., .7], 'A'),
                                    (self.task.goal_xy_m, [.1, 1., .3, .7], 'B')):
            scene = self.renderer.scene
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                               np.array([self.task.goal_radius_m, .012, .012]),
                               np.array([*point, self.sim.tree.terrain_height(*point)+.015]),
                               np.eye(3).ravel(), np.array(color, np.float32))
            geom.label = label
            scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = self.render_model = self.render_data = None
        if self.sim is not None:
            self.sim.robot_controller = None
        self.sim = self.controller = None
        self.done = True


gym.register('Thekenyos/SpotNavigation-v0', entry_point='treesim.spot_navigation:SpotNavigationEnv',
             nondeterministic=True)
