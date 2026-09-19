from collections import deque
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .basket_kiwi_env import BasketKiwiEnv, BasketTask
from .visual_servo import APPROACH_STANDOFF_M, optical_transform, pinhole_intrinsics, project_points, transform_points


SCOPE = 'Camera-policy assisted harvesting; wrist RGB-D only; frozen RELIC; not calibrated sensing or physical grasp validation'
HAND_COLOR_XYZ = np.array([.13806, .0202, .02452])
HAND_COLOR_RPY = np.array([-1.41372, 0., -1.5708])
HAND_DEPTH_XYZ = np.array([.13495, 0., .00799])
HAND_DEPTH_RPY = np.array([0., 1.41372, 0.])
HAND_POSITION = HAND_COLOR_XYZ
TOF_POSITION = HAND_DEPTH_XYZ
HAND_PITCH = np.pi/2-1.41372
# Boston Dynamics gripper spec, vertical FOV. RGB 60.2x46.4, Depth 55.9x44.
# https://dev.bostondynamics.com/docs/concepts/arm/arm_specification
RGB_FOVY_DEG = 46.4
DEPTH_FOVY_DEG = 44.
MIN_DEPTH_M = .15
CAMERA_COUNT = 2
RGB_CHANNELS = 3
ACTOR_KEYS = frozenset({'rgb', 'depth', 'proprio', 'phase', 'has_grasped', 'has_placed', 'age'})
PRIVILEGED_KEYS = frozenset({'privileged', 'fruit', 'target', 'target_xyz', 'distance', 'deposited',
                             'picked', 'attachment', 'harvest', 'estimate', 'target_index'})
COLLECT_FORWARD_M = (.75, 1.15, 1.60, 2.05, 2.50, 2.95)
S0V_RANGE_M = (.20, .40)


@dataclass(frozen=True)
class VisionConfig:
    size: int = 64
    history: int = 4
    latency_steps: int = 1
    noise: bool = True
    fruit_jitter_m: float = .18
    lesson: str = 'grab'
    cameras: tuple = ('ee_cam', 'ee_depth')

    def __post_init__(self):
        if self.size not in (64, 84, 96) or self.history not in (1, 4) or self.latency_steps not in (0, 1, 2):
            raise ValueError('Unsupported camera size, history or latency')
        if self.lesson not in ('grab', 'collect', 's0v') or not np.isfinite(self.fruit_jitter_m) or not 0 <= self.fruit_jitter_m <= .25:
            raise ValueError('Invalid visual lesson or fruit randomization')
        if self.cameras != ('ee_cam', 'ee_depth'):
            raise ValueError('Only wrist ee_cam RGB and ee_depth ToF are supported')


def camera_axes(pitch):
    forward = np.array([np.cos(pitch), 0., np.sin(pitch)])
    right = np.array([0., -1., 0.])
    return forward, np.cross(right, forward)


def urdf_rpy_xyaxes(rpy):
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    z = -rotation[:, 2]
    y = rotation[:, 1]
    x = np.cross(y, z)
    x = x/np.linalg.norm(x)
    y = np.cross(z, x)
    y = y/np.linalg.norm(y)
    return np.r_[x, y]


def add_hand_cameras_xml(wrist):
    ET.SubElement(wrist, 'camera', name='ee_cam', pos=' '.join(map(str, HAND_COLOR_XYZ)),
                  xyaxes=' '.join(map(str, urdf_rpy_xyaxes(HAND_COLOR_RPY))), fovy=str(RGB_FOVY_DEG))
    ET.SubElement(wrist, 'camera', name='ee_depth', pos=' '.join(map(str, HAND_DEPTH_XYZ)),
                  xyaxes=' '.join(map(str, urdf_rpy_xyaxes(HAND_DEPTH_RPY))), fovy=str(DEPTH_FOVY_DEG))


def depth_measurement(depth, rng, noise):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth < 3.)
    clean = np.where(valid, depth, 0.)
    if noise:
        clean = clean+rng.normal(0., .005+.003*clean**2, clean.shape)
        valid &= rng.random(depth.shape) >= .03
    valid &= (clean >= MIN_DEPTH_M) & (clean < 3.)
    return np.stack([np.where(valid, clean/3., 0.), valid]).astype(np.float32)


def assert_actor_observation(obs):
    leaked = set(obs).intersection(PRIVILEGED_KEYS)
    if leaked:
        raise AssertionError(f'Privileged keys in actor observation: {sorted(leaked)}')
    extra = set(obs)-ACTOR_KEYS
    if extra:
        raise AssertionError(f'Unknown actor observation keys: {sorted(extra)}')


def kiwi_camera_uv(model, data, cam_id, size, fruit_pos):
    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    local = rotation.T@(np.asarray(fruit_pos)-data.cam_xpos[cam_id])
    depth = -local[2]
    if depth <= 1e-6:
        return np.array([-1., -1.]), False, np.inf
    focal = size/(2*np.tan(np.deg2rad(model.cam_fovy[cam_id])/2))
    pixel = np.array([size/2+focal*local[0]/depth, size/2-focal*local[1]/depth])
    in_view = bool(np.all((pixel >= 0) & (pixel < size)))
    centre = (size-1)/2
    norm = float(np.linalg.norm((pixel-centre)/max(size/2, 1e-6)))
    return pixel, in_view, norm


class VisualTaskPhysics(BasketKiwiEnv):
    hand_cameras = True

    def __init__(self, relic, *, vision, view_weight=0., **kwargs):
        self.vision = vision
        if not np.isfinite(view_weight) or not 0 <= view_weight <= 4:
            raise ValueError('View shaping weight must be finite in [0, 4]')
        self.view_weight = view_weight
        super().__init__(relic, **kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (61,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (10,), np.float32)
        self.orientation_command = np.zeros(3)
        self.in_view_streak = 0
        self.view_loss_fired = False
        self.had_view = False
        if vision.lesson in ('grab', 's0v') and self.task.start_phase != 'pick':
            raise ValueError('Grab lesson must start with an unpicked fruit')

    def _build(self):
        from . import assisted_kiwi_env as ake
        original = ake.ET.SubElement

        def subelement(parent, tag, attrib=None, **extra):
            attrib = {} if attrib is None else dict(attrib)
            attrib.update(extra)
            element = original(parent, tag, attrib)
            if tag == 'site' and attrib.get('name') == 'tcp':
                add_hand_cameras_xml(parent)
            return element

        ake.ET.SubElement = subelement
        try:
            super()._build()
        finally:
            ake.ET.SubElement = original
        self.ee_cam = self.model.camera('ee_cam').id
        self.ee_depth = self.model.camera('ee_depth').id
        if not np.allclose(self.model.cam_fovy[self.ee_cam], RGB_FOVY_DEG) or not np.allclose(self.model.cam_fovy[self.ee_depth], DEPTH_FOVY_DEG):
            raise RuntimeError('ee_cam fovy must match the Boston Dynamics gripper specification')

    def privileged_observation(self):
        return BasketKiwiEnv._observation(self)

    def actor_proprio(self):
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        tcp = self.data.site_xpos[self.tcp]
        ee_pos = rotation.T@(tcp-self.data.xpos[self.chassis])
        ee_rot = (rotation.T@self.data.xmat[self.wrist].reshape(3, 3))[:, :2].ravel()
        arm = self.data.qpos[self.qids[12:19]]
        lo, hi = self.joint_limits[-1]
        gripper = float((self.data.qpos[self.qids[-1]]-lo)/max(hi-lo, 1e-6))
        gravity = rotation.T@[0., 0., -1.]
        return np.r_[ee_pos, ee_rot, arm, gripper, self.previous_action[:3], gravity[:2]].astype(np.float32)

    def actor_flags(self):
        phase = np.array([self.phase == name for name in ('approach', 'carry', 'settle')], dtype=np.float32)
        grasped = np.array([float(self.captured)], dtype=np.float32)
        placed = np.array([float(bool(self.deposited.any()))], dtype=np.float32)
        return phase, grasped, placed

    def _observation(self):
        return self.actor_proprio()

    def _view_terms(self):
        terms = dict(in_view=0., centering=0., view_loss=0.)
        if self.view_weight <= 0 or self.phase != 'approach' or self.model is None:
            return terms
        _, in_view, norm = kiwi_camera_uv(self.model, self.data, self.ee_cam, self.vision.size,
                                          self.data.xpos[self.target_body])
        if in_view:
            terms['in_view'] = self.view_weight*.5
            terms['centering'] = self.view_weight*.5*(1-np.tanh(2*min(norm, 8.)))
            self.in_view_streak += 1
            self.had_view = True
            self.view_loss_fired = False
        else:
            if self.had_view and self.in_view_streak >= 10 and not self.view_loss_fired:
                terms['view_loss'] = -self.view_weight
                self.view_loss_fired = True
            self.in_view_streak = 0
        return terms

    def reset(self, *, seed=None, options=None):
        self.fruit_contact_substeps = 0
        self.peak_fruit_contact_force = self.max_fruit_penetration = 0.
        self.orientation_command[:] = 0.
        self.in_view_streak = 0
        self.view_loss_fired = False
        self.had_view = False
        if self.model is not None:
            self.fruit_positions[:] = [self.original_site_pos[self.model.site(f'anchor_{i}').id] for i in range(6)]
        super().reset(seed=seed, options=options)
        offsets = self.np_random.uniform(-self.vision.fruit_jitter_m, self.vision.fruit_jitter_m, (6, 3))
        offsets[:, 2] *= .4
        for i in np.flatnonzero(~self.picked):
            self._set_fruit(i, self.original_site_pos[self.model.site(f'anchor_{i}').id]+offsets[i])
        self.collect_forward_m = ()
        if self.vision.lesson == 'collect' and self.task.start_phase == 'pick':
            self._place_collect_targets()
        mujoco.mj_forward(self.model, self.data)
        self._select_candidate()
        self.previous_distance = self.best_distance = self._distance()
        return self._observation(), self._info()

    def _set_fruit(self, index, position):
        anchor = self.model.site(f'anchor_{index}').id
        q = self.model.jnt_qposadr[self.model.joint(f'assisted_kiwi_{index}').id]
        self.data.qpos[q:q+3] = position
        self.model.site_pos[anchor] = position
        self.fruit_positions[index] = np.asarray(position, dtype=float)

    def _place_collect_targets(self):
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        base = self.data.xpos[self.chassis]
        chosen = np.flatnonzero(~self.picked)[:self.task.picks]
        forwards = []
        for rank, index in enumerate(chosen):
            forward = COLLECT_FORWARD_M[rank]+self.np_random.uniform(-.03, .03)
            lateral = self.np_random.uniform(-.08, .08)
            hang = self.original_site_pos[self.model.site(f'anchor_{index}').id][2]
            self._set_fruit(index, base+rotation@[forward, lateral, hang-base[2]])
            forwards.append(float(forward))
        self.collect_forward_m = tuple(forwards)
        for index in np.flatnonzero(~self.picked)[self.task.picks:]:
            original = self.original_site_pos[self.model.site(f'anchor_{index}').id]
            self._set_fruit(index, original+[0., 2.5, 0.])

    def _select_candidate(self):
        if self.phase != 'approach':
            return
        remaining = np.flatnonzero(~self.picked)
        distances = np.linalg.norm(self.data.xpos[self.fruit_bodies[remaining]]-self.data.site_xpos[self.tcp], axis=1)
        self.target_index = int(remaining[np.argmin(distances)])
        self.target_body = int(self.fruit_bodies[self.target_index])
        self.target = self.fruit_positions[self.target_index].copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (10,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Visual actions require ten finite values in [-1, 1]')
        self.orientation_command = action[6:9].copy()
        self._select_candidate()
        obs, reward, term, trunc, info = super().step(np.r_[action[:6], action[9]])
        extra = self._view_terms()
        info.setdefault('reward_terms', {}).update(extra)
        if self.view_weight > 0 and self.guidance_weight > 0:
            for name in ('progress', 'capture'):
                if name in info['reward_terms']:
                    info['reward_terms'][name] *= .5
        if extra['in_view'] or extra['centering'] or extra['view_loss']:
            reward = float(sum(info['reward_terms'].values()))
        info['kiwi_in_view'] = extra['in_view'] > 0
        info['in_view_streak'] = self.in_view_streak
        return obs, reward, term, trunc, info

    def _avoid_basket(self, velocity, nullspace):
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, angular, self.tcp)
        projected = angular[:, self.dofs[12:18]]@nullspace
        desired = self.data.xmat[self.chassis].reshape(3, 3)@(self.orientation_command*.6)
        correction = projected.T@np.linalg.solve(projected@projected.T+.03*np.eye(3), desired)
        return super()._avoid_basket(velocity+nullspace@correction, nullspace)

    def _physics_step(self):
        super()._physics_step()
        if not self.harvest_active:
            return
        touched = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = map(int, contact.geom)
            if not ((a in self.fruit_geoms and b in self.arm_geoms) or (b in self.fruit_geoms and a in self.arm_geoms)):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self.contact_force)
            self.peak_fruit_contact_force = max(self.peak_fruit_contact_force, float(self.contact_force[0]))
            self.max_fruit_penetration = max(self.max_fruit_penetration, -float(contact.dist))
            touched = True
        self.fruit_contact_substeps += int(touched)

    def _info(self):
        info = super()._info()
        info.update(scope=SCOPE, lesson=self.vision.lesson,
                    fruit_contact_time_s=self.fruit_contact_substeps/self.task.physics_hz,
                    peak_fruit_contact_force_N=self.peak_fruit_contact_force, max_fruit_penetration_m=self.max_fruit_penetration)
        if self.vision.lesson == 'grab':
            info['success'] = bool(self.captured and self.hold_ticks*.02 >= self.task.hold_time_s and not self.failure)
            if info['success']:
                info['outcome'] = 'visual_assisted_grasp'
        info['pregrasp'] = bool(self._distance() <= APPROACH_STANDOFF_M and not self.failure)
        if self.vision.lesson == 'collect' and self.task.start_phase == 'pick':
            info['collect_forward_m'] = list(getattr(self, 'collect_forward_m', ()))
        return info


class RobotCameras:
    names = ('ee_cam', 'ee_depth')

    def __init__(self, env, config):
        self.env, self.config = env, config
        self.renderer = mujoco.Renderer(env.model, height=config.size, width=config.size)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.option.sitegroup[:] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._store_defaults()

    def _store_defaults(self):
        model = self.env.model
        self.nominal_cam_pos = model.cam_pos.copy()
        self.nominal_cam_quat = model.cam_quat.copy()
        self.nominal_light_pos = model.light_pos.copy()
        self.nominal_light_dir = model.light_dir.copy()
        self.nominal_light_diffuse = model.light_diffuse.copy()
        self.nominal_geom_rgba = model.geom_rgba.copy()

    def reset(self, rng):
        model = self.env.model
        model.cam_pos[:] = self.nominal_cam_pos
        model.cam_quat[:] = self.nominal_cam_quat
        model.light_pos[:] = self.nominal_light_pos
        model.light_dir[:] = self.nominal_light_dir
        model.light_diffuse[:] = self.nominal_light_diffuse
        model.geom_rgba[:] = self.nominal_geom_rgba
        self.fovs = np.array([RGB_FOVY_DEG, DEPTH_FOVY_DEG])
        self.gain = np.ones((1, 1, 3))
        self.contrast = 1.
        if self.config.noise:
            if model.ncam >= 2:
                model.cam_pos[:] = self.nominal_cam_pos+rng.uniform(-.01, .01, self.nominal_cam_pos.shape)
                jitter = rng.uniform(-2., 2., (model.ncam, 3))
                for i in range(model.ncam):
                    base = Rotation.from_quat(self.nominal_cam_quat[i][[1, 2, 3, 0]])
                    composed = Rotation.from_euler('xyz', np.deg2rad(jitter[i]))*base
                    model.cam_quat[i] = composed.as_quat()[[3, 0, 1, 2]]
            if model.nlight:
                model.light_pos[:] = self.nominal_light_pos+rng.uniform(-.4, .4, self.nominal_light_pos.shape)
                model.light_diffuse[:] = np.clip(self.nominal_light_diffuse*rng.uniform(.6, 1.4, self.nominal_light_diffuse.shape), 0., 1.)
            model.geom_rgba[:, :3] = np.clip(self.nominal_geom_rgba[:, :3]*rng.uniform(.7, 1.3, (model.ngeom, 3)), 0., 1.)
            self.gain = rng.uniform(.75, 1.25, (1, 1, 3))
            self.contrast = float(rng.uniform(.85, 1.15))
        mujoco.mj_forward(model, self.env.data)

    def pose(self, index):
        if index not in range(CAMERA_COUNT):
            raise ValueError('ee_cam is camera 0 and ee_depth is camera 1')
        env = self.env
        if env.model.ncam >= 2:
            cam = env.model.camera(self.names[index]).id
            rotation = env.data.cam_xmat[cam].reshape(3, 3)
            return env.data.cam_xpos[cam].copy(), -rotation[:, 2], rotation[:, 1]
        rotation = env.data.xmat[env.wrist].reshape(3, 3)
        local = (HAND_POSITION, TOF_POSITION)[index]
        forward, up = camera_axes(HAND_PITCH)
        return env.data.xpos[env.wrist]+rotation@local, rotation@forward, rotation@up

    def _view(self, index):
        if self.env.model.ncam >= 2:
            self.renderer.update_scene(self.env.data, camera=self.names[index], scene_option=self.option)
            return
        self.renderer.update_scene(self.env.data, camera=self.camera, scene_option=self.option)
        position, forward, up = self.pose(index)
        for camera in self.renderer.scene.camera:
            camera.pos[:] = position
            camera.forward[:] = forward
            camera.up[:] = up
            camera.frustum_near = self.env.model.vis.map.znear*self.env.model.stat.extent
            camera.frustum_far = self.env.model.vis.map.zfar*self.env.model.stat.extent
            camera.frustum_bottom = -camera.frustum_near*np.tan(np.deg2rad(self.fovs[index]/2))
            camera.frustum_top = -camera.frustum_bottom
            camera.frustum_center = 0.

    def _render(self, index, depth=False):
        original_near = self.env.model.vis.map.znear
        self.env.model.vis.map.znear = .005/self.env.model.stat.extent
        try:
            if depth:
                self.renderer.enable_depth_rendering()
            else:
                self.renderer.disable_depth_rendering()
            self._view(index)
            return self.renderer.render().copy()
        finally:
            self.env.model.vis.map.znear = original_near
            self.renderer.disable_depth_rendering()

    def calibration(self):
        return dict(camera_to_world=np.stack([optical_transform(*self.pose(i)) for i in range(CAMERA_COUNT)]),
                    intrinsics=np.stack([pinhole_intrinsics(self.config.size, fov) for fov in self.fovs]),
                    grasp_reference_world=self.env.data.site_xpos[self.env.tcp]+APPROACH_STANDOFF_M*self.env.data.xmat[self.env.wrist].reshape(3, 3)[:, 0])

    def capture(self, rng):
        image = self._render(0).astype(np.float32)
        image = self.contrast*(image-127.5)+127.5
        image = image*self.gain
        if self.config.noise:
            image += rng.normal(0., 1.5, image.shape)
        rgb = np.clip(image, 0, 255).astype(np.uint8).transpose(2, 0, 1)
        depth = depth_measurement(self._render(1, depth=True), rng, self.config.noise)
        return rgb, depth

    def close(self):
        self.renderer.close()


class VisualKiwiEnv(gym.Wrapper):
    physics_class = VisualTaskPhysics

    def __init__(self, relic, *, task=None, vision=None, render_mode=None, guidance_weight=1., view_weight=0.):
        self.vision = vision or VisionConfig()
        super().__init__(self.physics_class(relic, vision=self.vision, task=task or BasketTask(picks=1),
                                          render_mode=render_mode, guidance_weight=guidance_weight,
                                          view_weight=view_weight))
        c = self.vision
        actor = {
            'rgb': gym.spaces.Box(0, 255, (RGB_CHANNELS*c.history, c.size, c.size), np.uint8),
            'depth': gym.spaces.Box(0., 1., (2*c.history, c.size, c.size), np.float32),
            'proprio': gym.spaces.Box(-np.inf, np.inf, (22,), np.float32),
            'phase': gym.spaces.Box(0., 1., (3,), np.float32),
            'has_grasped': gym.spaces.Box(0., 1., (1,), np.float32),
            'has_placed': gym.spaces.Box(0., 1., (1,), np.float32),
            'age': gym.spaces.Box(0., 120., (c.history,), np.float32),
        }
        self.actor_observation_space = gym.spaces.Dict(actor)
        self.observation_space = gym.spaces.Dict(dict(actor, privileged=gym.spaces.Box(-np.inf, np.inf, (99,), np.float32)))
        self.cameras = None
        self.ablation = 'none'
        self.frames = deque(maxlen=c.history)
        self.pending = deque(maxlen=c.latency_steps+1)

    @property
    def stage(self):
        return self.env.stage

    def set_stage(self, stage):
        self.env.set_stage(stage)

    def set_ablation(self, mode):
        if mode not in ('none', 'all', 'hand', 'tof'):
            raise ValueError('Unknown sensor ablation')
        self.ablation = mode

    def actor_observation(self, proprio=None):
        rgb, depth = self._masked_images(np.concatenate([p[0] for p in self.frames]),
                                         np.concatenate([p[1] for p in self.frames]))
        proprio = (proprio if proprio is not None else self.env.actor_proprio()).copy()
        if self.vision.noise:
            proprio += self.sensor_rng.normal(0., .002, proprio.shape)
        phase, grasped, placed = self.env.actor_flags()
        actor = dict(rgb=rgb, depth=depth, proprio=proprio.astype(np.float32), phase=phase,
                     has_grasped=grasped, has_placed=placed,
                     age=np.array([min(120., (self.env.steps-p[2])*.1) for p in self.frames], dtype=np.float32))
        assert_actor_observation(actor)
        if not self.actor_observation_space.contains(actor):
            raise RuntimeError('Invalid visual actor observation')
        return actor

    def _policy_observation(self, proprio):
        actor = self.actor_observation(proprio)
        privileged = self.env.privileged_observation()
        result = dict(actor, privileged=privileged)
        if not self.observation_space.contains(result):
            raise RuntimeError('Invalid visual policy observation')
        return result

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([int(self.env.np_random.integers(2**31)), 917]))
        if self.cameras is None:
            self.cameras = RobotCameras(self.env, self.vision)
        self.cameras.reset(self.sensor_rng)
        packet = (*self.cameras.capture(self.sensor_rng), self.env.steps, self.cameras.calibration())
        self.last_packet = packet
        self.frames.clear()
        self.pending.clear()
        self.frames.extend([packet]*self.vision.history)
        self.pending.extend([packet]*(self.vision.latency_steps+1))
        return self._policy_observation(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if not self.vision.noise or self.sensor_rng.random() >= .05:
            self.last_packet = (*self.cameras.capture(self.sensor_rng), self.env.steps, self.cameras.calibration())
        self.pending.append(self.last_packet)
        self.frames.append(self.pending[0])
        return self._policy_observation(obs), reward, term, trunc, info

    def _masked_images(self, rgb, depth):
        rgb, depth = rgb.copy(), depth.copy()
        if self.ablation in ('all', 'hand'):
            rgb[:] = 0
        if self.ablation in ('all', 'tof'):
            depth[:] = 0
        return rgb, depth

    def _info(self):
        return self.env._info()

    def sensor_packet(self):
        rgb, tof, step, calibration = self.frames[-1]
        rgb, depth = self._masked_images(rgb, tof)
        env = self.env
        base = np.eye(4)
        base[:3, :3] = env.data.xmat[env.chassis].reshape(3, 3)
        base[:3, 3] = env.data.xpos[env.chassis]
        return dict(rgb=rgb.copy(), tof=depth.copy(), age_s=(env.steps-step)*.1, capture_step=step,
                    camera_to_base=np.linalg.inv(base)[None]@calibration['camera_to_world'],
                    intrinsics=calibration['intrinsics'].copy(),
                    tcp_base=base[:3, :3].T@(env.data.site_xpos[env.tcp]-base[:3, 3]),
                    hand_forward_base=base[:3, :3].T@env.data.xmat[env.wrist].reshape(3, 3)[:, 0],
                    body_velocity=env._base_velocity().copy())

    def sensor_image(self):
        from PIL import Image, ImageDraw
        rgb, tof = self.frames[-1][:2]
        depth = np.repeat((tof[0]*255).astype(np.uint8)[..., None], 3, axis=2)
        depth[tof[1] == 0] = [120, 0, 120]
        tiles = [rgb[:RGB_CHANNELS].transpose(1, 2, 0), depth]
        image = Image.new('RGB', (512, 282), (18, 24, 32))
        for i, (tile, label) in enumerate(zip(tiles, ('Hand RGB', 'Hand ToF | purple=invalid'))):
            image.paste(Image.fromarray(tile).resize((256, 256)), (256*i, 26))
            ImageDraw.Draw(image).text((256*i+5, 5), label, fill='white')
            calibration = self.frames[-1][3]
            point = transform_points(calibration['grasp_reference_world'], np.linalg.inv(calibration['camera_to_world'][i]))
            pixel, valid = project_points(point, calibration['intrinsics'][i])
            if valid and np.all(pixel >= 0) and np.all(pixel < self.vision.size):
                x, y = pixel*256/self.vision.size+[256*i, 26]
                draw = ImageDraw.Draw(image)
                draw.line((x-5, y, x+5, y), fill='white', width=2)
                draw.line((x, y-5, x, y+5), fill='white', width=2)
        ImageDraw.Draw(image).text((5, 266), f'White cross: gripper reference + {100*APPROACH_STANDOFF_M:g}cm; display only', fill='white', stroke_width=1, stroke_fill='black')
        return image

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
        super().close()


gym.register('Thekenyos/VisualKiwi-v0', entry_point='treesim.visual_kiwi_env:VisualKiwiEnv')
