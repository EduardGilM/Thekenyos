from collections import deque

import gymnasium as gym
import numpy as np

from .assisted_kiwi_env import AssistedKiwiEnv
from .visual_kiwi_env import RobotCameras, VisionConfig
from .visual_servo import estimate_fruit


SCOPE = ('Estimated-target assisted harvesting; camera fruit XYZ replaces simulator fruit '
         'coordinates; artificial grip weld; not contact-only grasp')
LOST_POINT_BASE_M = np.array([2., 0., 1.2])
CONTROL_DT_S = .1


def propagate_estimate(point, body_velocity, dt=CONTROL_DT_S):
    point = np.asarray(point, dtype=float)
    vel = np.asarray(body_velocity, dtype=float)
    if point.shape != (3,) or vel.shape != (6,) or not np.isfinite(point).all() or not np.isfinite(vel).all():
        raise ValueError('Estimate propagation needs a finite body-frame point and 6-D velocity')
    if not np.isfinite(dt) or dt < 0 or dt > 1:
        raise ValueError('Invalid estimate propagation timestep')
    return point-dt*(vel[:3]+np.cross(vel[3:], point))


def select_estimate(estimates, previous, tcp_base):
    tcp_base = np.asarray(tcp_base, dtype=float)
    previous = None if previous is None else np.asarray(previous, dtype=float)
    if tcp_base.shape != (3,) or (previous is not None and previous.shape != (3,)):
        raise ValueError('Estimate selection needs body-frame TCP and optional previous point')
    candidates = [item for item in estimates if abs(item['point'][1]) < .6 and item['point'][2] > .65]
    if previous is not None:
        near = [item for item in candidates if np.linalg.norm(item['point']-previous) < .3]
        if near:
            candidates = near
    if not candidates:
        return None
    hand = [item for item in candidates if item['source'] == 'hand_rgb_tof']
    reference = tcp_base if previous is None else previous
    return min(hand or candidates, key=lambda item: np.linalg.norm(item['point']-reference))


def fruit_terms(point_base, tcp_base):
    point_base = np.asarray(point_base, dtype=np.float32)
    tcp_base = np.asarray(tcp_base, dtype=np.float32)
    if point_base.shape != (3,) or tcp_base.shape != (3,):
        raise ValueError('Fruit terms need body-frame fruit and TCP points')
    return np.r_[point_base-tcp_base, point_base].astype(np.float32)


def replace_fruit_observation(obs, point_base, tcp_base):
    obs = np.asarray(obs, dtype=np.float32).copy()
    if obs.shape[-1] < 6:
        raise ValueError('Assisted observations store fruit XYZ in the first six values')
    obs[:6] = fruit_terms(point_base, tcp_base)
    return obs


class EstimatedTarget(gym.Wrapper):
    def __init__(self, env, vision=None):
        super().__init__(env)
        self.vision = vision or VisionConfig(size=96, lesson='grab')
        if tuple(env.observation_space.shape) not in ((78,), (99,)):
            raise ValueError('Estimated-target wrap expects the 78-D grab or 99-D basket observation')
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (78,), np.float32)
        self.cameras = None
        self.ablation = 'none'
        self.frames = deque(maxlen=max(1, self.vision.history))
        self.pending = deque(maxlen=self.vision.latency_steps+1)
        self.estimate_base = None
        self.estimate_source = 'none'
        self.estimate_valid = False

    @property
    def stage(self):
        return self.env.stage

    def set_stage(self, stage):
        self.env.set_stage(stage)

    def set_ablation(self, mode):
        if mode not in ('none', 'all', 'hand', 'tof'):
            raise ValueError('Unknown sensor ablation')
        self.ablation = mode

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        inner = self.unwrapped
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([int(inner.np_random.integers(2**31)), 419]))
        if self.cameras is None:
            self.cameras = RobotCameras(inner, self.vision)
        self.cameras.reset(self.sensor_rng)
        packet = (*self.cameras.capture(self.sensor_rng), inner.steps, self.cameras.calibration())
        self.last_packet = packet
        self.frames.clear()
        self.pending.clear()
        self.frames.extend([packet]*self.frames.maxlen)
        self.pending.extend([packet]*(self.vision.latency_steps+1))
        self.estimate_base = None
        self.estimate_source = 'none'
        self.estimate_valid = False
        return self._observe(obs), self._with_estimate(info)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        inner = self.unwrapped
        if not self.vision.noise or self.sensor_rng.random() >= .05:
            self.last_packet = (*self.cameras.capture(self.sensor_rng), inner.steps, self.cameras.calibration())
        self.pending.append(self.last_packet)
        self.frames.append(self.pending[0])
        return self._observe(obs), reward, term, trunc, self._with_estimate(info)

    def _masked_images(self, rgb, tof):
        rgb, tof = rgb.copy(), tof.copy()
        if self.ablation in ('all', 'hand'):
            rgb[:] = 0
        if self.ablation in ('all', 'tof'):
            tof[:] = 0
        return rgb, tof

    def sensor_packet(self):
        rgb, tof, step, calibration = self.frames[-1]
        rgb, tof = self._masked_images(rgb, tof)
        env = self.unwrapped
        base = np.eye(4)
        base[:3, :3] = env.data.xmat[env.chassis].reshape(3, 3)
        base[:3, 3] = env.data.xpos[env.chassis]
        return dict(rgb=rgb.copy(), tof=tof.copy(), age_s=(env.steps-step)*.1, capture_step=step,
                    camera_to_base=np.linalg.inv(base)[None]@calibration['camera_to_world'],
                    intrinsics=calibration['intrinsics'].copy(),
                    tcp_base=base[:3, :3].T@(env.data.site_xpos[env.tcp]-base[:3, 3]),
                    hand_forward_base=base[:3, :3].T@env.data.xmat[env.wrist].reshape(3, 3)[:, 0],
                    body_velocity=env._base_velocity().copy())

    def _true_fruit_base(self):
        env = self.unwrapped
        rotation = env.data.xmat[env.chassis].reshape(3, 3)
        return rotation.T@(env.data.xpos[env.target_body]-env.data.xpos[env.chassis])

    def _observe(self, obs):
        env = self.unwrapped
        if self.estimate_base is not None:
            self.estimate_base = propagate_estimate(self.estimate_base, env._base_velocity())
        selected = select_estimate(estimate_fruit(self.sensor_packet()), self.estimate_base, obs[6:9])
        if selected is None:
            self.estimate_valid = False
            self.estimate_source = 'odometry' if self.estimate_base is not None else 'none'
        else:
            point = np.asarray(selected['point'], dtype=float)
            self.estimate_base = point.copy() if self.estimate_base is None else .5*(self.estimate_base+point)
            self.estimate_source = selected['source']
            self.estimate_valid = True
        used = LOST_POINT_BASE_M if self.estimate_base is None else self.estimate_base
        if getattr(env, 'captured', False):
            used = np.asarray(obs[6:9], dtype=float)
            self.estimate_source = 'grip_tcp'
        patched = replace_fruit_observation(obs, used, obs[6:9])[:78]
        if patched.shape != self.observation_space.shape or not np.isfinite(patched).all():
            raise RuntimeError(f'Invalid estimated observation: {patched.shape}')
        return patched

    def _with_estimate(self, info):
        used = LOST_POINT_BASE_M if self.estimate_base is None else self.estimate_base
        return dict(info, scope=SCOPE, estimate_valid=self.estimate_valid, estimate_source=self.estimate_source,
                    estimate_base_m=np.asarray(used, dtype=float).tolist(),
                    estimate_error_m=float(np.linalg.norm(used-self._true_fruit_base())),
                    physical_grasp_validated=False)

    def _info(self):
        return self._with_estimate(self.env._info())

    def sensor_image(self):
        from PIL import Image, ImageDraw
        rgb, tof = self._masked_images(*self.frames[-1][:2])
        depth = np.repeat((tof[0]*255).astype(np.uint8)[..., None], 3, axis=2)
        depth[tof[1] == 0] = [120, 0, 120]
        tiles = [rgb[:3].transpose(1, 2, 0), depth]
        image = Image.new('RGB', (512, 282), (18, 24, 32))
        for i, (tile, label) in enumerate(zip(tiles, ('Hand RGB', 'Hand ToF | purple=invalid'))):
            image.paste(Image.fromarray(tile).resize((256, 256)), (256*i, 26))
            ImageDraw.Draw(image).text((256*i+5, 5), label, fill='white')
        ImageDraw.Draw(image).text((5, 266), f'Estimate {self.estimate_source}; actor sees camera XYZ, not simulator fruit',
                                   fill='white')
        return image

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
        super().close()


def EstimatedKiwiEnv(relic, *, vision=None, **kwargs):
    return EstimatedTarget(AssistedKiwiEnv(relic, **kwargs), vision=vision)


gym.register('Thekenyos/EstimatedKiwi-v0', entry_point='treesim.estimated_kiwi_env:EstimatedKiwiEnv')
