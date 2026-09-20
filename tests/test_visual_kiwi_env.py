from dataclasses import asdict
import importlib.util
import os
from types import SimpleNamespace
import unittest

import gymnasium as gym
import mujoco
import numpy as np

from treesim.basket_kiwi_env import BasketTask
from treesim.visual_kiwi_env import (
    CAMERA_COUNT, RGB_CHANNELS, RGB_FOVY_DEG, RobotCameras, VisionConfig,
    VisualKiwiEnv, camera_axes, depth_measurement,
)


class VisualContractTest(unittest.TestCase):
    def test_config_validation(self):
        for kwargs in ({'size': 32}, {'history': 2}, {'latency_steps': -1}, {'lesson': 'oracle'},
                       {'fruit_jitter_m': np.nan}, {'fruit_jitter_m': .3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VisionConfig(**kwargs)

    def test_depth_invalid_and_metric_scale(self):
        depth = np.array([[np.nan, np.inf, 0., .07, .5, 2.99, 3.1]], dtype=np.float32)
        measured = depth_measurement(depth, np.random.default_rng(1), False)
        np.testing.assert_array_equal(measured[1], [[0, 0, 0, 0, 1, 1, 0]])
        np.testing.assert_allclose(measured[0]*3, [[0, 0, 0, 0, .5, 2.99, 0]])
        self.assertTrue(np.isfinite(depth_measurement(depth, np.random.default_rng(1), True)).all())

    def test_hand_only_rgb_layout(self):
        import treesim.visual_kiwi_env as visual
        self.assertFalse(hasattr(visual, 'HEAD_POSITION'))
        self.assertEqual(CAMERA_COUNT, 2)
        self.assertEqual(RGB_CHANNELS, 3)
        c = VisionConfig()
        self.assertEqual(c.history, 4)
        self.assertEqual((RGB_CHANNELS*c.history, c.size, c.size), (12, 64, 64))
        dummy = SimpleNamespace(ablation='none')
        with self.assertRaises(ValueError):
            VisualKiwiEnv.set_ablation(dummy, 'head')
        VisualKiwiEnv.set_ablation(dummy, 'hand')
        self.assertEqual(dummy.ablation, 'hand')

    def test_fruit_color_mask_accepts_tan_kiwi_and_rejects_leaves(self):
        from treesim.visual_servo import brown_mask
        kiwi = np.zeros((8, 8, 3), np.uint8)
        kiwi[..., 0], kiwi[..., 1], kiwi[..., 2] = 160, 90, 40
        self.assertTrue(brown_mask(kiwi).all())
        leaf = np.zeros((8, 8, 3), np.uint8)
        leaf[..., 0], leaf[..., 1], leaf[..., 2] = 40, 120, 30
        self.assertFalse(brown_mask(leaf).any())

    def test_fruit_color_mask_accepts_tan_kiwi_and_rejects_leaves(self):
        from treesim.visual_servo import brown_mask
        kiwi = np.zeros((8, 8, 3), np.uint8)
        kiwi[..., 0], kiwi[..., 1], kiwi[..., 2] = 160, 90, 40
        self.assertTrue(brown_mask(kiwi).all())
        leaf = np.zeros((8, 8, 3), np.uint8)
        leaf[..., 0], leaf[..., 1], leaf[..., 2] = 40, 120, 30
        self.assertFalse(brown_mask(leaf).any())

    def test_camera_axes_orthonormal(self):
        for pitch in np.linspace(-np.pi, np.pi, 25):
            forward, up = camera_axes(pitch)
            self.assertAlmostEqual(forward@up, 0.)
            self.assertAlmostEqual(np.linalg.norm(forward), 1.)
            self.assertAlmostEqual(np.linalg.norm(up), 1.)

    def test_camera_to_gripper_backprojection_includes_offset(self):
        from treesim.assisted_kiwi_env import TCP_OFFSET
        from treesim.visual_kiwi_env import TOF_POSITION
        from treesim.visual_servo import backproject_depth, optical_transform, pinhole_intrinsics, project_points, transform_points
        forward, up = camera_axes(np.pi/2-1.41372)
        transform = optical_transform(TOF_POSITION, forward, up)
        intrinsics = pinhole_intrinsics(64, 75.)
        points = backproject_depth(np.full((64, 64), .25), intrinsics)
        pixels, valid = project_points(points, intrinsics)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(pixels[20, 40], [40, 20])
        wrist_point = transform_points(points[20, 40], transform)
        np.testing.assert_allclose(transform_points(wrist_point, np.linalg.inv(transform)), points[20, 40])
        tcp_camera = transform_points(TCP_OFFSET, np.linalg.inv(transform))
        self.assertAlmostEqual(tcp_camera[2], .0588433, places=5)
        self.assertGreater(abs(np.linalg.norm(wrist_point-TCP_OFFSET)-.25), .025)

    def test_body_first_stops_translation_without_depth_and_stale_data(self):
        from unittest.mock import patch
        from treesim.visual_servo import BodyFirstServo
        servo = BodyFirstServo(.5)
        packet = dict(age_s=0., tcp_base=np.array([.3, 0., .8]), body_velocity=np.zeros(6),
                      camera_to_base=np.repeat(np.eye(4)[None], 2, axis=0))
        hand = dict(point=np.array([.7, .1, .9]), source='hand_rgb_tof')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[hand]):
            for _ in range(3):
                action = servo.action(packet)
            self.assertGreater(action[0], 0.)
            np.testing.assert_array_equal(action[3:6], 0.)
            packet['age_s'] = .3
            action = servo.action(packet)
            np.testing.assert_array_equal(action[:-1], 0.)
            self.assertEqual(action[-1], -1.)
        packet['age_s'] = 0.
        with patch('treesim.visual_servo.estimate_fruit', return_value=[]):
            np.testing.assert_array_equal(servo.action(packet), [0.]*9+[-1.])
        servo = BodyFirstServo(.5, 'fixed')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[hand]):
            np.testing.assert_array_equal(servo.action(packet)[3:6], 0.)
        servo.reset()
        close = dict(point=np.array([.3, 0., .915]), source='hand_rgb_tof')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[close]):
            for _ in range(5):
                action = servo.action(packet)
            self.assertEqual(action[-1], 1.)
            np.testing.assert_array_equal(servo.action(packet), [0.]*9+[1.])

    def test_registered_depth_separates_touching_brown_objects(self):
        from treesim.visual_servo import estimate_fruit, pinhole_intrinsics
        rows, cols = np.indices((64, 64))
        fruit = (rows-32)**2+(cols-32)**2 <= 6**2
        branch = (cols == 39) & (rows < 48)
        rgb = np.zeros((3, 64, 64), dtype=np.uint8)
        rgb[:, fruit | branch] = np.array([100, 60, 20])[:, None]
        depth = np.where(fruit, .3, np.where(branch, .9, 0.))
        cameras = np.repeat(np.eye(4)[None], 2, axis=0)
        cameras[:, 2, 3] = .8
        packet = dict(rgb=rgb, tof=np.stack([depth/3, depth > 0]), camera_to_base=cameras,
                      intrinsics=np.repeat(pinhole_intrinsics(64, 75.)[None], 2, axis=0))
        estimates = estimate_fruit(packet)
        self.assertEqual(len(estimates), 1)
        self.assertEqual(estimates[0]['source'], 'hand_rgb_tof')
        np.testing.assert_allclose(estimates[0]['point'], [0., 0., 1.13], atol=.015)
        packet['tof'][:] = 0
        self.assertEqual(estimate_fruit(packet), [])
        packet['tof'] = np.stack([depth/3, depth > 0])
        with self.assertRaises(ValueError):
            estimate_fruit(dict(packet, rgb=np.zeros((6, 64, 64), np.uint8)))

    def test_repeated_frames_do_not_bypass_sensing(self):
        from unittest.mock import patch
        from treesim.visual_servo import BodyFirstServo, pinhole_intrinsics
        packet = dict(age_s=0., rgb=np.full((3, 64, 64), 100, np.uint8),
                      tof=np.zeros((2, 64, 64), np.float32), tcp_base=np.array([.4, 0., .8]),
                      body_velocity=np.zeros(6), camera_to_base=np.repeat(np.eye(4)[None], 2, axis=0),
                      intrinsics=np.repeat(pinhole_intrinsics(64, 75.)[None], 2, axis=0), capture_step=0)
        servo = BodyFirstServo()
        np.testing.assert_array_equal(servo.action(packet), [0.]*9+[-1.])
        candidate = dict(point=np.array([.7, 0., .9]), source='hand_rgb_tof')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[candidate]):
            for _ in range(8):
                np.testing.assert_array_equal(servo.action(packet)[:-1], 0.)
        self.assertEqual(servo.hits, 1)

    def test_hand_depth_handoff_does_not_revert_to_monocular_range(self):
        from unittest.mock import patch
        from treesim.visual_servo import BodyFirstServo
        servo = BodyFirstServo(mode='fixed')
        packet = dict(age_s=0., tcp_base=np.array([.3, 0., .8]), body_velocity=np.zeros(6),
                      camera_to_base=np.repeat(np.eye(4)[None], 2, axis=0))
        hand = dict(point=np.array([.4, 0., 1.1]), source='hand_rgb_tof')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[hand]):
            for _ in range(3):
                servo.action(packet)
        lost = dict(point=np.array([.42, 0., 1.1]), source='size_prior')
        with patch('treesim.visual_servo.estimate_fruit', return_value=[lost]):
            np.testing.assert_array_equal(servo.action(packet), [0.]*9+[-1.])
        self.assertTrue(servo.hand_tracking)

    def test_collect_forward_layout_is_strictly_increasing(self):
        from treesim.visual_kiwi_env import COLLECT_FORWARD_M
        self.assertEqual(len(COLLECT_FORWARD_M), 6)
        self.assertTrue(np.all(np.diff(COLLECT_FORWARD_M) > .3))
        self.assertGreater(COLLECT_FORWARD_M[0], .6)

    def test_checkpoint_sensor_contract(self):
        import copy
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved = dict(source_sha256={}, task=asdict(BasketTask()), vision=asdict(VisionConfig()), training_config={})
        new = copy.deepcopy(saved)
        new['vision']['lesson'] = 'collect'
        new['task']['time_limit_s'] = 15.
        checkpoint_compatible(saved, new, 'warm_start')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, new, 'resume')
        changed = copy.deepcopy(new)
        changed['task']['capture_radius_m'] = .14
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, changed, 'warm_start')
        new['vision']['latency_steps'] = 0
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, new, 'warm_start')
        new.pop('vision')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, new, 'warm_start')

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('stable_baselines3'), 'Needs RL stack')
    def test_cnn_updates_all_sensor_encoders(self):
        import torch
        from stable_baselines3 import PPO
        from treesim.visual_kiwi_policy import ActorVisualExtractor, AsymmetricVisualPolicy
        space = gym.spaces.Dict(dict(
            rgb=gym.spaces.Box(0, 255, (12, 64, 64), np.uint8),
            depth=gym.spaces.Box(0., 1., (8, 64, 64), np.float32),
            proprio=gym.spaces.Box(-np.inf, np.inf, (22,), np.float32),
            phase=gym.spaces.Box(0., 1., (3,), np.float32),
            has_grasped=gym.spaces.Box(0., 1., (1,), np.float32),
            has_placed=gym.spaces.Box(0., 1., (1,), np.float32),
            age=gym.spaces.Box(0., 120., (4,), np.float32),
            privileged=gym.spaces.Box(-np.inf, np.inf, (99,), np.float32),
        ))
        class Dummy(gym.Env):
            metadata = {}
            def __init__(self):
                self.observation_space, self.action_space = space, gym.spaces.Box(-1., 1., (10,), np.float32)
            def reset(self, *, seed=None, options=None):
                return space.sample(), {}
            def step(self, action):
                return space.sample(), 0., False, True, {}
        env = Dummy()
        self.addCleanup(env.close)
        model = PPO(AsymmetricVisualPolicy, env, n_steps=2, batch_size=2,
                    policy_kwargs=dict(features_extractor_class=ActorVisualExtractor,
                                       share_features_extractor=False, ortho_init=False))
        sample = env.observation_space.sample()
        sample['proprio'][:] = 0
        obs, _ = model.policy.obs_to_tensor(sample)
        action, values, _ = model.policy(obs)
        (action.sum()+values.sum()).backward()
        for encoder in model.policy.pi_features_extractor.encoders.values():
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in encoder.parameters()), 0.)
        self.assertIsNotNone(model.policy.vf_features_extractor.mlp[0].weight.grad)


@unittest.skipUnless(os.environ.get('ASSISTED_KIWI_RELIC'), 'Set ASSISTED_KIWI_RELIC for EGL/physics checks')
class VisualIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = VisualKiwiEnv(os.environ['ASSISTED_KIWI_RELIC'], **kwargs)
        self.addCleanup(env.close)
        return env

    def test_rendered_depth_plane_is_one_metre(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="5 5 .1"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cameras = RobotCameras(SimpleNamespace(model=model, data=data), VisionConfig(noise=False))
        self.addCleanup(cameras.close)
        cameras.reset(np.random.default_rng(1))
        cameras.pose = lambda index: (np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
        _, depth = cameras.capture(np.random.default_rng(1))
        np.testing.assert_allclose(depth[0]*3, 1., atol=1e-5)
        np.testing.assert_array_equal(depth[1], 1.)

    def test_off_axis_render_matches_camera_projection(self):
        from treesim.visual_servo import pinhole_intrinsics, project_points
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="5 5 .1" rgba="0 0 1 1"/>'
                                              '<geom type="box" size=".01 .01 .01" pos=".06 .08 .7" rgba="1 0 0 1"/>'
                                              '</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cameras = RobotCameras(SimpleNamespace(model=model, data=data), VisionConfig(noise=False))
        self.addCleanup(cameras.close)
        cameras.reset(np.random.default_rng(1))
        cameras.pose = lambda index: (np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
        rgb, _ = cameras.capture(np.random.default_rng(1))
        rows, cols = np.nonzero(rgb[0].astype(float) > rgb[2].astype(float)+20)
        expected, _ = project_points(np.array([.06, -.08, .3]), pinhole_intrinsics(64, RGB_FOVY_DEG))
        np.testing.assert_allclose([cols.mean(), rows.mean()], expected, atol=1.2)

    def test_rendered_occlusion_is_not_filtered(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="5 5 .1"/>'
                                              '<geom type="box" size=".1 .1 .1" pos="0 0 .5"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cameras = RobotCameras(SimpleNamespace(model=model, data=data), VisionConfig(noise=False))
        self.addCleanup(cameras.close)
        cameras.reset(np.random.default_rng(1))
        cameras.pose = lambda index: (np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
        _, depth = cameras.capture(np.random.default_rng(1))
        self.assertAlmostEqual(float(depth[0, 32, 32]*3), .4, places=5)
        self.assertAlmostEqual(float(depth[0, 0, 0]*3), 1., places=5)

    def test_close_surface_is_not_clipped_from_rgb(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><visual><map znear=".1"/></visual><worldbody>'
                                              '<geom type="plane" size="5 5 .1" rgba="0 0 1 1"/>'
                                              '<geom type="box" size=".02 .02 .005" pos="0 0 .975" rgba="1 0 0 1"/>'
                                              '</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        original_near = model.vis.map.znear
        cameras = RobotCameras(SimpleNamespace(model=model, data=data), VisionConfig(noise=False))
        self.addCleanup(cameras.close)
        cameras.reset(np.random.default_rng(1))
        cameras.pose = lambda index: (np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([0., 1., 0.]))
        rgb, depth = cameras.capture(np.random.default_rng(1))
        self.assertGreater(int(rgb[0, 32, 32]), int(rgb[2, 32, 32]))
        self.assertEqual(depth[1, 32, 32], 0.)
        self.assertEqual(model.vis.map.znear, original_near)

    def test_seeded_reset_and_delay(self):
        env = self.make_env(vision=VisionConfig(noise=False), task=BasketTask(time_limit_s=.3))
        first, info = env.reset(seed=11)
        self.assertEqual(set(first), {'rgb', 'depth', 'proprio', 'phase', 'has_grasped', 'has_placed', 'age', 'privileged'})
        self.assertTrue(env.observation_space.contains(first))
        self.assertEqual(first['rgb'].shape, (RGB_CHANNELS*env.vision.history, env.vision.size, env.vision.size))
        with self.assertRaises(ValueError):
            env.cameras.pose(2)
        wrist = env.env.data.xpos[env.env.wrist]
        self.assertLess(np.linalg.norm(env.cameras.pose(0)[0]-wrist), .2)
        self.assertLess(np.linalg.norm(env.cameras.pose(1)[0]-wrist), .2)
        second, _, _, _, _ = env.step(np.zeros(10))
        np.testing.assert_array_equal(second['rgb'], first['rgb'])
        np.testing.assert_allclose(second['age'], .1)
        again, again_info = env.reset(seed=11)
        for name in first:
            np.testing.assert_array_equal(first[name], again[name])
        self.assertEqual(info, again_info)
        self.assertGreater(np.std(first['rgb']), 5.)

    def test_no_oracle_fields_or_target_tracking(self):
        env = self.make_env(vision=VisionConfig(noise=False))
        env.reset(seed=11)
        raw = env.env._observation().copy()
        images = env.cameras.capture(np.random.default_rng(2))
        poses = [env.cameras.pose(i) for i in range(CAMERA_COUNT)]
        env.env.target_index = (env.env.target_index+1) % 6
        env.env.target_body = int(env.env.fruit_bodies[env.env.target_index])
        env.env.target[:] = 99
        env.env.captured = not env.env.captured
        env.env.phase = 'settle'
        env.env.hold_ticks = 111
        env.env.deposited[:] = True
        np.testing.assert_array_equal(raw, env.env._observation())
        for before, after in zip(images, env.cameras.capture(np.random.default_rng(2))):
            np.testing.assert_array_equal(before, after)
        for i in range(CAMERA_COUNT):
            np.testing.assert_allclose(poses[i], env.cameras.pose(i))

    def test_fruit_motion_changes_images_not_proprio_or_camera_pose(self):
        env = self.make_env(vision=VisionConfig(noise=False))
        env.reset(seed=11)
        raw = env.env._observation().copy()
        before = env.cameras.capture(np.random.default_rng(1))
        poses = [env.cameras.pose(i) for i in range(CAMERA_COUNT)]
        for i in range(6):
            q = env.env.model.jnt_qposadr[env.env.model.joint(f'assisted_kiwi_{i}').id]
            env.env.data.qpos[q:q+3] += [0., .3, -.2]
        mujoco.mj_forward(env.env.model, env.env.data)
        after = env.cameras.capture(np.random.default_rng(1))
        np.testing.assert_allclose(raw, env.env._observation(), atol=1e-6)
        self.assertTrue(np.any(before[0] != after[0]))
        for i in range(CAMERA_COUNT):
            np.testing.assert_allclose(poses[i], env.cameras.pose(i))
        env.set_ablation('all')
        observation = env.actor_observation(raw)
        self.assertFalse(observation['rgb'].any())
        self.assertFalse(observation['depth'].any())
        np.testing.assert_array_equal(observation['proprio'], raw)

    def test_sensor_packet_is_delayed_and_uses_only_robot_geometry(self):
        from treesim.visual_servo import estimate_fruit
        env = self.make_env(vision=VisionConfig(noise=False))
        env.reset(seed=11)
        calibration = env.cameras.calibration()
        env.step(np.array([.5, 0., 0., 0., 0., 0., .2, 0., 0., -1.]))
        packet = env.sensor_packet()
        self.assertAlmostEqual(packet['age_s'], .1)
        rotation = env.env.data.xmat[env.env.chassis].reshape(3, 3)
        base = env.env.data.xpos[env.env.chassis]
        expected = (calibration['camera_to_world'][:, :3, 3]-base)@rotation
        np.testing.assert_allclose(packet['camera_to_base'][:, :3, 3], expected)
        env.env.target[:] = 99.
        env.env.phase = 'settle'
        for key, value in packet.items():
            np.testing.assert_array_equal(value, env.sensor_packet()[key])
        env.set_ablation('all')
        self.assertEqual(estimate_fruit(env.sensor_packet()), [])
        self.assertFalse(env.sensor_packet()['rgb'].any())
        self.assertFalse(env.sensor_packet()['tof'].any())

    def test_camera_ablation_does_not_change_physics_or_reward(self):
        env = self.make_env(task=BasketTask(time_limit_s=.3))
        trajectories = []
        for mode in ('none', 'all'):
            env.set_ablation(mode)
            env.reset(seed=11)
            trajectory = []
            for _ in range(3):
                _, reward, term, trunc, info = env.step(np.zeros(10))
                trajectory.append((env.env.data.qpos.copy(), reward, term, trunc))
            trajectories.append(trajectory)
        for a, b in zip(*trajectories):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])

    def test_camera_aim_commands_and_invalid_actions(self):
        env = self.make_env()
        env.reset(seed=11)
        for action in (np.zeros(7), np.full(10, np.nan), np.full(10, 2.)):
            with self.assertRaises(ValueError):
                env.step(action)
        results = []
        for aim in (0., .5):
            env.reset(seed=11)
            action = np.zeros(10)
            action[6:9] = aim
            initial = env.env.targets[12:18].copy()
            env.step(action)
            results.append(env.env.targets[12:18].copy())
            self.assertLessEqual(np.max(np.abs(results[-1]-initial)), .080001)
        self.assertGreater(np.linalg.norm(results[1]-results[0]), .001)

    def test_collect_pick_staggers_requested_fruit_in_front_of_the_robot(self):
        from treesim.visual_kiwi_env import COLLECT_FORWARD_M
        env = self.make_env(vision=VisionConfig(lesson='collect', noise=False),
                            task=BasketTask(start_phase='pick', picks=3, time_limit_s=8.))
        obs, info = env.reset(seed=11)
        e = env.env
        rotation = e.data.xmat[e.chassis].reshape(3, 3)
        forwards = []
        for index in np.flatnonzero(~e.picked)[:3]:
            delta = rotation.T@(e.data.xpos[e.fruit_bodies[index]]-e.data.xpos[e.chassis])
            forwards.append(delta[0])
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(info['collect_forward_m']), 3)
        self.assertTrue(np.all(np.diff(forwards) > .25), forwards)
        np.testing.assert_allclose(forwards, COLLECT_FORWARD_M[:3], atol=.12)
        self.assertGreater(forwards[0], e.task.capture_radius_m+.2)
        self.assertNotIn('target', obs)
        self.assertEqual(info['deposited_count'], 0)
        self.assertEqual(info['start_phase'], 'pick')
        self.assertEqual(info['lesson'], 'collect')

    def test_visual_collect_preserves_free_deposit_at_two_timesteps(self):
        for hz in (1000, 2000):
            env = self.make_env(vision=VisionConfig(lesson='collect'),
                                task=BasketTask(start_phase='release', picks=1, physics_hz=hz))
            env.reset(seed=11)
            for step in range(100):
                _, _, term, trunc, info = env.step(np.array([0., 0., 0., 0., 0., .3 if step < 10 else 0., 0., 0., 0., -1.]))
                if term or trunc:
                    break
            self.assertTrue(info['success'], info)
            self.assertFalse(env.env.data.eq_active[env.env.grips].any())
            self.assertGreaterEqual(info['settle_times_s'][env.env.target_index], .5)

    def test_search_spawn_starts_inside_hand_depth_and_detects(self):
        from treesim.visual_kiwi_env import kiwi_camera_uv
        env = self.make_env(vision=VisionConfig(noise=False), search_spawn=True, search_rewards=True,
                            view_weight=1., guidance_weight=1.)
        _, info = env.reset(seed=21)
        physics = env.env
        self.assertTrue(info['search_spawn'])
        _, depth_in_view, _ = kiwi_camera_uv(physics.model, physics.data, physics.ee_depth,
                                             env.vision.size, physics.data.xpos[physics.target_body])
        self.assertTrue(depth_in_view)
        _, _, _, _, stepped = env.step(np.zeros(10))
        self.assertTrue(stepped['search_target_detected'])
        self.assertGreater(stepped['reward_terms']['tracking'], 0.)
        self.assertNotIn('target', env.actor_observation())


if __name__ == '__main__':
    unittest.main()
