import itertools
import os
import unittest

import numpy as np

from treesim.basket_kiwi_env import BasketTask
from treesim.stationary_kiwi_env import GAPS_M, OFF_AXIS_M, StationaryKiwiEnv, arm_action, fruit_offset_m, scripted_reach
from treesim.visual_kiwi_env import VisionConfig


class StationaryContractTest(unittest.TestCase):
    def test_categorical_actions_never_command_body_or_wrist_rotation(self):
        for action in itertools.product(range(3), range(3), range(3), range(2)):
            decoded = arm_action(action)
            np.testing.assert_array_equal(decoded[[0, 1, 2, 6, 7, 8]], 0.)
            self.assertIn(decoded[-1], (-1., 1.))
            self.assertLessEqual(np.max(np.abs(decoded[3:6])), .45)
        for action in ([1, 1, 1], [1, 1, 1, 2], [3, 1, 1, 0], [.5, 1, 1, 0], [np.nan]*4):
            with self.assertRaises(ValueError):
                arm_action(action)

    def test_promotion_requires_s0v_then_repeated_arm_only_success(self):
        from scripts.train_assisted_kiwi import Curriculum, score
        curriculum = Curriculum(0, require_stationary=True)
        miss = dict(stage=0, episodes=8, successes=6, stationary_successes=6, mean_best_distance_m=.1)
        self.assertFalse(curriculum.update(miss))
        self.assertEqual(curriculum.stage, 0)
        s0v = dict(stage=0, episodes=8, successes=7, stationary_successes=7, mean_best_distance_m=.1)
        self.assertGreater(score(s0v), score(miss))
        self.assertTrue(curriculum.update(s0v))
        self.assertEqual(curriculum.stage, 1)
        later = dict(stage=1, episodes=8, successes=6, stationary_successes=6, mean_best_distance_m=.1)
        self.assertFalse(curriculum.update(later))
        self.assertTrue(curriculum.update(later))
        self.assertEqual(curriculum.stage, 2)

    def test_off_axis_offset_exceeds_assist_radius(self):
        rng = np.random.default_rng(0)
        for level in range(3):
            lo, hi = OFF_AXIS_M[level]
            self.assertGreater(lo, .12)
            self.assertLess(hi, GAPS_M[level])
            self.assertGreater(lo, GAPS_M[level]*np.tan(np.deg2rad(22.)))
            for _ in range(20):
                offset, radius = fruit_offset_m(level, rng)
                self.assertGreaterEqual(radius, lo)
                self.assertLessEqual(radius, hi)
                self.assertAlmostEqual(offset[0], GAPS_M[level], delta=.005)
                self.assertAlmostEqual(np.linalg.norm(offset[1:]), radius, delta=1e-9)

    def test_checkpoint_action_contract_is_explicit(self):
        from scripts.train_assisted_kiwi import checkpoint_compatible
        old = dict(source_sha256={}, task={}, training_config={})
        with self.assertRaises(ValueError):
            checkpoint_compatible(old, dict(old, stationary=True), 'warm_start')

    def test_encoder_warm_start_allows_stationary_camera_to_walking_collect(self):
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved = dict(source_sha256={'treesim/visual_kiwi_policy.py': 'enc'},
                     task={'capture_radius_m': .12}, vision={'size': 64, 'history': 3, 'lesson': 'grab'},
                     training_config={}, stationary=True)
        manifest = dict(source_sha256={'treesim/visual_kiwi_policy.py': 'enc', 'treesim/visual_kiwi_env.py': 'collect'},
                        task={'capture_radius_m': .12, 'picks': 3, 'start_phase': 'pick', 'time_limit_s': 90.},
                        vision={'size': 64, 'history': 3, 'lesson': 'collect'}, training_config={}, stationary=False)
        checkpoint_compatible(saved, manifest, 'encoder')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'warm_start')
        manifest['vision']['size'] = 96
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'encoder')

    @unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'Needs torch')
    def test_encoder_transfer_copies_vision_not_action_head(self):
        import torch
        from types import SimpleNamespace
        from scripts.train_assisted_kiwi import transfer_visual_encoder
        class Enc(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor([value]))
        source, dest = Enc(2.), Enc(0.)
        previous = SimpleNamespace(policy=SimpleNamespace(pi_features_extractor=source, features_extractor=source))
        model = SimpleNamespace(policy=SimpleNamespace(features_extractor=dest, action_net=SimpleNamespace(bias=torch.nn.Parameter(torch.tensor([9.])))))
        transfer_visual_encoder(model, previous)
        torch.testing.assert_close(dest.w, source.w)
        self.assertEqual(float(model.policy.action_net.bias.detach()), 9.)


@unittest.skipUnless(os.environ.get('ASSISTED_KIWI_RELIC'), 'Set ASSISTED_KIWI_RELIC for physical/camera checks')
class StationaryIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = StationaryKiwiEnv(os.environ['ASSISTED_KIWI_RELIC'], **kwargs)
        self.addCleanup(env.close)
        return env

    def test_robot_pose_stays_identical_while_fruit_distance_increases(self):
        env = self.make_env(vision=VisionConfig(noise=False))
        poses, gaps = [], []
        for stage in range(4):
            env.set_stage(stage)
            obs, info = env.reset(seed=11)
            e = env.env
            poses.append(np.r_[e.data.qpos[e.base_q:e.base_q+7], e.data.qpos[e.qids]])
            gaps.append(info['distance_m'])
            self.assertEqual(info['stage'], stage)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertFalse(info['walking_commands_enabled'])
            self.assertFalse(info['success'])
            if stage == 0:
                self.assertTrue(info['s0v'])
                self.assertGreaterEqual(info['distance_m'], .20)
                self.assertLessEqual(info['distance_m'], .40)
            else:
                self.assertGreater(info['distance_m'], e.task.capture_radius_m+.08)
                self.assertGreater(info['off_axis_m'], e.task.capture_radius_m)
                self.assertAlmostEqual(info['nominal_tcp_gap_m'], GAPS_M[stage-1])
        for pose in poses[1:]:
            np.testing.assert_allclose(pose, poses[0], atol=1e-10)
        self.assertTrue(np.all(np.diff(gaps) > .05))
        env.set_stage(0)
        first, _ = env.reset(seed=11)
        env.step([1, 1, 1, 0])
        again, _ = env.reset(seed=11)
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])

    def test_easy_fruit_is_visible_in_registered_hand_depth(self):
        from treesim.visual_servo import estimate_fruit
        env = self.make_env(vision=VisionConfig(noise=False))
        env.reset(seed=11)
        packet = env.sensor_packet()
        candidates = [r for r in estimate_fruit(packet) if r['source'] == 'hand_rgb_tof']
        self.assertTrue(candidates)
        e = env.env
        target = e.data.xmat[e.chassis].reshape(3, 3).T@(e.data.xpos[e.target_body]-e.data.xpos[e.chassis])
        self.assertLess(min(np.linalg.norm(r['point']-target) for r in candidates), .05)

    def test_passive_closure_does_not_solve_the_lesson(self):
        env = self.make_env()
        for seed in (11, 12, 13):
            env.reset(seed=seed)
            while True:
                _, _, term, trunc, info = env.step([1, 1, 1, 1])
                if term or trunc:
                    break
            self.assertFalse(info['stationary_success'], info)
            self.assertFalse(info['success'], info)

    def test_open_loop_tcp_reach_cannot_capture_off_axis_fruit(self):
        env = self.make_env()
        env.set_stage(1)
        for seed in (11, 12, 13):
            env.reset(seed=seed)
            while True:
                _, _, term, trunc, info = env.step([2, 1, 1, 1])
                if term or trunc:
                    break
            self.assertFalse(info['stationary_success'], info)
            self.assertGreater(info['best_distance_m'], env.env.task.capture_radius_m)

    def test_scripted_arm_capture_and_zero_gait_commands_at_both_rates(self):
        for hz in (1000, 2000):
            env = self.make_env(task=BasketTask(picks=1, time_limit_s=6., physics_hz=hz))
            env.reset(seed=11)
            commands = []
            tick = env.env._gait_tick
            def measured(command):
                commands.append(command.copy())
                return tick(command)
            env.env._gait_tick = measured
            for _ in range(60):
                _, _, term, trunc, info = env.step(scripted_reach(env.env))
                if term or trunc:
                    break
            self.assertTrue(info['stationary_success'], info)
            np.testing.assert_array_equal(commands, np.zeros_like(commands))
            self.assertLess(info['base_travel_m'], .1)


if __name__ == '__main__':
    unittest.main()
