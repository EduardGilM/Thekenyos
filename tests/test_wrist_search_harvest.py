import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.train_wrist_search_harvest import collect_stages, encoder_is_trained, stages, successful_checkpoint
from treesim.visual_kiwi_env import (DEPTH_FOVY_DEG, MIN_DEPTH_M, SEARCH_ACQUISITION,
                                    SEARCH_BODY_SPEED_PENALTY, SEARCH_CENTERING_CLIP, SEARCH_CLOSE,
                                    SEARCH_IN_VIEW_FOV_SCALE, SEARCH_TRACKING, SEARCH_VIEW_LOSS,
                                    SEARCH_VIEW_STREAK, S0V_RANGE_M, VisualKiwiEnv, gated_search_rewards,
                                    in_view_fruit_position)
from treesim.visual_servo import pinhole_intrinsics


def _wrapper():
    wrapper = object.__new__(VisualKiwiEnv)
    wrapper.search_weight = 1.
    wrapper.vision = SimpleNamespace(size=64)
    wrapper.search_acquisition_paid = False
    wrapper.search_view_streak = 0
    wrapper.search_view_loss_fired = False
    wrapper.search_previous_potential = None
    wrapper.last_action = np.zeros(10)
    data = SimpleNamespace(xmat=np.repeat(np.eye(3)[None], 2, axis=0).reshape(2, 9),
                           xpos=np.array([[0., 0., 0.], [0., 0., 1.]]))
    wrapper.env = SimpleNamespace(phase='approach', data=data, chassis=0, target_body=1,
                                  kiwi_tof_valid=False, kiwi_in_view=False,
                                  _base_velocity=lambda: np.zeros(6))
    packet = dict(camera_to_base=np.repeat(np.eye(4)[None], 2, axis=0),
                  intrinsics=np.repeat(pinhole_intrinsics(64, 44.)[None], 2, axis=0),
                  rgb=np.zeros((3, 64, 64), np.uint8),
                  tof=np.zeros((2, 64, 64), np.float32))
    wrapper.sensor_packet = lambda: packet
    wrapper._packet = packet
    return wrapper


class WristSearchHarvestTest(unittest.TestCase):
    def test_curriculum_uses_search_only_for_walking_grab(self):
        lessons = stages()
        self.assertEqual([item['name'] for item in lessons],
                         ['search-grab', 'collect-1', 'collect-3'])
        self.assertEqual([item['name'] for item in stages(grab_only=True)], ['search-grab'])
        self.assertIn('--search-rewards', lessons[0]['args'])
        self.assertIn('--fixed-stage', lessons[0]['args'])
        self.assertEqual(lessons[0]['args'][lessons[0]['args'].index('--episode-seconds')+1], '8')
        self.assertEqual(lessons[0]['args'][lessons[0]['args'].index('--gripper-std')+1], '.25')
        for item in lessons[1:]:
            self.assertNotIn('--search-rewards', item['args'])
            self.assertNotIn('--fixed-stage', item['args'])
            self.assertEqual(item['args'][item['args'].index('--view-weight')+1], '0')
            self.assertIn('--deposit-mix', item['args'])
            self.assertEqual(item['args'][item['args'].index('--deposit-shaping')+1], '1')
        self.assertEqual(lessons[-1]['args'][lessons[-1]['args'].index('--picks')+1], '3')
        self.assertEqual([item['name'] for item in collect_stages()],
                         ['collect-1', 'collect-3'])
        self.assertTrue(all('--visual-lesson' in item['args'] and item['args'][item['args'].index('--visual-lesson')+1] == 'collect'
                            for item in collect_stages()))

    def test_only_heldout_success_returns_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory/'policy.zip'
            checkpoint.write_bytes(b'weights')
            summary = dict(recommended_checkpoint=checkpoint.name,
                           heldout=[dict(trained_successes=0, episodes=8)])
            (directory/'summary.json').write_text(json.dumps(summary))
            self.assertIsNone(successful_checkpoint(directory))
            summary['heldout'][0]['trained_successes'] = 1
            (directory/'summary.json').write_text(json.dumps(summary))
            self.assertEqual(successful_checkpoint(directory), checkpoint)

    def test_untrained_initial_encoder_is_not_used(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)/'initial-policy.zip'
            path.write_bytes(b'weights')
            (path.with_suffix('.json')).write_text(json.dumps(dict(steps=0)))
            self.assertFalse(encoder_is_trained(path))
            (path.with_suffix('.json')).write_text(json.dumps(dict(steps=4096)))
            self.assertTrue(encoder_is_trained(path))

    def test_search_rewards_use_hand_rgb_tof_detection(self):
        wrapper = _wrapper()
        detection = dict(source='hand_rgb_tof', point=np.array([0., 0., 1.]))
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[detection]):
            first = wrapper._search_reward_terms()
            second = wrapper._search_reward_terms()
        self.assertEqual(first['acquisition'], SEARCH_ACQUISITION)
        self.assertEqual(first['tracking'], SEARCH_TRACKING)
        self.assertEqual(first['close'], 0.)
        self.assertEqual(first['body_speed'], 0.)
        self.assertEqual(second['acquisition'], 0.)
        self.assertLessEqual(abs(second['centering']), SEARCH_CENTERING_CLIP)
        wrapper.search_view_streak = SEARCH_VIEW_STREAK
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[]):
            lost = wrapper._search_reward_terms()
        self.assertEqual(lost['view_loss'], SEARCH_VIEW_LOSS)

    def test_close_command_and_body_speed_shape_while_tracking(self):
        wrapper = _wrapper()
        wrapper.last_action[9] = 1.
        wrapper.env._base_velocity = lambda: np.array([.4, 0., 0.])
        detection = dict(source='hand_rgb_tof', point=np.array([0., 0., 1.]))
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[detection]):
            terms = wrapper._search_reward_terms()
        self.assertEqual(terms['close'], SEARCH_CLOSE)
        self.assertAlmostEqual(terms['body_speed'], -SEARCH_BODY_SPEED_PENALTY*.4)

    def test_rgb_hold_inside_tof_cutoff_does_not_fire_view_loss(self):
        wrapper = _wrapper()
        wrapper.search_acquisition_paid = True
        wrapper.search_view_streak = SEARCH_VIEW_STREAK
        wrapper.env.data.xpos[1] = np.array([0., 0., MIN_DEPTH_M/2])
        rgb = np.zeros((3, 64, 64), np.uint8)
        rgb[0, 32, 32] = 80
        rgb[1, 32, 32] = 40
        rgb[2, 32, 32] = 20
        wrapper._packet['rgb'] = rgb
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[]):
            terms = wrapper._search_reward_terms()
        self.assertEqual(terms['view_loss'], 0.)
        self.assertEqual(terms['tracking'], SEARCH_TRACKING)
        self.assertGreater(wrapper.search_view_streak, SEARCH_VIEW_STREAK)

    def test_rgb_without_prior_tof_lock_is_not_acquisition(self):
        wrapper = _wrapper()
        wrapper.env.data.xpos[1] = np.array([0., 0., MIN_DEPTH_M/2])
        rgb = np.zeros((3, 64, 64), np.uint8)
        rgb[0, 32, 32] = 80
        rgb[1, 32, 32] = 40
        rgb[2, 32, 32] = 20
        wrapper._packet['rgb'] = rgb
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[]):
            terms = wrapper._search_reward_terms()
        self.assertEqual(terms['acquisition'], 0.)
        self.assertEqual(terms['tracking'], 0.)
        self.assertFalse(wrapper.search_acquisition_paid)

    def test_privileged_tof_in_view_counts_as_acquisition(self):
        wrapper = _wrapper()
        wrapper.env.kiwi_tof_valid = True
        with mock.patch('treesim.visual_servo.estimate_fruit', return_value=[]):
            terms = wrapper._search_reward_terms()
        self.assertEqual(terms['acquisition'], SEARCH_ACQUISITION)
        self.assertEqual(terms['tracking'], SEARCH_TRACKING)
        self.assertTrue(wrapper.search_acquisition_paid)

    def test_search_spawn_stays_inside_wrist_depth_fov(self):
        half = np.deg2rad(DEPTH_FOVY_DEG/2)
        max_radius = S0V_RANGE_M[1]*np.tan(half)*SEARCH_IN_VIEW_FOV_SCALE
        self.assertLess(max_radius, S0V_RANGE_M[1]*np.tan(half))
        physics = SimpleNamespace(ee_depth=0,
                                  data=SimpleNamespace(cam_xpos=np.array([[0., 0., 0.]]),
                                                       cam_xmat=np.diag([1., -1., -1.]).ravel()[None]))
        rng = np.random.default_rng(0)
        for _ in range(32):
            position, dist, radius = in_view_fruit_position(physics, rng)
            self.assertGreaterEqual(dist, S0V_RANGE_M[0])
            self.assertLessEqual(dist, S0V_RANGE_M[1])
            self.assertLessEqual(radius, dist*np.tan(half)*SEARCH_IN_VIEW_FOV_SCALE+1e-9)
            self.assertAlmostEqual(position[2], dist)

    def test_privileged_progress_is_zero_until_hand_detection(self):
        undetected = gated_search_rewards(dict(progress=1.2, time=-.015),
                                          dict(acquisition=0., tracking=0., centering=0.,
                                               view_loss=0., close=0., body_speed=0.))
        self.assertEqual(undetected['progress'], 0.)
        self.assertEqual(undetected['time'], -.015)
        locked = gated_search_rewards(dict(progress=.4, time=-.015),
                                      dict(acquisition=SEARCH_ACQUISITION, tracking=SEARCH_TRACKING,
                                           centering=0., view_loss=0., close=0., body_speed=0.))
        self.assertEqual(locked['progress'], .4)
        self.assertEqual(locked['tracking'], SEARCH_TRACKING)
        self.assertEqual(locked['acquisition'], SEARCH_ACQUISITION)

    def test_search_rewards_are_idle_while_carrying(self):
        wrapper = _wrapper()
        wrapper.env.phase = 'carry'
        wrapper.env.kiwi_tof_valid = True
        terms = wrapper._search_reward_terms()
        self.assertEqual(terms['tracking'], 0.)
        self.assertEqual(terms['acquisition'], 0.)


if __name__ == '__main__':
    unittest.main()
