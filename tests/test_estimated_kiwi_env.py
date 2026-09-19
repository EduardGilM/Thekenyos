import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from treesim.estimated_kiwi_env import (LOST_POINT_BASE_M, fruit_terms, propagate_estimate,
                                        replace_fruit_observation, select_estimate)


class EstimatedTargetContractTest(unittest.TestCase):
    def test_select_estimate_prefers_hand_depth_and_rejects_empty(self):
        tcp = np.array([.3, 0., .8])
        prior = dict(point=np.array([.7, .1, .9]), source='size_prior')
        hand = dict(point=np.array([.35, .02, .92]), source='hand_rgb_tof')
        self.assertIsNone(select_estimate([], None, tcp))
        selected = select_estimate([prior, hand], None, tcp)
        self.assertEqual(selected['source'], 'hand_rgb_tof')
        far = dict(point=np.array([.7, .8, .9]), source='size_prior')
        self.assertIsNone(select_estimate([far], None, tcp))
        previous = np.array([.72, .08, .91])
        associated = select_estimate([prior, dict(point=np.array([1.2, .1, .9]), source='size_prior')],
                                     previous, tcp)
        np.testing.assert_allclose(associated['point'], prior['point'])

    def test_lost_observation_does_not_copy_simulator_fruit(self):
        privileged = np.arange(78, dtype=np.float32)
        tcp = np.array([.25, .01, .7], dtype=np.float32)
        patched = replace_fruit_observation(privileged, LOST_POINT_BASE_M, tcp)
        self.assertFalse(np.allclose(patched[:6], privileged[:6]))
        np.testing.assert_allclose(patched[:6], fruit_terms(LOST_POINT_BASE_M, tcp))
        np.testing.assert_array_equal(patched[6:], privileged[6:])

    def test_known_estimate_writes_body_and_tcp_terms_only(self):
        privileged = np.ones(78, dtype=np.float32)
        privileged[:6] = [9., 8., 7., 6., 5., 4.]
        point, tcp = np.array([.8, -.1, 1.1]), np.array([.4, 0., .75])
        patched = replace_fruit_observation(privileged, point, tcp)
        np.testing.assert_allclose(patched[3:6], point)
        np.testing.assert_allclose(patched[:3], point-tcp)
        np.testing.assert_array_equal(patched[6:], privileged[6:])

    def test_odometry_hold_moves_a_world_fixed_point_in_the_body_frame(self):
        point = np.array([1., .2, 1.1])
        moved = propagate_estimate(point, np.array([.5, 0., 0., 0., 0., 0.]), dt=.1)
        np.testing.assert_allclose(moved, [.95, .2, 1.1])
        yawed = propagate_estimate(np.array([1., 0., 1.]), np.array([0., 0., 0., 0., 0., np.pi]), dt=.1)
        self.assertGreater(abs(yawed[1]), .2)

    def test_warm_start_from_privileged_checkpoint_allows_new_estimator_files(self):
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved = dict(source_sha256={'treesim/assisted_kiwi_env.py': 'env', 'treesim/spot.py': 'spot',
                                    'scripts/train_assisted_kiwi.py': 'old'},
                     task={'stage': 2, 'capture_radius_m': .12, 'hold_time_s': .3, 'physics_hz': 1000,
                           'time_limit_s': 15.},
                     training_config={'num_envs': 4})
        manifest = dict(source_sha256={'treesim/assisted_kiwi_env.py': 'env', 'treesim/spot.py': 'spot',
                                       'scripts/train_assisted_kiwi.py': 'new',
                                       'treesim/estimated_kiwi_env.py': 'est',
                                       'treesim/visual_kiwi_env.py': 'vis',
                                       'treesim/visual_servo.py': 'servo'},
                        task={'stage': 0, 'capture_radius_m': .12, 'hold_time_s': .3, 'physics_hz': 1000,
                              'time_limit_s': 30.},
                        training_config={'num_envs': 4}, estimated_target=True)
        checkpoint_compatible(saved, manifest, 'warm_start')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'resume')
        changed = copy.deepcopy(saved)
        changed['source_sha256']['treesim/assisted_kiwi_env.py'] = 'other-physics'
        with self.assertRaises(ValueError):
            checkpoint_compatible(changed, manifest, 'warm_start')
        resume = copy.deepcopy(manifest)
        checkpoint_compatible(resume, manifest, 'resume')
        resume['source_sha256']['treesim/estimated_kiwi_env.py'] = 'changed'
        with self.assertRaises(ValueError):
            checkpoint_compatible(resume, manifest, 'resume')

    def test_workspace_wrapper_forwards_camera_ablation(self):
        import gymnasium as gym
        from scripts.train_assisted_kiwi import WorkspaceApproach

        class Inner(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(-1., 1., (78,), np.float32)
                self.action_space = gym.spaces.Box(-1., 1., (7,), np.float32)
                self.mode = 'none'

            def set_ablation(self, mode):
                self.mode = mode

        inner = Inner()
        WorkspaceApproach(inner, 0.).set_ablation('tof')
        self.assertEqual(inner.mode, 'tof')

    def test_basket_actor_does_not_see_privileged_extras(self):
        obs = np.arange(99, dtype=np.float32)
        tcp = np.array([.25, .01, .7], dtype=np.float32)
        patched = replace_fruit_observation(obs, LOST_POINT_BASE_M, tcp)[:78]
        self.assertEqual(patched.shape, (78,))
        self.assertFalse(np.allclose(patched[:6], obs[:6]))
        np.testing.assert_array_equal(patched[6:], obs[6:78])

    def test_warm_start_from_estimated_grab_to_estimated_basket(self):
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved = dict(source_sha256={'treesim/assisted_kiwi_env.py': 'env', 'treesim/spot.py': 'spot',
                                    'treesim/estimated_kiwi_env.py': 'old-est',
                                    'scripts/train_assisted_kiwi.py': 'old'},
                     task={'stage': 0, 'capture_radius_m': .12, 'hold_time_s': .3, 'physics_hz': 1000,
                           'time_limit_s': 30.},
                     training_config={'num_envs': 4}, estimated_target=True)
        manifest = dict(source_sha256={'treesim/assisted_kiwi_env.py': 'env', 'treesim/spot.py': 'spot',
                                       'treesim/estimated_kiwi_env.py': 'new-est',
                                       'treesim/basket_kiwi_env.py': 'basket', 'treesim/basket.py': 'geom',
                                       'scripts/train_assisted_kiwi.py': 'new'},
                        task={'stage': 0, 'capture_radius_m': .12, 'hold_time_s': .3, 'physics_hz': 1000,
                              'time_limit_s': 90., 'picks': 1, 'start_phase': 'pick', 'settle_time_s': .5},
                        training_config={'num_envs': 4}, estimated_target=True)
        checkpoint_compatible(saved, manifest, 'warm_start')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'resume')

    def test_queue_keeps_seven_actions_and_warm_starts_privileged_weights(self):
        from pathlib import Path
        from unittest import mock
        from scripts.queue_estimated_target import harvest_job_pids, train_args
        args = train_args(Path('/relic'), Path('/out'), Path('/ckpt.zip'))
        self.assertIn('--estimate', args)
        self.assertIn('--warm-start', args)
        self.assertNotIn('--vision', args)
        self.assertNotIn('--stationary', args)
        self.assertNotIn('--basket', args)
        self.assertEqual(args[args.index('--workspace-weight')+1], '1')
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for pid, command in ((1, b'python\0scripts/queue_visual_harvest.py\0'),
                                 (2, b'python\0scripts/queue_estimated_target.py\0'),
                                 (3, b'python\0scripts/train_assisted_kiwi.py\0--vision\0')):
                path = root/str(pid)/'cmdline'
                path.parent.mkdir()
                path.write_bytes(command)
            with mock.patch.object(Path, 'glob', return_value=[root/'1'/'cmdline', root/'2'/'cmdline', root/'3'/'cmdline']):
                self.assertEqual(harvest_job_pids(), [1, 3])


@unittest.skipUnless(os.environ.get('ASSISTED_KIWI_RELIC'), 'Set ASSISTED_KIWI_RELIC for EGL/physics checks')
class EstimatedTargetIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        from treesim.assisted_kiwi_env import AssistedKiwiEnv
        from treesim.estimated_kiwi_env import EstimatedTarget
        from treesim.visual_kiwi_env import VisionConfig
        env = EstimatedTarget(AssistedKiwiEnv(os.environ['ASSISTED_KIWI_RELIC'], **kwargs),
                              vision=VisionConfig(size=64, noise=False, history=1, latency_steps=0))
        self.addCleanup(env.close)
        return env

    def test_actor_observation_does_not_restore_simulator_fruit_when_undetected(self):
        env = self.make_env()
        with patch('treesim.estimated_kiwi_env.estimate_fruit', return_value=[]):
            obs, info = env.reset(seed=7)
        privileged = env.unwrapped._observation()
        self.assertEqual(obs.shape, (78,))
        self.assertEqual(env.action_space.shape, (7,))
        self.assertFalse(np.allclose(obs[:6], privileged[:6]))
        np.testing.assert_allclose(obs[:6], fruit_terms(LOST_POINT_BASE_M, obs[6:9]), atol=1e-5)
        self.assertFalse(info['estimate_valid'])
        self.assertEqual(info['estimate_source'], 'none')
        self.assertNotIn('target_position', obs)

    def test_forced_camera_estimate_replaces_privileged_xyz(self):
        env = self.make_env()
        fake = dict(point=np.array([1.05, .12, 1.15]), source='hand_rgb_tof')
        with patch('treesim.estimated_kiwi_env.estimate_fruit', return_value=[fake]):
            obs, info = env.reset(seed=8)
        privileged = env.unwrapped._observation()
        np.testing.assert_allclose(obs[3:6], fake['point'], atol=1e-5)
        self.assertFalse(np.allclose(obs[:6], privileged[:6]))
        self.assertTrue(info['estimate_valid'])
        self.assertGreater(info['estimate_error_m'], 0.)
