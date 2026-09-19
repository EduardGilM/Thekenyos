import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from treesim.basket import CENTER, SIZE, WALL
from treesim.basket_kiwi_env import BasketKiwiEnv, BasketTask, fruit_in_basket


class BasketTaskTest(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({'picks': 0}, {'picks': 7}, {'start_phase': 'unknown'}, {'settle_time_s': 0.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BasketTask(**kwargs)

    def test_basket_checkpoint_requires_basket_physics_but_allows_curriculum(self):
        import copy
        from dataclasses import asdict
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved = dict(source_sha256={'treesim/assisted_kiwi_env.py': 'base', 'treesim/spot.py': 'spot',
                                   'treesim/basket_kiwi_env.py': 'task', 'treesim/basket.py': 'basket',
                                   'scripts/train_assisted_kiwi.py': 'trainer'},
                     task=asdict(BasketTask(picks=1, start_phase='release')), training_config={})
        manifest = copy.deepcopy(saved)
        manifest['task'].update(picks=6, start_phase='pick')
        checkpoint_compatible(saved, manifest, 'warm_start')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'resume')
        for name in ('treesim/basket_kiwi_env.py', 'treesim/basket.py'):
            changed = copy.deepcopy(manifest)
            changed['source_sha256'][name] = 'changed'
            with self.assertRaises(ValueError):
                checkpoint_compatible(saved, changed, 'warm_start')

    def test_full_rotated_ellipsoid_containment(self):
        base = np.array([1., -2., .6])
        rotation = Rotation.from_euler('xyz', [.1, -.15, .7]).as_matrix()
        center = base+rotation@(CENTER+[0., 0., .08])
        self.assertTrue(fruit_in_basket(center, rotation, base, rotation))
        for offset in ([.3, 0., .08], [0., 0., -.04], [0., 0., SIZE[2]]):
            self.assertFalse(fruit_in_basket(base+rotation@(CENTER+offset), rotation, base, rotation))
        touching = CENTER+[SIZE[0]/2-WALL-.03, 0., WALL/2+.04]
        self.assertTrue(fruit_in_basket(touching, np.eye(3), np.zeros(3), np.eye(3)))
        self.assertTrue(fruit_in_basket(touching+[.000005, 0., 0.], np.eye(3), np.zeros(3), np.eye(3)))
        self.assertFalse(fruit_in_basket(touching+[.0003, 0., 0.], np.eye(3), np.zeros(3), np.eye(3)))
        self.assertFalse(fruit_in_basket(touching+[0., 0., -.0003], np.eye(3), np.zeros(3), np.eye(3)))


@unittest.skipUnless(os.environ.get('ASSISTED_KIWI_RELIC'), 'Set ASSISTED_KIWI_RELIC for physical checks')
class BasketIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = BasketKiwiEnv(os.environ['ASSISTED_KIWI_RELIC'], task=BasketTask(**kwargs))
        self.addCleanup(env.close)
        return env

    def test_basket_mass_observations_and_seeded_reset(self):
        env = self.make_env(time_limit_s=.2)
        obs, info = env.reset(seed=11)
        self.assertEqual(obs.shape, (99,))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(env.basket_geoms), 5)
        self.assertAlmostEqual(env.basket_mass_added, 1.2, places=5)
        self.assertEqual(info['deposited_count'], 0)
        self.assertFalse(info['success'])
        for action in ([0], [np.nan]*7, [2.]*7):
            with self.assertRaises(ValueError):
                env.step(action)
        env.step(np.zeros(7))
        _, _, term, trunc, _ = env.step(np.zeros(7))
        self.assertFalse(term)
        self.assertTrue(trunc)
        again, again_info = env.reset(seed=11)
        np.testing.assert_array_equal(obs, again)
        self.assertEqual(info, again_info)

    def test_gait_velocity_uses_body_axes_not_basket_inertia_axes(self):
        import mujoco
        env = self.make_env()
        env.reset(seed=11)
        env.data.qvel[:] = 0.
        env.data.qvel[:3] = [.2, -.1, .05]
        mujoco.mj_forward(env.model, env.data)
        rotation = env.data.xmat[env.chassis].reshape(3, 3)
        np.testing.assert_allclose(env._base_velocity()[:3], rotation.T@[.2, -.1, .05], atol=1e-10)
        np.testing.assert_allclose(env._base_velocity()[3:], 0., atol=1e-10)

    def test_grab_does_not_finish_episode(self):
        env = self.make_env(picks=2)
        obs, _ = env.reset(seed=11)
        for _ in range(100):
            action = np.zeros(7)
            action[3:6] = np.clip(obs[:3]*5., -1., 1.)
            action[6] = 1. if np.linalg.norm(obs[:3]) < .108 else -1.
            obs, _, term, trunc, info = env.step(action)
            self.assertFalse(term or trunc, info)
            if env.captured and env.hold_ticks*.02 >= env.task.hold_time_s:
                break
        self.assertTrue(env.captured)
        self.assertEqual(info['phase'], 'carry')
        self.assertEqual(info['deposited_count'], 0)
        self.assertFalse(info['success'])

    def test_release_fixture_requires_real_free_settling(self):
        for hz in (1000, 2000):
            env = self.make_env(start_phase='release', picks=1, physics_hz=hz)
            _, info = env.reset(seed=11)
            self.assertTrue(env.captured)
            self.assertFalse(info['success'])
            for step in range(100):
                action = np.zeros(7)
                action[5] = .3 if step < 10 else 0.
                action[6] = -1.
                _, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            self.assertTrue(info['success'], info)
            self.assertEqual(info['deposited_count'], 1)
            self.assertFalse(env.data.eq_active[env.grips].any())
            self.assertFalse(env.data.eq_active[env.stems[env.target_index]])
            self.assertGreaterEqual(info['settle_times_s'][env.target_index], .5)

    def test_next_target_preserves_payload_and_episode(self):
        env = self.make_env(start_phase='release', picks=2)
        env.reset(seed=11)
        first = env.target_index
        for step in range(100):
            _, _, term, trunc, info = env.step(np.array([0., 0., 0., 0., 0., .3 if step < 10 else 0., -1.]))
            self.assertFalse(term or trunc, info)
            if info['deposited_count']:
                break
        self.assertEqual(info['deposited_count'], 1)
        self.assertNotEqual(env.target_index, first)
        self.assertEqual(info['phase'], 'approach')
        self.assertFalse(env.data.eq_active[env.stems[first]])
        self.assertFalse(env.data.eq_active[env.grips[first]])
        self.assertGreater(info['elapsed_s'], .5)
        env.data.xfrc_applied[env.fruit_bodies[first], :3] = [80., 0., 80.]
        for _ in range(20):
            _, _, term, trunc, info = env.step(np.zeros(7))
            if term or trunc:
                break
        self.assertTrue(term)
        self.assertEqual(info['outcome'], 'spilled')
        self.assertEqual(info['spilled_ids'], [first])
        self.assertEqual(info['retained_count'], 0)
        self.assertEqual(info['reward_terms']['spill'], -15.)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))
        env.reset(seed=11)
        self.assertFalse(env.deposited.any())
        self.assertFalse(env.spilled.any())

    def test_dropped_fruit_is_terminal(self):
        import mujoco
        env = self.make_env(start_phase='carry')
        env.reset(seed=11)
        env.data.eq_active[env.grips[env.target_index]] = False
        env.captured = False
        env.released[env.target_index] = True
        env.phase = 'settle'
        q = env.model.jnt_qposadr[env.model.joint(f'assisted_kiwi_{env.target_index}').id]
        env.data.qpos[q:q+3] = [3., 0., .02]
        mujoco.mj_forward(env.model, env.data)
        _, _, term, trunc, info = env.step(np.zeros(7))
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertFalse(info['success'])
        self.assertEqual(info['outcome'], 'dropped')


if __name__ == '__main__':
    unittest.main()
