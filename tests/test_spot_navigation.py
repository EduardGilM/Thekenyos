import os
import unittest

import numpy as np

try:
    from treesim.spot_navigation import NavigationTask, SpotNavigationEnv, goal_action
except ModuleNotFoundError as exc:
    if exc.name != 'gymnasium':
        raise
    SpotNavigationEnv = None


@unittest.skipUnless(SpotNavigationEnv, 'Optional Gymnasium dependency required')
class NavigationTest(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({'start_xy_m': (np.nan, 0)}, {'goal_xy_m': (0,)},
                       {'terrain_amplitude_m': -1}, {'terrain_extent_m': 2},
                       {'time_limit_s': 0}, {'goal_radius_m': np.inf},
                       {'goal_xy_m': (-2, 0)}, {'physics_hz': 999}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NavigationTask(**kwargs)
        env = SpotNavigationEnv('/unused')
        for action in ([0, 0], [2, 0, 0], [np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                env.validate_action(action)
        np.testing.assert_array_equal(env.validate_action([1, -1, 1]), [.4, -.25, .7])
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])

    def test_body_frame_goal_controller(self):
        action = goal_action({'goal_body_xy_m': np.array([3., 0.])})
        self.assertGreater(action[0], 0)
        self.assertEqual(action[2], 0)
        action = goal_action({'goal_body_xy_m': np.array([0., 3.])})
        self.assertGreater(action[2], 0)
        self.assertAlmostEqual(action[0], 0, places=6)
        np.testing.assert_array_equal(goal_action({'goal_body_xy_m': np.zeros(2)}), np.zeros(3))


@unittest.skipUnless(SpotNavigationEnv and os.environ.get('SPOT_NAV_RELIC'),
                     'Requires Gymnasium and SPOT_NAV_RELIC for physical integration checks')
class NavigationIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        import gymnasium as gym
        return gym.make('Thekenyos/SpotNavigation-v0', relic=os.environ['SPOT_NAV_RELIC'],
                        device=os.environ.get('SPOT_NAV_DEVICE', 'cpu'), **kwargs).unwrapped

    def test_reset_and_actions(self):
        from gymnasium.utils.env_checker import check_env
        env = self.make_env(task=NavigationTask(time_limit_s=.04))
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)
        first, info = env.reset(seed=42)
        heights = env.controller.heights.numpy().copy()
        for action in ([0, 0], [np.nan, 0, 0], [0, 2, 0]):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.sim.sim_time, 0)
        env.step([0, 0, 0])
        second, second_info = env.reset(seed=42)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(info, second_info)
        np.testing.assert_array_equal(heights, env.controller.heights.numpy())
        env.step([0, 0, 0])
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['outcome'], 'timeout')
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])
        env.reset(seed=43)
        self.assertFalse(np.array_equal(heights, env.controller.heights.numpy()))

    def test_graph_matches_eager_and_goal_hold(self):
        from dataclasses import replace
        env = self.make_env()
        self.addCleanup(env.close)
        env.reset(seed=42)
        if env.sim.model.device.is_cuda:
            self.assertIsNotNone(env.sim._graph)
        actions = ([0, 0, 0], [.2, 0, 0], [.2, .1, 0], [.2, 0, .1])
        captured = [env.step(action)[0] for action in actions]
        env.reset(seed=42)
        env.sim._graph = None
        for action, expected in zip(actions, captured):
            actual = env.step(action)[0]
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-3, rtol=1e-4)
        for _ in range(50):
            env.step([0, 0, 0])
        env.task = replace(env.task, start_xy_m=(-3.5, 0.), goal_xy_m=tuple(env.pose[:2]))
        env.previous_distance = 0.
        for _ in range(100):
            _, reward, terminated, truncated, info = env.step([0, 0, 0])
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['success'])
        self.assertGreaterEqual(info['stable_goal_time_s'], .5)
        self.assertEqual(info['reward_terms']['success'], 10.)
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])

    def test_render_model_preserves_physics_and_terrain(self):
        import mujoco
        env = self.make_env()
        self.addCleanup(env.close)
        env.reset(seed=42)
        physics = env.sim.solver.mj_model
        masses = env.sim.model.body_mass.numpy().copy()
        initial = env.sim.body_q_np().copy()
        model = env._render_model()
        self.assertEqual(model.nmocap, env.sim.model.body_count)
        self.assertEqual(model.nq, 0)
        np.testing.assert_array_equal(model.hfield_data, physics.hfield_data)
        np.testing.assert_array_equal(model.hfield_size, physics.hfield_size)
        data = mujoco.MjData(model)
        data.mocap_pos[env.render_mocap_ids] = initial[:, :3]
        data.mocap_quat[env.render_mocap_ids] = initial[:, [6, 3, 4, 5]]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[env.render_body_ids], initial[:, :3], atol=1e-6)
        np.testing.assert_allclose(data.xquat[env.render_body_ids], initial[:, [6, 3, 4, 5]], atol=1e-6)
        np.testing.assert_array_equal(env.sim.body_q_np(), initial)
        np.testing.assert_array_equal(env.sim.model.body_mass.numpy(), masses)

    def test_boundary_and_nonfinite_latches(self):
        env = self.make_env()
        self.addCleanup(env.close)
        for index, value, outcome in ((0, 4.1, 'out_of_bounds'), (2, np.nan, 'nonfinite')):
            env.reset(seed=42)
            state = env.sim.state_0
            poses = state.body_q.numpy()
            changed = poses.copy()
            changed[env.chassis, index] = value
            state.body_q.assign(changed)
            env.controller.check(state)
            state.body_q.assign(poses)
            env.controller.check(state)
            if outcome == 'nonfinite':
                with self.assertRaises(RuntimeError):
                    env.step([0, 0, 0])
                self.assertTrue(env.done)
                self.assertEqual(env.outcome, outcome)
            else:
                _, _, terminated, truncated, info = env.step([0, 0, 0])
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info['outcome'], outcome)
                self.assertEqual(info['reward_terms']['failure'], -10.)

    def test_failure_latch(self):
        import warp as wp
        env = self.make_env()
        self.addCleanup(env.close)
        env.reset(seed=42)
        state = env.sim.state_0
        poses = state.body_q.numpy()
        broken = poses.copy()
        broken[env.chassis, 2] = .1
        state.body_q.assign(broken)
        env.controller.check(state)
        state.body_q.assign(poses)
        env.controller.check(state)
        wp.synchronize_device(env.sim.model.device)
        self.assertTrue(env.controller.failures.numpy()[0])
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome'], 'fall')
        env.reset(seed=42)
        self.assertFalse(env.controller.failures.numpy().any())


if __name__ == '__main__':
    unittest.main()
