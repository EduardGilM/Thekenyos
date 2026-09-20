import os
import unittest

import numpy as np

from treesim.assisted_kiwi_env import AssistedTask, AssistedKiwiEnv, can_capture


class AssistedTaskTest(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({'stage': -1}, {'stage': 3}, {'physics_hz': 500},
                       {'capture_radius_m': np.nan}, {'capture_radius_m': 1.},
                       {'hold_time_s': 0.}, {'time_limit_s': np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AssistedTask(**kwargs)
        env = AssistedKiwiEnv('/unused')
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))

    def test_capture_requires_closure_and_slow_motion(self):
        self.assertTrue(can_capture(.1, True, -.5, .1))
        for args in ((.1, False, -.5, .1), (.1, True, -1., .1),
                     (.1, True, -.5, 2.), (np.nan, True, -.5, .1)):
            self.assertFalse(can_capture(*args))


@unittest.skipUnless(os.environ.get('ASSISTED_KIWI_RELIC'), 'Set ASSISTED_KIWI_RELIC for physical checks')
class AssistedIntegrationTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = AssistedKiwiEnv(os.environ['ASSISTED_KIWI_RELIC'], **kwargs)
        self.addCleanup(env.close)
        return env

    def test_reset_api_actions_timeout(self):
        from gymnasium.utils.env_checker import check_env
        env = self.make_env(task=AssistedTask(time_limit_s=.2))
        check_env(env, skip_render_check=True)
        first, info = env.reset(seed=42)
        before = env.data.time
        for action in ([0], [np.nan]*7, [2.]*7):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(before, env.data.time)
        env.step(np.zeros(7))
        _, _, term, trunc, _ = env.step(np.zeros(7))
        self.assertFalse(term)
        self.assertTrue(trunc)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(7))
        second, second_info = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info, second_info)
        env.reset(seed=43)
        self.assertFalse(np.array_equal(first, env._observation()))

    def test_assist_is_explicit_and_reset_clears_it(self):
        env = self.make_env()
        env.reset(seed=42)
        self.assertFalse(env.captured)
        self.assertTrue(env.data.eq_active[env.stems].all())
        self.assertFalse(env.data.eq_active[env.grips].any())
        positions = env.data.qpos.copy()
        env._capture()
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.site_xpos[env.hand_sites[env.target_index]],
                                   env.data.xpos[env.target_body], atol=1e-10)
        np.testing.assert_array_equal(positions, env.data.qpos)
        self.assertFalse(env.data.eq_active[env.stems[env.target_index]])
        self.assertTrue(env.data.eq_active[env.grips[env.target_index]])
        self.assertFalse(env._info()['physical_grasp_validated'])
        env.reset(seed=42)
        self.assertFalse(env.data.eq_active[env.grips].any())
        self.assertTrue(env.data.eq_active[env.stems].all())
        np.testing.assert_array_equal(env.model.eq_data, env.original_eq)

    def test_no_capture_with_open_hand_or_distant_fruit(self):
        env = self.make_env()
        env.reset(seed=42)
        env.step(np.zeros(7))
        self.assertFalse(env.captured)
        env.set_stage(2)
        env.reset(seed=42)
        for _ in range(5):
            _, _, term, trunc, _ = env.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
            self.assertFalse(env.captured)
            if term or trunc:
                break

    def test_scripted_assisted_capture_at_two_timesteps(self):
        results = []
        for hz in (1000, 2000):
            env = self.make_env(task=AssistedTask(physics_hz=hz, time_limit_s=20.))
            obs, _ = env.reset(seed=11)
            for _ in range(200):
                action = np.zeros(7)
                action[3:6] = np.clip(obs[:3]*5., -1., 1.)
                action[6] = 1. if np.linalg.norm(obs[:3]) < .108 else -1.
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            self.assertTrue(info['success'], info)
            self.assertTrue(env.data.eq_active[env.grips[env.target_index]])
            self.assertFalse(info['physical_grasp_validated'])
            with self.assertRaises(RuntimeError):
                env.step(action)
            results.append(info['distance_m'])
        self.assertLess(abs(results[0]-results[1]), .02)

    def test_floating_spot_walks_toward_target(self):
        env = self.make_env(task=AssistedTask(stage=2))
        env.reset(seed=42)
        initial = env.data.xpos[env.chassis].copy()
        for _ in range(50):
            _, _, term, trunc, info = env.step(np.array([.7, 0., 0., 0., 0., 0., -1.]))
            self.assertFalse(term or trunc, info)
        self.assertGreater(env.data.xpos[env.chassis, 0]-initial[0], .4)
        self.assertFalse(env.captured)

    def test_uncaptured_fruit_stays_in_canopy(self):
        env = self.make_env()
        env.reset(seed=11)
        for _ in range(20):
            _, _, term, trunc, _ = env.step(np.zeros(7))
            np.testing.assert_allclose(env.data.xpos[env.fruit_bodies], env.fruit_positions, atol=.002, rtol=0)
            if term or trunc:
                break

    def test_episode_stage_changes_only_on_reset(self):
        env = self.make_env()
        env.reset(seed=42)
        env.set_stage(2)
        self.assertEqual(env._info()['stage'], 0)
        env.reset(seed=42)
        self.assertEqual(env._info()['stage'], 2)

    def test_workspace_reward_does_not_change_physics(self):
        from scripts.train_assisted_kiwi import WorkspaceApproach
        env = self.make_env(task=AssistedTask(stage=2, time_limit_s=.5))
        rollouts = []
        for weight in (0., 1.):
            wrapped = WorkspaceApproach(env, weight)
            obs, _ = wrapped.reset(seed=42)
            transitions = []
            for _ in range(5):
                obs, reward, term, trunc, info = wrapped.step(np.array([.7, 0., 0., .2, 0., .1, -1.]))
                transitions.append((obs.copy(), term, trunc, info['success'], reward-info['reward_terms']['workspace']))
            rollouts.append(transitions)
        for before, after in zip(*rollouts):
            np.testing.assert_array_equal(before[0], after[0])
            self.assertEqual(before[1:4], after[1:4])
            self.assertAlmostEqual(before[4], after[4])

    def test_fall_is_latched(self):
        env = self.make_env()
        env.reset(seed=42)
        env.data.qpos[env.base_q+2] = .15
        _, _, term, trunc, info = env.step(np.zeros(7))
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertFalse(info['success'])
        self.assertEqual(info['outcome'], 'fall')


class TrainingHarnessTest(unittest.TestCase):
    def manifest(self):
        return dict(source_sha256={'treesim/assisted_kiwi_env.py': 'env', 'treesim/spot.py': 'spot',
                                   'scripts/train_assisted_kiwi.py': 'trainer'},
                    task={'stage': 0, 'capture_radius_m': .12, 'hold_time_s': .3, 'physics_hz': 1000},
                    training_config={'num_envs': 4, 'learning_rate': .0001})

    def test_visual_env_drops_unknown_keywords(self):
        from scripts.train_assisted_kiwi import visual_env
        class Env:
            def __init__(self, relic, *, task=None, vision=None):
                self.relic, self.task, self.vision = relic, task, vision
        env = visual_env(Env, 'relic', task=1, vision=2, search_spawn=True)
        self.assertEqual((env.relic, env.task, env.vision), ('relic', 1, 2))
        self.assertFalse(hasattr(env, 'search_spawn'))

    def test_warm_start_preserves_environment_contract(self):
        import copy
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved, manifest = self.manifest(), self.manifest()
        saved['source_sha256']['scripts/train_assisted_kiwi.py'] = 'old-trainer'
        checkpoint_compatible(saved, manifest, 'warm_start')
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'resume')
        for field in ('treesim/assisted_kiwi_env.py', 'treesim/spot.py'):
            changed = copy.deepcopy(saved)
            changed['source_sha256'][field] = 'changed-physics'
            with self.assertRaises(ValueError):
                checkpoint_compatible(changed, manifest, 'warm_start')
        changed = copy.deepcopy(saved)
        changed['task']['capture_radius_m'] = .15
        with self.assertRaises(ValueError):
            checkpoint_compatible(changed, manifest, 'warm_start')
        manifest['task']['stage'] = 1
        checkpoint_compatible(saved, manifest, 'warm_start')

    def test_resume_requires_matching_training_settings(self):
        from scripts.train_assisted_kiwi import checkpoint_compatible
        saved, manifest = self.manifest(), self.manifest()
        checkpoint_compatible(saved, manifest, 'resume')
        manifest['training_config']['num_envs'] = 8
        with self.assertRaises(ValueError):
            checkpoint_compatible(saved, manifest, 'resume')

    def test_curriculum_requires_consecutive_matching_evaluations(self):
        from scripts.train_assisted_kiwi import Curriculum
        curriculum = Curriculum(0)
        passed = dict(stage=0, episodes=8, successes=6)
        failed = dict(stage=0, episodes=8, successes=5)
        self.assertFalse(curriculum.update(passed))
        self.assertFalse(curriculum.update(failed))
        self.assertFalse(curriculum.update(passed))
        self.assertTrue(curriculum.update(passed))
        self.assertEqual(curriculum.stage, 1)
        with self.assertRaises(ValueError):
            curriculum.update(passed)
        fixed = Curriculum(0)
        for _ in range(4):
            self.assertFalse(fixed.update(passed, fixed=True))
        self.assertEqual(fixed.stage, 0)

    def test_checkpoint_selection_prioritizes_success_over_distance(self):
        from scripts.train_assisted_kiwi import score
        self.assertGreater(score(dict(successes=6, episodes=8, mean_best_distance_m=.11)),
                           score(dict(successes=5, episodes=8, mean_best_distance_m=.01)))

    def test_search_lesson_stays_close_and_requires_view_weight(self):
        from argparse import Namespace
        from scripts.train_assisted_kiwi import configure_search_lesson

        def error(message):
            raise ValueError(message)

        args = Namespace(search_rewards=True, vision=True, stationary=False, visual_lesson='grab',
                         view_weight=1., fixed_stage=False)
        configure_search_lesson(args, error)
        self.assertTrue(args.fixed_stage)
        args.view_weight = 0.
        with self.assertRaises(ValueError):
            configure_search_lesson(args, error)
        args.view_weight = 1.
        args.stationary = True
        with self.assertRaises(ValueError):
            configure_search_lesson(args, error)
        args.stationary = False
        args.visual_lesson = 'collect'
        args.fixed_stage = False
        configure_search_lesson(args, error)
        self.assertFalse(args.fixed_stage)

    def test_workspace_error_requires_body_approach_and_alignment(self):
        from scripts.train_assisted_kiwi import workspace_error
        for target in ([.45, 0.], [.6, .1], [.75, -.15]):
            self.assertAlmostEqual(workspace_error(target), 0.)
        self.assertAlmostEqual(workspace_error([1.5, 0.]), .75)
        self.assertAlmostEqual(workspace_error([.2, 0.]), .25)
        self.assertAlmostEqual(workspace_error([.6, .4]), .25)

    def test_workspace_curriculum_requires_combined_success(self):
        from scripts.train_assisted_kiwi import Curriculum
        curriculum = Curriculum(1, require_workspace=True)
        report = dict(stage=1, episodes=8, successes=8, combined_successes=5)
        self.assertFalse(curriculum.update(report))
        self.assertFalse(curriculum.update(report))
        report['combined_successes'] = 6
        self.assertFalse(curriculum.update(report))
        self.assertTrue(curriculum.update(report))
        self.assertEqual(curriculum.stage, 2)

    def test_workspace_shaping_preserves_dynamics_and_grasp_rules(self):
        import gymnasium as gym
        from types import SimpleNamespace
        from scripts.train_assisted_kiwi import WorkspaceApproach
        class Env(gym.Env):
            stage = 2
            target = np.array([1.5, 0., 1.47])
            chassis = 0
            def reset(self, seed=None, options=None):
                self.data = SimpleNamespace(xpos=np.array([[0., 0., .6]]), xmat=np.eye(3)[None].reshape(1, 9))
                return np.zeros(78), dict(elapsed_s=0., success=False)
            def _base_velocity(self):
                return np.zeros(6)
            def step(self, action):
                self.last_action = action.copy()
                self.data.xpos[0, 0] += float(action[0])
                return np.zeros(78), 1., False, False, dict(elapsed_s=.1, success=False, reward_terms={'original': 1.})
        for weight in (-1., np.nan, 5.):
            with self.assertRaises(ValueError):
                WorkspaceApproach(Env(), weight=weight)
        env = WorkspaceApproach(Env(), weight=1.)
        _, initial = env.reset(seed=42)
        arm_only = np.array([0., 0., 0., 1., 1., 1., -1.])
        self.assertLess(env.step(arm_only)[1], 1.)
        self.assertEqual(env.error, initial['workspace_error_m'])
        action = np.array([.3, 0., 0., 1., 1., 1., -1.])
        _, reward, term, trunc, info = env.step(action)
        np.testing.assert_array_equal(env.unwrapped.last_action, action)
        self.assertGreater(reward, 1.)
        self.assertLess(info['workspace_error_m'], initial['workspace_error_m'])
        self.assertFalse(term or trunc or info['success'])
        env.step(action)
        _, _, _, _, info = env.step(action)
        self.assertTrue(info['workspace_reached'])
        self.assertFalse(info['workspace_settled'])
        self.assertGreater(info['base_path_m'], .8)
        env.step(np.zeros(7))
        _, bonus, _, _, info = env.step(np.zeros(7))
        self.assertTrue(info['workspace_settled'])
        self.assertEqual(bonus, 3.)
        self.assertEqual(env.step(np.zeros(7))[1], 1.)
        _, reset = env.reset(seed=42)
        self.assertFalse(reset['workspace_reached'])
        self.assertEqual(reset['base_path_m'], 0.)
        plain = WorkspaceApproach(Env(), weight=0.)
        plain.reset(seed=42)
        self.assertEqual(plain.step(action)[1], 1.)

    def test_deposit_mix_probabilities_must_sum_to_one(self):
        from scripts.train_assisted_kiwi import parse_deposit_mix
        self.assertEqual(parse_deposit_mix('pick:1'), dict(pick=1., carry=0., release=0.))
        mix = parse_deposit_mix('pick:0.5,carry:0.3,release:0.2')
        self.assertAlmostEqual(sum(mix.values()), 1.)
        self.assertAlmostEqual(mix['carry'], .3)
        for text in ('pick:0.5', 'grab:1', 'pick:-0.1,carry:0.6,release:0.5'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_deposit_mix(text)

    def test_deposit_shaping_pays_hold_far_and_open_near_basket(self):
        from scripts.train_assisted_kiwi import (DEPOSIT_HOLD, DEPOSIT_LINGER, DEPOSIT_NEAR_M,
                                                DEPOSIT_OPEN_NEAR, deposit_shaping_terms)
        far = deposit_shaping_terms(dict(phase='carry', basket_distance_m=DEPOSIT_NEAR_M+.2, events=[]), 1.)
        self.assertAlmostEqual(far['hold'], DEPOSIT_HOLD)
        self.assertEqual(far['linger'], 0.)
        self.assertEqual(far['open_near'], 0.)
        near = deposit_shaping_terms(dict(phase='carry', basket_distance_m=DEPOSIT_NEAR_M-.05, events=[]), 1.)
        self.assertEqual(near['hold'], 0.)
        self.assertAlmostEqual(near['linger'], -DEPOSIT_LINGER)
        opened = deposit_shaping_terms(dict(phase='settle', basket_distance_m=.1,
                                            events=[dict(event='released', fruit=0)]), 1.)
        self.assertAlmostEqual(opened['open_near'], DEPOSIT_OPEN_NEAR)
        self.assertEqual(deposit_shaping_terms(dict(phase='approach', basket_distance_m=.8, events=[]), 1.)['hold'], 0.)
        self.assertEqual(deposit_shaping_terms(dict(phase='carry', basket_distance_m=.8, events=[]), 0.)['hold'], 0.)

    def test_deposit_curriculum_samples_start_phase_and_adds_shaping(self):
        import gymnasium as gym
        from dataclasses import dataclass
        from scripts.train_assisted_kiwi import DepositCurriculum

        @dataclass(frozen=True)
        class Task:
            start_phase: str = 'pick'

        class Env(gym.Env):
            def __init__(self):
                super().__init__()
                self.task = Task()
                self.stage = 0
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
                self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
            def set_stage(self, stage):
                self.stage = stage
            def reset(self, seed=None, options=None):
                return np.zeros(4), dict(phase=self.task.start_phase, start_phase=self.task.start_phase,
                                         basket_distance_m=.8, events=[], reward_terms={'time': -.015})
            def step(self, action):
                info = dict(phase='carry', basket_distance_m=.8, events=[], reward_terms={'time': -.015})
                return np.zeros(4), -.015, False, False, info

        env = DepositCurriculum(Env(), 'pick:0,carry:1,release:0', weight=1.)
        _, info = env.reset(seed=11)
        self.assertEqual(env.unwrapped.task.start_phase, 'carry')
        self.assertEqual(info['train_start_phase'], 'carry')
        env.set_stage(1)
        self.assertEqual(env.unwrapped.stage, 1)
        _, reward, _, _, stepped = env.step(np.zeros(4))
        self.assertGreater(reward, -.015)
        self.assertGreater(stepped['reward_terms']['hold'], 0.)
        off = DepositCurriculum(Env(), dict(pick=1., carry=0., release=0.), weight=0.)
        off.reset(seed=11)
        self.assertEqual(off.step(np.zeros(4))[1], -.015)

    def test_deposit_curriculum_softens_dropped_oracle_penalty(self):
        import gymnasium as gym
        from dataclasses import dataclass
        from scripts.train_assisted_kiwi import DEPOSIT_DROP, DEPOSIT_HOLD, DepositCurriculum

        @dataclass
        class Task:
            start_phase: str = 'pick'

        class Env(gym.Env):
            def __init__(self):
                super().__init__()
                self.task = Task()
                self.stage = 0
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
                self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
            def reset(self, seed=None, options=None):
                return np.zeros(4), {}
            def step(self, action):
                info = dict(phase='carry', outcome='dropped', basket_distance_m=.8, events=[],
                            reward_terms={'failure': -10., 'time': -.015})
                return np.zeros(4), -10.015, True, False, info

        env = DepositCurriculum(Env(), dict(pick=1., carry=0., release=0.), weight=1.)
        _, reward, terminated, _, info = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertEqual(info['reward_terms']['failure'], DEPOSIT_DROP)
        self.assertAlmostEqual(reward, DEPOSIT_DROP-.015+DEPOSIT_HOLD)

    def test_deposit_hold_cannot_outscore_a_basket_deposit(self):
        import gymnasium as gym
        from dataclasses import dataclass
        from scripts.train_assisted_kiwi import DEPOSIT_HOLD_CAP, DepositCurriculum

        @dataclass
        class Task:
            start_phase: str = 'pick'

        class Env(gym.Env):
            def __init__(self):
                super().__init__()
                self.task = Task()
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
                self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
            def reset(self, seed=None, options=None):
                return np.zeros(4), {}
            def step(self, action):
                info = dict(phase='carry', outcome='running', basket_distance_m=.8, events=[],
                            reward_terms={'time': -.015})
                return np.zeros(4), -.015, False, False, info

        env = DepositCurriculum(Env(), dict(pick=0., carry=1., release=0.), weight=1.)
        env.reset(seed=0)
        paid = 0.
        for _ in range(900):
            _, reward, _, _, info = env.step(np.zeros(4))
            paid += info['reward_terms']['hold']
        self.assertLessEqual(paid, DEPOSIT_HOLD_CAP+1e-9)
        self.assertLess(paid, 15.)

    def test_evaluation_aggregates_all_seeds(self):
        from pathlib import Path
        import tempfile
        from scripts.train_assisted_kiwi import evaluate
        class Policy:
            def predict(self, obs, deterministic):
                self.assert_deterministic = deterministic
                return np.zeros(7), None
        class Env:
            stage = 0
            def reset(self, seed):
                self.seed = seed
                return np.zeros(78), {}
            def step(self, action):
                success = self.seed % 2 == 0
                return np.zeros(78), 1., success, not success, dict(
                    success=success, best_distance_m=.1, base_travel_m=.5,
                    target_index=self.seed % 6, outcome='assisted_grasp' if success else 'timeout')
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate(Policy(), Env(), [0, 1, 2, 3], Path(directory), 'test', False)
            self.assertEqual(report['successes'], 2)
            self.assertEqual(report['episodes'], 4)
            self.assertEqual(report['targets'], [0, 1, 2, 3])
            with self.assertRaises(FileExistsError):
                evaluate(Policy(), Env(), [0], Path(directory), 'test', False)


if __name__ == '__main__':
    unittest.main()
