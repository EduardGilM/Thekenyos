import unittest

import numpy as np

from treesim.kiwi_rl.curriculum import (
    STAGES, evaluate_skills, evaluation_horizon_s, first_unsatisfied_stage,
    fruit_block_reason, next_stage, promotion_ready, sample_world_skills,
    stage_named,
)


class CurriculumTest(unittest.TestCase):
    def test_six_blueprint_stages_in_order(self):
        names = [stage.name for stage in STAGES]
        self.assertEqual(names, [
            'deposit_pixels', 'grasp_detach', 'stationary_harvest',
            'visual_approach', 'multi_harvest', 'generalise',
        ])
        self.assertEqual([stage.goal for stage in STAGES], [
            'DEPOSIT_ONLY', 'DETACH_ONLY', 'HARVEST', 'HARVEST', 'HARVEST', 'HARVEST',
        ])
        self.assertFalse(STAGES[0].allow_locomotion)
        self.assertTrue(STAGES[3].allow_locomotion)
        self.assertTrue(STAGES[4].continue_after_success)
        self.assertEqual(STAGES[4].fruit_count, 5)
        self.assertTrue(STAGES[5].randomize_layout)
        self.assertFalse(STAGES[0].randomize_layout)
        self.assertEqual(next_stage(STAGES[0]).name, 'grasp_detach')
        self.assertIsNone(next_stage(STAGES[-1]))

    def test_sample_mix_keeps_previous_skills(self):
        rng = np.random.default_rng(0)
        skills = sample_world_skills(stage_named('grasp_detach'), 4000, rng)
        self.assertEqual(skills['goal_id'].shape, (4000,))
        detach = float(np.mean(skills['goal_id'] == 1))
        deposit = float(np.mean(skills['goal_id'] == 0))
        self.assertGreater(detach, 0.65)
        self.assertGreater(deposit, 0.15)
        self.assertTrue(np.all(skills['timeout_s'][skills['goal_id'] == 0] == 30.0))
        self.assertTrue(np.all(skills['timeout_s'][skills['goal_id'] == 1] == 60.0))

    def test_evaluation_disables_guidance_and_uses_primary_goal(self):
        skills = evaluate_skills(stage_named('stationary_harvest'), 8)
        self.assertTrue(np.all(skills['goal_id'] == 2))
        self.assertTrue(np.all(skills['guidance_weight'] == 0.0))
        self.assertTrue(np.all(skills['primary_mask']))
        self.assertTrue(np.all(skills['continue_after_success'] == 0))
        self.assertTrue(np.all(skills['required_harvests'] == 1))

    def test_multi_harvest_eval_continues_and_caps_horizon(self):
        skills = evaluate_skills(stage_named('multi_harvest'), 4)
        self.assertTrue(np.all(skills['continue_after_success'] == 1))
        self.assertTrue(np.all(skills['required_harvests'] == 5))
        self.assertTrue(np.all(skills['guidance_weight'] == 0.0))
        self.assertAlmostEqual(evaluation_horizon_s(stage_named('deposit_pixels')), 30.0)
        self.assertAlmostEqual(evaluation_horizon_s(stage_named('stationary_harvest')), 45.0)
        self.assertAlmostEqual(evaluation_horizon_s(stage_named('multi_harvest')), 90.0)
        general = evaluate_skills(stage_named('generalise'), 8)
        self.assertTrue(np.all(general['randomize_layout'] == 1))
        self.assertTrue(np.any(np.abs(general['layout_dx_m']) > 0))

    def test_mix_continue_only_on_harvest_goal(self):
        rng = np.random.default_rng(1)
        skills = sample_world_skills(stage_named('multi_harvest'), 4000, rng)
        harvest = skills['goal_id'] == 2
        self.assertTrue(np.all(skills['continue_after_success'][harvest] == 1))
        self.assertTrue(np.all(skills['continue_after_success'][~harvest] == 0))
        self.assertTrue(np.all(skills['required_harvests'][harvest] == 5))
        self.assertTrue(np.all(skills['required_harvests'][~harvest] == 1))

    def test_promotion_needs_two_consecutive_gates(self):
        stage = stage_named('deposit_pixels')
        self.assertFalse(promotion_ready([0.91], stage))
        self.assertFalse(promotion_ready([0.91, 0.5], stage))
        self.assertTrue(promotion_ready([0.5, 0.91, 0.92], stage))
        self.assertFalse(promotion_ready([0.91, 0.92], stage, episodes_seen=10))
        self.assertTrue(promotion_ready([0.91, 0.92], stage, episodes_seen=200))
        with self.assertRaises(ValueError):
            promotion_ready([1.2, 1.0], stage)

    def test_one_fruit_scene_blocks_multi_harvest_not_deposit(self):
        start = stage_named('deposit_pixels')
        self.assertIsNone(first_unsatisfied_stage(start, 5))
        blocked = first_unsatisfied_stage(start, 1)
        self.assertEqual(blocked.name, 'multi_harvest')
        self.assertEqual(first_unsatisfied_stage(stage_named('visual_approach'), 1).name,
                         'multi_harvest')
        self.assertEqual(first_unsatisfied_stage(stage_named('multi_harvest'), 1).name,
                         'multi_harvest')
        self.assertIsNone(first_unsatisfied_stage(stage_named('multi_harvest'), 5))
        reason = fruit_block_reason(start, 1)
        self.assertIn('--fruit-count 5', reason)
        self.assertIsNone(fruit_block_reason(start, 5))


if __name__ == '__main__':
    unittest.main()
