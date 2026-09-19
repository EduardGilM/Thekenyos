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

    def test_speedrun_shortens_eval_not_gates_or_timeouts(self):
        deposit = stage_named('deposit_pixels')
        harvest = stage_named('stationary_harvest')
        multi = stage_named('multi_harvest')
        self.assertAlmostEqual(evaluation_horizon_s(deposit), 30.0)
        self.assertAlmostEqual(evaluation_horizon_s(deposit, profile='speedrun'), 8.0)
        self.assertAlmostEqual(evaluation_horizon_s(harvest, profile='speedrun'), 20.0)
        self.assertAlmostEqual(evaluation_horizon_s(multi, profile='speedrun'), 45.0)
        self.assertEqual(deposit.gate_success_rate, 0.90)
        self.assertEqual(deposit.gate_episodes, 200)
        self.assertEqual(deposit.budget_s, 30.0)
        from treesim.kiwi_rl.curriculum import (
            SPEEDRUN_EVAL_CAP_S, SPEEDRUN_PRESET, apply_speedrun_preset,
            idle_locomotion_mask, summarise_stage,
        )
        self.assertEqual(set(SPEEDRUN_EVAL_CAP_S), {stage.name for stage in STAGES})
        preset = apply_speedrun_preset({'eval_every': 50, 'gate_success_rate': 0.90})
        self.assertEqual(preset['eval_every'], 100)
        self.assertEqual(preset['eval_profile'], 'speedrun')
        self.assertEqual(preset['gate_success_rate'], 0.90)
        self.assertNotIn('gate_episodes', SPEEDRUN_PRESET)
        self.assertIsNone(idle_locomotion_mask(10, True))
        self.assertIsNone(idle_locomotion_mask(7, False))
        mask = idle_locomotion_mask(10, False)
        np.testing.assert_array_equal(mask[:3], 0)
        np.testing.assert_array_equal(mask[3:], 1)
        summary = summarise_stage(deposit, profile='speedrun')
        self.assertEqual(summary['eval_profile'], 'speedrun')
        self.assertEqual(summary['evaluation_horizon_s'], 8.0)
        self.assertFalse(summary['training_ready'])
        with self.assertRaises(ValueError):
            evaluation_horizon_s(deposit, profile='cheat')

    def test_easy_preset_does_not_change_gates_or_weld(self):
        from treesim.kiwi_rl.curriculum import EASY_PRESET, apply_easy_preset
        from treesim.kiwi_rl.reach_teacher import hover_tcp_local_m
        from treesim.basket import CENTER, SIZE
        deposit = stage_named('deposit_pixels')
        preset = apply_easy_preset({'gate_success_rate': deposit.gate_success_rate,
                                    'gate_episodes': deposit.gate_episodes})
        self.assertEqual(preset['teacher_mix'], 0.4)
        self.assertEqual(preset['shaping_coef'], 5.0)
        self.assertEqual(preset['default_shaping_coef'], 2.0)
        self.assertEqual(preset['gate_success_rate'], 0.90)
        self.assertEqual(preset['gate_episodes'], 200)
        self.assertNotIn('weld', EASY_PRESET)
        self.assertEqual(deposit.gate_success_rate, 0.90)
        local = hover_tcp_local_m(0.12)
        np.testing.assert_allclose(local, CENTER + np.array([0.0, 0.0, SIZE[2] + 0.12]))
        self.assertGreater(float(local[2]), float(CENTER[2] + SIZE[2]))
        drop_z = float(local[2] - EASY_PRESET['drop_offset_m'])
        self.assertGreater(drop_z, float(CENTER[2] + SIZE[2]))
        self.assertEqual(EASY_PRESET['drop_offset_m'], 0.06)
        with self.assertRaises(ValueError):
            hover_tcp_local_m(0.0)
        with self.assertRaises(ValueError):
            hover_tcp_local_m(-0.1)

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
