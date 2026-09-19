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
        from treesim.kiwi_rl.curriculum import (
            EASY_PRESET, apply_easy_preset, easy_start_far_frac, easy_teacher_mix,
        )
        from treesim.kiwi_rl.reach_teacher import (
            basket_chassis_aabb_m, easy_start_local_m, hold_close_fracs, hover_tcp_local_m,
            jaw_hold_q, random_easy_start_local_m, select_hold_close, tcp_outside_basket,
        )
        from treesim.basket import CENTER, SIZE
        deposit = stage_named('deposit_pixels')
        preset = apply_easy_preset({'gate_success_rate': deposit.gate_success_rate,
                                    'gate_episodes': deposit.gate_episodes})
        self.assertEqual(preset['teacher_mix'], 1.0)
        self.assertEqual(preset['teacher_horizon_updates'], 60)
        self.assertEqual(preset['shaping_coef'], 5.0)
        self.assertEqual(preset['default_shaping_coef'], 2.0)
        self.assertEqual(preset['start_margin_m'], 0.40)
        self.assertEqual(preset['start_clearance_m'], 0.28)
        self.assertEqual(preset['n_start_poses'], 24)
        self.assertEqual(preset['n_hold_levels'], 8)
        self.assertEqual(preset['hold_close_min'], 0.40)
        self.assertEqual(preset['hold_close_max'], 1.00)
        self.assertEqual(preset['far_horizon_updates'], 200)
        self.assertEqual(preset['gate_success_rate'], 0.90)
        self.assertEqual(preset['gate_episodes'], 200)
        self.assertNotIn('weld', EASY_PRESET)
        self.assertNotIn('drop_offset_m', EASY_PRESET)
        self.assertNotIn('airdrop_above_rim_m', EASY_PRESET)
        self.assertEqual(deposit.gate_success_rate, 0.90)
        hover = hover_tcp_local_m(0.12)
        np.testing.assert_allclose(hover, CENTER + np.array([0.0, 0.0, SIZE[2] + 0.12]))
        self.assertGreater(float(hover[2]), float(CENTER[2] + SIZE[2]))
        self.assertFalse(tcp_outside_basket(hover, margin_m=0.04, above_rim_m=0.0))
        lo, hi = basket_chassis_aabb_m()
        np.testing.assert_allclose(hi[2], CENTER[2] + SIZE[2])
        near = easy_start_local_m(0.0)
        far = easy_start_local_m(1.0)
        self.assertTrue(tcp_outside_basket(near, margin_m=0.04, above_rim_m=0.0))
        self.assertTrue(tcp_outside_basket(far, margin_m=0.04, above_rim_m=0.0))
        self.assertGreater(float(near[0]), float(hi[0]))
        self.assertGreater(float(far[0]), float(near[0]))
        self.assertGreater(float(np.linalg.norm(far[:2] - CENTER[:2])),
                           float(np.linalg.norm(near[:2] - CENTER[:2])))
        self.assertEqual(easy_start_far_frac(0), 0.0)
        self.assertEqual(easy_start_far_frac(100), 0.5)
        self.assertEqual(easy_start_far_frac(200), 1.0)
        self.assertEqual(easy_start_far_frac(800), 1.0)
        self.assertAlmostEqual(easy_teacher_mix(0), 1.0)
        self.assertAlmostEqual(easy_teacher_mix(30), 0.5)
        self.assertAlmostEqual(easy_teacher_mix(60), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(90), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(0, start_mix=0.0), 0.0)
        fracs = hold_close_fracs()
        self.assertEqual(len(fracs), 8)
        self.assertAlmostEqual(float(fracs[0]), 0.40)
        self.assertAlmostEqual(float(fracs[-1]), 1.00)
        self.assertAlmostEqual(jaw_hold_q(0.0, 0.0, -1.5), 0.0)
        self.assertAlmostEqual(jaw_hold_q(1.0, 0.0, -1.5), -1.5)
        picked = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.2, 'max_load_N': 1.0, 'retained': False},
            {'close_frac': 0.7, 'slip_m': 0.02, 'max_load_N': 9.0, 'retained': True},
        ])
        self.assertAlmostEqual(picked['close_frac'], 0.7)
        rng = np.random.default_rng(0)
        sampled = [random_easy_start_local_m(rng) for _ in range(8)]
        lo, hi = basket_chassis_aabb_m()
        xs = [float(p[0]) for p in sampled]
        self.assertGreater(max(xs) - min(xs), 0.02)
        for pose in sampled:
            self.assertTrue(tcp_outside_basket(pose, margin_m=0.04, above_rim_m=0.0))
            self.assertGreaterEqual(float(pose[0]), float(hi[0] + 0.40) - 1e-9)
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_runtime.py').read_text(encoding='utf-8')
        self.assertIn('def _apply_easy_start', src)
        self.assertIn('def _apply_easy_jaw_hold', src)
        self.assertIn('def _run_hold_sweep', src)
        self.assertIn('self._easy_jaw_hold_next.assign', src)
        self.assertIn('fruit_equality', src)
        self.assertIn('grasp_local', src)
        self.assertIn('grasp_local_near_tcp', src)
        self.assertNotIn('norm(local)) > 0.12', src)
        self.assertIn('use_pocket', src)
        self.assertIn('set_easy_progress', src)
        self.assertNotIn('def _easy_airdrop', src)
        teacher_src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'reach_teacher.py').read_text(encoding='utf-8')
        self.assertIn('def sweep_jaw_hold', teacher_src)
        self.assertIn('def grasp_pocket_world_m', teacher_src)
        self.assertIn('def _pad_geom_ids', teacher_src)
        self.assertIn('kinematic jaw hold during sweep', teacher_src)
        self.assertIn('model.opt.timestep', teacher_src)
        self.assertIn('fruit_equality', teacher_src)
        self.assertIn('if over:', src)
        self.assertIn('out_applied[world, joint] = 0.0', src)
        scene_src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_scene.py').read_text(encoding='utf-8')
        self.assertIn('def reattach_basket_collision_geoms', scene_src)
        self.assertIn('def enable_arm_basket_contact_pairs', scene_src)
        self.assertIn('basket_shell', scene_src)
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
