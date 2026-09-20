import inspect
import unittest

import numpy as np

from treesim.kiwi_rl.curriculum import (
    STAGES, apply_easy_hover_cohort, evaluate_skills, evaluation_horizon_s,
    first_unsatisfied_stage, fruit_block_reason, next_stage, promotion_ready,
    sample_world_skills, stage_named,
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
            EASY_PRESET, HOLD_SWEEP_CLEARANCE_M, HOLD_SWEEP_MARGIN_M,
            apply_easy_preset, easy_start_far_frac, easy_teacher_mix,
        )
        from treesim.kiwi_rl.reach_teacher import (
            basket_chassis_aabb_m, easy_over_opening_local_m, easy_start_local_m,
            easy_start_side_y_m, hold_close_fracs, hover_tcp_local_m, jaw_hold_q,
            random_easy_start_local_m, select_hold_close, tcp_outside_basket,
            tcp_over_opening_above_rim,
        )
        from treesim.kiwi_rl.curriculum import sample_easy_start_indices
        from treesim.basket import CENTER, SIZE
        deposit = stage_named('deposit_pixels')
        preset = apply_easy_preset({'gate_success_rate': deposit.gate_success_rate,
                                    'gate_episodes': deposit.gate_episodes})
        self.assertEqual(preset['teacher_mix'], 1.0)
        self.assertEqual(preset['teacher_horizon_updates'], 6)
        self.assertEqual(preset['shaping_coef'], 25.0)
        self.assertEqual(preset['entropy_coef'], 0.001)
        self.assertEqual(preset['hover_clearance_m'], 0.28)
        self.assertEqual(preset['release_target_clearance_m'], 0.28)
        self.assertGreaterEqual(
            preset['release_max_above_rim_m'], preset['hover_clearance_m'])
        self.assertEqual(preset['release_target_inset_x_m'], 0.15)
        self.assertEqual(len(preset['safe_hover_arm_q']), 6)
        self.assertTrue(np.isfinite(preset['safe_hover_arm_q']).all())
        self.assertEqual(preset['default_shaping_coef'], 2.0)
        self.assertFalse(preset['start_over_opening'])
        self.assertTrue(preset['shape_hand_and_fruit'])
        self.assertTrue(preset['release_at_center'])
        self.assertTrue(preset['release_over_opening'])
        self.assertEqual(preset['release_opening_inset_m'], 0.04)
        self.assertEqual(preset['start_open_radius_m'], 0.06)
        self.assertEqual(preset['start_inset_x_m'], 0.0)
        self.assertEqual(preset['start_margin_m'], 0.32)
        self.assertEqual(preset['start_x_span_m'], 0.08)
        self.assertEqual(preset['start_clearance_m'], 0.10)
        self.assertEqual(preset['shaping_length_m'], 0.60)
        self.assertEqual(preset['default_shaping_length_m'], 0.25)
        self.assertEqual(preset['start_side_y_m'], 0.10)
        self.assertEqual(preset['start_y_span_m'], 0.04)
        self.assertEqual(preset['start_z_span_m'], 0.08)
        self.assertEqual(preset['deposit_reward'], 30.0)
        self.assertEqual(preset['fail_reward'], -30.0)
        self.assertEqual(preset['ppo_clip'], 0.2)
        self.assertEqual(preset['ppo_lr'], 5e-4)
        self.assertEqual(preset['ppo_epochs'], 4)
        self.assertEqual(preset['ppo_grad_clip'], 0.5)
        self.assertIsNone(preset['ppo_adv_std_cap'])
        self.assertEqual(preset['ppo_value_coef'], 0.5)
        self.assertEqual(preset['ppo_target_kl'], 0.05)
        self.assertFalse(preset['ppo_unclip_positive'])
        self.assertEqual(preset['ppo_success_repeat'], 8)
        self.assertEqual(preset['ppo_imitation_coef'], 0.5)
        self.assertEqual(preset['ppo_success_epochs'], 4)
        self.assertEqual(HOLD_SWEEP_MARGIN_M, 0.40)
        self.assertEqual(HOLD_SWEEP_CLEARANCE_M, 0.28)
        self.assertEqual(preset['n_start_poses'], 24)
        self.assertEqual(preset['n_hold_levels'], 10)
        self.assertEqual(preset['hold_close_min'], 0.25)
        self.assertEqual(preset['hold_close_max'], 0.70)
        self.assertEqual(preset['far_horizon_updates'], 2000)
        self.assertEqual(preset['far_frac_cap'], 0.25)
        self.assertEqual(preset['release_max_above_rim_m'], 0.30)
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
        self.assertTrue(tcp_over_opening_above_rim(hover_tcp_local_m(0.28)))
        self.assertFalse(tcp_over_opening_above_rim(hover, min_clearance_m=0.16))
        over = easy_over_opening_local_m()
        self.assertTrue(tcp_over_opening_above_rim(over))
        self.assertFalse(tcp_outside_basket(over, margin_m=0.04, above_rim_m=0.0))
        self.assertAlmostEqual(float(over[0]), float(CENTER[0]))
        self.assertAlmostEqual(float(over[1]), float(CENTER[1]))
        self.assertAlmostEqual(float(over[2]), float(CENTER[2] + SIZE[2] + 0.10))
        self.assertLess(float(np.linalg.norm(over[:2] - CENTER[:2])), 0.15)
        liner = np.array([float(CENTER[0]), float(CENTER[1]), float(CENTER[2] + 0.05)])
        self.assertFalse(tcp_over_opening_above_rim(liner, min_clearance_m=0.0))
        lo, hi = basket_chassis_aabb_m()
        np.testing.assert_allclose(hi[2], CENTER[2] + SIZE[2])
        near = easy_start_local_m(0.0)
        far = easy_start_local_m(1.0)
        self.assertFalse(tcp_over_opening_above_rim(near))
        self.assertAlmostEqual(easy_start_side_y_m(), 0.10)
        self.assertAlmostEqual(easy_start_side_y_m(side_y_m=0.0), 0.0)
        self.assertAlmostEqual(float(near[1]), float(CENTER[1] + 0.10))
        self.assertTrue(tcp_outside_basket(near, margin_m=0.04, above_rim_m=0.0))
        self.assertTrue(tcp_outside_basket(far, margin_m=0.04, above_rim_m=0.0))
        self.assertGreater(float(near[0]), float(hi[0]))
        self.assertGreater(float(far[0]), float(near[0]))
        self.assertGreater(float(np.linalg.norm(far[:2] - CENTER[:2])),
                           float(np.linalg.norm(near[:2] - CENTER[:2])))
        self.assertEqual(easy_start_far_frac(0), 0.0)
        self.assertAlmostEqual(easy_start_far_frac(100), 0.05)
        self.assertEqual(easy_start_far_frac(500), 0.25)
        self.assertEqual(easy_start_far_frac(800), 0.25)
        self.assertEqual(easy_start_far_frac(2000), 0.25)
        self.assertEqual(easy_start_far_frac(100, horizon=200, cap=1.0), 0.5)
        starts = sample_easy_start_indices(20, 4, 4, 0.25, np.random.default_rng(4))
        self.assertEqual(int(np.count_nonzero(starts == 4)), 15)
        self.assertEqual(int(np.count_nonzero(starts != 4)), 5)
        cohort = np.zeros(20, dtype=np.int32)
        cohort[:3] = 1
        starts = sample_easy_start_indices(
            20, 4, 4, 1.0, np.random.default_rng(4), cohort=cohort)
        np.testing.assert_array_equal(starts[:3], np.full(3, 4, dtype=np.int32))
        with self.assertRaises(ValueError):
            easy_start_far_frac(0, cap=1.5)
        self.assertAlmostEqual(easy_teacher_mix(0, horizon=60), 1.0)
        self.assertAlmostEqual(easy_teacher_mix(30, horizon=60), 0.5)
        self.assertAlmostEqual(easy_teacher_mix(60, horizon=60), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(90, horizon=60), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(0, start_mix=0.0), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(3), 0.5)
        self.assertAlmostEqual(easy_teacher_mix(6), 0.0)
        self.assertAlmostEqual(easy_teacher_mix(10, anneal_after=20, horizon=60), 1.0)
        self.assertAlmostEqual(easy_teacher_mix(20, anneal_after=20, horizon=60), 1.0)
        self.assertAlmostEqual(easy_teacher_mix(50, anneal_after=20, horizon=60), 0.5)
        self.assertAlmostEqual(easy_teacher_mix(80, anneal_after=20, horizon=60), 0.0)
        with self.assertRaises(ValueError):
            easy_teacher_mix(0, anneal_after=-1)
        fracs = hold_close_fracs()
        self.assertEqual(len(fracs), 10)
        self.assertAlmostEqual(float(fracs[0]), 0.25)
        self.assertAlmostEqual(float(fracs[-1]), 0.70)
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
            self.assertFalse(tcp_over_opening_above_rim(pose))
            self.assertGreaterEqual(float(pose[0]), float(hi[0] + 0.32) - 1e-9)
            self.assertLessEqual(float(pose[0]), float(hi[0] + 0.32 + 0.08) + 1e-9)
            self.assertGreaterEqual(float(pose[2]), float(hi[2] + 0.10) - 1e-9)
            self.assertLessEqual(float(pose[2]), float(hi[2] + 0.10 + 0.08) + 1e-9)
            self.assertGreaterEqual(abs(float(pose[1]) - float(CENTER[1])), 0.10 - 1e-9)
            self.assertLessEqual(abs(float(pose[1]) - float(CENTER[1])), 0.10 + 0.04 + 1e-9)
        over_samples = [easy_over_opening_local_m(rng) for _ in range(8)]
        for pose in over_samples:
            self.assertTrue(tcp_over_opening_above_rim(pose))
            self.assertFalse(tcp_outside_basket(pose, margin_m=0.04, above_rim_m=0.0))
            self.assertLess(float(np.linalg.norm(pose[:2] - CENTER[:2])), 0.15)
            self.assertGreaterEqual(float(pose[2]), float(hi[2] + 0.10) - 1e-9)
            self.assertLessEqual(float(pose[2]), float(hi[2] + 0.10 + 0.08) + 1e-9)
        with self.assertRaises(ValueError):
            easy_over_opening_local_m(clearance_m=0.0)
        with self.assertRaises(ValueError):
            easy_over_opening_local_m(radius_m=0.20)
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_runtime.py').read_text(encoding='utf-8')
        self.assertIn('def _apply_easy_start', src)
        self.assertIn('def _privileged_carry_action', src)
        self.assertIn('def set_carry_progress', src)
        self.assertIn('def _build_carry_catalog', src)
        self.assertIn('safe_hover_arm_q', src)
        self.assertIn('def _sample_grasp_locals', src)
        self.assertIn('random_grasp_offset_local_m', src)
        self.assertIn('_tcp_local_host', src)
        self.assertIn('grasp_offset_std_m', src)
        self.assertIn('plan_carry_joint_path', src)
        self.assertIn('def _apply_easy_jaw_hold', src)
        self.assertIn('def _scripted_jaw_hold', src)
        self.assertIn('def _pin_scripted_jaw', src)
        self.assertIn('def _adapt_scripted_jaw', src)
        self.assertIn('def _in_release_zone', src)
        self.assertIn('self._hold_sweep_q', src)
        self.assertIn('margin_m=HOLD_SWEEP_MARGIN_M', src)
        self.assertIn('clearance_m=HOLD_SWEEP_CLEARANCE_M', src)
        self.assertIn('self._shaping_length', src)
        self.assertIn('self._deposit_w', src)
        self.assertIn('self._fail_w', src)
        self.assertIn('timed_out[world] != 0 and success[world] == 0', src)
        self.assertIn('def _prefer_hover_after_success', src)
        self.assertIn('self._hover_start_index', src)
        self.assertIn('self._easy_catalog_n', src)
        self.assertIn('self._easy_hover_cohort', src)
        self.assertIn('sample_easy_start_indices', src)
        self.assertIn('self._easy_released', src)
        self.assertIn("'release_fired'", src)
        self.assertIn('def clear_easy_hover_starts', src)
        self.assertLess(
            src.index('self._prefer_hover_after_success(mask)'),
            src.index('self.task.reset(mask_wp)'),
        )
        self.assertIn('[1, 10000]', src)
        self.assertIn('[-10000, 0]', src)
        self.assertIn('[0, 50]', src)
        self.assertIn('side_y_m=0.0', src)
        self.assertIn('start_over_opening', src)
        self.assertIn('easy_over_opening_local_m', src)
        self.assertIn('tcp_over_opening_above_rim', src)
        self.assertIn('start_qs.append(np.asarray(drop_q', src)
        self.assertIn('shape_hand_fruit', src)
        self.assertIn('release_at_center', src)
        self.assertIn('release_over_opening', src)
        self.assertIn('self._open_half_xy', src)
        self.assertIn('self._open_max_above_rim_m', src)
        self.assertIn('open_max_above_rim_m', src)
        self.assertIn('far_frac_cap', src)
        self.assertIn('self._hover_offset', src)
        self.assertIn('self._release_offset', src)
        self.assertIn('potential_ref = 2 if aligned else 1', src)
        self.assertIn('target_world', src)
        self.assertIn('self._easy_pin', src)
        self.assertIn('qpos[world, jaw_qposadr] = hold', src)
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
        self.assertIn('def easy_start_side_y_m', teacher_src)
        self.assertIn('def easy_over_opening_local_m', teacher_src)
        self.assertIn('def carry_waypoints_local_m', teacher_src)
        self.assertIn('def random_carry_start_local_m', teacher_src)
        self.assertIn('def random_grasp_offset_local_m', teacher_src)
        self.assertIn('def wrist_clears_crate', teacher_src)
        self.assertIn('def plan_carry_joint_path', teacher_src)
        self.assertIn('def solve_tcp_axis', teacher_src)
        self.assertIn('def tcp_over_opening_above_rim', teacher_src)
        self.assertIn('def at_basket_center', teacher_src)
        self.assertIn('def over_opening_xy', teacher_src)
        self.assertIn('def opening_half_xy_m', teacher_src)
        self.assertIn('def level_wrist_local_m', teacher_src)
        self.assertNotIn('push_tcp_outside_basket', inspect.getsource(easy_over_opening_local_m))
        self.assertIn('def scripted_jaw_target', teacher_src)
        self.assertIn('def fruit_in_release_zone', teacher_src)
        self.assertIn('def adapt_scripted_hold_q', teacher_src)
        self.assertIn('def grasp_pocket_world_m', teacher_src)
        self.assertIn('def _pad_geom_ids', teacher_src)
        self.assertIn('def axial_mouth_local', teacher_src)
        self.assertIn('def jaw_open_closed_q', teacher_src)
        self.assertIn('def jaw_open_closed_from_gaps', teacher_src)
        self.assertIn('def pad_center_gap_m', teacher_src)
        self.assertIn('_hand_fruit_contact_load_n', teacher_src)
        self.assertIn('_apply_jaw_close_ctrl', teacher_src)
        self.assertIn('jaw_actuator', teacher_src)
        self.assertIn('inset_m=0.0', teacher_src)
        self.assertIn('warmup', teacher_src)
        self.assertIn('kinematic hold; actuator tracks', teacher_src)
        self.assertIn('jaw_open', teacher_src)
        self.assertIn('def _freejoint_addrs', teacher_src)
        self.assertIn('hand-frame', teacher_src)
        self.assertIn('jaw_open_closed_q', src)
        self.assertNotIn('self._jaw_closed = float(jaw_range[0]', src)
        self.assertIn('model.opt.timestep', teacher_src)
        self.assertIn('fruit_equality', teacher_src)
        self.assertIn('if over == 1:', src)
        self.assertIn('out_applied[world, joint] = 0.0', src)
        scene_src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_scene.py').read_text(encoding='utf-8')
        self.assertIn('def reattach_basket_collision_geoms', scene_src)
        self.assertIn('def enable_arm_basket_contact_pairs', scene_src)
        self.assertIn('basket_shell', scene_src)
        with self.assertRaises(ValueError):
            hover_tcp_local_m(0.0)
        with self.assertRaises(ValueError):
            hover_tcp_local_m(-0.1)

    def test_hover_cohort_keeps_success_worlds_on_hover_row(self):
        idx = apply_easy_hover_cohort([3, 11, 7, 0], [0, 1, 0, 1], 24)
        np.testing.assert_array_equal(idx, np.array([3, 24, 7, 24], dtype=np.int32))
        same = apply_easy_hover_cohort([2, 5], [0, 0], 24)
        np.testing.assert_array_equal(same, np.array([2, 5], dtype=np.int32))
        with self.assertRaises(ValueError):
            apply_easy_hover_cohort([1, 2], [1], 24)
        with self.assertRaises(ValueError):
            apply_easy_hover_cohort([1], [1], -1)

    def test_ik_demo_preset_keeps_fruit_free_and_mixes_starts(self):
        from treesim.kiwi_rl.curriculum import (
            IK_DEMO_PRESET, apply_ik_demo_preset, sample_carry_start_indices,
        )
        preset = apply_ik_demo_preset({'gate_success_rate': 0.90})
        self.assertEqual(preset['demo_updates'], 16)
        self.assertEqual(preset['rl_updates'], 4)
        self.assertEqual(preset['updates'], 20)
        self.assertEqual(preset['n_start_poses'], 48)
        self.assertEqual(preset['hard_start_frac'], 0.5)
        self.assertEqual(preset['grasp_inset_span_m'], 0.05)
        self.assertEqual(preset['grasp_lateral_span_m'], 0.018)
        self.assertEqual(preset['release_target_inset_x_m'], 0.0)
        self.assertGreater(preset['transit_clearance_m'], preset['release_clearance_m'])
        self.assertEqual(preset['gate_success_rate'], 0.90)
        self.assertNotIn('weld', IK_DEMO_PRESET)
        idx = sample_carry_start_indices(200, 48, np.random.default_rng(0))
        self.assertEqual(idx.shape, (200,))
        self.assertGreater(int(idx.max()), 10)
        self.assertGreater(int(np.unique(idx).size), 20)
        with self.assertRaises(ValueError):
            sample_carry_start_indices(0, 8, np.random.default_rng(0))

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
