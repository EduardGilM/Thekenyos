import os
from pathlib import Path
import unittest
import torch
from tests.test_reward_graph import state
from treesim.kiwi_rl.sequence_curriculum import (Curriculum, SequenceProgress, SEQUENCE_OBSERVATION_DIM,
    LIVE_CREDIT, TIME_BUDGET, FAILURE_COST, POSITION_HOLD_S, BELOW_RIM_STEP_COST)

HOLD = round(POSITION_HOLD_S / .02)


def sample(**values):
    extra = dict(finger_gap=.1, jaw_gap=.1, opposition=0., grasp_time=0., held_at_detach=False,
                 basket_horizontal=1., basket_height=.3, arm_q=[0.] * 6, grasp_depth=0.)
    extra.update(values)
    return state(**extra)


def advance(p, **values):
    now = sample(**values)
    return p.step(now, now['failed'] | now['success'])


POSITION = dict(enclosed=True, insertion=1., finger_gap=.025, jaw_gap=.01, opposition=.8)
GRIP = dict(POSITION, finger_gap=0., jaw_gap=0., bilateral=True, holding=True, grasp=True, grasp_time=.12)


class SequenceTests(unittest.TestCase):
    def test_contact_improves_reward_even_when_old_proxy_decreases(self):
        p = SequenceProgress(sample(), curriculum_stage=2)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **dict(POSITION, finger_gap=.03287, jaw_gap=.00738, closure=.9863))
        reward, done, _, _ = advance(p, **dict(POSITION, finger_gap=.01315, jaw_gap=.00777, closure=.7146))
        self.assertGreater(reward.item(), 0.)
        self.assertFalse(done.item())
        self.assertFalse(p.graph_completed['grip'].item())

    def test_ordered_full_cycle_and_valid_release_does_not_require_grip_forever(self):
        p = SequenceProgress(sample(), curriculum_stage=5)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **GRIP)
        advance(p, **GRIP, detached=True, held_at_detach=True)
        self.assertTrue(p.graph_valid_extract.item())
        OVER = dict(release_distance=.01, basket_horizontal=0., basket_height=.05)
        for _ in range(6): advance(p, **GRIP, detached=True, held_at_detach=True, **OVER)
        advance(p, detached=True, held_at_detach=True, touching=False, **OVER)
        self.assertTrue(p.graph_valid_release.item())
        reward, done, _, _ = advance(p, detached=True, held_at_detach=True, touching=False, basket_horizontal=0., basket_height=-.1)
        self.assertTrue(p.graph_success.item())
        self.assertTrue(p.prefix_success.item())
        self.assertTrue(done.item())
        self.assertGreater(reward.item(), .7)
        self.assertEqual(p.milestone_reward['deposit'].item(), 1.)
        self.assertTrue(all(v.item() for v in p.graph_completed.values()))

    def test_no_unheld_detach_laundering_or_drop_after_grasp(self):
        for had_grip in (False, True):
            p = SequenceProgress(sample(), curriculum_stage=5)
            if had_grip:
                for _ in range(HOLD): advance(p, **POSITION)
                advance(p, **GRIP)
            _, done, _, _ = advance(p, detached=True, held_at_detach=False)
            self.assertTrue(done.item())
            advance(p, **GRIP, detached=True, held_at_detach=False, basket_horizontal=0., basket_height=-.1)
            self.assertFalse(p.graph_success.item())
        p = SequenceProgress(sample(), curriculum_stage=5)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **GRIP)
        advance(p, **GRIP, detached=True, held_at_detach=True)
        for _ in range(7): _, done, _, _ = advance(p, detached=True, held_at_detach=True)
        self.assertTrue(done.item())
        self.assertFalse(p.graph_valid_release.item())

    def test_prefix_termination_history_reset_and_no_camping_income(self):
        p = SequenceProgress(sample(), curriculum_stage=1)
        for _ in range(HOLD): _, done, _, _ = advance(p, **POSITION)
        self.assertTrue(done.item())
        self.assertTrue(p.prefix_success.item())
        self.assertFalse(p.graph_success.item())
        p.curriculum_stage = 2
        p.reset(torch.tensor([True]), sample())
        self.assertEqual(p.curriculum_stage_ids.item(), 2)
        self.assertFalse(p.graph_completed['position'].item())
        self.assertEqual(p.observation().shape, (1, SEQUENCE_OBSERVATION_DIM))
        for _ in range(HOLD + 1): advance(p, **POSITION)
        self.assertTrue(p.graph_completed['position'].item())
        reward, done, _, _ = advance(p, **POSITION)
        self.assertLess(reward.item(), 0.)
        self.assertFalse(done.item())
        torch.testing.assert_close(sum(p.graph_progress_reward.values()),
            p.shaping_reward + sum(p.milestone_reward.values()))

    def test_small_regression_can_recover_without_threshold_penalty_or_new_bonus(self):
        p = SequenceProgress(sample(), curriculum_stage=3, stall_steps=2)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **GRIP)
        lost, done, _, _ = advance(p, **POSITION)
        recovered, _, _, _ = advance(p, **GRIP)
        self.assertFalse(done.item())
        self.assertGreater(lost.item(), -.5)
        self.assertLess((lost + p.gamma * recovered).item(), 0.)
        self.assertTrue(p.graph_completed['grip'].item())
        before = sum(v.item() for v in p.live_components.values())
        for _ in range(10):
            reward, done, _, _ = advance(p, **GRIP)
            if done.item(): break
        self.assertTrue(done.item(), 'Sustained inactivity ends the episode')
        self.assertTrue(p.timeout.item())
        self.assertFalse(p.failure.item())
        self.assertGreater(reward.item(), -.01, 'Stalling keeps the live credit; forfeiting it zeroed the shaping')
        self.assertGreater(sum(v.item() for v in p.live_components.values()), before - .01)

    def test_promotion_requires_repeated_episode_evidence_on_both_scenes(self):
        c = Curriculum()
        good = dict(episodes=32, prefix_success=29/32, physical_failure=0.)
        self.assertFalse(c.consider([good, dict(good, episodes=1)]))
        self.assertFalse(c.consider([good, good]))
        self.assertFalse(c.consider([good, dict(good, prefix_success=.8)]))
        self.assertFalse(c.consider([good, good]))
        self.assertTrue(c.consider([good, good]))
        self.assertEqual(c.level, 2)
        self.assertEqual(Curriculum(**c.state()).state(), c.state())

    def test_losing_position_after_earning_it_costs_reward_at_identical_jaw_distances(self):
        results = []
        for aligned in (True, False):
            p = SequenceProgress(sample(), curriculum_stage=2)
            for _ in range(HOLD): advance(p, **POSITION)
            r, done, _, _ = advance(p, **dict(POSITION, enclosed=aligned,
                insertion=float(aligned), opposition=.8 if aligned else 0.))
            self.assertFalse(done.item())
            self.assertTrue(p.graph_completed['position'].item())
            self.assertEqual(p.milestone_reward['position'].item(), 0.)
            results.append(r.item())
        self.assertLess(results[1], results[0] - .05)

    def test_physical_bilateral_contact_is_not_penalized_by_ambiguous_witness_geometry(self):
        p = SequenceProgress(sample(), curriculum_stage=2)
        for _ in range(HOLD): advance(p, **POSITION)
        reward, done, _, _ = advance(p, bilateral=True, finger_gap=0., jaw_gap=0.,
            grasp_time=.02, insertion=.8, opposition=-1., enclosed=False)
        self.assertGreater(reward.item(), 0.)
        self.assertFalse(done.item())
        self.assertFalse(p.graph_completed['grip'].item())

    def test_events_pay_once_and_do_not_change_after_unlock(self):
        for level in range(1, 6):
            p = SequenceProgress(sample(), curriculum_stage=level)
            paid = 0.
            for _ in range(HOLD + 1):
                _, done, _, _ = advance(p, **POSITION)
                paid += p.milestone_reward['position'].item()
                if done.item(): break
            self.assertEqual(paid, 1.)
            if level > 1:
                advance(p)
                for _ in range(4):
                    advance(p, **POSITION)
                    self.assertEqual(p.milestone_reward['position'].item(), 0.)

    def test_deadline_is_terminal_but_reports_timeout_and_buffer_cuts_do_not_end_task(self):
        from treesim.kiwi_rl.ppo import compute_gae_torch
        p = SequenceProgress(sample(), curriculum_stage=2, max_steps=5)
        for _ in range(4):
            _, done, trunc, _ = advance(p)
            self.assertFalse((done | trunc).item())
        r, done, trunc, _ = advance(p)
        self.assertTrue(done.item())
        self.assertFalse(trunc.item())
        self.assertTrue(p.timeout.item())
        self.assertFalse(p.failure.item())
        a = compute_gae_torch(r[None], torch.zeros(1, 1), torch.full((1, 1), 100.),
            done[None], trunc[None], p.gamma, .99)
        torch.testing.assert_close(a, r[None])

    def test_return_ordering_for_full_event_traces(self):
        # Exercise actual reward code on event traces, without simulating a grasp.
        def run(kind, level):
            p = SequenceProgress(sample(), curriculum_stage=level, stall_steps=2000)
            total = 0.
            initial_live = p.graph_score.item()
            events = 0.
            rim_cost = 0.
            for t in range(1, 1501):
                values = dict(POSITION)
                if kind == 'idle': values = {}
                if kind == 'grip' and t == 1500: values = dict(GRIP)
                if kind == 'fail_position' and t == 4: values['detached'] = True
                if kind == 'position_slide' and t >= 1: values['slip'] = .2
                if kind in ('drop', 'harvest'):
                    if t >= 10: values = dict(GRIP)
                    if t >= 20: values.update(detached=True, held_at_detach=True)
                    if t >= 30: values.update(release_distance=.01, basket_horizontal=0., basket_height=.07)
                    if kind == 'drop' and t == 40:
                        values = dict(detached=True, held_at_detach=True, ground_contact=True)
                    if kind == 'harvest' and t >= 1398:
                        values = dict(detached=True, held_at_detach=True, release_distance=.01, basket_horizontal=0., basket_height=.07)
                    if kind == 'harvest' and t == 1500: values.update(basket_height=-.1, touching=False)
                r, done, trunc, _ = advance(p, **values)
                total += r.item()
                rim_cost += BELOW_RIM_STEP_COST * p._below_rim.item()
                events += sum(v.item() for v in p.milestone_reward.values())
                if (done | trunc).item(): break
            live = sum(v.item() for v in p.live_components.values())
            expected = events + live - initial_live - TIME_BUDGET*t/1500 - FAILURE_COST*p.failure.item() - rim_cost
            self.assertAlmostEqual(total, expected, places=5)
            return total
        self.assertGreater(run('grip', 2), run('fail_position', 2))
        self.assertGreater(run('harvest', 5), run('drop', 5))
        self.assertGreater(run('wait', 2), run('idle', 2))
        self.assertGreater(run('wait', 2), run('fail_position', 2))

    def test_carry_pays_for_progress_along_the_joint_line_and_punishes_dipping_below_the_rim(self):
        from treesim.kiwi_rl.sequence_curriculum import CARRY_TARGET_Q
        target = torch.tensor(CARRY_TARGET_Q)
        p = SequenceProgress(sample(), curriculum_stage=5)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **GRIP)
        advance(p, **GRIP, detached=True, held_at_detach=True)
        for _ in range(8): advance(p, **GRIP, detached=True, held_at_detach=True, arm_q=[.1] * 6)
        self.assertTrue(p.graph_completed['extract'].item())
        torch.testing.assert_close(p.carry_start_q, torch.full((1, 6), .1))
        def quality(**values):
            return p.qualities(sample(**GRIP, detached=True, held_at_detach=True, **values))[3].item()
        start = torch.full((6,), .1)
        on_line = lambda s: (start + s * (target - start)).tolist()
        q0, q_half, q_end = quality(arm_q=on_line(0.)), quality(arm_q=on_line(.5)), quality(arm_q=on_line(1.))
        self.assertGreater(q_half, q0 + .3)
        self.assertGreater(q_end, .95)
        off = (start + .5 * (target - start) + torch.tensor([0., .5, 0., 0., 0., 0.])).tolist()
        self.assertLess(quality(arm_q=off), q_half - .01, 'Leaving the line costs credit')
        back = (start - .1 * (target - start)).tolist()
        self.assertLess(quality(arm_q=back), q0 - .05, 'Moving away from the target along the line costs credit')
        self.assertLess(quality(arm_q=on_line(.5), basket_height=-.1, basket_horizontal=.3), q_half * .3,
                        'Dipping below the rim beside the basket loses most credit')
        r_high, _, _, _ = advance(p, **GRIP, detached=True, held_at_detach=True, arm_q=on_line(.5), basket_horizontal=.3, basket_height=.3)
        r_dip, _, _, _ = advance(p, **GRIP, detached=True, held_at_detach=True, arm_q=on_line(.5), basket_horizontal=.3, basket_height=-.1)
        r_stay, _, _, _ = advance(p, **GRIP, detached=True, held_at_detach=True, arm_q=on_line(.5), basket_horizontal=.3, basket_height=-.1)
        self.assertLess(r_dip, -.05)
        self.assertLess(r_stay, -.001, 'Staying below the rim outside the basket keeps costing')

    def test_tip_grasps_are_not_secure_and_seating_deeper_pays(self):
        p = SequenceProgress(sample(), curriculum_stage=2)
        for _ in range(HOLD): advance(p, **POSITION)
        advance(p, **dict(GRIP, grasp_depth=.03))
        self.assertFalse(p.graph_grip.item(), 'A grasp 3 cm out on the fingertips is not secure')
        self.assertFalse(p.graph_completed['grip'].item())
        shallow = p.qualities(sample(**dict(GRIP, grasp_depth=.03)))[1].item()
        seated = p.qualities(sample(**dict(GRIP, grasp_depth=0.)))[1].item()
        self.assertGreater(seated, shallow + .1, 'Seating the fruit deeper must pay before the grip event')
        advance(p, **dict(GRIP, grasp_depth=.01))
        self.assertTrue(p.graph_grip.item())

    def test_rehearsal_keeps_earlier_objectives_and_resets_without_paying_again(self):
        initial = {k:v.expand(20, *v.shape[1:]).clone() for k,v in sample().items()}
        p = SequenceProgress(initial, curriculum_stage=4, rehearse=True)
        self.assertEqual(int((p.curriculum_stage_ids == 4).sum()), 16)
        self.assertEqual(set(p.curriculum_stage_ids.tolist()), {1, 2, 3, 4})
        p.graph_completed['position'].fill_(True)
        p.reset(torch.ones(20, dtype=torch.bool), initial)
        self.assertFalse(p.graph_completed['position'].any())
        self.assertEqual(int((p.curriculum_stage_ids == 4).sum()), 16)
        self.assertTrue((p.graph_score <= LIVE_CREDIT).all())


@unittest.skipUnless(os.environ.get('FAST_SCENE') and os.environ.get('FAST_STATES'), 'Real scene and recorded qpos required')
class SurfaceRegressionTest(unittest.TestCase):
    def test_native_and_gpu_distances_at_the_recorded_false_optimum(self):
        import numpy as np
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.reward_graph import SEQUENCE_PROFILE
        from treesim.kiwi_rl.surface_distance import JawSurfaceDistance
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            rt = FastRuntime(Path(os.environ['FAST_SCENE']), worlds=2, task_profile=SEQUENCE_PROFILE)
            qpos = np.load(os.environ['FAST_STATES'])['qpos'][86].copy()
            qid = rt.control.contract.qids[-1]
            poses = np.tile(qpos, (2, 1))
            poses[:, qid] = [-.883125007, -.637812495]
            rt.data.qpos.assign(poses)
            mw.forward(rt.gpu_model, rt.data)
            geometry = JawSurfaceDistance(rt)
            measured = geometry.signals(rt)
            # MJWarp uses native CCD. The exported CPU scene disables it;
            # legacy libccd gives an invalid witness outside this jaw hull.
            rt.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            data = mujoco.MjData(rt.model)
            expected = []
            pairs = geometry.pairs.numpy()
            for pose in poses:
                data.qpos[:] = pose
                mujoco.mj_kinematics(rt.model, data)
                expected.append([min(max(0., mujoco.mj_geomDistance(rt.model, data, int(g), int(f), 1., None))
                    for g, f in pairs[group]) for group in geometry.groups])
            got = torch.stack((measured['finger_gap'], measured['jaw_gap']), -1).cpu().numpy()
            np.testing.assert_allclose(got, expected, atol=3.e-4)
            self.assertGreater(got[0, 0], .025)
            self.assertLess(got[1, 0], .02)
            self.assertGreater(1/(1+got[1].sum()/.04), 1/(1+got[0].sum()/.04))
