"""Reward/v3 ledger: once-per-fruit, damage separate from spill."""
import unittest

import numpy as np

from treesim.kiwi_rl.rewards import (
    CARRY_LINE_POINTS, RewardEvaluator, RewardState, W_CARRY_LINE,
    approach_center_potential, carry_line_waypoints_world_m, pay_carry_line_once,
    shaping_step,
)


class LedgerTest(unittest.TestCase):
    def test_deposit_loss_spill_paid_once(self):
        ev = RewardEvaluator([10, 11])
        self.assertEqual(ev.event_deposit(10), 20.0)
        self.assertEqual(ev.event_deposit(10), 0.0)
        self.assertEqual(ev.event_loss(11), -25.0)
        self.assertEqual(ev.event_loss(11), 0.0)
        self.assertEqual(ev.event_spill(10), -25.0)
        self.assertEqual(ev.event_spill(10), 0.0)

    def test_loss_spill_cross_event_deduplication(self):
        ev = RewardEvaluator([7, 8])
        ev.register_preloaded(7)
        self.assertEqual(ev.event_spill(7), -25.)
        self.assertEqual(ev.event_loss(7), 0.)
        self.assertEqual(ev.event_deposit(7), 0.)
        self.assertEqual(ev.event_loss(8), -25.)
        self.assertEqual(ev.event_spill(8), 0.)
        self.assertEqual(ev.event_deposit(8), 0.)

    def test_preloaded_no_deposit_reward(self):
        ev = RewardEvaluator([7])
        ev.register_preloaded(7)
        self.assertEqual(ev.event_deposit(7), 0.0)
        self.assertEqual(ev.event_spill(7), -25.0)

    def test_damage_separate_and_validated(self):
        ev = RewardEvaluator([1])
        self.assertEqual(ev.damage_increment(0.1), -0.5)
        self.assertAlmostEqual(ev.ledger.damage_sum, 0.1)
        with self.assertRaises(ValueError):
            ev.damage_increment(-1.0)
        with self.assertRaises(ValueError):
            ev.event_deposit(999)

    def test_shaping_resets_baseline_on_ref_change(self):
        st = RewardState()
        r1, _ = shaping_step(st, 1.0, 0.99, "a")
        self.assertEqual(r1, 0.0)
        r2, _ = shaping_step(st, 0.5, 0.99, "a")
        self.assertGreater(r2, 0.0)
        r3, _ = shaping_step(st, 0.5, 0.99, "b")
        self.assertEqual(r3, 0.0)

    def test_approach_center_pays_fruit_and_hand(self):
        far = approach_center_potential([1.0, 0.0, 0.2], [1.0, 0.0, 0.5], [0.0, 0.0, 0.15], 0.60)
        near = approach_center_potential([0.05, 0.0, 0.2], [0.04, 0.0, 0.5], [0.0, 0.0, 0.15], 0.60)
        self.assertGreater(near, far)
        hand_only = approach_center_potential([1.0, 0.0, 0.2], [0.04, 0.0, 0.5], [0.0, 0.0, 0.15], 0.60)
        self.assertGreater(hand_only, far)
        from treesim.basket import CENTER, SIZE
        hover = np.asarray(CENTER, dtype=np.float64) + np.array([0.0, 0.0, float(SIZE[2]) + 0.28])
        at_hover = approach_center_potential(hover, hover, hover, 0.60)
        dive = approach_center_potential(CENTER, CENTER + np.array([0.0, 0.0, 0.10]), hover, 0.60)
        self.assertGreater(at_hover, dive)
        with self.assertRaises(ValueError):
            approach_center_potential([np.nan, 0, 0], [0, 0, 0], [0, 0, 0])

    def test_carry_line_pays_each_point_once(self):
        origin = np.array([0.0, 0.0, 1.5])
        target = np.array([0.0, 0.0, 0.6])
        points = carry_line_waypoints_world_m(origin, target, n_points=4)
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points[-1], target)
        self.assertGreater(float(np.linalg.norm(points[0] - origin)), 0.1)
        reward, mask, hits = pay_carry_line_once(
            points[1], origin, target, 0, n_points=4, radius_m=0.05, bonus=8.0)
        self.assertEqual(reward, 8.0)
        self.assertEqual(hits, 1)
        again, mask2, hits2 = pay_carry_line_once(
            points[1], origin, target, mask, n_points=4, radius_m=0.05, bonus=8.0)
        self.assertEqual(again, 0.0)
        self.assertEqual(hits2, 0)
        self.assertEqual(mask2, mask)
        next_r, mask3, hits3 = pay_carry_line_once(
            points[2], origin, target, mask2, n_points=4, radius_m=0.05, bonus=8.0)
        self.assertEqual(next_r, 8.0)
        self.assertEqual(hits3, 1)
        self.assertNotEqual(mask3, mask2)
        dropped, _, dropped_hits = pay_carry_line_once(
            points[3], origin, target, 0, n_points=4, grasped=False)
        self.assertEqual(dropped, 0.0)
        self.assertEqual(dropped_hits, 0)
        unarmed, _, unarmed_hits = pay_carry_line_once(
            points[0], origin, target, 0, n_points=4, armed=False)
        self.assertEqual(unarmed, 0.0)
        self.assertEqual(unarmed_hits, 0)
        miss, miss_mask, miss_hits = pay_carry_line_once(
            origin + np.array([2.0, 0.0, 0.0]), origin, target, 0, n_points=4)
        self.assertEqual(miss, 0.0)
        self.assertEqual(miss_mask, 0)
        self.assertEqual(miss_hits, 0)
        self.assertEqual(CARRY_LINE_POINTS, 8)
        self.assertEqual(W_CARRY_LINE, 8.0)
        with self.assertRaises(ValueError):
            carry_line_waypoints_world_m([np.nan, 0, 0], target)
        with self.assertRaises(ValueError):
            pay_carry_line_once(points[0], origin, target, 0, radius_m=0.0)


if __name__ == "__main__":
    unittest.main()
