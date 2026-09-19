"""Reward/v3 ledger: once-per-fruit, damage separate from spill."""
import unittest

import numpy as np

from treesim.kiwi_rl.rewards import (
    RewardEvaluator, RewardState, approach_center_potential, shaping_step,
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
        with self.assertRaises(ValueError):
            approach_center_potential([np.nan, 0, 0], [0, 0, 0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
