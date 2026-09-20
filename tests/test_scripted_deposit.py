import unittest
import torch
from treesim.kiwi_rl.scripted_deposit import ScriptedDeposit, HIGH_Q, LOW_Q, JAW_HOLD, JAW_OPEN


class Progress:
    def __init__(self, worlds):
        z = torch.zeros(worlds, dtype=torch.bool)
        self.graph_valid_extract = z.clone(); self.graph_grip = z.clone()
        self.graph_completed = {'extract': z.clone()}
        self.carry_start_q = torch.zeros(worlds, 6)


class ScriptedDepositTest(unittest.TestCase):
    def test_engages_only_after_held_extraction_and_follows_both_segments_then_releases(self):
        p = Progress(2); d = ScriptedDeposit(2, 'cpu', max_delta=.01)
        targets = torch.zeros(2, 6); policy = torch.full((2, 7), .3)
        out = d.act(p, targets, policy)
        torch.testing.assert_close(out, policy)
        p.graph_valid_extract[0] = True; p.graph_grip[0] = True; p.graph_completed['extract'][0] = True
        high = torch.tensor(HIGH_Q); low = torch.tensor(LOW_Q)
        for _ in range(2000):
            out = d.act(p, targets, policy)
            self.assertTrue(bool((out[0, :6].abs() <= 1.).all()))
            self.assertAlmostEqual(float(out[0, 6]), JAW_HOLD if not d.released[0] else JAW_OPEN, places=5)
            targets = targets + out[:, :6] * .01
            torch.testing.assert_close(out[1], policy[1])
            if d.released[0]: break
        self.assertTrue(bool(d.released[0]))
        self.assertLess(float((targets[0] - low).abs().max()), .03)
        for _ in range(600):
            out = d.act(p, targets, policy); targets = targets + out[:, :6] * .01
        self.assertLess(float((targets[0] - high).abs().max()), .03, 'Retracts to the high posture after release')
        self.assertAlmostEqual(float(out[0, 6]), JAW_OPEN, places=5)
        self.assertFalse(bool(d.active[1]))
        d.reset(torch.tensor([True, False]))
        self.assertFalse(bool(d.active[0])); self.assertFalse(bool(d.released[0]))

    def test_handover_keeps_a_harder_squeeze_and_starts_from_targets(self):
        p = Progress(1); d = ScriptedDeposit(1, 'cpu', max_delta=.01, settle_steps=3)
        p.graph_valid_extract[0] = True; p.graph_grip[0] = True; p.graph_completed['extract'][0] = True
        targets = torch.full((1, 6), .4)
        out = d.act(p, targets, torch.tensor([[0., 0., 0., 0., 0., 0., .95]]))
        self.assertAlmostEqual(float(out[0, 6]), .95)
        torch.testing.assert_close(out[0, :6], torch.zeros(6), msg='No arm motion during the settle hold')
        torch.testing.assert_close(d.start_q[0], targets[0])
        for _ in range(3): out = d.act(p, targets, torch.tensor([[0., 0., 0., 0., 0., 0., .2]]))
        self.assertAlmostEqual(float(out[0, 6]), .95, msg='The latched hold must not follow a later weaker command')
        self.assertGreater(float(out[0, :6].abs().sum()), 0., 'Motion starts after the settle hold')


if __name__ == '__main__':
    unittest.main()
