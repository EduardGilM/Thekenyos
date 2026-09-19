import unittest
import torch
from tests.test_reward_graph import state, step, progress
from tests.test_prerequisite_graph import HELD
from treesim.kiwi_rl.reward_graph import SOFT_GRAPH_PROFILE


def fresh(**kwargs):
    return progress(state(insertion=.2), reward_profile=SOFT_GRAPH_PROFILE, **kwargs)


class SoftGraphTest(unittest.TestCase):
    def test_loss_costs_twice_gain_and_recovery_is_not_repaid(self):
        p=fresh()
        gain=step(p,insertion=.4)[0].item()
        loss=step(p,insertion=.2)[0].item()
        recovery=step(p,insertion=.4)[0].item()
        self.assertGreater(gain,0)
        self.assertLess(loss,-2*gain)
        self.assertLess(recovery,0)
        step(p,insertion=.5)
        self.assertGreater(p.shaping_reward.item(),0)
        torch.testing.assert_close(sum(p.graph_progress_reward.values()),p.shaping_reward)

    def test_stall_does_not_erase_real_progress_and_timeout_bootstraps(self):
        p=fresh(stall_steps=1)
        step(p,insertion=.4)
        _,term,_,_=step(p,insertion=.4)
        self.assertTrue(term.item())
        self.assertEqual(p.shaping_reward.item(),0)
        p=fresh(max_steps=1)
        _,term,trunc,_=step(p,insertion=.4)
        self.assertFalse(term.item()); self.assertTrue(trunc.item())
        self.assertGreater(p.shaping_reward.item(),0)

    def test_invalid_extraction_penalized_once_without_early_reset(self):
        p=fresh()
        _,term,_,_=step(p,detached=True)
        self.assertFalse(term.item())
        self.assertLess(p.task_reward.item(),-2)
        self.assertFalse(p.graph_valid_extract.item())
        step(p,detached=True)
        self.assertAlmostEqual(p.task_reward.item(),-.001,places=6)
        _,term,_,_=step(p,detached=True,ground_contact=True)
        self.assertTrue(term.item()); self.assertLess(p.task_reward.item(),-5)
        self.assertFalse(p.graph_success.item())

    def test_controlled_extraction_and_valid_release_still_required(self):
        for valid in (False,True):
            p=fresh()
            for _ in range(6): step(p,**HELD)
            step(p,detached=True,release_distance=.02,**(HELD if valid else {}))
            step(p,detached=True,release_distance=.02)
            step(p,detached=True,settle_time=2.,success=True)
            self.assertEqual(p.graph_success.item(),valid)

    def test_reset_clears_peak_credit_locally(self):
        p=fresh(); step(p,insertion=.8)
        p.reset(torch.tensor([True]),state(insertion=.2))
        step(p,insertion=.4)
        self.assertGreater(p.shaping_reward.item(),0)

if __name__=='__main__': unittest.main()
