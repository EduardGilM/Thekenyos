import unittest
import torch
from tests.test_reward_graph import state,step,progress
from tests.test_prerequisite_graph import HELD
from treesim.kiwi_rl.reward_graph import CONTROL_GRAPH_PROFILE


def fresh(): return progress(state(insertion=.2),reward_profile=CONTROL_GRAPH_PROFILE)


class ControlGraphTest(unittest.TestCase):
    def test_approach_recovery_is_rewarded_without_profitable_cycle(self):
        p=fresh(); step(p,insertion=.5)
        down=step(p,insertion=.3)[0].item()
        up=step(p,insertion=.5)[0].item()
        self.assertLess(down,0);self.assertGreater(up,0)
        self.assertLess(down+p.gamma*up,0)
        self.assertAlmostEqual(p.graph_progress_reward['position'].item(),.999*1.-.6,places=5)

    def test_only_sustained_enclosure_gets_extra_position_loss(self):
        penalties=[]
        for ticks in (1,6):
            p=fresh()
            for _ in range(ticks):step(p,enclosed=True,insertion=1.)
            previous=p.graph_score.item()
            step(p,insertion=.9)
            expected=p.gamma*p.graph_score.item()-previous
            penalties.append(expected-p.shaping_reward.item())
        self.assertAlmostEqual(penalties[0],0,places=5)
        self.assertAlmostEqual(penalties[1],.25,places=5)

    def test_grip_loss_penalized_but_valid_release_is_not(self):
        for release in (False,True):
            p=fresh()
            for _ in range(6):step(p,**HELD)
            if release:step(p,**HELD,detached=True,release_distance=.02)
            previous=p.graph_score.item()
            step(p,enclosed=True,insertion=1.,detached=release,release_distance=.02)
            extra=p.gamma*p.graph_score.item()-previous-p.shaping_reward.item()
            self.assertAlmostEqual(extra,0 if release else .75,places=5)
            torch.testing.assert_close(sum(p.graph_progress_reward.values()),p.shaping_reward)

    def test_unheld_extraction_cannot_become_success(self):
        p=fresh();step(p,detached=True)
        self.assertFalse(p.graph_valid_extract.item())
        step(p,detached=True,success=True,settle_time=2.)
        self.assertFalse(p.graph_success.item())
        self.assertLess(p.task_reward.item(),0)
