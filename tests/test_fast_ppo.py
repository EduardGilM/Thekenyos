import importlib.util
from pathlib import Path
import sys
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
class FastPPOTest(unittest.TestCase):
    def test_recurrent_replay_resets_individual_world(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import build_policy, update
        from treesim.kiwi_rl.ppo import tanh_logprob
        torch.manual_seed(42)
        torch.set_num_threads(1)
        policy = build_policy()
        memory = torch.zeros(2,64)
        rows = []
        with torch.no_grad():
            for t in range(4):
                reset = torch.tensor([t in (0,2), t == 0])
                memory *= (~reset)[:,None]
                rgbd, r84 = torch.randn(2,5,16,16), torch.randn(2,84)
                mean, logstd, value, memory = policy(rgbd,r84,memory)
                raw = mean + .1 * torch.randn_like(mean)
                rows.append(dict(rgbd=rgbd, r84=r84, raw=raw,
                    logp=tanh_logprob(raw,mean,logstd), value=value,
                    reward=torch.tensor([t/10.,-t/10.]),
                    terminated=torch.tensor([t==1,False]), reset=reset))
        before = policy.mean.weight.detach().clone()
        result = update(policy, torch.optim.Adam(policy.parameters(),lr=3e-4), rows, torch.zeros(2), minibatch_worlds=1)
        self.assertLess(result['kl'], 1e-6)
        self.assertEqual(result['optimized_transitions'], 8)
        self.assertEqual(result['minibatches'], 2)
        self.assertFalse(torch.equal(before,policy.mean.weight))

    def test_evaluation_reports_mean_of_world_minima(self):
        import torch
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import evaluate
        rows = [dict(distance=torch.tensor(values), terminated=torch.tensor([False, False]),
                     success=torch.tensor([0, 0])) for values in ([.4, .8], [.1, .5])]
        with patch('train_fast.collect', return_value=(rows, None)):
            result = evaluate(None, None, None, 2, 2)
        self.assertAlmostEqual(result['evaluation/closest_distance_m'], .1)
        self.assertAlmostEqual(result['evaluation/mean_closest_distance_m'], .3)

if __name__ == '__main__':
    unittest.main()
