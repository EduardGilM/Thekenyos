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

    def test_entropy_expands_narrow_policy_without_rewards(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import update
        from treesim.kiwi_rl.ppo import tanh_logprob

        class ConstantPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.zeros(7))
                self.logstd = torch.nn.Parameter(torch.full((7,), -2.))
                self.value = torch.nn.Parameter(torch.zeros(()))

            def forward(self, rgbd, r84, memory):
                return (self.mean.expand(len(r84), -1), self.logstd,
                        self.value.expand(len(r84)), memory)

        torch.manual_seed(7)
        policy = ConstantPolicy()
        worlds = 1024
        r84 = torch.zeros(worlds, 84)
        with torch.no_grad():
            mean, logstd, value, _ = policy(None, r84, None)
            raw = mean + logstd.exp() * torch.randn_like(mean)
            row = dict(rgbd=torch.zeros(worlds, 1), r84=r84, raw=raw,
                       logp=tanh_logprob(raw, mean, logstd), value=value.clone(),
                       reward=torch.zeros(worlds), terminated=torch.ones(worlds, dtype=torch.bool),
                       reset=torch.ones(worlds, dtype=torch.bool))
        before = policy.logstd.detach().clone()
        result = update(policy, torch.optim.SGD(policy.parameters(), lr=.1), [row],
                        torch.zeros(worlds), gae_lambda=.99, entropy_coef=.001)
        self.assertTrue(bool((policy.logstd > before).all()))
        self.assertAlmostEqual(result['critic_loss'], 0.)
        self.assertAlmostEqual(result['explained_variance'], 0.)
        self.assertEqual(result['gae_lambda'], .99)
        self.assertAlmostEqual(result['action_std/6'], float(before[6].exp()), places=6)
        self.assertTrue(torch.isfinite(torch.tensor(result['entropy'])))

    def test_configurable_gae_and_metrics(self):
        import torch
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import build_policy, update
        from treesim.kiwi_rl.ppo import tanh_logprob, compute_gae_torch
        torch.manual_seed(3)
        policy = build_policy()
        rows = []
        memory = torch.zeros(2, 64)
        with torch.no_grad():
            for t in range(3):
                rgbd, r84 = torch.randn(2, 5, 16, 16), torch.randn(2, 84)
                mean, logstd, value, memory = policy(rgbd, r84, memory)
                raw = mean + logstd.exp() * torch.randn_like(mean)
                rows.append(dict(rgbd=rgbd, r84=r84, raw=raw,
                    logp=tanh_logprob(raw, mean, logstd), value=value,
                    reward=torch.tensor([0., float(t == 2)]),
                    terminated=torch.tensor([t == 2, t == 2]),
                    reset=torch.tensor([t == 0, t == 0])))
        with patch('treesim.kiwi_rl.ppo.compute_gae_torch', wraps=compute_gae_torch) as gae:
            result = update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4),
                            rows, torch.zeros(2), gae_lambda=.99, entropy_coef=.001)
        self.assertEqual(gae.call_args.args[-1], .99)
        for key in ('actor_loss', 'critic_loss', 'explained_variance', 'return_variance', 'entropy'):
            self.assertTrue(torch.isfinite(torch.tensor(result[key])), key)
        self.assertLess(result['kl'], 1e-6)

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
