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
        self.assertIn('entropy_per_dim', result)
        self.assertIn('entropy_gaussian', result)
        self.assertEqual(result['entropy_kind'], 'tanh_gaussian_differential_nats')

    def test_evaluation_reports_mean_of_world_minima(self):
        import torch
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import evaluate
        rows = [dict(distance=torch.tensor(values), terminated=torch.tensor([False, False]),
                     success=torch.tensor([0, 0])) for values in ([.4, .8], [.1, .5])]
        with patch('train_fast.collect', return_value=(rows, None, {})):
            result = evaluate(None, None, None, 2, 2)
        self.assertAlmostEqual(result['evaluation/closest_distance_m'], .1)
        self.assertAlmostEqual(result['evaluation/mean_closest_distance_m'], .3)


class FastTrainerCLITest(unittest.TestCase):
    def test_single_cli_defaults_to_deposit_pixels(self):
        import inspect
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import train_fast
        source = inspect.getsource(train_fast)
        self.assertEqual(source.count('\ndef main('), 1)
        self.assertEqual(source.count('\ndef run('), 1)
        self.assertIn("default='deposit_pixels'", source)
        self.assertIn('curriculum_stage', inspect.getsource(train_fast.run))
        self.assertIn('evaluate_mission', source)
        self.assertIn('drain_faults', source)
        self.assertIn('--speedrun', source)
        self.assertIn('should_persist_checkpoint', source)
        run_src = inspect.getsource(train_fast.run)
        self.assertIn('fruit-count', run_src)
        self.assertIn('curriculum_blocked', run_src)
        self.assertIn('eval_profile', run_src)
        self.assertNotIn('remaining curriculum through stage 6 needs', run_src)

    def test_speedrun_checkpoint_stride_skips_idle_updates(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import should_persist_checkpoint
        kw = dict(updates=2000, eval_every=100, checkpoint_every=50, video_every=50)
        self.assertFalse(should_persist_checkpoint(1, promoted=False, **kw))
        self.assertTrue(should_persist_checkpoint(50, promoted=False, **kw))
        self.assertTrue(should_persist_checkpoint(100, promoted=False, **kw))
        self.assertTrue(should_persist_checkpoint(7, promoted=True, **kw))
        self.assertTrue(should_persist_checkpoint(2000, promoted=False, **kw))
        self.assertTrue(should_persist_checkpoint(1, updates=2000, eval_every=50,
                                                 checkpoint_every=1, video_every=10, promoted=False))

    def test_speedrun_keeps_explicit_zero_video_every(self):
        import argparse
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import apply_speedrun_cli
        kept = apply_speedrun_cli(argparse.Namespace(
            speedrun=True, video_every=0, eval_every=50, checkpoint_every=1,
            entropy_coef=0.005, eval_profile='default', mask_idle_locomotion=True))
        self.assertEqual(kept.eval_every, 100)
        self.assertEqual(kept.checkpoint_every, 50)
        self.assertEqual(kept.entropy_coef, 0.01)
        self.assertEqual(kept.eval_profile, 'speedrun')
        self.assertEqual(kept.video_every, 0)
        filled = apply_speedrun_cli(argparse.Namespace(
            speedrun=True, video_every=10, eval_every=50, checkpoint_every=1,
            entropy_coef=0.005, eval_profile='default', mask_idle_locomotion=True))
        self.assertEqual(filled.video_every, 50)


if __name__ == '__main__':
    unittest.main()
