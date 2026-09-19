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
        self.assertIn('easy_teacher_mix', source)
        self.assertIn('--speedrun', source)
        self.assertIn('--easy', source)
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

    def test_easy_cli_fills_teacher_mix_but_keeps_explicit_zero(self):
        import argparse
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import apply_easy_cli
        filled = apply_easy_cli(argparse.Namespace(
            easy=True, teacher_mix=None, shaping_coef=None))
        self.assertEqual(filled.teacher_mix, 0.0)
        self.assertEqual(filled.shaping_coef, 5.0)
        kept = apply_easy_cli(argparse.Namespace(
            easy=True, teacher_mix=0.0, shaping_coef=2.0))
        self.assertEqual(kept.teacher_mix, 0.0)
        self.assertEqual(kept.shaping_coef, 2.0)
        off = apply_easy_cli(argparse.Namespace(
            easy=False, teacher_mix=None, shaping_coef=None))
        self.assertEqual(off.teacher_mix, 0.0)
        self.assertEqual(off.shaping_coef, 2.0)
        import inspect
        import train_fast
        self.assertNotIn('privileged_deposit_action', inspect.getsource(train_fast.evaluate_mission))
        collect_src = inspect.getsource(train_fast.collect)
        self.assertIn("row['ground_contact']", collect_src)
        self.assertIn("row['fallen']", collect_src)
        self.assertIn("row['hand_load_N']", collect_src)
        run_src = inspect.getsource(train_fast.run)
        self.assertIn('ground_contact_worlds', run_src)
        self.assertIn('basket_distance_mean_m', run_src)
        self.assertIn('easy_far_frac', run_src)
        self.assertIn('easy_teacher_mix', run_src)
        self.assertIn('teacher_anneal_after', run_src)
        self.assertIn('teacher_mask', collect_src)
        self.assertIn('easy_hold_close_mean', run_src)
        self.assertIn('hold_close_frac', run_src)
        runtime_src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_runtime.py').read_text(encoding='utf-8')
        self.assertIn('def _scripted_jaw_hold', runtime_src)
        self.assertIn('def _pin_scripted_jaw', runtime_src)
        self.assertIn('def _in_release_zone', runtime_src)
        self.assertIn('self._shaping_length', runtime_src)
        self.assertIn('self._deposit_w', runtime_src)
        self.assertIn('HOLD_SWEEP_CLEARANCE_M', runtime_src)
        self.assertIn('grasp_local_m', run_src)
        self.assertIn('hold_sweep_rows', run_src)
        self.assertIn('set_easy_progress', run_src)
        self.assertIn('hand_load_max_N', run_src)
        self.assertIn('latest.pt', run_src)
        self.assertIn('nonfinite_worlds', run_src)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_privileged_mix_uses_atanh_of_teacher_action(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import mix_privileged_actions
        raw = torch.zeros(2, 7)
        teacher = torch.full((2, 7), 0.5)
        mixed = mix_privileged_actions(raw, teacher, torch.tensor([True, False]))
        self.assertAlmostEqual(float(mixed[0, 0]), float(torch.atanh(torch.tensor(0.5))), places=5)
        self.assertEqual(float(mixed[1, 0]), 0.0)
        raw10 = torch.zeros(2, 10)
        raw10[:, :3] = 0.25
        mixed10 = mix_privileged_actions(raw10, teacher, torch.tensor([True, False]))
        self.assertEqual(float(mixed10[0, 0]), 0.25)
        self.assertAlmostEqual(float(mixed10[0, 3]), float(torch.atanh(torch.tensor(0.5))), places=5)
        self.assertEqual(float(mixed10[1, 3]), 0.0)


class FastRuntimeFaultTest(unittest.TestCase):
    def test_drain_faults_recovers_sparse_nonfinite(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / 'treesim' / 'kiwi_rl' / 'fast_runtime.py').read_text(encoding='utf-8')
        self.assertIn('FLAG_NONFINITE | FLAG_OVERFLOW | FLAG_BAD_ACTION', src)
        self.assertIn('NONFINITE_ABORT_FRACTION', src)
        self.assertIn('nonfinite_worlds', src)
        self.assertNotIn('flags={self._flags.numpy().tolist()}', src)


if __name__ == '__main__':
    unittest.main()
