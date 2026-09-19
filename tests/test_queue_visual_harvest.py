import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.queue_visual_harvest import heldout_succeeded, run_stage, select_load_args, stages, trainer_pids


class QueueVisualHarvestTest(unittest.TestCase):
    def test_curriculum_order_is_walk_then_one_fruit_then_three(self):
        names = [stage['name'] for stage in stages()]
        self.assertEqual(names, ['walk-grab', 'collect-1', 'collect-3'])
        grab, one, three = stages()
        self.assertEqual(grab['args'][grab['args'].index('--episode-seconds')+1], '30')
        self.assertEqual(one['args'][one['args'].index('--picks')+1], '1')
        self.assertEqual(three['args'][three['args'].index('--picks')+1], '3')
        self.assertNotIn('--picks', grab['args'])
        self.assertTrue(all('--stationary' not in stage['args'] for stage in stages()))
        self.assertTrue(all('--vision' in stage['args'] for stage in stages()))

    def test_failed_previous_stage_falls_back_to_stationary_encoder(self):
        stationary = Path('/tmp/stationary-policy.zip')
        self.assertEqual(select_load_args(None, stationary),
                         ['--encoder-warm-start', str(stationary.resolve())])
        with tempfile.TemporaryDirectory() as raw:
            previous = Path(raw)
            (previous/'summary.json').write_text(json.dumps({
                'recommended_checkpoint': 'policy-00008192.zip',
                'heldout': [{'trained_successes': 0, 'episodes': 8}],
            }))
            (previous/'policy-00008192.zip').write_bytes(b'ckpt')
            self.assertEqual(select_load_args(previous, stationary),
                             ['--encoder-warm-start', str(stationary)])

    def test_successful_heldout_uses_full_weight_warm_start(self):
        stationary = Path('/tmp/stationary-policy.zip')
        with tempfile.TemporaryDirectory() as raw:
            previous = Path(raw)
            (previous/'summary.json').write_text(json.dumps({
                'recommended_checkpoint': 'policy-00012288.zip',
                'heldout': [{'trained_successes': 3, 'episodes': 8}],
            }))
            checkpoint = previous/'policy-00012288.zip'
            checkpoint.write_bytes(b'ckpt')
            self.assertEqual(select_load_args(previous, stationary),
                             ['--warm-start', str(checkpoint.resolve())])
            self.assertTrue(heldout_succeeded(json.loads((previous/'summary.json').read_text())))

    def test_wait_for_pids_does_not_signal_processes(self):
        from scripts.queue_visual_harvest import wait_for_pids
        with tempfile.TemporaryDirectory() as raw:
            log = Path(raw)/'progress.jsonl'
            log.touch()
            with mock.patch('scripts.queue_visual_harvest.alive', side_effect=[True, False]), \
                 mock.patch('scripts.queue_visual_harvest.time.sleep') as sleep, \
                 mock.patch('scripts.queue_visual_harvest.os.kill') as kill:
                wait_for_pids([4321], log, poll_s=5)
            sleep.assert_called_once_with(5)
            kill.assert_not_called()
            events = [json.loads(line)['event'] for line in log.read_text().splitlines()]
            self.assertEqual(events, ['waiting', 'wait_heartbeat', 'wait_complete'])

    def test_trainer_pid_scan_ignores_the_queue_script(self):
        with tempfile.TemporaryDirectory() as raw:
            proc = Path(raw)
            (proc/'1'/'cmdline').parent.mkdir()
            (proc/'1'/'cmdline').write_bytes(b'python\0scripts/queue_visual_harvest.py\0')
            (proc/'9'/'cmdline').parent.mkdir()
            (proc/'9'/'cmdline').write_bytes(b'python\0scripts/train_assisted_kiwi.py\0--vision\0')
            with mock.patch.object(Path, 'glob', return_value=[proc/'1'/'cmdline', proc/'9'/'cmdline']):
                self.assertEqual(trainer_pids(), [9])
                self.assertEqual(trainer_pids(exclude=(9,)), [])

    @unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'Needs torch')
    def test_encoder_walking_init_starts_at_zero_mean(self):
        import math
        import torch
        from types import SimpleNamespace
        from scripts.train_assisted_kiwi import initialize_walking_policy
        policy = SimpleNamespace(action_net=torch.nn.Linear(8, 10),
                                 log_std=torch.nn.Parameter(torch.ones(10)))
        torch.nn.init.constant_(policy.action_net.weight, .4)
        torch.nn.init.constant_(policy.action_net.bias, .4)
        initialize_walking_policy(policy, .3, .4, encoder_transfer=True)
        self.assertEqual(float(policy.action_net.weight.abs().sum()), 0.)
        self.assertEqual(float(policy.action_net.bias.abs().sum()), 0.)
        self.assertAlmostEqual(float(policy.log_std[0]), math.log(.15))
        self.assertAlmostEqual(float(policy.log_std[5]), math.log(.15))
        self.assertAlmostEqual(float(policy.log_std[-1]), math.log(.4))
        initialize_walking_policy(policy, .3, .4, encoder_transfer=False)
        self.assertAlmostEqual(float(policy.log_std[0]), math.log(.3))

    def test_queue_does_not_precreate_the_trainer_output_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root/'visual-walk-grab-02'
            log = root/'progress.jsonl'
            log.touch()
            with mock.patch('scripts.queue_visual_harvest.subprocess.run',
                            return_value=mock.Mock(returncode=1)) as launched, \
                 mock.patch('scripts.queue_visual_harvest.emit'):
                with self.assertRaises(RuntimeError):
                    run_stage(Path('python'), root, root, output, ('--vision',),
                              ['--encoder-warm-start', 'ckpt.zip'], log)
            launched.assert_called_once()
            self.assertFalse(output.exists())
            self.assertIn('--output', launched.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
