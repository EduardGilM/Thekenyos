import json
import tempfile
import unittest
from pathlib import Path

from treesim.kiwi_rl.training_log import TrainingLog


class FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = []

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def finish(self, exit_code=None):
        self.finished.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


class TrainingLogTests(unittest.TestCase):
    def test_disabled_always_appends_jsonl_without_wandb(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = TrainingLog(tmp, {'exptseed': 7})
            log.log({'loss': 1.5, 'episodes': 2}, step=3)
            log.finish(success=True)
            self.assertEqual(json.loads((Path(tmp) / 'training.jsonl').read_text()),
                             {'step': 3, 'loss': 1.5, 'episodes': 2})

    def test_offline_uses_config_and_explicit_step(self):
        fake = FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            log = TrainingLog(tmp, {'approximations': {'fruit': 'coarse'}, 'worlds': 4, 'rates': {'control_hz': 25}},
                              wandb_mode='offline', wandb_module=fake, wandb_name='smoke')
            log.log({'reward_mean': 2.0}, step=9)
            log.finish(success=False)
        self.assertEqual(fake.calls[0]['dir'], str(Path(tmp).resolve()))
        self.assertEqual(fake.calls[0]['mode'], 'offline')
        self.assertEqual(fake.calls[0]['name'], 'smoke')
        self.assertEqual(fake.calls[0]['config']['exptseed'], None)
        self.assertEqual(fake.run.logged, [({'reward_mean': 2.0}, 9)])
        self.assertEqual(fake.run.finished, [1])

    def test_online_failure_is_not_silently_disabled(self):
        class Unavailable:
            def init(self, **kwargs):
                raise ConnectionError('unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'could not be started'):
                TrainingLog(tmp, wandb_mode='online', wandb_module=Unavailable())

    def test_resume_requires_existing_wandb_identity(self):
        fake = FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            TrainingLog(tmp, wandb_mode='online', wandb_module=fake,
                        wandb_run_id='existing-run')
        self.assertEqual(fake.calls[0]['id'], 'existing-run')
        self.assertEqual(fake.calls[0]['resume'], 'must')

    def test_checkpoint_is_opt_in(self):
        fake = FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            log = TrainingLog(tmp, wandb_mode='offline', wandb_module=fake)
            log.log_checkpoint(Path(tmp) / 'checkpoint.pt')
        self.assertFalse(hasattr(fake.run, 'save'))


if __name__ == '__main__':
    unittest.main()
