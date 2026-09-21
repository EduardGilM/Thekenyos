import unittest
from pathlib import Path


class HarvestContinueLaunchTest(unittest.TestCase):
    def test_live_profile_refuses_sentinels_and_harvest14(self):
        text = Path('scripts/launch_harvest_continue_3072.sh').read_text(encoding='utf-8')
        self.assertIn('WORLD=3072', text)
        self.assertIn('STEPS=64', text)
        self.assertIn('MB=384', text)
        self.assertIn('refusing 4096 worlds or 512 minibatch', text)
        self.assertIn('harvest14', text)
        self.assertIn('harvest5-checkpoint-0008', text)
        self.assertIn('--eval-profile speedrun', text)
        self.assertIn('--demo-updates 0', text)
        train_cmd = text.split('scripts/train_fast.py', 1)[1].split('watch_training.py', 1)[0]
        self.assertNotIn('--easy', train_cmd)
        self.assertNotIn('pkill jupyter', text.lower())
        self.assertIn('Never touch Jupyter 8080', text)
        self.assertIn('watch_training.py', text)
