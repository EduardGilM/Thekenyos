import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_harvest_fast import resume_history


class ResumeTests(unittest.TestCase):
    def test_resume_preserves_logged_boundary_and_rejects_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint.pt'
            config = {'role': 'teacher', 'worlds': 4096}
            manifest = {'config': config, 'update': 461}
            Path(str(checkpoint) + '.json').write_text(json.dumps(manifest))
            rows = [{'step': 0, 'evaluation/success': 0},
                    {'step': 461, 'update': 461, 'elapsed_seconds': 1787.,
                     'transitions': 120848384, 'completed_episodes': 390107}]
            (root / 'training.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            result = resume_history(root, checkpoint, config)
            self.assertEqual(result['index'], 461)
            self.assertEqual(result['latest']['transitions'], 120848384)
            self.assertEqual(result['latest']['elapsed_seconds'], 1787.)
            with self.assertRaisesRegex(ValueError, 'worlds'):
                resume_history(root, checkpoint, dict(config, worlds=2048))
            manifest['update'] = 460
            Path(str(checkpoint) + '.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'silent rollback'):
                resume_history(root, checkpoint, config)
