import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from treesim.kiwi_rl import ppo


@unittest.skipUnless(torch is not None, 'Torch required')
class CheckpointTest(unittest.TestCase):
    def test_roundtrip_restores_optimizer_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            model(torch.ones(1, 3)).square().sum().backward()
            optimizer.step()
            path = Path(directory) / 'step-1.pt'
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            ppo.save_checkpoint(path, {'actor': model}, {'actor': optimizer}, {}, {'schema': 'test'})
            with torch.no_grad():
                model.weight.zero_()
            ppo.load_checkpoint(path, {'actor': model}, {'actor': optimizer}, expected_meta={'schema': 'test'})
            for name, value in expected.items():
                torch.testing.assert_close(model.state_dict()[name], value)
            self.assertTrue(optimizer.state)
            with self.assertRaises(ValueError):
                ppo.load_checkpoint(path, {'actor': model}, expected_meta={'schema': 'other'})
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaises(ValueError):
                ppo.load_checkpoint(path, {'actor': model})

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1)
            path = Path(directory) / 'checkpoint.pt'
            ppo.save_checkpoint(path, {'actor': model}, {}, {}, {})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ppo.save_checkpoint(path, {'actor': model}, {}, {}, {})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(json.loads(Path(str(path) + '.json').read_text())['sha256']), 64)
            ppo.save_checkpoint(path, {'actor': model}, {}, {}, {'schema': 'replaced'},
                                replace=True)
            self.assertNotEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(Path(str(path) + '.json').read_text())['schema'], 'replaced')

    def test_single_sample_loss_is_finite(self):
        value = torch.zeros(1, requires_grad=True)
        result = ppo.ppo_epoch_loss(value, value, value.detach(), torch.ones(1),
                                    torch.ones(1), value, .2, 1., 0., 0.)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == '__main__':
    unittest.main()
