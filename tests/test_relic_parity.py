"""G1 parity gate helpers (numpy-only parts run everywhere).

The full ONNX gate needs onnxruntime + the external RELIC checkout and
is skipped unless RELIC_PATH is set; CI without weights still checks
the sampler contract.
"""
import os
import unittest

import numpy as np

from scripts.check_relic_parity import sample_valid_r84


class ParitySamplerTest(unittest.TestCase):
    def test_valid_range_and_reproducible(self):
        a = sample_valid_r84(64, seed=0)
        b = sample_valid_r84(64, seed=0)
        self.assertEqual(a.shape, (64, 84))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a, b)
        # gravity slice is unit norm
        n = np.linalg.norm(a[:, 6:9], axis=1)
        np.testing.assert_allclose(n, 1.0, rtol=1e-5)


class OnnxGateTest(unittest.TestCase):
    def test_trainable_import_matches_nonzero_observations(self):
        relic = os.environ.get('RELIC_PATH')
        if not relic:
            self.skipTest('External RELIC checkout required')
        try:
            import torch
        except ImportError:
            self.skipTest('Torch required')
        from pathlib import Path
        from treesim.kiwi_rl.models_torch import load_trainable_relic_actor, load_relic_jit_actor
        path = Path(relic) / 'source/relic/relic/assets/spot/pretrained/policy.pt'
        reference = load_relic_jit_actor(str(path))
        trainable = load_trainable_relic_actor(str(path))
        x = torch.from_numpy(sample_valid_r84(128, 42))
        torch.testing.assert_close(trainable(x), reference(x), rtol=0, atol=1e-5)
        self.assertTrue(all(p.requires_grad for p in trainable.parameters()))
        loss = trainable(x).square().mean()
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in trainable.parameters()), 0.)

    def test_live_onnx_gate(self):
        relic = os.environ.get("RELIC_PATH")
        if not relic:
            self.skipTest("needs external RELIC checkout (RELIC_PATH)")
        from pathlib import Path
        from scripts.check_relic_parity import run_parity
        onnx_path = (Path(relic) / "source/relic/relic/assets/spot"
                     / "pretrained/policy.onnx")
        if not onnx_path.is_file():
            self.skipTest("no weights in external checkout")
        try:
            res = run_parity(Path(relic), samples=32, seed=0)
        except ImportError:
            self.skipTest("needs onnxruntime")
        self.assertTrue(res["pass"], res)


if __name__ == "__main__":
    unittest.main()
