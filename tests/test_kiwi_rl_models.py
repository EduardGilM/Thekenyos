"""Model interfaces: exact dims, valid distributions, G1 parity gate."""
import unittest

import numpy as np

from treesim.kiwi_rl import models as M
from treesim.kiwi_rl import schemas as S


class ModelsTest(unittest.TestCase):
    def test_vision_and_map_shapes(self):
        v3 = M.VisionEncoderV3(seed=0)
        z = v3.forward(np.zeros((2, 5, 240, 320), dtype=np.float32))
        self.assertEqual(z.shape, (2, 256))
        me = M.MapEncoder()
        self.assertEqual(
            me.forward(np.zeros((2, 3, 64, 64), dtype=np.float32)).shape,
            (2, 128))
        with self.assertRaises(ValueError):
            v3.forward(np.zeros((2, 3, 240, 320), dtype=np.float32))

    def test_n3_m3_dims_and_logprobs(self):
        n3 = M.make_n3(seed=0)
        h = np.zeros((1, 4, 256), dtype=np.float32)
        mu, logits, h2 = n3.forward(np.zeros((4, 488), dtype=np.float32), h)
        self.assertEqual(mu.shape, (4, 3))
        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(h2.shape, (1, 4, 256))
        m3 = M.make_m3(seed=0)
        mu, logits, _ = m3.forward(np.zeros((4, 382), dtype=np.float32), h)
        self.assertEqual(mu.shape, (4, 8))
        self.assertEqual(logits.shape, (4, 3))
        lp = M.gaussian_logprob_tanh(mu, mu, m3.logstd)
        self.assertTrue(np.isfinite(lp).all() and lp.shape == (4,))
        le = M.categorical_logprob(logits, np.zeros(4, dtype=np.int64))
        self.assertTrue(np.isfinite(le).all())

    def test_scale_aware_parity_preserves_near_zero_accuracy(self):
        reference = np.full((2, 12), 50., dtype=np.float32)
        self.assertTrue(M.onnx_pytorch_parity(reference, reference + 8e-5)['pass'])
        reference.fill(0.)
        self.assertFalse(M.onnx_pytorch_parity(reference, reference + 2e-5)['pass'])
        with self.assertRaises(ValueError):
            M.onnx_pytorch_parity(reference, reference, rtol=1e-3)
        with self.assertRaises(ValueError):
            M.onnx_pytorch_parity(reference, np.full_like(reference, np.nan))

    def test_g1_placeholder_and_parity(self):
        g1 = M.GaitPolicyG1()
        a = g1.act(np.zeros((2, 84), dtype=np.float32))
        self.assertEqual(a.shape, (2, 12))
        q = g1.apply_to_targets(np.zeros(12), a[0])
        self.assertEqual(q.shape, (12,))
        ok = M.onnx_pytorch_parity(a, a.copy())
        self.assertTrue(ok["pass"])
        bad = M.onnx_pytorch_parity(a, a + 1.0)
        self.assertFalse(bad["pass"])
        self.assertEqual(S.R84_DIM, 84)


if __name__ == "__main__":
    unittest.main()
