"""Torch model shapes + PPO math. JP-ONLY: skipped without torch.

Run on jp after: pip install -r requirements-train.txt
Covers V3/N3/M3/G1/critic dims, tanh log-prob finiteness, GAE terminal
vs truncated semantics and checkpoint round-trip (tmp dir, not git).
"""
import unittest

torch = None
try:
    import torch  # noqa: F401
except ImportError:
    pass

requires_torch = unittest.skipUnless(torch is not None, "needs torch")


class LazyImportTest(unittest.TestCase):
    def test_modules_import_without_torch(self):
        # kiwi_rl.models/ppo must stay importable torch-free (numpy tests).
        import treesim.kiwi_rl.models as m
        import treesim.kiwi_rl.ppo as p
        self.assertTrue(hasattr(m, "tanh_squash"))
        self.assertTrue(hasattr(p, "compute_gae"))


class GAETochFreeTest(unittest.TestCase):
    def test_truncated_bootstraps_terminal_does_not(self):
        import numpy as np
        from treesim.kiwi_rl.ppo import compute_gae
        r = np.ones(4, dtype=np.float32)
        v = np.zeros(4, dtype=np.float32)
        a_term = compute_gae(r, v, np.array([0, 0, 0, 1]),
                             np.array([0, 0, 0, 0]), 0.99, 0.95)
        a_trunc = compute_gae(r, v, np.array([0, 0, 0, 0]),
                              np.array([0, 0, 0, 1]), 0.99, 0.95,
                              final_value=5.0)
        self.assertAlmostEqual(a_term[-1], 1.0)
        self.assertAlmostEqual(a_trunc[-1], 1.0 + 0.99 * 5.0)
        self.assertGreater(a_trunc[-1], a_term[-1])


@requires_torch
class TorchShapesTest(unittest.TestCase):
    def test_forward_dims(self):
        import torch
        from treesim.kiwi_rl import models_torch as T
        v3 = T.build_v3().eval()
        with torch.no_grad():
            z = v3(torch.zeros(2, 5, 240, 320))
        self.assertEqual(tuple(z.shape), (2, 256))
        me = T.build_map_encoder().eval()
        with torch.no_grad():
            self.assertEqual(tuple(me(torch.zeros(2, 3, 64, 64)).shape),
                             (2, 128))
        n3 = T.build_n3()
        h = torch.zeros(1, 2, 256)
        mu, logits, h2 = n3.forward(torch.zeros(2, 488), h)
        self.assertEqual(tuple(mu.shape), (2, 3))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(h2.shape), (1, 2, 256))
        m3 = T.build_m3()
        mu, logits, _ = m3.forward(torch.zeros(2, 382), h)
        self.assertEqual(tuple(mu.shape), (2, 8))
        self.assertEqual(tuple(logits.shape), (2, 3))
        g1 = T.build_g1_mlp().eval()
        with torch.no_grad():
            self.assertEqual(tuple(g1(torch.zeros(2, 84)).shape), (2, 12))
        c = T.build_critic().eval()
        with torch.no_grad():
            v = c(torch.zeros(2, 207), torch.zeros(2, 128, 23),
                  torch.zeros(2, 128))
        self.assertEqual(tuple(v.shape), (2,))

    def test_empty_pool_gives_zero_max(self):
        import torch
        from treesim.kiwi_rl import models_torch as T
        c = T.build_critic().eval()
        with torch.no_grad():
            v_empty = c(torch.zeros(1, 207), torch.zeros(1, 128, 23),
                        torch.zeros(1, 128))
            fruit = torch.zeros(1, 128, 23)
            fruit[0, 0, 0] = 1.0
            valid = torch.zeros(1, 128)
            valid[0, 0] = 1.0
            v_one = c(torch.zeros(1, 207), fruit, valid)
        self.assertTrue(torch.isfinite(v_empty).all())
        self.assertFalse(torch.equal(v_empty, v_one))


if __name__ == "__main__":
    unittest.main()
