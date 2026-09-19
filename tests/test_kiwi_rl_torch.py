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


class ContinuingGAETest(unittest.TestCase):
    def test_rollout_boundary_bootstraps(self):
        from treesim.kiwi_rl.ppo import compute_gae
        result = compute_gae([0.], [0.], [False], [False], .99, .95, final_value=5.)
        self.assertAlmostEqual(float(result[0]), 4.95, places=5)

    def test_timeout_does_not_leak_next_episode(self):
        from treesim.kiwi_rl.ppo import compute_gae
        result = compute_gae([0., 100.], [0., 0.], [False, True], [True, False],
                             .99, .95, next_values=[5., 0.])
        self.assertAlmostEqual(float(result[0]), 4.95, places=5)
        self.assertAlmostEqual(float(result[1]), 100.)

    def test_missing_midrollout_timeout_value_rejected(self):
        from treesim.kiwi_rl.ppo import compute_gae
        with self.assertRaises(ValueError):
            compute_gae([0., 1.], [0., 0.], [False, True], [True, False], .99, .95)


@requires_torch
class LearningRegressionTest(unittest.TestCase):
    def test_vision_parameters_registered_and_updated(self):
        from treesim.kiwi_rl.models_torch import build_v3
        model = build_v3()
        self.assertGreater(sum(p.numel() for p in model.parameters()), 1000)
        self.assertGreater(len(model.state_dict()), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = next(model.parameters()).detach().clone()
        loss = model(torch.randn(1, 5, 64, 64)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.parameters())))
        clone = build_v3()
        clone.load_state_dict(model.state_dict())
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[name]))

    def test_logprob_matches_transformed_distribution(self):
        import numpy as np
        from torch.distributions import Normal, TransformedDistribution, TanhTransform
        from treesim.kiwi_rl.ppo import tanh_logprob
        from treesim.kiwi_rl.models import gaussian_logprob_tanh
        u = torch.tensor([[1., -.5, 2.]], dtype=torch.float64)
        mu, ls = torch.zeros_like(u), torch.zeros(3, dtype=torch.float64)
        reference = TransformedDistribution(Normal(mu, ls.exp()), [TanhTransform()])
        expected = reference.log_prob(u.tanh()).sum(-1)
        torch.testing.assert_close(tanh_logprob(u, mu, ls), expected)
        np.testing.assert_allclose(gaussian_logprob_tanh(u.numpy(), mu.numpy(), ls.numpy()),
                                   expected.numpy(), atol=1e-6)
        mask = torch.tensor([1., 0., 1.], dtype=torch.float64)
        sliced = tanh_logprob(u[:, [0, 2]], mu[:, [0, 2]], ls[[0, 2]])
        torch.testing.assert_close(tanh_logprob(u, mu, ls, dim_mask=mask), sliced)

    def test_saturated_logprob_stays_finite(self):
        from treesim.kiwi_rl.ppo import tanh_logprob
        u = torch.tensor([[100., -100.]], requires_grad=True)
        loss = tanh_logprob(u, torch.zeros_like(u), torch.zeros(2)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(u.grad).all())


if __name__ == "__main__":
    unittest.main()
