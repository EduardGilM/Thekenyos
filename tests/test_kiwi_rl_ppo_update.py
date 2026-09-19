import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from treesim.kiwi_rl import ppo
from treesim.kiwi_rl.models_torch import build_student


@unittest.skipUnless(torch is not None, 'Torch required')
class RecurrentPPOTest(unittest.TestCase):
    def make_batch(self):
        torch.manual_seed(17)
        model = build_student(hidden_size=32, intent_dim=4)
        time, batch = 4, 2
        observations = dict(rgbd=torch.randn(time, batch, 5, 32, 32),
            proprio=torch.zeros(time, batch, 85), context=torch.zeros(time, batch, 14),
            basket=torch.zeros(time, batch, 16), local_map=torch.zeros(time, batch, 3, 64, 64),
            previous_actions=torch.zeros(time, batch, 11), previous_events=torch.zeros(time, batch, 5))
        resets = torch.zeros(time, batch, dtype=torch.bool)
        resets[0] = True
        memory = torch.zeros(batch, 32)
        with torch.no_grad():
            output = ppo.unroll_policy(model, observations, memory, resets)
            result = dict(kind='factual_on_policy', observations=observations, initial_memory=memory,
                          resets=resets, burn_in=1)
            for head in ('n', 'm'):
                samples = [ppo.sample_hybrid({key: value[t] for key, value in output.items()}, head) for t in range(time)]
                result[head] = {key: torch.stack([sample[key] for sample in samples]) for key in ('raw', 'event', 'log_prob')}
                result[head].update(active=torch.ones(time, batch, dtype=torch.bool),
                                    advantage=torch.tensor([[1., -1.], [.8, -.6], [.5, -.2], [.2, -.1]]))
        return model, result

    def test_device_gae_matches_independent_numpy_implementation(self):
        generator = np.random.default_rng(42)
        rewards, values, next_values = [generator.normal(size=(7, 3)).astype(np.float32) for _ in range(3)]
        terminated, truncated = np.zeros((7, 3), dtype=bool), np.zeros((7, 3), dtype=bool)
        terminated[4, 0], truncated[2, 1] = True, True
        expected = ppo.compute_gae(rewards, values, terminated, truncated, .99, .95, next_values=next_values)
        actual = ppo.compute_gae_torch(*map(torch.from_numpy, (rewards, values, next_values, terminated, truncated)), .99, .95)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_real_update_reaches_vision_and_preserves_inactive_head(self):
        model, batch = self.make_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch['n']['active'].zero_()
        vision = next(model.vision.parameters()).detach().clone()
        navigation = model.n_mean.weight.detach().clone()
        result = ppo.recurrent_actor_step(model, optimizer, batch)
        self.assertTrue(result['updated'])
        self.assertEqual(result['samples'], 6)
        self.assertFalse(torch.equal(vision, next(model.vision.parameters())))
        self.assertTrue(torch.equal(navigation, model.n_mean.weight))
        self.assertIsNone(model.n_mean.weight.grad)

    def test_counterfactual_data_is_rejected(self):
        model, batch = self.make_batch()
        batch['kind'] = 'counterfactual'
        with self.assertRaises(ValueError):
            ppo.recurrent_actor_step(model, torch.optim.Adam(model.parameters()), batch)

    def test_peaked_gaussian_entropy_is_negative(self):
        torch.manual_seed(0)
        logstd = torch.full((4, 10), -1.6)
        entropy = ppo.gaussian_entropy(logstd)
        expected = 10 * (0.5 * (1.0 + float(np.log(2.0 * np.pi))) - 1.6)
        self.assertTrue(bool((entropy < 0).all()))
        np.testing.assert_allclose(entropy.numpy(), expected, rtol=1e-5, atol=1e-5)
        mu = torch.zeros(4, 10)
        raw = mu + logstd.exp() * torch.randn_like(mu)
        tanh_h = ppo.tanh_gaussian_entropy(logstd, raw=raw, mu=mu)
        self.assertTrue(bool(torch.isfinite(tanh_h).all()))
        self.assertLess(float(tanh_h.mean()), 0.0)

    def test_kl_guard_stops_before_optimizer_step(self):
        model, batch = self.make_batch()
        batch['m']['log_prob'] += 10.
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = ppo.recurrent_actor_step(model, torch.optim.Adam(model.parameters()), batch)
        self.assertFalse(result['updated'])
        self.assertEqual(result['reason'], 'KL limit')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
