import unittest
try:
    import torch
except ImportError:
    torch = None

from treesim.kiwi_rl import models_torch as models


def inputs(batch=2):
    return dict(rgbd=torch.randn(batch, 5, 64, 64), proprio=torch.zeros(batch, 85),
                context=torch.zeros(batch, 14), basket=torch.zeros(batch, 16),
                local_map=torch.zeros(batch, 3, 64, 64), previous_actions=torch.zeros(batch, 11),
                previous_events=torch.zeros(batch, 5))


@unittest.skipUnless(torch is not None, 'Torch required')
class SharedBeliefTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_memory_is_shared_and_reset_is_per_environment(self):
        model = models.build_student(hidden_size=64, intent_dim=8)
        obs = inputs()
        h = torch.zeros(2, 64)
        first = model(**obs, memory=h)
        second = model(**obs, memory=first['memory'])
        self.assertFalse(torch.equal(first['m_mu'], second['m_mu']))
        self.assertFalse(torch.equal(first['n_mu'], second['n_mu']))
        reset = model(**obs, memory=first['memory'], reset=torch.tensor([True, False]))
        torch.testing.assert_close(first['memory'][0], reset['memory'][0])
        torch.testing.assert_close(second['memory'][1], reset['memory'][1])

    def test_gradient_reaches_vision_belief_and_intent(self):
        model = models.build_student(hidden_size=64, intent_dim=8)
        out = model(**inputs(), memory=torch.zeros(2, 64))
        (out['m_mu'].square().mean() + out['n_mu'].square().mean()).backward()
        for module in (model.vision, model.belief, model.m_intent, model.n_intent):
            gradient = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            self.assertGreater(gradient, 0.)

    def test_privileged_input_is_rejected_by_student(self):
        model = models.build_student(hidden_size=64, intent_dim=8)
        with self.assertRaises(TypeError):
            model(**inputs(), memory=torch.zeros(2, 64), fruit_truth=torch.ones(2, 23))

    def test_intent_intervention_does_not_change_memory(self):
        model = models.build_student(hidden_size=64, intent_dim=8)
        obs = inputs()
        out = model(**obs, memory=torch.zeros(2, 64))
        changed = model(**obs, memory=torch.zeros(2, 64),
                        intent_override={'m': torch.ones(2, 8)})
        torch.testing.assert_close(out['memory'], changed['memory'])
        torch.testing.assert_close(out['n_mu'], changed['n_mu'])
        self.assertFalse(torch.equal(out['m_mu'], changed['m_mu']))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
    def test_cuda_update_changes_registered_weights(self):
        model = models.build_student(hidden_size=64, intent_dim=8).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        obs = {name: value.to('cuda') for name, value in inputs().items()}
        before = next(model.vision.parameters()).detach().clone()
        out = model(**obs, memory=torch.zeros(2, 64, device='cuda'))
        loss = out['m_mu'].square().mean() + out['n_mu'].square().mean()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.vision.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_all_heads_and_memory_roundtrip(self):
        model = models.build_student(hidden_size=64, intent_dim=8)
        restored = models.build_student(hidden_size=64, intent_dim=8)
        restored.load_state_dict(model.state_dict())
        obs = inputs()
        a = model(**obs, memory=torch.zeros(2, 64))
        b = restored(**obs, memory=torch.zeros(2, 64))
        for key in a:
            torch.testing.assert_close(a[key], b[key])
        self.assertEqual(tuple(a['n_mu'].shape), (2, 3))
        self.assertEqual(tuple(a['m_mu'].shape), (2, 8))
        self.assertEqual(tuple(a['n_events'].shape), (2, 2))
        self.assertEqual(tuple(a['m_events'].shape), (2, 3))


if __name__ == '__main__':
    unittest.main()
