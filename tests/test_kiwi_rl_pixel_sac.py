"""Sparse wrist-camera SAC: shared observation path, no image rewards."""
import inspect
import unittest

import numpy as np

from treesim.kiwi_rl.pixel_obs import (CAMERA_CHANNELS, FRAME_STACK, VECTOR_DIM, build_observation,
                                       empty_wrist_state, next_observation_state, record_demo_transition)
from treesim.kiwi_rl.rewards import SPARSE_TERM_NAMES, compute_reward_batch

try:
    import torch
except ImportError:
    torch = None


def _painted_state(seed, height=16, width=16, phase='MANIPULATE'):
    rng = np.random.default_rng(seed)
    state = empty_wrist_state(height, width, phase)
    state['wrist_rgb'] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    state['wrist_depth_m'] = rng.uniform(0.1, 1.5, (height, width)).astype(np.float32)
    state['wrist_valid'] = rng.random((height, width)) > .1
    state['proprio'] = np.zeros(85, dtype=np.float32)
    state['proprio'][0] = .1
    state['latches'] = np.array([0., 1., 0., 0.], dtype=np.float32)
    return state


def _privileged(batch=2, fruit=3):
    return dict(
        deposited=np.zeros((batch, fruit), dtype=bool),
        lost=np.zeros((batch, fruit), dtype=bool),
        spilled=np.zeros((batch, fruit), dtype=bool),
        damage=np.zeros(batch, dtype=np.float32),
        fallen=np.zeros(batch, dtype=bool),
        dt_s=np.full(batch, .04, dtype=np.float32),
    )


class WristObservationTest(unittest.TestCase):
    def test_build_observation_replays_demo_state_bit_for_bit(self):
        state = _painted_state(3)
        state = next_observation_state(state)
        state['wrist_rgb'][4, 4] = [180, 90, 20]
        next_state = next_observation_state(state)
        next_state['wrist_rgb'][5, 5] = [10, 20, 30]
        next_state['phase'] = 'VERIFY'
        demo = record_demo_transition(state, np.zeros(8, np.float32), next_state,
                                      _privileged(1, 2), _privileged(1, 2))
        replayed = build_observation(demo['obs_state'])
        np.testing.assert_array_equal(replayed['camera'], demo['observation']['camera'])
        np.testing.assert_array_equal(replayed['vector'], demo['observation']['vector'])
        self.assertEqual(replayed['camera'].shape[0], CAMERA_CHANNELS * FRAME_STACK)
        self.assertEqual(replayed['vector'].shape, (VECTOR_DIM,))

    def test_observation_rejects_fruit_coordinates(self):
        state = _painted_state(1)
        state['fruit_xyz'] = np.array([.4, 0., 1.], dtype=np.float32)
        with self.assertRaises(ValueError):
            build_observation(state)


class SparseRewardBatchTest(unittest.TestCase):
    def test_compute_reward_batch_signature_accepts_no_image_tensor(self):
        names = list(inspect.signature(compute_reward_batch).parameters)
        self.assertEqual(names, ['privileged', 'action', 'next_privileged'])
        self.assertTrue(all('image' not in n and 'rgb' not in n and 'camera' not in n and 'pixel' not in n
                            for n in names))

    def test_sparse_terms_exclude_view_losses_and_relabel_from_privileged(self):
        prev, nxt = _privileged(), _privileged()
        nxt['deposited'][0, 1] = True
        nxt['damage'][1] = .2
        nxt['fallen'][1] = True
        reward, terms = compute_reward_batch(prev, np.zeros((2, 8), np.float32), nxt)
        self.assertEqual(tuple(terms), SPARSE_TERM_NAMES)
        self.assertNotIn('in_view', terms)
        self.assertNotIn('centering', terms)
        self.assertNotIn('view_loss', terms)
        self.assertAlmostEqual(float(terms['deposit'][0]), 20.)
        self.assertAlmostEqual(float(terms['deposit'][1]), 0.)
        self.assertAlmostEqual(float(terms['damage'][1]), -1.)
        self.assertAlmostEqual(float(terms['fall'][1]), -100.)
        np.testing.assert_allclose(reward[0], sum(float(terms[name][0]) for name in SPARSE_TERM_NAMES))
        np.testing.assert_allclose(reward[1], sum(float(terms[name][1]) for name in SPARSE_TERM_NAMES))


@unittest.skipUnless(torch is not None, 'needs torch')
class PixelSACTest(unittest.TestCase):
    def test_actor_loss_does_not_train_encoder(self):
        from treesim.kiwi_rl.pixel_sac import PixelSAC
        torch.manual_seed(0)
        sac = PixelSAC(16, 16, action_dim=8)
        camera = torch.zeros(2, CAMERA_CHANNELS * FRAME_STACK, 16, 16)
        vector = torch.zeros(2, VECTOR_DIM)
        sac.actor_loss(camera, vector).backward()
        encoder_grad = sum(p.grad.abs().sum().item() for p in sac.encoder.parameters() if p.grad is not None)
        actor_grad = sum(p.grad.abs().sum().item() for p in sac.actor.parameters() if p.grad is not None)
        self.assertEqual(encoder_grad, 0.)
        self.assertGreater(actor_grad, 0.)
        sac.q1.zero_grad(set_to_none=True)
        sac.q2.zero_grad(set_to_none=True)
        sac.actor.zero_grad(set_to_none=True)
        q1, _ = sac.q(camera, vector, torch.zeros(2, 8), shift=False)
        q1.mean().backward()
        critic_encoder = sum(p.grad.abs().sum().item() for p in sac.encoder.parameters() if p.grad is not None)
        self.assertGreater(critic_encoder, 0.)

    def test_drq_shifts_two_target_samples_independently_including_demos(self):
        from treesim.kiwi_rl.pixel_sac import PixelSAC, random_shift
        torch.manual_seed(1)
        images = torch.arange(2 * 15 * 16 * 16, dtype=torch.float32).reshape(2, 15, 16, 16)
        gen = torch.Generator().manual_seed(11)
        a = random_shift(images, generator=gen)
        gen = torch.Generator().manual_seed(12)
        b = random_shift(images, generator=gen)
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), tuple(images.shape))
        sac = PixelSAC(16, 16, action_dim=8)
        camera = torch.zeros(2, CAMERA_CHANNELS * FRAME_STACK, 16, 16)
        camera[:, 0, 0, 0] = 1.
        first, second = sac.critic_target_samples(camera, torch.zeros(2, VECTOR_DIM),
                                                  torch.zeros(2, 8), generator=torch.Generator().manual_seed(3))
        self.assertEqual(tuple(first[0].shape), (2,))
        self.assertEqual(tuple(second[1].shape), (2,))


if __name__ == '__main__':
    unittest.main()
