import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'Torch required')
class PhysicalTrainingSmokeTest(unittest.TestCase):
    def _rollout(self, steps=3, worlds=2):
        from scripts.train_physical_smoke import build_policy
        from treesim.kiwi_rl.ppo import tanh_logprob

        policy = build_policy()
        memory = torch.zeros(worlds, 64)
        rows = []
        generator = torch.Generator().manual_seed(7)
        for index in range(steps):
            rgbd = torch.rand(worlds, 5, 32, 32, generator=generator)
            r84 = torch.randn(worlds, 84, generator=generator)
            with torch.no_grad():
                mean, logstd, value, next_memory = policy(rgbd, r84, memory)
                raw = mean.detach() + .3
                logp = tanh_logprob(raw, mean, logstd).detach()
            rows.append(dict(rgbd=rgbd, r84=r84, raw=raw, logp=logp,
                             value=value.detach(), reward=torch.full((worlds,), float(index)),
                             applied=torch.zeros(worlds, 7), distance=torch.ones(worlds)))
            memory = next_memory.detach()
        with torch.no_grad():
            _, _, bootstrap, _ = policy(rows[-1]['rgbd'], rows[-1]['r84'], memory)
        return policy, rows, bootstrap

    def test_factual_replay_update_changes_vision_and_action(self):
        from scripts.train_physical_smoke import update

        policy, rows, bootstrap = self._rollout()
        result = update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), rows, bootstrap)
        self.assertLess(result['kl'], .02)
        self.assertTrue(result['vision_changed'])
        self.assertTrue(result['action_changed'])
        self.assertGreater(result['reward_std'], 0.)

    def test_replay_mismatch_is_rejected(self):
        from scripts.train_physical_smoke import update

        policy, rows, bootstrap = self._rollout()
        rows[0]['logp'] = rows[0]['logp'] + 1.
        with self.assertRaisesRegex(RuntimeError, 'KL'):
            update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), rows, bootstrap)

    def test_constant_measured_reward_is_rejected(self):
        from scripts.train_physical_smoke import update

        policy, rows, bootstrap = self._rollout()
        for row in rows:
            row['reward'].zero_()
        with self.assertRaisesRegex(RuntimeError, 'rewards do not vary'):
            update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), rows, bootstrap)

    def test_imitation_update_uses_teacher_labels_and_reports_distance(self):
        from scripts.train_physical_smoke import imitation_update

        policy, rows, _ = self._rollout()
        for row in rows:
            row['terminated'] = torch.zeros_like(row['reward'], dtype=torch.bool)
            row['teacher_label'] = torch.full_like(row['raw'], .2)
            row['teacher_distance_before'] = torch.full((len(row['reward']),), .4).numpy()
            row['teacher_distance_after'] = torch.full((len(row['reward']),), .3).numpy()
        result = imitation_update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), rows)
        self.assertTrue(result['vision_changed'])
        self.assertTrue(result['action_changed'])
        self.assertAlmostEqual(result['teacher_distance_change'], -.1, places=6)
        self.assertTrue(result['teacher_improved'])
        rows[-1]['terminated'][0] = True
        result = imitation_update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), rows)
        self.assertFalse(result['teacher_improved'])


if __name__ == '__main__':
    unittest.main()
