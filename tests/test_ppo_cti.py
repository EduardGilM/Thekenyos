import unittest

import torch

from treesim.kiwi_rl.ppo_cti import (
    DecisionBatch, PPODecisionQueue, _policy_snapshot, _slice, choose_worlds,
)


class PPODecisionQueueTest(unittest.TestCase):
    def batch(self, version, priority):
        result = DecisionBatch(snapshot=None, source_world_ids=(2, 5),
            source_episode_ids=torch.tensor([4, 7]), source_tick=64,
            policy_version=version, policy_state={}, initial_memory=torch.zeros(2, 64),
            initial_progress={}, priority=priority,
            factual_steps=[{'valid': torch.ones(2, dtype=torch.bool),
                            'active': torch.ones(2, dtype=torch.bool),
                            'ending': torch.zeros(2, dtype=torch.bool)} for _ in range(8)])
        return result

    def test_world_selection_mixes_nearest_and_rotating_worlds(self):
        distance = torch.tensor([.4, .3, .2, .1, .5, .6, .7, .8])
        first, cursor = choose_worlds(distance, 4, 0)
        second, _ = choose_worlds(distance, 4, cursor)
        self.assertEqual(first[:2], [3, 2])
        self.assertEqual(len(set(first)), 4)
        self.assertNotEqual(set(first[2:]), set(second[2:]))

    def test_queue_is_bounded_prioritized_and_expires_old_policy_versions(self):
        queue = PPODecisionQueue(worlds=2, max_batches=2, max_age=2)
        for version, priority in ((0, 1.), (1, 3.), (2, 2.)):
            self.assertTrue(queue.finish(self.batch(version, priority)))
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.metrics()['total_collected_roots'], 6)
        self.assertEqual(queue.metrics()['total_dropped_roots'], 2)
        selected = queue.pop(3)
        self.assertEqual(selected.policy_version, 1)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.pop(5), None)

    def test_batch_exposes_per_source_metadata_and_factual_rows(self):
        batch = self.batch(6, 0.)
        self.assertEqual(batch.sources[1], dict(source_world=5, source_episode=7,
            source_tick=64, policy_version=6))
        self.assertEqual(len(batch.factual), 8)

    def test_world_slices_and_policy_snapshot_are_stable(self):
        source = torch.arange(4096 * 3).reshape(4096, 3)
        picked = _slice(source, torch.tensor([7, 9]))
        self.assertEqual(tuple(picked.shape), (2, 3))
        policy = torch.nn.Linear(3, 2)
        frozen = _policy_snapshot(policy)
        with torch.no_grad():
            policy.weight.add_(1)
        self.assertFalse(torch.equal(frozen['weight'], policy.weight.cpu()))

    def test_factual_record_marks_world_inactive_after_its_episode_ends(self):
        from types import SimpleNamespace
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue

        runtime = SimpleNamespace(device_name='cpu',
            data=SimpleNamespace(qpos=torch.zeros(3, 2), qvel=torch.zeros(3, 2)),
            control=SimpleNamespace(targets=torch.zeros(3, 2), previous=torch.zeros(3, 2)))
        queue = PPODecisionQueue(worlds=2)
        batch = self.batch(0, 0.)
        batch.factual_steps.clear()
        batch.source_world_ids = (0, 2)
        active = torch.ones(3, dtype=torch.bool)
        def signals():
            return {name: torch.zeros(3, dtype=torch.bool) for name in
                    ('touching', 'grasp', 'detached', 'success', 'failed')}
        def record(ended):
            now = signals()
            queue.record(batch, obs=torch.zeros(3, 4), raw_action=torch.zeros(3, 7),
                action=torch.zeros(3, 7), gait_action=torch.zeros(3, 12),
                next_obs=torch.zeros(3, 4), runtime=runtime, signals=now,
                reward=torch.zeros(3), task_reward=torch.zeros(3), shaping_reward=torch.zeros(3),
                terminated=ended, truncated=torch.zeros(3, dtype=torch.bool),
                stalled=torch.zeros(3, dtype=torch.bool), episode_ids=torch.zeros(3, dtype=torch.long),
                active_mask=active)
            active[ended] = False
        record(torch.tensor([True, False, False]))
        record(torch.zeros(3, dtype=torch.bool))
        self.assertEqual(batch.factual[0]['active'].tolist(), [True, True])
        self.assertEqual(batch.factual[0]['ending'].tolist(), [True, False])
        self.assertEqual(batch.factual[1]['active'].tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
