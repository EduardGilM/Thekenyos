import importlib.util
import os
import unittest

import torch

from treesim.kiwi_rl.selective_cti import _choose_winners, update_cti


class SelectiveChoiceTest(unittest.TestCase):
    def test_requires_resolved_baseline_and_categorical_gain(self):
        scores = torch.tensor([[0., 0., 0., 0.], [5., 2., 1., 10.]])
        resolved = torch.tensor([[False, True, True, True], [True, True, True, True]])
        physical_failure = torch.tensor([[False, False, True, True],
                                         [False, False, False, True]])
        success = torch.zeros_like(resolved)
        self.assertEqual(_choose_winners(scores, resolved, physical_failure, success).tolist(),
                         [-1, -1, 1, -1])

    def test_success_repair_still_requires_material_reward_improvement(self):
        scores = torch.tensor([[0., 0.], [.25, 1.]])
        resolved = torch.ones_like(scores, dtype=torch.bool)
        failed = torch.tensor([[True, False], [False, False]])
        success = torch.tensor([[False, False], [True, True]])
        self.assertEqual(_choose_winners(scores, resolved, failed, success,
                                         minimum_gain=.5).tolist(), [-1, 1])

    def test_later_stall_cannot_win_by_discounted_return(self):
        # Neither branch succeeds. The baseline did not physically fail, so a
        # later stall with a higher discounted return is not a repair.
        scores = torch.tensor([[0.], [3.]])
        ended = torch.ones_like(scores, dtype=torch.bool)
        physical_failure = torch.zeros_like(ended)
        success = torch.zeros_like(ended)
        self.assertEqual(_choose_winners(scores, ended, physical_failure, success).tolist(), [-1])


class _TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(32 + 84 + 64, 7)

    def forward(self, privileged, r84, memory):
        mean = self.actor(torch.cat((privileged, r84, memory), dim=-1))
        return mean, torch.zeros(7, device=mean.device), torch.zeros(len(mean), device=mean.device), memory


class CTIUpdateTest(unittest.TestCase):
    def test_updates_only_selected_bounded_action_rows(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        row = dict(privileged=torch.zeros(3, 32), r84=torch.zeros(3, 84),
                   memory=torch.zeros(3, 64), action=torch.ones(3, 7) * .5,
                   mask=torch.tensor([True, False, True]),
                   success_mask=torch.tensor([True, False, False]))
        before = policy.actor.bias.detach().clone()
        metrics = update_cti(policy, optimizer, [row], coef=.1)
        self.assertTrue(metrics['updated'])
        self.assertEqual(metrics['selected_worlds'], 2)
        self.assertEqual(metrics['target_actions'], 2)
        self.assertFalse(torch.equal(before, policy.actor.bias))

    def test_rejects_unbounded_targets_and_nonfinite_coefficient(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        row = dict(privileged=torch.zeros(1, 32), r84=torch.zeros(1, 84),
                   memory=torch.zeros(1, 64), action=torch.ones(1, 7) * 2,
                   mask=torch.ones(1, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, 'finite and bounded'):
            update_cti(policy, optimizer, [row])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            update_cti(policy, optimizer, [], coef=float('nan'))


class CTISegmentTest(unittest.TestCase):
    def test_targets_are_the_actual_winning_segment_not_resampled_actions(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from treesim.kiwi_rl.selective_cti import SelectiveCTI
        from treesim.kiwi_rl.harvest_training import EpisodeProgress
        from tests.test_harvest_training import state

        rt=SimpleNamespace(worlds=1,device_name='cpu',control_dt=.02,tick=0)
        rt.check=lambda: None
        rt.reset=lambda mask: None
        cti=SelectiveCTI(rt,None,perturb_steps=2)
        recorded=[]
        def restore_root(root):
            rt.tick=0
            torch.manual_seed(19)
            recorded.append([])
            return torch.zeros(1,64),EpisodeProgress(state(),stall_steps=9,max_steps=10),torch.zeros(1)
        cti._restore_app=restore_root
        def act(policy,memory,*,explore,delta):
            action=.1*torch.rand(1,7)+delta
            recorded[-1].append(action.clone())
            rt.safe=bool(action.mean()>.4)
            return torch.full((1,84),float(rt.tick)),torch.zeros(1,32),memory.clone(),action,memory+1
        cti._act=act
        def step(action):
            rt.tick+=1
            return None,None,torch.tensor([rt.tick==2]),None
        rt.step=step
        def observe(runtime):
            now=state()
            now['success'][:]=rt.tick==2 and rt.safe
            now['failed'][:]=rt.tick==2 and not rt.safe
            return now
        with patch('treesim.kiwi_rl.selective_cti.signals',side_effect=observe):
            examples,metrics=cti._episode(None,None,torch.tensor([True]),torch.zeros(1),
                                        [torch.zeros(1,7),torch.full((1,7),.5)])
        self.assertEqual(metrics['selected_worlds'],1)
        self.assertEqual(len(examples),2)
        for i,row in enumerate(examples):
            torch.testing.assert_close(row['action'],recorded[1][i])
            torch.testing.assert_close(recorded[1][i]-recorded[0][i],torch.full((1,7),.5))
            self.assertEqual(row['r84'][0,0].item(),i)
            self.assertTrue(row['mask'].item())


@unittest.skipUnless(os.environ.get('FAST_SCENE') and os.environ.get('GAIT_CHECKPOINT') and
                     all(importlib.util.find_spec(name) for name in ('warp', 'mujoco', 'mujoco_warp')) and
                     torch.cuda.is_available(), 'FAST_SCENE, GAIT_CHECKPOINT and CUDA stack required')
class CTIGpuIntegrationTest(unittest.TestCase):
    def test_two_world_bounded_replay_and_auxiliary_update(self):
        import warp as wp
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        from treesim.kiwi_rl.selective_cti import SelectiveCTI

        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            runtime = FastRuntime(os.environ['FAST_SCENE'], worlds=2, device='cuda:0')
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            policy = build_privileged_policy().cuda().eval()
            cti = SelectiveCTI(runtime, gait, stall_seconds=.1, max_episode_seconds=.2,
                               perturb_steps=2)
            examples, metrics = cti.run(policy)
            self.assertGreater(metrics['transitions'], 0)
            self.assertTrue(torch.isfinite(torch.tensor(metrics['seconds'])))
            if examples:
                result = update_cti(policy, torch.optim.Adam(policy.parameters(), lr=1e-4), examples)
                self.assertTrue(result['updated'])


if __name__ == '__main__':
    unittest.main()
