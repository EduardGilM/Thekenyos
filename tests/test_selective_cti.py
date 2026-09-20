import importlib.util
import json
import os
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from treesim.kiwi_rl.harvest_training import EpisodeProgress
from treesim.kiwi_rl.selective_cti import (
    SelectiveCTI, _choose_winners, sample_policy_sequences, update_cti,
)
def state(distance=.2, basket=.8, grasp=False, detached=False):
    return dict(distance=torch.tensor([distance]), basket_distance=torch.tensor([basket]),
                grasp=torch.tensor([grasp]), holding=torch.tensor([grasp]),
                detached=torch.tensor([detached]), touching=torch.tensor([grasp]),
                stem_force=torch.tensor([0.]), success=torch.tensor([False]),
                failed=torch.tensor([False]))


def outcome(distance=.2, holding=False, detached=False, score=0., **changes):
    result = dict(score=torch.tensor([score]), distance=torch.tensor([distance]),
                  basket_distance=torch.tensor([.8]), holding=torch.tensor([holding]),
                  detached=torch.tensor([detached]), success=torch.tensor([False]),
                  failed=torch.tensor([False]), numerical=torch.tensor([False]),
                  lost_fruit=torch.tensor([False]), retained=torch.tensor([True]),
                  triggered=torch.tensor([True]))
    result['retained_potential'] = EpisodeProgress.potential(result)
    for key, value in changes.items():
        result[key] = torch.tensor([value])
    return result


class SelectiveChoiceTest(unittest.TestCase):
    def test_retained_partial_grasp_can_win_without_task_success(self):
        baseline, candidate = outcome(score=.05), outcome(holding=True, score=1.8)
        winner, _ = _choose_winners([baseline, candidate], state())
        self.assertEqual(winner.tolist(), [1])

    def test_transient_touch_and_no_progress_cannot_win(self):
        for candidate in (outcome(score=9.), outcome(holding=True, score=9., retained=False)):
            winner, reasons = _choose_winners([outcome(), candidate], state())
            self.assertEqual(winner.tolist(), [-1])
            self.assertTrue(reasons[1][0])

    def test_invalid_factual_replay_rejects_even_a_successful_candidate(self):
        baseline=outcome(replay_invalid=True)
        candidate=outcome(holding=True,score=20.,success=True)
        winner,reasons=_choose_winners([baseline,candidate],state())
        self.assertEqual(winner.item(),-1)
        self.assertIn('factual_replay_mismatch',reasons[1][0])

    def test_run_requires_actual_ppo_batch(self):
        rt=SimpleNamespace(worlds=1,control_dt=.02)
        with self.assertRaisesRegex(ValueError,'actual PPO'):
            SelectiveCTI(rt,None).run(None,None)

    def test_failure_or_lost_fruit_cannot_win(self):
        for changes in ({'failed': True}, {'numerical': True}, {'lost_fruit': True}):
            candidate = outcome(holding=True, score=5., **changes)
            winner, _ = _choose_winners([outcome(), candidate], state())
            self.assertEqual(winner.tolist(), [-1])

    def test_retained_approach_and_gain_threshold(self):
        candidates = [outcome(distance=.19, score=1.), outcome(distance=.17, score=.05),
                      outcome(distance=.17, score=.2)]
        for candidate, expected in zip(candidates, (-1, -1, 1)):
            winner, _ = _choose_winners([outcome(), candidate], state())
            self.assertEqual(winner.item(), expected)

    def test_progress_must_beat_root_and_baseline(self):
        for root, baseline in ((state(grasp=True), outcome()), (state(), outcome(holding=True))):
            winner, _ = _choose_winners([baseline, outcome(holding=True, score=4.)], root)
            self.assertEqual(winner.item(), -1)


class PolicyExplorationTest(unittest.TestCase):
    def test_sampled_sequences_are_seeded_coherent_all_joint_and_both_jaw_signs(self):
        first = sample_policy_sequences(12, 100, device='cpu', seed=421)
        repeated = sample_policy_sequences(12, 100, device='cpu', seed=421)
        self.assertEqual({candidate['block_steps'] for candidate in first}, {8, 32, 100})
        self.assertEqual({candidate['scale'] for candidate in first}, {.5, 1., 2.})
        stacked = torch.stack([candidate['residual'] for candidate in first])
        self.assertTrue(torch.all(stacked.abs().sum(dim=(0, 1)) > 0))
        jaw = stacked[:, :, 6]
        self.assertTrue((jaw > 0).any())
        self.assertTrue((jaw < 0).any())
        for left, right in zip(first, repeated):
            torch.testing.assert_close(left['residual'], right['residual'], rtol=0, atol=0)
            self.assertEqual(left['noise_seed'], right['noise_seed'])

    def test_policy_residual_opens_and_closes_jaw_without_special_action_override(self):
        runtime = SimpleNamespace(worlds=1, device_name='cpu', control_dt=.02)
        runtime.observe = lambda: torch.zeros(1, 84)
        runtime.set_gait_actions = lambda actions: None
        cti = SelectiveCTI(runtime, lambda obs: torch.zeros(1, 12))
        policy = lambda priv, obs, memory: (torch.zeros(1, 7), torch.zeros(7),
                                             torch.zeros(1), memory)
        memory = torch.zeros(1, 64)
        residual = torch.tensor([.3, -.2, .1, .4, -.3, .2, 1.])
        with patch('treesim.kiwi_rl.selective_cti.privileged_observation',
                   return_value=torch.zeros(1, 32)):
            _, _, _, closing, _ = cti._act(policy, memory, explore=True, noise=residual)
            _, _, _, opening, _ = cti._act(policy, memory, explore=True, noise=-residual)
        self.assertTrue((closing[0, :6] != 0).all())
        self.assertGreater(closing[0, 6].item(), 0)
        self.assertLess(opening[0, 6].item(), 0)


class _TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(32 + 84 + 64, 7)

    def forward(self, privileged, r84, memory):
        mean = self.actor(torch.cat((privileged, r84, memory), dim=-1))
        return mean, torch.zeros(7, device=mean.device), torch.zeros(len(mean), device=mean.device), memory + 1


def training_row():
    return dict(privileged=torch.zeros(3, 32), r84=torch.zeros(3, 84),
                memory=torch.zeros(3, 64), action=torch.ones(3, 7) * .5,
                mask=torch.tensor([True, False, True]),
                success_mask=torch.tensor([True, False, False]))


def residual_candidate(steps=2):
    return dict(label='policy-residual-test', residual=torch.ones(steps, 7),
                noise_seed=19, scale=1., std_floor=0., block_steps=steps)


class CTIUpdateTest(unittest.TestCase):
    def test_updates_only_selected_bounded_action_rows(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        row = training_row()
        row['action'][1] = float('nan')  # Masked values must not poison gradients.
        before = policy.actor.bias.detach().clone()
        metrics = update_cti(policy, optimizer, [row])
        self.assertTrue(metrics['updated'])
        self.assertLessEqual(metrics['kl'], .01)
        self.assertEqual(metrics['selected_worlds'], 2)
        self.assertEqual(metrics['target_actions'], 2)
        self.assertIsNotNone(metrics['target_error_before'])
        self.assertIsNotNone(metrics['target_error_after'])
        self.assertLess(metrics['target_error_after'], metrics['target_error_before'])
        self.assertFalse(torch.equal(before, policy.actor.bias))

    def test_rejects_unbounded_targets_and_nonfinite_coefficient(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        row = training_row()
        row['action'][:] = 2.
        with self.assertRaisesRegex(ValueError, 'finite and bounded'):
            update_cti(policy, optimizer, [row])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            update_cti(policy, optimizer, [], coef=float('nan'))

    def test_rejects_update_without_nontrivial_target_error_reduction(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.)
        row = training_row()
        before = deepcopy(policy.state_dict())
        metrics = update_cti(policy, optimizer, [row])
        self.assertTrue(metrics['rejected_update'])
        self.assertFalse(metrics['updated'])
        self.assertTrue(metrics['no_target_improvement'])
        self.assertAlmostEqual(metrics['target_error_after'], metrics['target_error_before'], places=7)
        self.assertEqual(optimizer.state_dict()['state'], {})
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_factual_kl_rejection_restores_policy_and_adam_state(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        row = training_row()
        self.assertTrue(update_cti(policy, optimizer, [row])['updated'])
        optimizer.param_groups[0]['lr'] = 1.
        policy_before, opt_before = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
        anchor = {**row, 'privileged': torch.ones(3, 32)}
        metrics = update_cti(policy, optimizer, [row], anchor_rows=[anchor])
        self.assertTrue(metrics['rejected_update'])
        self.assertFalse(metrics['updated'])
        self.assertGreater(metrics['kl'], .01)
        self.assertAlmostEqual(metrics['target_error_after'], metrics['target_error_before'], places=7)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, policy_before[key], rtol=0, atol=0)
        actual = optimizer.state_dict()
        self.assertEqual(actual['param_groups'], opt_before['param_groups'])
        for parameter, states in actual['state'].items():
            for name, value in states.items():
                torch.testing.assert_close(value, opt_before['state'][parameter][name], rtol=0, atol=0)

    def test_recurrent_anchor_memory_is_fixed_before_update(self):
        policy = _TinyTeacher()
        memories = []
        original = policy.forward
        def record(priv, obs, memory):
            memories.append(memory.clone())
            return original(priv, obs, memory)
        policy.forward = record
        row = training_row()
        anchors = [{key: value for key, value in row.items() if key != 'memory'} for _ in range(2)]
        anchors[0]['initial_memory'] = torch.zeros(3, 64)
        anchors[0]['reset'] = torch.tensor([False, True, False])
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        self.assertTrue(update_cti(policy, optimizer, [row], anchor_rows=anchors)['updated'])
        # Target-error measurements add one forward pass before and after the
        # optimizer step. The recurrent-anchor passes remain paired around it.
        torch.testing.assert_close(memories[1], memories[-3])
        torch.testing.assert_close(memories[2], memories[-2])
        torch.testing.assert_close(memories[2], torch.ones(3, 64))

    def test_nonfinite_post_step_kl_rolls_back_and_remains_json_serializable(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        before = deepcopy(policy.state_dict())
        step = optimizer.step
        def corrupt_step():
            step()
            with torch.no_grad():
                policy.actor.bias.fill_(float('nan'))
        optimizer.step = corrupt_step
        metrics = update_cti(policy, optimizer, [training_row()])
        self.assertFalse(metrics['updated'])
        self.assertTrue(metrics['rejected_update'])
        self.assertTrue(metrics['nonfinite_update'])
        self.assertAlmostEqual(metrics['target_error_after'], metrics['target_error_before'], places=7)
        json.dumps(metrics, allow_nan=False)
        self.assertEqual(optimizer.state_dict()['state'], {})
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_nonfinite_gradient_does_not_mutate_policy(self):
        policy = _TinyTeacher()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        before = deepcopy(policy.state_dict())
        hook = policy.actor.bias.register_hook(lambda grad: grad * float('nan'))
        metrics = update_cti(policy, optimizer, [training_row()])
        hook.remove()
        self.assertTrue(metrics['rejected_update'])
        self.assertTrue(metrics['nonfinite_update'])
        self.assertAlmostEqual(metrics['target_error_after'], metrics['target_error_before'], places=7)
        json.dumps(metrics, allow_nan=False)
        self.assertEqual(optimizer.state_dict()['state'], {})
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key])


class CTISegmentTest(unittest.TestCase):
    def fixture(self, *, stall_steps=100, max_steps=100, reject_confirmation=False, transient=False):
        rt = SimpleNamespace(worlds=1, device_name='cpu', control_dt=.02, tick=0, kind=None)
        rt.check = lambda: None
        rt.reset = lambda mask: None
        cti = SelectiveCTI(rt, None, perturb_steps=2, horizon_seconds=.12,
                           retention_seconds=.04)
        recorded = []
        def restore_root(root):
            rt.tick, rt.kind = 0, None
            torch.manual_seed(19)
            recorded.append([])
            return torch.zeros(1, 64), EpisodeProgress(state(), stall_steps=stall_steps,
                                                      max_steps=max_steps), torch.zeros(1)
        cti._restore_app = restore_root
        def act(policy, memory, *, explore, noise=None, noise_scale=1., std_floor=0.):
            if rt.tick == 0:
                rt.kind = noise is not None
            action = .1 * torch.rand(1, 7)
            if noise is not None:
                action += .5
            recorded[-1].append(action.clone())
            return torch.full((1, 84), float(rt.tick)), torch.zeros(1, 32), memory.clone(), action, memory + 1
        cti._act = act
        def step(action):
            rt.tick += 1
            return None, None, torch.tensor([False]), None
        rt.step = step
        def observe(runtime):
            holding = bool(rt.kind) and rt.tick > 0
            if reject_confirmation and torch.initial_seed() == 91009:
                holding = False
            if transient and rt.tick >= 5:
                holding = False
            return state(grasp=holding)
        return cti, rt, recorded, observe

    def test_actual_confirmed_segment_labels_and_matched_noise(self):
        cti, rt, recorded, observe = self.fixture()
        candidate = residual_candidate()
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, candidate])
        self.assertEqual(metrics['selected_worlds'], 1)
        self.assertEqual(len(examples), 2)
        confirmed = [row for row in metrics['branch_records']
                     if row['pass_name'] == 'confirmation' and row['candidate'] == 1]
        self.assertEqual(confirmed[0]['selected_candidate'], 1)
        sampled = next(row for row in metrics['branch_records']
                       if row['pass_name'] == 'search' and row['candidate'] == 1)
        self.assertEqual(sampled['noise_seed'], candidate['noise_seed'])
        self.assertEqual(sampled['block_steps'], candidate['block_steps'])
        # Restore before initial signals; then search pair; then confirmation pair.
        for i, row in enumerate(examples):
            torch.testing.assert_close(row['action'], recorded[4][i])
            torch.testing.assert_close(recorded[4][i] - recorded[3][i], torch.full((1, 7), .5))
            self.assertFalse(torch.equal(recorded[4][i], recorded[2][i]))
            self.assertEqual(row['r84'][0, 0].item(), i)
        json.dumps(metrics['branch_records'], allow_nan=False)

    def test_second_seed_must_confirm_improvement(self):
        cti, rt, recorded, observe = self.fixture(reject_confirmation=True)
        candidate = residual_candidate()
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, candidate])
        self.assertEqual(metrics['provisional_worlds'], 1)
        self.assertEqual(metrics['selected_worlds'], 0)
        self.assertEqual(examples, [])

    def test_grasp_retained_at_horizon_can_win_even_if_it_would_later_stall(self):
        # Baseline stalls on step 6; the new grasp moves its stall to step 7.
        # Step 6 is an explicit leaf, so the grasp is useful partial progress.
        cti, rt, recorded, observe = self.fixture(stall_steps=6)
        candidate = residual_candidate()
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, candidate])
        self.assertEqual(metrics['selected_worlds'], 1)
        self.assertTrue(examples)
        self.assertTrue(metrics['branch_records'][0]['stalled'])
        self.assertTrue(metrics['branch_records'][1]['horizon'])
        self.assertGreater(metrics['branch_records'][1]['leaf'], 2.)

    def test_horizon_keeps_leaf_real_stall_clears_it(self):
        for stall_steps, expected_stall in ((100, False), (3, True)):
            cti, rt, recorded, observe = self.fixture(stall_steps=stall_steps)
            with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
                result, _, _ = cti._branch(None, None, torch.tensor([True]), residual_candidate())
            self.assertEqual(result['stalled'].item(), expected_stall)
            self.assertEqual(result['horizon'].item(), not expected_stall)
            if expected_stall:
                self.assertEqual(result['leaf'].item(), 0.)
            else:
                self.assertGreater(result['leaf'].item(), 2.)

    def test_timeout_before_policy_continuation_cannot_supply_partial_labels(self):
        cti, rt, recorded, observe = self.fixture(max_steps=3)
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, residual_candidate()])
        self.assertEqual(examples, [])
        self.assertIn('incomplete_retention', metrics['branch_records'][1]['rejection_reasons'])

    def test_transient_grasp_lost_before_leaf_is_rejected(self):
        cti, rt, recorded, observe = self.fixture(transient=True)
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, residual_candidate()])
        self.assertEqual(examples, [])
        candidate = metrics['branch_records'][1]
        self.assertIn('no_retained_progress', candidate['rejection_reasons'])
        self.assertEqual(candidate['hold_fraction'], 0.)

    def test_numerical_failure_rejects_branch(self):
        cti, rt, recorded, observe = self.fixture()
        def fail():
            raise RuntimeError('GPU numerical failure: flagged_worlds=1')
        rt.check = fail
        with patch('treesim.kiwi_rl.selective_cti.signals', side_effect=observe):
            examples, metrics = cti._episode(None, None, torch.tensor([True]), torch.zeros(1),
                                             [None, residual_candidate()])
        self.assertEqual(examples, [])
        self.assertTrue(metrics['branch_records'][1]['numerical'])
        json.dumps(metrics['branch_records'], allow_nan=False)

@unittest.skipUnless(os.environ.get('CTI_GPU_TEST') == '1' and os.environ.get('FAST_SCENE') and
                     os.environ.get('GAIT_CHECKPOINT') and
                     all(importlib.util.find_spec(name) for name in ('warp', 'mujoco', 'mujoco_warp')) and
                     torch.cuda.is_available(), 'explicit CTI_GPU_TEST=1, scene, gait and CUDA required')
class CTIGpuIntegrationTest(unittest.TestCase):
    def test_two_world_bounded_replay_and_auxiliary_update(self):
        import warp as wp
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            source = FastRuntime(os.environ['FAST_SCENE'], worlds=4, device='cuda:0')
            runtime = FastRuntime(os.environ['FAST_SCENE'], worlds=2, device='cuda:0')
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            policy = build_privileged_policy().cuda().eval()
            from treesim.kiwi_rl.harvest_training import HarvestCollector
            from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
            collector=HarvestCollector(source,role='teacher',reward_profile='milestone-harvest/v1')
            queue=PPODecisionQueue(worlds=2,segment_steps=128)
            collector.collect(policy,gait,128,cti_queue=queue,policy_version=7)
            batch=queue.pop(7)
            cti=SelectiveCTI(runtime,gait)
            examples,metrics=cti.run(policy,batch)
            self.assertEqual(metrics['source'],'ppo')
            self.assertEqual(metrics['source_policy_version'],7)
            self.assertGreater(metrics['branch_replay_checked_steps'],0)
            self.assertEqual(metrics['branch_replay_rejected_worlds'],0)
            self.assertTrue(all(r['source']=='ppo' for r in metrics['branch_records']))
            json.dumps(metrics['branch_records'],allow_nan=False)
            if examples:
                result=update_cti(policy,torch.optim.Adam(policy.parameters(),lr=1e-4),examples)
                self.assertTrue(result['updated'] or result['rejected_update'])
            # A fabricated reference trajectory must never produce teaching targets.
            batch.factual[0]['qpos'] += 1.
            bad_examples,bad_metrics=cti.run(policy,batch)
            self.assertEqual(bad_examples,[])
            self.assertEqual(bad_metrics['branch_replay_rejected_worlds'],2)


if __name__ == '__main__':
    unittest.main()
