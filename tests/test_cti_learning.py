from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import torch

from treesim.kiwi_rl.cti_learning import normal_log_prob, vtrace_targets, update_branch_cti


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.zeros(7))
        self.critic = torch.nn.Parameter(torch.zeros(()))
        self.logstd = torch.nn.Parameter(torch.zeros(7))

    def forward(self, privileged, r84, memory):
        return (self.actor.expand(len(privileged), -1), self.logstd,
                self.critic.expand(len(privileged)), memory)


def branch(policy, worlds=3, steps=2, reward=-5.):
    rows=[]
    for tick in range(steps):
        raw=torch.full((worlds,7),.4)
        rows.append(dict(privileged=torch.zeros(worlds,32),r84=torch.zeros(worlds,84),
            memory=torch.zeros(worlds,64),action=raw.tanh(),raw_action=raw,
            behavior_log_prob=normal_log_prob(raw,policy.actor.detach(),policy.logstd.detach()),
            reward=torch.full((worlds,),reward),terminated=torch.full((worlds,),tick==steps-1),
            truncated=torch.zeros(worlds,dtype=torch.bool),mask=torch.ones(worlds,dtype=torch.bool),
            physical_failure=torch.full((worlds,),tick==steps-1)))
    return rows


def assert_nested_equal(test, actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    elif isinstance(actual, dict):
        test.assertEqual(actual.keys(),expected.keys())
        for key in actual:assert_nested_equal(test,actual[key],expected[key])
    elif isinstance(actual, (list,tuple)):
        test.assertEqual(len(actual),len(expected))
        for left,right in zip(actual,expected):assert_nested_equal(test,left,right)
    else:test.assertEqual(actual,expected)


class VTraceTest(unittest.TestCase):
    def test_on_policy_matches_bootstrapped_mc_and_one_step_td(self):
        rewards=torch.tensor([[1.],[2.]])
        values=torch.tensor([[3.],[4.]],requires_grad=True)
        targets,adv=vtrace_targets(rewards,values,torch.tensor([5.]),torch.zeros(2,1),
                                  torch.full((2,1),.9),torch.ones(2,1,dtype=torch.bool))
        torch.testing.assert_close(targets,torch.tensor([[6.85],[6.5]]))
        torch.testing.assert_close(adv,targets-values.detach())
        self.assertFalse(targets.requires_grad);self.assertFalse(adv.requires_grad)
        td,pg=vtrace_targets(rewards[:1],values[:1],torch.tensor([4.]),torch.zeros(1,1),
                            torch.tensor([[.9]]),torch.ones(1,1,dtype=torch.bool))
        torch.testing.assert_close(td,torch.tensor([[4.6]]));torch.testing.assert_close(pg,torch.tensor([[1.6]]))

    def test_terminal_zero_bootstrap_truncation_retains_value(self):
        targets,_=vtrace_targets(torch.ones(1,2),torch.zeros(1,2),torch.tensor([float('nan'),10.]),
            torch.zeros(1,2),torch.tensor([[0.,.9]]),torch.ones(1,2,dtype=torch.bool))
        torch.testing.assert_close(targets,torch.tensor([[1.,10.]]))

    def test_variable_world_lengths_ignore_padding_and_do_not_leak(self):
        mask=torch.tensor([[True,True],[False,True],[False,True]])
        rewards=torch.tensor([[1.,2.],[float('nan'),3.],[float('nan'),4.]])
        values=torch.tensor([[7.,8.],[float('nan'),9.],[float('nan'),10.]])
        targets,adv=vtrace_targets(rewards,values,torch.tensor([5.,6.]),torch.zeros(3,2),
            torch.ones(3,2),mask)
        torch.testing.assert_close(targets,torch.tensor([[6.,15.],[0.,13.],[0.,10.]]))
        self.assertTrue(torch.isfinite(adv).all())
        with self.assertRaisesRegex(ValueError,'prefix'):
            vtrace_targets(torch.zeros(3,1),torch.zeros(3,1),torch.zeros(1),torch.zeros(3,1),
                torch.ones(3,1),torch.tensor([[True],[False],[True]]))

    def test_importance_correction_uses_proposal_probability_and_clips(self):
        # rho=.5 scales both the local TD and future trace; rho=2 clips at1.
        targets,pg=vtrace_targets(torch.tensor([[1.,1.],[2.,2.]]),torch.zeros(2,2),torch.zeros(2),
            torch.tensor([[math.log(.5),math.log(2.)],[math.log(.5),math.log(2.)]]),
            torch.tensor([[.9,.9],[0.,0.]]),torch.ones(2,2,dtype=torch.bool))
        torch.testing.assert_close(targets,torch.tensor([[.95,2.8],[1.,2.]]))
        torch.testing.assert_close(pg,targets)
        # Large positive log ratios cannot overflow after clipping.
        targets,_=vtrace_targets(torch.ones(1,1),torch.zeros(1,1),torch.zeros(1),torch.full((1,1),1e30),
                                torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(targets.item(),1.)

    def test_normal_proposal_density_matches_torch(self):
        raw=torch.tensor([[.4,-.7]]);mean=torch.tensor([[.2,.1]]);std=torch.tensor([[.8,1.3]])
        expected=torch.distributions.Normal(mean,std).log_prob(raw).sum(-1)
        torch.testing.assert_close(normal_log_prob(raw,mean,std.log()),expected)


class BranchLearningTest(unittest.TestCase):
    def test_failure_only_batch_updates_actor_and_critic(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.01)
        rows=branch(policy)
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(3))
        self.assertTrue(metrics['updated']);self.assertFalse(metrics['rejected_update'])
        self.assertEqual(metrics['valid_transitions'],6);self.assertEqual(metrics['physical_failures'],3)
        self.assertLess(policy.actor.mean().item(),0.)
        self.assertLess(policy.critic.item(),0.)
        self.assertGreater(metrics['value_loss'],0.)
        self.assertAlmostEqual(metrics['effective_weight'],1.)
        self.assertLessEqual(metrics['kl'],.01)
        self.assertEqual(metrics['epochs'],2);self.assertEqual(metrics['optimizer_steps'],2)

    def test_invalid_roots_and_nonfinite_padding_are_ignored(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.005)
        rows=branch(policy)
        for row in rows:
            row['mask'][1]=False
            for key in ('privileged','r84','memory','raw_action','behavior_log_prob','reward'):
                row[key][1]=float('nan')
        metrics=update_branch_cti(policy,optimizer,rows,torch.tensor([0.,float('nan'),0.]))
        self.assertTrue(metrics['updated']);self.assertEqual(metrics['valid_transitions'],4)
        self.assertTrue(torch.isfinite(policy.actor).all())

    def test_terminated_and_truncated_paths_use_correct_bootstrap(self):
        for terminated,expected_value_direction in ((True,-1),(False,1)):
            policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.005)
            rows=branch(policy,worlds=1,steps=1,reward=-1.)
            rows[0]['terminated'][:]=terminated;rows[0]['truncated'][:]=not terminated
            metrics=update_branch_cti(policy,optimizer,rows,torch.tensor([10.]))
            self.assertTrue(metrics['updated'])
            self.assertGreater(policy.critic.item()*expected_value_direction,0.)

    def test_importance_underflow_is_reported_without_fictitious_signal(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        rows=branch(policy,worlds=1,steps=1)
        rows[0]['behavior_log_prob'][:]=1e6
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(1))
        self.assertEqual(metrics['importance_underflow'],1)
        self.assertEqual(metrics['effective_weight'],0.)
        self.assertEqual(metrics['actor_loss'],0.)
        self.assertEqual(metrics['value_loss'],0.)
        self.assertEqual(policy.critic.item(),0.)
        self.assertEqual(policy.actor.abs().sum().item(),0.)

    def test_world_minibatches_include_every_valid_transition_each_epoch(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        rows=branch(policy,worlds=5,steps=3)
        rows[1]['terminated'][0]=True;rows[2]['mask'][0]=False
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(5),minibatch_worlds=2)
        self.assertTrue(metrics['updated']);self.assertEqual(metrics['valid_transitions'],14)
        self.assertEqual(metrics['optimizer_steps'],6)

    def test_kl_violation_restores_policy_and_populated_adam(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        self.assertTrue(update_branch_cti(policy,optimizer,branch(policy),torch.zeros(3))['updated'])
        old_policy=deepcopy(policy.state_dict());old_adam=deepcopy(optimizer.state_dict())
        # A factual guard rejection must restore even pre-existing Adam moments.
        with patch('treesim.kiwi_rl.cti_learning._anchor_kl',return_value=.1):
            metrics=update_branch_cti(policy,optimizer,branch(policy),torch.zeros(3))
        self.assertTrue(metrics['rejected_update']);self.assertFalse(metrics['updated'])
        self.assertEqual(metrics['attempts'],3);self.assertEqual(metrics['optimizer_steps'],0)
        self.assertEqual(metrics['attempted_optimizer_steps'],6)
        assert_nested_equal(self,policy.state_dict(),old_policy)
        assert_nested_equal(self,optimizer.state_dict(),old_adam)

    def test_actual_high_lr_violates_kl_and_rolls_back(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=10.)
        old=deepcopy(policy.state_dict());adam=deepcopy(optimizer.state_dict())
        metrics=update_branch_cti(policy,optimizer,branch(policy),torch.zeros(3))
        self.assertTrue(metrics['rejected_update']);self.assertGreater(metrics['kl'],.01)
        assert_nested_equal(self,policy.state_dict(),old);assert_nested_equal(self,optimizer.state_dict(),adam)

    def test_backtracking_accepts_only_guarded_attempt(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        with patch('treesim.kiwi_rl.cti_learning._anchor_kl',side_effect=[.04,.009]):
            metrics=update_branch_cti(policy,optimizer,branch(policy),torch.zeros(3))
        self.assertTrue(metrics['updated']);self.assertEqual(metrics['attempts'],2)
        self.assertEqual(metrics['lr_scale'],.25);self.assertEqual(optimizer.param_groups[0]['lr'],.001)

    def test_nonfinite_optimizer_step_rolls_back_everything(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        original=deepcopy(policy.state_dict());adam=deepcopy(optimizer.state_dict())
        actual_step=optimizer.step
        def corrupt_step():
            actual_step()
            with torch.no_grad():policy.critic.fill_(float('nan'))
        with patch.object(optimizer,'step',side_effect=corrupt_step):
            metrics=update_branch_cti(policy,optimizer,branch(policy),torch.zeros(3))
        self.assertTrue(metrics['rejected_update']);self.assertTrue(metrics['nonfinite_update'])
        assert_nested_equal(self,policy.state_dict(),original)
        assert_nested_equal(self,optimizer.state_dict(),adam)

    def test_actual_privileged_policy_actor_and_critic_learn(self):
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        torch.manual_seed(17)
        policy=build_privileged_policy();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        rows=[];memory=torch.zeros(2,64)
        with torch.no_grad():
            for tick in range(2):
                priv=torch.randn(2,32);obs=torch.randn(2,84)
                mean,logstd,_,next_memory=policy(priv,obs,memory)
                raw=mean+.4*logstd.exp()
                rows.append(dict(privileged=priv,r84=obs,memory=memory.clone(),action=raw.tanh(),
                    raw_action=raw,behavior_log_prob=normal_log_prob(raw,mean,logstd),
                    reward=torch.full((2,),-2.),terminated=torch.full((2,),tick==1),
                    truncated=torch.zeros(2,dtype=torch.bool),mask=torch.ones(2,dtype=torch.bool)))
                memory=next_memory
        actor=policy.mean.weight.detach().clone();critic=policy.value.weight.detach().clone()
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(2))
        self.assertTrue(metrics['updated']);self.assertFalse(torch.equal(actor,policy.mean.weight))
        self.assertFalse(torch.equal(critic,policy.value.weight))

    def test_factual_anchor_rows_support_initial_memory_and_resets(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        rows=branch(policy)
        anchors=[dict(privileged=r['privileged'],r84=r['r84'],reset=torch.zeros(3,dtype=torch.bool)) for r in rows]
        anchors[0]['initial_memory']=torch.zeros(3,64)
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(3),anchor_rows=anchors)
        self.assertTrue(metrics['updated']);self.assertLessEqual(metrics['kl'],.01)
        with self.assertRaisesRegex(ValueError,'empty'):
            update_branch_cti(policy,optimizer,rows,torch.zeros(3),anchor_rows=[])

    def test_nonfinite_valid_data_rejected_before_mutation(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        rows=branch(policy);rows[0]['reward'][0]=float('nan')
        original=deepcopy(policy.state_dict())
        metrics=update_branch_cti(policy,optimizer,rows,torch.zeros(3))
        self.assertTrue(metrics['rejected_update']);self.assertTrue(metrics['nonfinite_update'])
        assert_nested_equal(self,policy.state_dict(),original)

    def test_reset_episode_cannot_leak_into_previous_trace(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        rows=branch(policy);rows[0]['terminated'][:]=True
        with self.assertRaisesRegex(ValueError,'split episodes'):
            update_branch_cti(policy,optimizer,rows,torch.zeros(3))

    def test_empty_input_and_no_valid_roots_do_not_mutate(self):
        policy=TinyPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        self.assertFalse(update_branch_cti(policy,optimizer,[],torch.zeros(0))['updated'])
        rows=branch(policy)
        for row in rows:row['mask'].zero_()
        self.assertFalse(update_branch_cti(policy,optimizer,rows,torch.zeros(3))['updated'])
        self.assertFalse(optimizer.state)

if __name__=='__main__':unittest.main()
