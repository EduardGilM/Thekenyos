"""Branch search correctness, plus opt-in actual PPO-to-CTI GPU integration."""
import os
from copy import deepcopy
from types import SimpleNamespace
import unittest
import torch
from treesim.kiwi_rl.branch_cti import initial_plans, refine_plans, expand_application, BranchCTI
from treesim.kiwi_rl.counterfactual import capture_worlds, restore_worlds
from tests.test_counterfactual_replay import WorldSnapshotTest
from unittest.mock import patch


class BranchSearchTest(unittest.TestCase):
    def test_replicated_world_restore_preserves_root_mapping_and_source(self):
        fixtures=WorldSnapshotTest()
        source,target=fixtures.runtime(8),fixtures.runtime(130)
        before=source.data.qpos.copy()
        with patch('treesim.kiwi_rl.counterfactual._world_profile',return_value=('same',)):
            snapshot=capture_worlds(source,[6,1])
            mapping=torch.arange(2).repeat(65)
            restore_worlds(target,snapshot,source_indices=mapping)
            torch.testing.assert_close(torch.from_numpy(target.data.qpos),torch.from_numpy(before[[6,1]]).repeat(65,1))
            torch.testing.assert_close(torch.from_numpy(source.data.qpos),torch.from_numpy(before))
            with self.assertRaises(ValueError): restore_worlds(target,snapshot,source_indices=[0,2])

    def test_graph_and_memory_expansion_is_deep_and_world_aligned(self):
        app=dict(memory=torch.tensor([[1.],[2.]]),progress=SimpleNamespace(
            graph_dwell=torch.tensor([.1,.2]),previous={'holding':torch.tensor([True,False])},gamma=.999))
        memory,progress=expand_application(app,torch.tensor([1,0,1]))
        self.assertEqual(memory.flatten().tolist(),[2.,1.,2.])
        self.assertEqual(progress.previous['holding'].tolist(),[False,True,False])
        progress.graph_dwell[0]=9.
        self.assertAlmostEqual(app['progress'].graph_dwell[1].item(),.2)

    def test_refinement_uses_returns_even_when_every_branch_is_a_failure(self):
        gen=torch.Generator().manual_seed(7)
        plans=initial_plans(2,12,5,generator=gen,device='cpu')
        scores=-torch.arange(13.).flip(0)[:,None].repeat(1,2)-1
        new=refine_plans(plans,scores,torch.ones_like(scores,dtype=torch.bool),generator=gen)
        torch.testing.assert_close(new[2],plans[12])
        self.assertTrue((new[:2]==0).all())
        self.assertFalse(torch.equal(new[2:],plans[2:]))
        self.assertTrue((plans[1:,:,:,6]>0).any() and (plans[1:,:,:,6]<0).any())
        self.assertTrue((plans[1:].abs().sum((0,1,2))>0).all())


@unittest.skipUnless(os.environ.get('CTI_GPU_TEST')=='1' and os.environ.get('FAST_SCENE') and os.environ.get('GAIT_CHECKPOINT'), 'GPU opt-in required')
class BranchGpuTest(unittest.TestCase):
    def test_actual_graph_ppo_roots_replay_branch_and_learn(self):
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.harvest_training import HarvestCollector,GRAPH_PROFILE
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
        from treesim.kiwi_rl.cti_learning import update_branch_cti
        torch.set_num_threads(1);torch.manual_seed(42);wp.init();stream=torch.cuda.Stream()
        with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
            rt=FastRuntime(os.environ['FAST_SCENE'],worlds=8,task_profile=GRAPH_PROFILE)
            search_rt=FastRuntime(os.environ['FAST_SCENE'],worlds=12,task_profile=GRAPH_PROFILE)
            policy=build_privileged_policy().cuda()
            gait=load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            rt.prepare_settled_reset(gait)
            search_rt.prepare_settled_reset(gait)
            collector=HarvestCollector(rt,role='teacher',reward_profile=GRAPH_PROFILE)
            queue=PPODecisionQueue(worlds=4,segment_steps=16)
            factual,_,_=collector.collect(policy,gait,16,cti_queue=queue,policy_version=0)
            batch=queue.pop(0)
            self.assertIsNotNone(batch)
            search=BranchCTI(search_rt,gait,roots=4,alternatives=2,search_iterations=2,
                horizon_seconds=.64,intervention_seconds=.32,block_steps=8)
            (rows,bootstrap),metrics=search.run(policy,batch)
            self.assertEqual(metrics['branch_replay_rejected_worlds'],0,metrics)
            self.assertGreater(metrics['branch_replay_checked_steps'],0)
            self.assertEqual(metrics['valid_branches'],24)
            self.assertGreater(metrics['valid_transitions'],0)
            self.assertEqual(len(rows),32)
            self.assertTrue(all(record['source']=='ppo' for record in metrics['branch_records']))
            self.assertTrue(torch.isfinite(torch.stack([r['behavior_log_prob'] for r in rows])).all())
            before=deepcopy(policy.state_dict())
            result=update_branch_cti(policy,torch.optim.Adam(policy.parameters(),lr=1e-4),rows,bootstrap,anchor_rows=factual[:4])
            self.assertTrue(result['updated'],result)
            self.assertGreater(result['optimizer_steps'],0)
            self.assertFalse(torch.equal(before['value.weight'],policy.state_dict()['value.weight']))
            self.assertFalse(torch.equal(before['mean.weight'],policy.state_dict()['mean.weight']))
            # Incorrect factual evidence must reject all descendants of that root.
            batch.factual[0]['targets']+=.1
            (_, _),bad=search.run(policy,batch)
            self.assertEqual(bad['valid_transitions'],0)
            self.assertEqual(bad['branch_replay_rejected_worlds'],4)
