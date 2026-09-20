import unittest
from copy import deepcopy
import torch
from tests.test_reward_graph import state, step, progress
from treesim.kiwi_rl.reward_graph import GRAPH_PROFILE
from treesim.kiwi_rl.graph_training import acceptance, graph_rank, protect_skills

HELD=dict(enclosed=True,insertion=1.,closure=1.,bilateral=True,holding=True,touching=True,max_load=1.)

def ready():
    p=progress(state(insertion=.2),reward_profile=GRAPH_PROFILE)
    for _ in range(6): step(p,**HELD)
    return p

class PrerequisiteGraphTest(unittest.TestCase):
    def test_unheld_detachment_is_failure_not_progress(self):
        p=progress(state(insertion=.5),reward_profile=GRAPH_PROFILE)
        reward,term,_,_=step(p,detached=True)
        self.assertTrue(term.item());self.assertLess(reward.item(),-5)
        self.assertEqual(p.graph_score.item(),0)
        self.assertFalse(p.graph_completed['extract'].item())
        self.assertFalse(p.graph_valid_extract.item())

    def test_grasp_required_before_and_during_detach(self):
        for changes in ({'bilateral':False},{'slip':.2},{'holding':False}):
            p=ready();step(p,**{**HELD,**changes},detached=True)
            self.assertTrue(p.graph_invalid_extract.item())
        p=ready();step(p,**HELD,detached=True)
        self.assertTrue(p.graph_valid_extract.item())
        self.assertGreaterEqual(p.graph_score.item(),6.)

    def test_lost_prerequisite_outweighs_small_progress(self):
        p=ready();gain=step(p,**HELD,stem_force=1.)[0].item()
        loss=step(p,enclosed=True,insertion=1.,closure=1.)[0].item()
        self.assertGreater(gain,0);self.assertLess(loss,-gain)
        self.assertEqual(p.graph_regressions['grip'].item(),1)

    def test_carry_loses_credit_and_recovery_cannot_profit(self):
        p=ready();step(p,**HELD,detached=True)
        total=0.
        for t in range(8):
            changes={} if t==0 else HELD
            reward,*_=step(p,detached=True,**changes)
            total+=p.gamma**t*reward.item()
            if t==0:self.assertEqual(p.graph_score.item(),0.)
        self.assertLess(total,0.)

    def test_basket_release_valid_but_transit_release_invalid(self):
        for distance,valid in ((.02,True),(.5,False)):
            p=ready();step(p,**HELD,detached=True,release_distance=distance)
            step(p,detached=True,release_distance=distance)
            self.assertEqual(p.graph_valid_release.item(),valid)
            step(p,detached=True,settle_time=2.,success=True)
            self.assertEqual(p.graph_success.item(),valid)

    def test_uncontrolled_fruit_cannot_skip_to_success(self):
        p=ready();step(p,detached=True,success=True,settle_time=2.)
        self.assertFalse(p.graph_success.item());self.assertLess(p.task_reward.item(),0.)

    def test_ground_drop_zero_terminal_potential(self):
        p=ready();step(p,**HELD,detached=True)
        previous=p.graph_score.item()
        reward,term,_,_=step(p,detached=True,ground_contact=True)
        self.assertTrue(term.item());self.assertLess(reward.item(),-5.)
        self.assertAlmostEqual(p.shaping_reward.item(),-previous,places=5)
        torch.testing.assert_close(sum(p.graph_progress_reward.values()),p.shaping_reward)

    def test_stall_zero_potential_timeout_bootstraps(self):
        p=progress(state(insertion=.5),reward_profile=GRAPH_PROFILE,stall_steps=1)
        reward,term,_,_=step(p,insertion=.5)
        self.assertTrue(term.item());self.assertAlmostEqual(p.shaping_reward.item(),-1.)
        p=progress(state(insertion=.5),reward_profile=GRAPH_PROFILE,max_steps=1)
        _,term,trunc,_=step(p,insertion=.5)
        self.assertFalse(term.item());self.assertTrue(trunc.item())
        self.assertAlmostEqual(p.shaping_reward.item(),-.001,places=5)

    def test_world_reset_and_replay_latches(self):
        p=ready();step(p,**HELD,detached=True);saved=deepcopy(p)
        p.reset(torch.tensor([True]),state())
        self.assertFalse(p.graph_valid_extract.item());self.assertTrue(saved.graph_valid_extract.item())
        self.assertTrue(all(not v.item() for v in p.graph_regressions.values()))

    def test_candidate_cannot_trade_away_previous_skill(self):
        old={'completed/position':1.,'completed/grip':.8,'success':0.,'closest_distance_m':.01}
        newer=dict(old,success=.2);newer['completed/grip']=.4
        promoted,reasons=acceptance(newer,old)
        self.assertFalse(promoted);self.assertIn('completed/grip',reasons)
        newer=dict(old,success=.2)
        self.assertTrue(acceptance(newer,old)[0])
        self.assertFalse(acceptance(old,old)[0])
        self.assertFalse(acceptance(dict(old,success=.01),old)[0])
        scene_old=dict(old,**{'training_scene/episodes':96,'training_scene/completed/grip':.8})
        scene_bad=dict(scene_old,success=.3,**{'training_scene/completed/grip':.4})
        self.assertIn('training_scene/completed/grip',acceptance(scene_bad,scene_old)[1])
        drop=dict(old,graph_score=100.,ground_drop=1.)
        self.assertLess(graph_rank(drop),graph_rank(old))

    def test_small_promotions_cannot_ratchet_away_a_skill(self):
        first={'completed/grip':.8,'success':0.,'closest_distance_m':.01}
        second=dict(first,success=.1,**{'completed/grip':.77})
        self.assertTrue(acceptance(second,first)[0])
        floor=protect_skills(first,second)
        third=dict(second,success=.2,**{'completed/grip':.74})
        self.assertFalse(acceptance(third,second,floor=floor)[0])

    def test_practice_outcomes_do_not_inflate_full_task_rates(self):
        from treesim.kiwi_rl.harvest_training import episode_metrics
        row=dict(success=False,grasp=False,detached=False,physical_failure=False,stalled=True,timeout=False,duration_s=8.,closest_distance_m=.1)
        result=episode_metrics([row,dict(row,practice=True,success=True)])
        self.assertEqual(result['episodes'],1);self.assertEqual(result['success'],0)
        self.assertEqual(result['practice/success'],1)

import os
@unittest.skipUnless(os.environ.get('GRAPH_GPU_TEST')=='1','GPU opt-in required')
class PracticeGpuTest(unittest.TestCase):
    def test_reachable_boundary_restore_and_new_replay(self):
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.ppo import load_checkpoint
        from treesim.kiwi_rl.harvest_training import HarvestCollector
        from treesim.kiwi_rl.graph_training import StagePractice
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
        from treesim.kiwi_rl.branch_cti import BranchCTI
        torch.set_num_threads(1);torch.manual_seed(42);wp.init();stream=torch.cuda.Stream()
        with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
            rt=FastRuntime(os.environ['FAST_SCENE'],worlds=16,arm_speed_rad_s=.5,task_profile=GRAPH_PROFILE)
            gait=load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval();rt.prepare_settled_reset(gait)
            policy=build_privileged_policy().cuda()
            load_checkpoint(os.environ['GRAPH_CHECKPOINT'],{'teacher':policy})
            collector=HarvestCollector(rt,role='teacher',reward_profile=GRAPH_PROFILE,stall_seconds=8.)
            collector.practice=StagePractice()
            for _ in range(12):collector.collect(policy,gait,128,deterministic=False,store=False)
            self.assertTrue(collector.practice.bank)
            self.assertGreater(collector.practice.starts,0)
            rt.check()
            # A factual root taken after practice resets must reproduce exactly
            # under the same branch validator as ordinary starts.
            queue=PPODecisionQueue(worlds=4,segment_steps=16)
            collector.collect(policy,gait,16,cti_queue=queue,policy_version=0)
            batch=queue.pop(0)
            search_rt=FastRuntime(os.environ['FAST_SCENE'],worlds=12,arm_speed_rad_s=.5,task_profile=GRAPH_PROFILE)
            search_rt.prepare_settled_reset(gait)
            search=BranchCTI(search_rt,gait,roots=4,alternatives=2,search_iterations=1,horizon_seconds=.32,intervention_seconds=.16,block_steps=8)
            (rows,_),metrics=search.run(policy,batch)
            self.assertGreater(metrics['valid_transitions'],0)
            # Contact-heavy roots may diverge across GPU world layouts. Every
            # descendant of a rejected root must be excluded, not tolerated.
            rejected={r['world'] for r in metrics['branch_records'] if not r['eligible']}
            self.assertEqual(len(rejected),metrics['branch_replay_rejected_worlds'])
            self.assertEqual(metrics['valid_branches'],3*(4-len(rejected)))
            for root in rejected:
                self.assertFalse(any(bool(row['mask'][root::4].any()) for row in rows))
