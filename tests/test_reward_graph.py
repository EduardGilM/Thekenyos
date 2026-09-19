import unittest
from copy import deepcopy
import torch
from treesim.kiwi_rl.reward_graph import GRAPH_PROFILE, approach_score, enclosure_features, ellipsoid_extent, release_position_distance
from treesim.kiwi_rl.harvest_training import EpisodeProgress
from treesim.kiwi_rl.selective_cti import _choose_winners
from tests.test_selective_cti import outcome


def state(**changes):
    values = dict(distance=.5, basket_distance=.8, grasp=False, holding=False,
        detached=False, touching=False, stem_force=0., success=False, failed=False,
        settle_time=0., enclosed=False, insertion=0., closure=0., slip=0.,
        bilateral=False, max_load=0., damage=0., ground_contact=False, release_distance=1.)
    values.update(changes)
    return {key: torch.tensor([value]) for key, value in values.items()}


def progress(initial=None, **kwargs):
    return EpisodeProgress(initial or state(), reward_profile=GRAPH_PROFILE, stall_steps=1000, **kwargs)


def step(p, **kwargs):
    return p.step(state(**kwargs), torch.tensor([False]))


class GraphTest(unittest.TestCase):
    def test_exact_approach_saturation_and_regression(self):
        torch.testing.assert_close(approach_score(torch.tensor([.4,.5,.6,1.5])),torch.tensor([1.,1.,.9,0.]))
        p=progress();step(p,distance=.1);self.assertEqual(p.graph_score.item(),1.)
        step(p,distance=.6);self.assertLess(p.graph_score.item(),1.)

    def test_collision_enclosure_extent_empty_closure(self):
        finger=torch.tensor([[[-.1,-.1,.05],[.1,.1,.07]]])
        jaw=torch.tensor([[[-.1,-.1,-.07],[.1,.1,-.05]]])
        extent=torch.tensor([[.03,.03,.04]])
        inside,insertion,closure=enclosure_features(torch.zeros(1,3),extent,finger,jaw)
        self.assertTrue(inside.item());self.assertEqual(insertion.item(),1.);self.assertGreater(closure.item(),0)
        outside,_,closure=enclosure_features(torch.tensor([[.3,0.,0.]]),extent,finger,jaw)
        self.assertFalse(outside.item());self.assertEqual(closure.item(),0.)
        closed=finger.clone();closed[:,:,2]-=.1
        inside,_,closure=enclosure_features(torch.zeros(1,3),extent,closed,jaw)
        self.assertFalse(inside.item());self.assertEqual(closure.item(),0.)
        # Rigid scene rotation cancels in wrist-relative transforms.
        r=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
        torch.testing.assert_close(ellipsoid_extent(r.transpose(1,2)@r,extent),extent)

    def test_measured_bilateral_side_grasp_is_inside_open_jaws(self):
        # Saved GPU checkpoint205 pose, with 5.24 N finger / 2.24 N jaw load.
        center=torch.tensor([[.2172552,-.0362215,-.0237873]])
        extent=torch.tensor([[.0307458,.0274551,.0324795]])
        finger=torch.tensor([[[.1394505,-.047,.0018425],[.2480924,.047,.0459613]]])
        jaw=torch.tensor([[[.1461,-.040797,-.059955],[.235654,.040774,-.032153]]])
        inside,_,_=enclosure_features(center,extent,finger,jaw)
        self.assertTrue(inside.item())
        for displacement in ([.1,0.,0.],[0.,-.05,0.],[0.,0.,.07]):
            inside,_,closure=enclosure_features(center+torch.tensor([displacement]),extent,finger,jaw)
            self.assertFalse(inside.item());self.assertEqual(closure.item(),0.)

    def test_grip_dwell_slip_opening_and_overload(self):
        p=progress();held=dict(enclosed=True,insertion=1.,closure=1.,bilateral=True,holding=True,touching=True,max_load=1.)
        for _ in range(6):step(p,**held)
        self.assertTrue(p.graph_grip.item());self.assertEqual(p.graph_score.item(),3.)
        step(p,**{**held,'slip':.1});self.assertFalse(p.graph_grip.item());self.assertLess(p.graph_score.item(),3.)
        step(p,**{**held,'bilateral':False,'holding':False,'enclosed':False,'insertion':.5})
        self.assertLess(p.graph_score.item(),2.)
        step(p,**{**held,'max_load':16.,'stem_force':9.});self.assertEqual(p.graph_score.item(),0.)

    def test_grade_order_terminal_partial_and_irreversible_detach(self):
        p=progress();held=dict(enclosed=True,insertion=1.,closure=1.,bilateral=True,holding=True,touching=True,max_load=1.,stem_force=7.)
        for _ in range(6):step(p,**held)
        attached=p.graph_score.item()
        step(p,**held,detached=True);secured=p.graph_score.item()
        reward,term,_,_=step(p,detached=True,ground_contact=True)
        self.assertTrue(term.item());self.assertTrue(p.graph_ground_drop.item())
        self.assertGreater(p.graph_score.item(),attached);self.assertLess(p.graph_score.item(),secured)
        self.assertAlmostEqual(p.task_reward.item(),-.001,places=6)
        step(p);self.assertTrue(p.graph_detached.item())

    def test_release_target_above_basket_accounts_for_fruit_extent(self):
        from treesim.basket import CENTER, SIZE
        extent=torch.tensor([[.03,.03,.04]])
        base=torch.as_tensor(CENTER,dtype=torch.float32)[None].clone()
        above=base.clone();above[:,2]+=SIZE[2]+extent[:,2]+.04
        self.assertAlmostEqual(release_position_distance(above,extent).item(),0.,places=7)
        self.assertGreater(release_position_distance(base,extent).item(),.1)
        outside=above.clone();outside[:,0]+=SIZE[0]
        self.assertGreater(release_position_distance(outside,extent).item(),0.)

    def test_settle_grade_reset_and_no_wrong_location_release_credit(self):
        p=progress();step(p,detached=True);self.assertAlmostEqual(p.graph_score.item(),3.6,places=5)
        step(p,detached=True,settle_time=1.);self.assertEqual(p.graph_score.item(),5.5)
        step(p,detached=True,settle_time=0.);self.assertAlmostEqual(p.graph_score.item(),3.6,places=5)
        step(p,detached=True,settle_time=2.,success=True);self.assertEqual(p.graph_score.item(),6.)

    def test_discounted_cycle_no_profit_and_guidance_independent(self):
        p=progress(guidance=0.);total=0.
        for i,d in enumerate((.8,.5,.8,.5)):
            reward,*_=step(p,distance=d);total+=p.gamma**i*reward.item()
        self.assertLess(total,0.)
        self.assertEqual(p.graph_score.item(),1.)

    def test_mask_reset_and_copy_all_graph_state(self):
        initial={k:v.repeat(2) for k,v in state().items()};p=progress(initial)
        now={k:v.repeat(2) for k,v in state(detached=True).items()}
        p.step(now,torch.zeros(2,dtype=torch.bool));copy=deepcopy(p)
        p.reset(torch.tensor([True,False]),initial)
        self.assertFalse(p.graph_detached[0]);self.assertTrue(p.graph_detached[1])
        self.assertTrue(copy.graph_detached.all())
        for k,v in vars(copy).items():
            if k.startswith('graph_'): self.assertEqual(v.shape,(2,))

    def test_ppo_queue_preserves_every_graph_tensor_and_configuration(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
        initial={k:v.repeat(2) for k,v in state().items()}
        p=progress(initial,control_dt=.04,gamma=.99)
        p.graph_dwell[:]=torch.tensor([.04,.08]);p.graph_best[:]=torch.tensor([2.,3.])
        collector=SimpleNamespace(progress=p,memory=torch.zeros(2,64),reset_mask=torch.zeros(2,dtype=torch.bool),
            episode_ids=torch.arange(2),tick=7)
        runtime=SimpleNamespace(device_name='cpu')
        with patch('treesim.kiwi_rl.harvest_training.signals',return_value=initial), patch(
                'treesim.kiwi_rl.counterfactual.capture_worlds',return_value=None):
            batch=PPODecisionQueue(worlds=2).begin(runtime,collector,torch.nn.Linear(1,1),1)
        restored=batch.initial_progress
        self.assertEqual(restored.control_dt,.04);self.assertEqual(restored.gamma,.99)
        self.assertEqual(restored.reward_profile,GRAPH_PROFILE)
        for key,value in vars(p).items():
            if key.startswith('graph_'):
                torch.testing.assert_close(getattr(restored,key),value[list(batch.source_world_ids)])
        p.graph_dwell.zero_();self.assertGreater(restored.graph_dwell.sum().item(),0)

    def test_ppo_and_cti_discounted_grade_rank_match(self):
        root=state();root['graph_score']=torch.tensor([1.])
        outcomes=[]
        for ending in (dict(),dict(detached=True,ground_contact=True)):
            p=progress();total=0.
            for tick in range(3):
                reward,*_=step(p,**(ending if tick==2 else {}))
                total+=p.gamma**tick*reward.item()
            expected=sum(p.gamma**t*(-.001) for t in range(3))+p.gamma**3*p.graph_score.item()-1.
            self.assertAlmostEqual(total,expected,places=5)
            outcomes.append(outcome(score=total,graph_score=p.graph_score.item(),replay_invalid=False,
                                    lost_fruit=bool(ending),retained=False))
        winner,_=_choose_winners(outcomes,root);self.assertEqual(winner.item(),1)

    def test_cti_same_return_safe_drop_and_invalid_replay(self):
        root=state();root['graph_score']=torch.tensor([1.])
        base=outcome(score=0.,graph_score=1.,replay_invalid=False)
        drop=outcome(score=2.,graph_score=3.6,detached=True,lost_fruit=True,retained=False,replay_invalid=False)
        winner,_=_choose_winners([base,drop],root);self.assertEqual(winner.item(),1)
        held=outcome(score=2.5,graph_score=4.,holding=True,detached=True,replay_invalid=False)
        winner,_=_choose_winners([base,drop,held],root);self.assertEqual(winner.item(),2)
        base['replay_invalid'][:]=True
        winner,reasons=_choose_winners([base,drop],root);self.assertEqual(winner.item(),-1)
        self.assertIn('factual_replay_mismatch',reasons[1][0])

if __name__=='__main__':unittest.main()
