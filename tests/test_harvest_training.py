import os
import unittest
import torch
from treesim.kiwi_rl.harvest_training import EpisodeProgress


def state(distance=.2,basket=.8,grasp=False,detached=False,force=0.):
    return dict(distance=torch.tensor([distance]),basket_distance=torch.tensor([basket]),
        grasp=torch.tensor([grasp]),detached=torch.tensor([detached]),touching=torch.tensor([grasp]),
        stem_force=torch.tensor([force]),success=torch.tensor([False]),failed=torch.tensor([False]))


class ProgressTest(unittest.TestCase):
    def test_stalls_on_no_new_best_even_if_it_oscillates(self):
        progress=EpisodeProgress(state(),stall_steps=3,max_steps=20)
        for distance in (.21,.20):
            _,term,_,_=progress.step(state(distance),torch.tensor([False]));self.assertFalse(term.item())
        _,term,_,stall=progress.step(state(.201),torch.tensor([False]))
        self.assertTrue(term.item());self.assertTrue(stall.item())

    def test_new_grasp_pull_and_carry_extend_episode_without_repeat_bonus(self):
        progress=EpisodeProgress(state(),stall_steps=3,max_steps=20)
        zero=torch.tensor([False])
        progress.step(state(),zero); progress.step(state(),zero)
        reward,term,_,_=progress.step(state(grasp=True),zero)
        self.assertFalse(term.item());self.assertGreater(reward.item(),1.)
        repeat,*_=progress.step(state(grasp=True),zero);self.assertLess(repeat.item(),.01)
        progress.step(state(grasp=True,force=1.),zero);self.assertEqual(progress.stale.item(),0)
        progress.step(state(grasp=True,detached=True),zero)
        progress.step(state(grasp=True,detached=True,basket=.7),zero)
        self.assertEqual(progress.stale.item(),0)
        # Timeout is a truncation, not a physical or stalled terminal.
        progress.max_steps=int(progress.age.item())+1
        _,term,truncated,_=progress.step(state(grasp=True,detached=True,basket=.6),zero)
        self.assertFalse(term.item());self.assertTrue(truncated.item())

    def test_reset_clears_episode_history(self):
        p=EpisodeProgress(state(),stall_steps=3,max_steps=20)
        p.step(state(grasp=True,detached=True),torch.tensor([False]))
        p.reset(torch.tensor([True]),state())
        self.assertFalse(p.ever_grasp.item());self.assertEqual(p.age.item(),0)

    def test_past_grasp_does_not_make_unheld_stem_load_progress(self):
        p=EpisodeProgress(state(grasp=True),stall_steps=3,max_steps=20)
        zero=torch.tensor([False])
        p.step(state(grasp=True),zero)
        slipped=state(grasp=True,force=2.)
        slipped['touching']=torch.tensor([False])
        p.step(slipped,zero)
        self.assertEqual(p.stale.item(),1)
        self.assertEqual(p.best_force.item(),0)


@unittest.skipUnless(os.environ.get('FAST_SCENE') and os.environ.get('GAIT_CHECKPOINT'),'JP scene and gait required')
class PersistentGpuTest(unittest.TestCase):
    def test_two_buffers_keep_physics_and_memory_and_replay_exactly(self):
        import sys
        from pathlib import Path
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.harvest_training import HarvestCollector
        from treesim.kiwi_rl.control import load_gait_artifact
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from train_fast import update
        from train_physical_smoke import build_policy
        wp.init();stream=torch.cuda.Stream()
        with torch.cuda.stream(stream),wp.ScopedStream(wp.stream_from_torch(stream)):
            rt=FastRuntime(os.environ['FAST_SCENE'],worlds=2,camera='hand_camera')
            policy=build_policy().cuda(); gait=load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            c=HarvestCollector(rt,stall_seconds=4,max_episode_seconds=30)
            c.collect(policy,gait,64,deterministic=True)
            previous=c.memory.clone()
            rows,bootstrap,episodes=c.collect(policy,gait,64,deterministic=True)
            self.assertFalse(episodes)
            torch.testing.assert_close(rows[0]['initial_memory'],previous)
            self.assertFalse(rows[0]['reset'].any())
            torch.testing.assert_close(wp.to_torch(rt.data.time),torch.full_like(wp.to_torch(rt.data.time),2.56),atol=1e-4,rtol=1e-4)
            result=update(policy,torch.optim.Adam(policy.parameters(),lr=1e-4),rows,bootstrap,1)
            self.assertLess(result['kl'],1e-6)
