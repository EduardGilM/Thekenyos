import os
import unittest
from unittest.mock import patch
import torch
from treesim.kiwi_rl.harvest_training import EpisodeProgress


def state(distance=.2,basket=.8,grasp=False,detached=False,force=0.):
    return dict(distance=torch.tensor([distance]),basket_distance=torch.tensor([basket]),
        grasp=torch.tensor([grasp]),holding=torch.tensor([grasp]),detached=torch.tensor([detached]),touching=torch.tensor([grasp]),
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
        slipped['holding']=torch.tensor([False])
        p.step(slipped,zero)
        self.assertEqual(p.stale.item(),1)
        self.assertEqual(p.best_force.item(),0)

    def test_damaging_detachment_never_gets_positive_guidance(self):
        for grasp in (False, True):
            p=EpisodeProgress(state(),stall_steps=3,max_steps=20)
            hit=state(distance=.01,grasp=grasp,detached=True)
            hit['touching']=torch.tensor([True])
            hit['failed']=torch.tensor([True])
            reward,term,_,_=p.step(hit,torch.tensor([True]))
            self.assertTrue(term.item())
            self.assertLessEqual(reward.item(),-5.)

    def test_touching_without_current_grasp_does_not_earn_detachment_bonus(self):
        p=EpisodeProgress(state(),stall_steps=3,max_steps=20)
        hit=state(detached=True,grasp=True)
        hit['holding']=torch.tensor([False])
        # An earlier grasp can be logged, but it is not a retained detachment.
        p.ever_grasp[:]=True
        reward,*_=p.step(hit,torch.tensor([False]))
        self.assertLess(reward.item(),0.)

    def test_checkpoint_selection_rejects_destructive_detachment(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from train_harvest_fast import evaluation_score
        intact=dict(success=0.,physical_failure=0.,grasp=0.,closest_distance_m=.2)
        destructive=dict(success=0.,physical_failure=1.,grasp=1.,closest_distance_m=.01)
        self.assertGreater(evaluation_score(intact),evaluation_score(destructive))


class _CameraRuntime:
    """Small CPU fake for collector camera delivery and reset semantics."""
    def __init__(self, *, end_first_world=False):
        self.worlds, self.device_name, self.control_dt = 2, 'cpu', .02
        self.chassis, self.fruit_body = 0, 1
        self.data = type('Data', (), {})()
        self.data.xpos = torch.tensor([[[0., 0., 0.], [.2, 0., 0.]],
                                       [[0., 0., 0.], [.2, 0., 0.]]])
        self.data.xmat = torch.eye(3).repeat(2, 2, 1, 1)
        self._distance = torch.tensor([.2, .2])
        self.task = type('Task', (), {})()
        for name in ('ever_grasped', 'stable_grasp', 'detached', 'hand_contact',
                     'success', 'failed'):
            setattr(self.task, name, torch.zeros(2, dtype=torch.bool))
        self.task.stem_force = torch.zeros(2)
        self.camera_frames = 0
        self.step_count = 0
        self.end_first_world = end_first_world

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(2, dtype=torch.bool)
        self.task.success[mask] = False
        self.task.failed[mask] = False

    def observe(self):
        return torch.zeros(2, 84)

    def pixels(self):
        self.camera_frames += 1
        return torch.stack([torch.full((5, 2, 2), self.camera_frames * 10 + world)
                            for world in range(self.worlds)])

    def set_gait_actions(self, actions):
        pass

    def step(self, actions):
        done = torch.zeros(2, dtype=torch.bool)
        if self.end_first_world and self.step_count == 0:
            self.task.success[0] = True
            done[0] = True
        self.step_count += 1
        return None, None, done, {}

    def check(self):
        pass


class _CameraPolicy:
    def __init__(self):
        self.inputs = []

    def __call__(self, rgbd, obs, memory):
        self.inputs.append(rgbd.clone())
        return torch.zeros(2, 7), torch.zeros(7), torch.zeros(2), memory


class CameraClockTest(unittest.TestCase):
    def setUp(self):
        import warp as wp
        self.wp_patch = patch.object(wp, 'to_torch', side_effect=lambda value: value)
        self.wp_patch.start()

    def tearDown(self):
        self.wp_patch.stop()

    @staticmethod
    def gait(obs):
        return torch.zeros(2, 12)

    def test_masked_episode_reset_keeps_other_world_camera_frame(self):
        from treesim.kiwi_rl.harvest_training import HarvestCollector
        runtime = _CameraRuntime(end_first_world=True)
        policy = _CameraPolicy()
        collector = HarvestCollector(runtime, role='student', stall_seconds=10., max_episode_seconds=20.)
        collector.collect(policy, self.gait, 2, deterministic=True)
        self.assertTrue(torch.equal(policy.inputs[0][1], policy.inputs[1][1]))
        self.assertTrue(torch.all(policy.inputs[1][0] == 20))
        self.assertTrue(torch.all(policy.inputs[1][1] == 11))

    def test_odd_tick_bootstrap_uses_held_camera_frame(self):
        from treesim.kiwi_rl.harvest_training import HarvestCollector
        runtime = _CameraRuntime()
        policy = _CameraPolicy()
        collector = HarvestCollector(runtime, role='student', stall_seconds=10., max_episode_seconds=20.)
        collector.collect(policy, self.gait, 1, deterministic=True)
        self.assertEqual(runtime.camera_frames, 1)
        self.assertTrue(torch.equal(policy.inputs[0], policy.inputs[1]))


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
