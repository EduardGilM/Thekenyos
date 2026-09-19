import importlib.util
import os
import unittest

@unittest.skipUnless(os.environ.get('FAST_SCENE') and all(importlib.util.find_spec(n) for n in
                     ('torch','warp','mujoco','mujoco_warp')), 'FAST_SCENE and GPU stack required')
class FastRuntimeTest(unittest.TestCase):
    def test_actions_masked_reset_and_latched_failure(self):
        import numpy as np
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            runtime=FastRuntime(os.environ['FAST_SCENE'],worlds=2)
            before=runtime.control.targets.numpy().copy()
            actions=torch.full((2,7),.5,device='cuda:0')
            runtime.step(actions)
            expected=np.clip(before[:,12:]+.025,runtime._action_lower.numpy()[12:],runtime._action_upper.numpy()[12:])
            np.testing.assert_allclose(runtime.control.targets.numpy()[:,12:],expected,atol=1e-6)
            runtime.check()
            np.testing.assert_allclose(runtime.data.time.numpy(), .02, atol=1e-6)
            self.assertGreater(float(np.max(np.abs(runtime.data.qpos.numpy()[0]-runtime._initial_qpos.numpy()))),1e-6)
            state={name:getattr(runtime.data,name).numpy().copy() for name in ('qpos','qvel','time')}
            reward=runtime._reward.numpy().copy()
            runtime.reset(torch.tensor([True,False],device='cuda:0'))
            for name,value in state.items():
                np.testing.assert_array_equal(getattr(runtime.data,name).numpy()[1],value[1])
            np.testing.assert_array_equal(runtime._reward.numpy()[1],reward[1])
            np.testing.assert_allclose(runtime.data.qpos.numpy()[0],runtime._initial_qpos.numpy(),atol=1e-6)
            actions[0,0]=float('nan')
            runtime.step(actions)
            runtime.reset()
            with self.assertRaisesRegex(RuntimeError,'numerical failure'):
                runtime.check()

    def test_deposit_skill_reset_places_free_fruit_at_tcp(self):
        import numpy as np
        import torch
        import warp as wp
        from treesim.kiwi_rl.curriculum import evaluate_skills, stage_named
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            runtime = FastRuntime(os.environ['FAST_SCENE'], worlds=2)
            skills = evaluate_skills(stage_named('deposit_pixels'), 2)
            runtime.configure_skills(skills)
            runtime.reset()
            runtime.check()
            qpos = runtime.data.qpos.numpy()
            tcp = runtime.data.site_xpos.numpy()[:, runtime.tcp_site]
            fruit = qpos[:, runtime._fruit_qposadr:runtime._fruit_qposadr + 3]
            np.testing.assert_allclose(fruit, tcp, atol=5e-3)
            np.testing.assert_array_equal(runtime.task.detached.numpy(), [1, 1])
            np.testing.assert_array_equal(runtime.task.grasped.numpy(), [1, 1])
            np.testing.assert_array_equal(runtime.task.goal.numpy(), [0, 0])
            if runtime.task.equality_id >= 0:
                self.assertFalse(bool(runtime.task.eq_active.numpy()[0, runtime.task.equality_id]))
                self.assertFalse(bool(runtime.task.eq_active.numpy()[1, runtime.task.equality_id]))
            self.assertGreater(float(np.max(np.abs(qpos[0] - runtime._initial_qpos.numpy()))), 1e-6)

if __name__=='__main__':
    unittest.main()
