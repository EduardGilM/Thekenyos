import importlib.util
import inspect
import os
import unittest


class FastRuntimeContractTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('warp'), 'Warp required')
    def test_capacity_and_arm_speed_contract(self):
        from treesim.kiwi_rl.fast_runtime import FastRuntime, _OVERFLOW_BITS

        params = inspect.signature(FastRuntime).parameters
        self.assertEqual(params['arm_speed_rad_s'].default, 2.5)
        self.assertEqual(params['solver_iterations'].default, 100)
        self.assertEqual(_OVERFLOW_BITS['NJMAX_NNZ'], 2)
        self.assertEqual(_OVERFLOW_BITS['BROADPHASE'], 4)
        self.assertEqual(_OVERFLOW_BITS['NARROWPHASE'], 8)

    @unittest.skipUnless(importlib.util.find_spec('warp'), 'Warp required')
    def test_diagnostics_decode_backend_bits_without_world_dump(self):
        import numpy as np
        from treesim.kiwi_rl.fast_runtime import FastRuntime

        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)
            def numpy(self):
                return self.value

        class Efc:
            J_rowadr = Array([[0, 3]])
            J_rownnz = Array([[3, 2]])

        class Data:
            overflow = Array([0, 0])
            nacon = Array([9])
            nefc = Array([2, 0])
            naconmax = 16
            njmax = 4
            njmax_nnz = 8
            efc = Efc()

        runtime = FastRuntime.__new__(FastRuntime)
        runtime.data = Data()
        result = runtime._diagnostics(np.array([2 | (2 << 3), 0]))
        self.assertEqual(result['flagged_world_count'], 1)
        self.assertEqual(result['overflow_bit_counts'], {'NJMAX_NNZ': 1})
        self.assertEqual(result['needed']['constraints_max'], 2)
        self.assertEqual(result['needed']['constraint_nnz_lower_bound'], 5)

@unittest.skipUnless(os.environ.get('FAST_SCENE') and all(importlib.util.find_spec(n) for n in
                     ('torch','warp','mujoco','mujoco_warp')), 'FAST_SCENE and GPU stack required')
class FastRuntimeTest(unittest.TestCase):
    def test_camera_default_is_four_to_three(self):
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        self.assertEqual(inspect.signature(FastRuntime).parameters['resolution'].default, (64, 48))

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

    def test_solver_limit_stays_identifiable_after_backend_reset(self):
        import numpy as np
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            runtime=FastRuntime(os.environ['FAST_SCENE'],worlds=2,arm_speed_rad_s=.5)
            self.assertEqual(runtime.model.opt.iterations,100)
            before=runtime.control.targets.numpy().copy()
            runtime.step(torch.ones((2,7),device='cuda:0'))
            expected=np.clip(before[:,12:]+.01,runtime._action_lower.numpy()[12:],runtime._action_upper.numpy()[12:])
            np.testing.assert_allclose(runtime.control.targets.numpy()[:,12:],expected,atol=1e-6)
            runtime.data.overflow.assign(np.array([512,0],dtype=np.int32))
            runtime._latch()
            runtime.data.overflow.zero_()
            runtime.reset()
            with self.assertRaisesRegex(RuntimeError,"ITERATIONS.*1"):
                runtime.check()

if __name__=='__main__':
    unittest.main()
