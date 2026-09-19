import importlib.util
import os
import unittest

import numpy as np

from treesim.kiwi_rl import counterfactual


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        np.copyto(self.value, value)


class _Part:
    def __init__(self, **values):
        self.__dict__.update(values)


class RuntimeSnapshotTest(unittest.TestCase):
    def runtime(self):
        runtime = _Part()
        runtime.model = _Part(randomized_mass=np.array([1., 2.]), opt=_Part(timestep=.005, iterations=80))
        runtime.gpu_model = _Part()
        runtime.data = _Part(qpos=_Array([[1., 2.]]), solver=_Part(warmstart=_Array([[3.]])))
        runtime.control = _Part(targets=_Array([[4.]]), previous=_Array([[5.]]))
        runtime.task = _Part(eq_active=_Array([[1]]), grasp_time=_Array([.4]))
        runtime.rig = _Part(rgb=[_Array([[[.1]]])], depth=[_Array([[[.2]]])],
                            valid=[_Array([[[1]]])], frame_id=7,
                            timestamp_s=.14, available=True)
        return runtime

    def test_restores_all_buffers_rng_and_application_state_in_place(self):
        import random
        runtime = self.runtime()
        buffer = runtime.data.qpos
        generator = np.random.default_rng(92)
        random.seed(7)
        np.random.seed(9)
        snapshot = counterfactual.capture(runtime, application_state={'memory': [1, 2], 'tick': 7},
                                          rng_generators={'policy': generator})
        expected_python, expected_numpy, expected_generator = random.random(), np.random.rand(), generator.random()
        expected_values = [runtime.data.qpos.numpy(), runtime.data.solver.warmstart.numpy(),
                           runtime.control.targets.numpy(), runtime.task.eq_active.numpy(),
                           runtime.rig.rgb[0].numpy(), runtime.model.randomized_mass.copy()]
        runtime.data.qpos.assign([[10., 20.]])
        runtime.data.solver.warmstart.assign([[30.]])
        runtime.control.targets.assign([[40.]])
        runtime.task.eq_active.assign([[0]])
        runtime.rig.rgb[0].assign([[[.9]]])
        runtime.model.randomized_mass[:] = [8., 9.]
        runtime.rig.frame_id, runtime.rig.timestamp_s, runtime.rig.available = 100, 2., False
        generator.random()

        app = counterfactual.restore(runtime, snapshot, rng_generators={'policy': generator})
        self.assertIs(runtime.data.qpos, buffer)
        for actual, expected in zip((runtime.data.qpos.numpy(), runtime.data.solver.warmstart.numpy(),
                                     runtime.control.targets.numpy(), runtime.task.eq_active.numpy(),
                                     runtime.rig.rgb[0].numpy(), runtime.model.randomized_mass), expected_values):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual((runtime.rig.frame_id, runtime.rig.timestamp_s, runtime.rig.available), (7, .14, True))
        self.assertEqual(app, {'memory': [1, 2], 'tick': 7})
        self.assertEqual((random.random(), np.random.rand(), generator.random()),
                         (expected_python, expected_numpy, expected_generator))

    def test_rejects_changed_scalar_physics_profile(self):
        runtime = self.runtime()
        snapshot = counterfactual.capture(runtime)
        runtime.model.opt.iterations = 120
        with self.assertRaisesRegex(ValueError, 'physics profile changed'):
            counterfactual.restore(runtime, snapshot)

@unittest.skipUnless(os.environ.get('FAST_SCENE') and all(importlib.util.find_spec(n) for n in
                     ('torch', 'warp', 'mujoco', 'mujoco_warp')),
                     'FAST_SCENE and CUDA MuJoCo-Warp stack required')
class MidContactReplayTest(unittest.TestCase):
    def test_same_actions_replay_from_midcontact_snapshot(self):
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime

        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            rt = FastRuntime(os.environ['FAST_SCENE'], worlds=2, camera='hand_camera')
            # Put the released fruit inside a real jaw collider so the snapshot
            # is guaranteed to contain a fruit/hand contact, independent of
            # where the exported canopy places its anchor.
            rt.task.eq_active.zero_()
            kinds = rt.task.kind.numpy()
            jaw_geoms = np.flatnonzero(np.isin(kinds, (4, 5)))
            self.assertGreater(len(jaw_geoms), 0, 'scene has no finger/jaw collision geometry')
            qpos = rt.data.qpos.numpy()
            joint = int(rt.model.body_jntadr[rt.fruit_body])
            qadr = int(rt.model.jnt_qposadr[joint])
            geom_id = int(jaw_geoms[0])
            geom_positions = rt.data.geom_xpos.numpy()
            for world in range(rt.worlds):
                qpos[world, qadr:qadr + 3] = geom_positions[world, geom_id]
            rt.data.qpos.assign(qpos)
            actions = torch.zeros((rt.worlds, 7), dtype=torch.float32, device=rt.device_name)
            for _ in range(100):
                rt.step(actions)
                if np.all(rt.task.hand_contact.numpy() != 0):
                    break
            self.assertTrue(np.all(rt.task.hand_contact.numpy() != 0),
                                f'fruit did not reach hand contact; pos={rt.data.xpos.numpy()[0, rt.fruit_body]}, '
                                f'nacon={rt.data.nacon.numpy()}, flags={rt._flags.numpy()}')
            fruit_geom = int(rt.task.fruit_geom.numpy()[0])
            contacts = rt.data.contact.geom.numpy()
            active = int(rt.data.nacon.numpy()[0])
            self.assertTrue(any(fruit_geom in pair for pair in contacts[:active]),
                            'contact buffer has no active fruit contact pair')
            state = {'memory': torch.arange(64, device=rt.device_name).reshape(1, 64),
                     'camera_tick': 12, 'progress': {'age': [3, 3]}}
            frame0 = rt.pixels().clone()
            snapshot_frame_id = rt.rig.frame_id
            snapshot = counterfactual.capture(rt, application_state=state)
            jaw_cap = rt.control.jaw_cap.numpy().copy()
            rt.control.jaw_cap.zero_()
            restored_state = counterfactual.restore(rt, snapshot)
            self.assertEqual(restored_state['camera_tick'], 12)
            np.testing.assert_array_equal(rt.control.jaw_cap.numpy(), jaw_cap)
            self.assertEqual(rt.rig.frame_id, snapshot_frame_id)
            torch.testing.assert_close(rt.rig.tensor('hand_camera'), frame0, rtol=0, atol=0)
            actions = torch.full((rt.worlds, 7), .05, dtype=torch.float32, device=rt.device_name)
            first_outputs = []
            first_frames = []
            for _ in range(2):
                result = rt.step(actions)
                first_outputs.append((result[1].clone(), result[2].clone()))
                first_frames.append(rt.pixels().clone())
            first_qpos = rt.data.qpos.numpy().copy()
            first_qvel = rt.data.qvel.numpy().copy()
            first_time = rt.data.time.numpy().copy()
            first_task = {key: value.numpy().copy() for key, value in rt.task.outputs().items()}

            restored_state = counterfactual.restore(rt, snapshot)
            self.assertEqual(restored_state['camera_tick'], 12)
            np.testing.assert_array_equal(rt.control.jaw_cap.numpy(), jaw_cap)
            self.assertEqual(rt.rig.frame_id, snapshot_frame_id)
            torch.testing.assert_close(rt.rig.tensor('hand_camera'), frame0, rtol=0, atol=0)
            second_outputs = []
            second_frames = []
            for _ in range(2):
                result = rt.step(actions)
                second_outputs.append((result[1].clone(), result[2].clone()))
                second_frames.append(rt.pixels().clone())
            # CUDA solver reductions can differ by a few float32 ulps at this
            # deliberately overlapping contact; discrete task state is exact.
            np.testing.assert_allclose(rt.data.qpos.numpy(), first_qpos, rtol=0, atol=1e-5)
            np.testing.assert_allclose(rt.data.qvel.numpy(), first_qvel, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(rt.data.time.numpy(), first_time)
            for key, expected in first_task.items():
                np.testing.assert_array_equal(rt.task.outputs()[key].numpy(), expected)
            for (expected_reward, expected_done), (actual_reward, actual_done) in zip(first_outputs, second_outputs):
                torch.testing.assert_close(expected_reward, actual_reward, rtol=0, atol=1e-6)
                torch.testing.assert_close(expected_done, actual_done, rtol=0, atol=0)
            for expected, actual in zip(first_frames, second_frames):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(rt.rig.frame_id, snapshot_frame_id + 2)
            rt.check()


if __name__ == '__main__':
    unittest.main()
