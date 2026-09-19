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

class WorldSnapshotTest(unittest.TestCase):
    def runtime(self, worlds):
        from dataclasses import make_dataclass
        from types import SimpleNamespace
        annotation = SimpleNamespace(shape=('nworld', 'width'))
        constraint = make_dataclass('Constraint', [('force', annotation)])
        data = make_dataclass('Data', [('qpos', annotation), ('efc', constraint),
                                      ('shared', SimpleNamespace(shape=('nq',)))])
        rt = _Part(worlds=worlds, manifest={'model_sha256': 'same-model'}, profile=('same',))
        rt.data = data(np.arange(worlds * 3, dtype=np.float32).reshape(worlds, 3),
                       constraint(np.arange(worlds * 2, dtype=np.float32).reshape(worlds, 2)),
                       np.arange(worlds, dtype=np.float32))
        rt.control, rt.task = _Part(), _Part()
        for owner, names in ((rt, counterfactual._RUNTIME_WORLD),
                             (rt.control, counterfactual._CONTROL_WORLD),
                             (rt.task, counterfactual._TASK_WORLD)):
            for name in names:
                setattr(owner, name, np.arange(worlds, dtype=np.float32).copy())
        rt.control.qids = np.arange(worlds, dtype=np.int32)
        return rt

    def test_subset_is_immutable_source_unchanged_and_target_buffers_preserved(self):
        import torch
        from unittest.mock import patch
        source, target = self.runtime(8), self.runtime(3)
        target.data.qpos[:] = -1
        buffer = target.data.qpos
        app = {'memory': torch.tensor([[6.], [1.]]), 'tick': 8}
        with patch.object(counterfactual, '_world_profile', side_effect=lambda rt: rt.profile):
            snapshot = counterfactual.capture_worlds(source, [6, 1], application_state=app)
            source.data.qpos[6] += 100
            app['memory'] += 100
            restored = counterfactual.restore_worlds(target, snapshot, [2, 0])
        self.assertIs(buffer, target.data.qpos)
        np.testing.assert_array_equal(target.data.qpos, [[3, 4, 5], [-1, -1, -1], [18, 19, 20]])
        np.testing.assert_array_equal(target.control.targets, [1, 1, 6])
        np.testing.assert_array_equal(target.control.qids, [0, 1, 2])
        torch.testing.assert_close(restored['memory'], torch.tensor([[6.], [1.]]))
        self.assertEqual(snapshot.source_world_ids, (6, 1))
        self.assertNotIn('data.shared', snapshot.arrays)
        self.assertTrue(all(value.shape[0] == 2 for value in snapshot.arrays.values()))

    def test_rejects_unknown_buffers_and_invalid_ids(self):
        from unittest.mock import patch
        source = self.runtime(8)
        with patch.object(counterfactual, '_world_profile', return_value=('same',)):
            for ids in ([1, 1], [-1], [8], [], [1.5]):
                with self.assertRaises(ValueError):
                    counterfactual.capture_worlds(source, ids)
            source.task.new_state = np.zeros(8)
            with self.assertRaisesRegex(ValueError, 'Unclassified runtime array'):
                counterfactual.capture_worlds(source, [1])

    def test_validates_all_layouts_before_mutating_any_target_buffer(self):
        from unittest.mock import patch
        source, target = self.runtime(8), self.runtime(2)
        before = target.data.qpos.copy()
        with patch.object(counterfactual, '_world_profile', side_effect=lambda rt: rt.profile):
            snapshot = counterfactual.capture_worlds(source, [6, 1])
            snapshot.arrays['task.failed'] = snapshot.arrays['task.failed'].double()
            with self.assertRaisesRegex(ValueError, 'buffer layout mismatch'):
                counterfactual.restore_worlds(target, snapshot)
            np.testing.assert_array_equal(target.data.qpos, before)
            target.manifest['model_sha256'] = 'different-model'
            with self.assertRaisesRegex(ValueError, 'model or physics profile mismatch'):
                counterfactual.restore_worlds(target, snapshot)


@unittest.skipUnless(os.environ.get('COUNTERFACTUAL_WORLD_GPU_TEST') == '1' and
                     os.environ.get('FAST_SCENE') and os.environ.get('GAIT_CHECKPOINT'),
                     'explicit world-transfer GPU opt-in, scene and gait required')
class CrossRuntimeReplayTest(unittest.TestCase):
    def test_64_step_factual_policy_replay_across_world_counts(self):
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy, privileged_observation
        from treesim.kiwi_rl.harvest_training import signals
        source_worlds = int(os.environ.get('CTI_REPLAY_SOURCE_WORLDS', '8'))
        target_worlds = int(os.environ.get('CTI_REPLAY_TARGET_WORLDS', '2'))
        self.assertGreater(source_worlds, target_worlds)
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)), torch.no_grad():
            torch.manual_seed(712)
            source = FastRuntime(os.environ['FAST_SCENE'], worlds=source_worlds)
            target = FastRuntime(os.environ['FAST_SCENE'], worlds=target_worlds)
            policy = build_privileged_policy().cuda().eval()
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            memory = torch.zeros(source_worlds, 64, device='cuda:0')
            ids = torch.linspace(source_worlds - 1, 0, target_worlds, device='cuda:0').long()
            def act():
                nonlocal memory
                obs = source.observe().clone()
                mean, logstd, _, memory = policy(privileged_observation(source, obs), obs, memory)
                action = (mean + logstd.exp() * torch.randn_like(mean)).tanh()
                gait_action = gait(obs)
                source.set_gait_actions(gait_action)
                return action, gait_action
            for _ in range(8):
                action, _ = act()
                source.step(action)
                source.check()
            snapshot = counterfactual.capture_worlds(source, ids, application_state={'memory': memory[ids].clone()})
            root_obs = source.observe()[ids].clone()
            steps = []
            for _ in range(64):
                action, gait_action = act()
                _, reward, done, _ = source.step(action)
                source.check()
                steps.append(dict(action=action[ids].clone(), gait=gait_action[ids].clone(),
                    qpos=wp.to_torch(source.data.qpos)[ids].clone(),
                    qvel=wp.to_torch(source.data.qvel)[ids].clone(),
                    targets=wp.to_torch(source.control.targets)[ids].clone(),
                    observation=source.observe()[ids].clone(),
                    state={key: value[ids].clone() for key, value in signals(source).items()},
                    done=done[ids].clone(), reward=reward[ids].clone()))
            restored = counterfactual.restore_worlds(target, snapshot)
            torch.testing.assert_close(restored['memory'], snapshot.application_state['memory'], rtol=0, atol=0)
            torch.testing.assert_close(target.observe(), root_obs, rtol=0, atol=1e-6)
            maxima = dict(qpos=0., qvel=0., targets=0.)
            for tick, expected in enumerate(steps):
                target.set_gait_actions(expected['gait'])
                _, reward, done, _ = target.step(expected['action'])
                target.check()
                for key, array, tolerance in (('qpos', target.data.qpos, 1e-4),
                                              ('qvel', target.data.qvel, 1e-3),
                                              ('targets', target.control.targets, 1e-6)):
                    actual = wp.to_torch(array)
                    maxima[key] = max(maxima[key], float((actual - expected[key]).abs().max()))
                    torch.testing.assert_close(actual, expected[key], rtol=0, atol=tolerance,
                                               msg=f'{key} factual replay mismatch at tick {tick}')
                for key, value in signals(target).items():
                    if value.dtype == torch.bool:
                        torch.testing.assert_close(value, expected['state'][key], rtol=0, atol=0,
                                                   msg=f'{key} discrete replay mismatch at tick {tick}')
                torch.testing.assert_close(done, expected['done'], rtol=0, atol=0)
            print(f'WORLD_REPLAY source={source_worlds} target={target_worlds} steps=64 maxima={maxima}', flush=True)


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
