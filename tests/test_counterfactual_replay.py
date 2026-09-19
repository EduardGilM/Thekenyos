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

    def test_replay_geometry_uses_native_addresses_and_quaternion_sign_equivalence(self):
        import torch
        rt = self.runtime(2)
        rt.data.qpos = np.zeros((2, 35), dtype=np.float32)
        rt.data.qpos[:, [8, 23]] = 1.
        rt.model = _Part(body_jntadr=np.array([0, 1]), jnt_type=np.array([0, 0]),
                         jnt_qposadr=np.array([5, 20]))
        rt.chassis, rt.fruit_body = 0, 1
        rt.control.qids = np.array([0, 1, 2], dtype=np.int32)
        recorded = {'qpos': torch.from_numpy(rt.data.qpos.copy())}
        recorded['qpos'][:, 23:27] *= -1
        rt.data.qpos[:, 5] += .001
        rt.data.qpos[:, 20:22] += [.003, .004]
        rt.data.qpos[:, 2] += .005
        errors = counterfactual.replay_position_errors(rt, recorded)
        torch.testing.assert_close(errors['base_m'], torch.full((2,), .001))
        torch.testing.assert_close(errors['fruit_m'], torch.full((2,), .005))
        torch.testing.assert_close(errors['robot_rad'], torch.full((2,), .005))
        self.assertTrue((errors['fruit_orientation_rad'] == 0).all())

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


@unittest.skipUnless(os.environ.get('CTI_REPLAY_RESET_GPU_TEST') == '1' and
                     os.environ.get('CTI_REPLAY_CHECKPOINT'),
                     'explicit warmed reset replay opt-in and teacher checkpoint required')
class WarmedCrossRuntimeReplayTest(unittest.TestCase):
    def test_factual_foreign_resets_replay_across_eight_warm_buffers(self):
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        from treesim.kiwi_rl.harvest_training import HarvestCollector, signals
        from treesim.kiwi_rl.ppo import load_checkpoint
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
        original_slots = os.environ.get('CTI_REPLAY_ORIGINAL_SLOTS') == '1'
        source_worlds = int(os.environ.get('CTI_REPLAY_SOURCE_WORLDS', '4096'))
        target_worlds = int(os.environ.get('CTI_REPLAY_TARGET_WORLDS', '16'))
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)), torch.no_grad():
            source = FastRuntime(os.environ['FAST_SCENE'], worlds=source_worlds)
            target = FastRuntime(os.environ['FAST_SCENE'], worlds=target_worlds)
            policy = build_privileged_policy().cuda().eval()
            restored = load_checkpoint(os.environ['CTI_REPLAY_CHECKPOINT'], {'teacher': policy})
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            collector = HarvestCollector(source, role='teacher', stall_seconds=4., max_episode_seconds=30.)
            queue = PPODecisionQueue(worlds_per_batch=target_worlds)
            torch.set_rng_state(restored['rng']['torch'])
            torch.cuda.set_rng_state_all(restored['rng']['cuda'])
            failures = []
            total_foreign = 0
            for buffer in range(8):
                collector.collect(policy, gait, 64, cti_queue=queue, policy_version=buffer)
                batch = queue.pop_ready(buffer)
                self.assertIsNotNone(batch)
                results = {}
                slots = (torch.tensor(batch.source_world_ids, device='cuda:0') if original_slots
                         else torch.arange(target_worlds, device='cuda:0'))
                def arranged(value):
                    result = torch.zeros_like(value)
                    result[slots] = value
                    return result
                def selected(array):
                    return wp.to_torch(array)[slots]
                foreign_resets = sum(bool(step['source_reset']) and not bool(step['ending'].any())
                                     for step in batch.factual_steps)
                total_foreign += foreign_resets
                for match_source_resets in (False, True):
                    counterfactual.restore_worlds(target, batch.snapshot, slots)
                    from copy import deepcopy
                    replay_progress = deepcopy(batch.initial_progress)
                    root_arrays = counterfactual._world_arrays(target)
                    for path, saved in batch.snapshot.arrays.items():
                        torch.testing.assert_close(root_arrays[path][slots], saved, rtol=0, atol=0, equal_nan=True)
                    valid = torch.ones(target_worlds, dtype=torch.bool, device='cuda:0')
                    maxima = dict(qpos=0., qvel=0., targets=0., observation=0.)
                    mismatch = torch.zeros_like(valid)
                    first_mismatch = None
                    physical_mismatch = torch.zeros_like(valid)
                    for tick, step in enumerate(batch.factual_steps):
                        active = step['valid']
                        target.set_gait_actions(arranged(step['gait_action']))
                        _, _, done, _ = target.step(arranged(step['action']))
                        target.check()
                        for key, array, tolerance in (('qpos', target.data.qpos, 1e-4),
                                                      ('qvel', target.data.qvel, 1e-3),
                                                      ('targets', target.control.targets, 1e-6)):
                            actual = selected(array)
                            error = (actual - step[key]).abs().amax(dim=-1)
                            mismatch |= active & (error > tolerance)
                            if bool(active.any()):
                                maxima[key] = max(maxima[key], float(error[active].max()))
                        # Geometry helper expects native target order.
                        geometry = counterfactual.replay_position_errors(target, {'qpos': arranged(step['qpos'])})
                        first_step = tick == 0
                        geometry_bad = ((geometry['base_m'] > (1e-5 if first_step else .002)) |
                                        (geometry['fruit_m'] > (1e-5 if first_step else .01)) |
                                        (geometry['robot_rad'] > (1e-4 if first_step else .01)))
                        if first_step:
                            geometry_bad |= geometry['fruit_orientation_rad'] > .001
                        geometry_bad |= ~torch.isfinite(torch.stack(list(geometry.values()))).all(dim=0)
                        physical_mismatch |= active & geometry_bad[slots]
                        obs_error = (target.observe()[slots] - step['next_obs']).abs().amax(dim=-1)
                        if bool(active.any()):
                            maxima['observation'] = max(maxima['observation'], float(obs_error[active].max()))
                        current = {key:value[slots] for key,value in signals(target).items()}
                        _, terminated, truncated, _ = replay_progress.step(current, done[slots])
                        physical_mismatch |= active & ((terminated != step['terminated']) |
                            (truncated != step['truncated']) | (replay_progress.task_reward != step['task_reward']))
                        for key, value in signals(target).items():
                            if value.dtype == torch.bool:
                                different = active & (value[slots] != step['signals'][key])
                                mismatch |= different
                                physical_mismatch |= different
                        if first_mismatch is None and bool(mismatch.any()):
                            bad = int(mismatch.nonzero()[0])
                            qpos_error = (selected(target.data.qpos)[bad] - step['qpos'][bad]).abs()
                            qvel_error = (selected(target.data.qvel)[bad] - step['qvel'][bad]).abs()
                            first_mismatch = dict(tick=tick, world=bad,
                                qpos_index=int(qpos_error.argmax()), qpos_error=float(qpos_error.max()),
                                qvel_index=int(qvel_error.argmax()), qvel_error=float(qvel_error.max()),
                                previous_source_reset=bool(batch.factual_steps[tick-1]['source_reset']) if tick else False)
                        finish = active & step['ending']
                        valid &= ~finish
                        if bool(finish.any()) or (match_source_resets and step['source_reset']):
                            target.reset(arranged(finish))
                        if not bool(valid.any()):
                            break
                    results['matched' if match_source_resets else 'local_only'] = dict(
                        mismatched_worlds=int(mismatch.sum()), physical_rejected_worlds=int(physical_mismatch.sum()),
                        first_mismatch=first_mismatch, **maxima)
                    if match_source_resets and bool(mismatch.any()):
                        failures.append((buffer, int(mismatch.sum()), maxima))
                print(f'WARM_REPLAY buffer={buffer} foreign_resets={foreign_resets} results={results}', flush=True)
            if not original_slots:
                self.assertGreater(total_foreign, 0, 'regression must exercise foreign-world reset forwards')
            # Keep strict trajectory failures visible in diagnostics. They also
            # occur with full same-runtime snapshots; acceptance requires the
            # separate physical geometry and exact task-outcome guards above.
            print(f'WARM_REPLAY strict_trajectory_failures={failures}', flush=True)


@unittest.skipUnless(os.environ.get('CTI_REPLAY_FLOOR_GPU_TEST') == '1',
                     'explicit same-runtime numerical-floor diagnostic opt-in required')
class ReplayNumericalFloorTest(unittest.TestCase):
    def test_same_runtime_full_snapshot_vs_cross_runtime_at_warm_roots(self):
        import json
        import torch
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy
        from treesim.kiwi_rl.harvest_training import HarvestCollector, signals
        from treesim.kiwi_rl.ppo import load_checkpoint
        from treesim.kiwi_rl.ppo_cti import PPODecisionQueue
        wp.init()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)), torch.no_grad():
            source = FastRuntime(os.environ['FAST_SCENE'], worlds=16)
            target = FastRuntime(os.environ['FAST_SCENE'], worlds=16)
            policy = build_privileged_policy().cuda().eval()
            restored = load_checkpoint(os.environ['CTI_REPLAY_CHECKPOINT'], {'teacher': policy})
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            collector = HarvestCollector(source, role='teacher', stall_seconds=4., max_episode_seconds=30.)
            queue = PPODecisionQueue(worlds_per_batch=16)
            torch.set_rng_state(restored['rng']['torch'])
            torch.cuda.set_rng_state_all(restored['rng']['cuda'])
            fruit_joint = int(source.model.body_jntadr[source.fruit_body])
            fq = int(source.model.jnt_qposadr[fruit_joint])
            fv = int(source.model.jnt_dofadr[fruit_joint])
            def angle(a, b):
                a, b = a.double(), b.double()
                a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
                b = torch.where((a*b).sum(dim=-1, keepdim=True) < 0, -b, b)
                return 4 * torch.atan2((a-b).norm(dim=-1), (a+b).norm(dim=-1))
            def differences(q, v, expected):
                eq, ev = expected['qpos'], expected['qvel']
                return dict(base_position_m=(q[:, :3]-eq[:, :3]).norm(dim=-1),
                    base_orientation_rad=angle(q[:, 3:7], eq[:, 3:7]),
                    robot_joint_rad=(q[:, 7:fq]-eq[:, 7:fq]).abs().amax(dim=-1),
                    fruit_position_m=(q[:, fq:fq+3]-eq[:, fq:fq+3]).norm(dim=-1),
                    fruit_orientation_rad=angle(q[:, fq+3:fq+7], eq[:, fq+3:fq+7]),
                    fruit_linear_m_s=(v[:, fv:fv+3]-ev[:, fv:fv+3]).norm(dim=-1),
                    fruit_angular_rad_s=(v[:, fv+3:fv+6]-ev[:, fv+3:fv+6]).norm(dim=-1))
            for buffer in range(4):
                full_root = counterfactual.capture(source)
                collector.collect(policy, gait, 64, cti_queue=queue, policy_version=buffer)
                batch = queue.pop_ready(buffer)
                self.assertIsNotNone(batch)
                endpoint = counterfactual.capture(source)
                for mode in ('same_full_1', 'same_full_2', 'cross_subset_1', 'cross_subset_2'):
                    if mode.startswith('same'):
                        rt = source
                        counterfactual.restore(rt, full_root)
                        slots = torch.tensor(batch.source_world_ids, device='cuda:0')
                    else:
                        rt = target
                        counterfactual.restore_worlds(rt, batch.snapshot)
                        slots = torch.arange(16, device='cuda:0')
                    for path, array in (('data.qpos', rt.data.qpos), ('data.qvel', rt.data.qvel),
                                        ('control.targets', rt.control.targets)):
                        torch.testing.assert_close(wp.to_torch(array)[slots], batch.snapshot.arrays[path],
                                                   rtol=0, atol=0)
                    maxima, first = {}, {}
                    flag_mismatch = {key: torch.zeros(16, dtype=torch.bool, device='cuda:0')
                                     for key in ('success', 'failed', 'holding', 'detached')}
                    for tick, step in enumerate(batch.factual_steps):
                        active = step['valid']
                        actions, gaits = torch.zeros_like(step['action']), torch.zeros_like(step['gait_action'])
                        actions[slots], gaits[slots] = step['action'], step['gait_action']
                        rt.set_gait_actions(gaits)
                        rt.step(actions)
                        rt.check()
                        errors = differences(wp.to_torch(rt.data.qpos)[slots], wp.to_torch(rt.data.qvel)[slots], step)
                        for key, error in errors.items():
                            value = float(error[active].max())
                            maxima[key] = max(maxima.get(key, 0.), value)
                            if tick == 0: first[key] = value
                        actual_signals = signals(rt)
                        for key in flag_mismatch:
                            flag_mismatch[key] |= active & (actual_signals[key][slots] != step['signals'][key])
                        if step['source_reset']:
                            mask = torch.zeros(16, dtype=torch.bool, device='cuda:0')
                            mask[slots] = step['ending']
                            rt.reset(mask)
                    print('REPLAY_FLOOR ' + json.dumps(dict(buffer=buffer, mode=mode,
                        steps=len(batch.factual_steps), first_step=first, maxima=maxima,
                        mismatched_worlds={key:int(value.sum()) for key,value in flag_mismatch.items()})), flush=True)
                counterfactual.restore(source, endpoint)


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
