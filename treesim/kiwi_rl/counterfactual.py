"""Exact same-runtime snapshots for controlled counterfactual branches.

Snapshots are process-local checkpoints, not portable files. They retain every
mutable array reachable from the fast runtime's model, data, controller, task,
and camera outputs, plus explicit application state and RNG state. Restore
copies into existing arrays so captured Warp graphs keep their buffers.
"""
from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np


@dataclass
class RuntimeSnapshot:
    arrays: tuple
    attributes: dict
    application_state: object
    rng_state: dict
    signature: tuple
    runtime_identity: int
    physics_profile: tuple


def _array(value):
    # Warp arrays expose both methods. Torch tensors are handled as explicit
    # application state to avoid confusing device aliases with runtime buffers.
    return isinstance(value, np.ndarray) or (
        callable(getattr(value, 'numpy', None)) and callable(getattr(value, 'assign', None)))


def _runtime_arrays(runtime):
    roots = [runtime.model, runtime.gpu_model, runtime.data,
             runtime.control, runtime.task]
    # Runtime-owned scratch/state arrays include reward baselines, action
    # staging, numerical latches, and initial templates. The captured graph
    # keeps using these same objects after restore.
    for name, value in vars(runtime).items():
        if name not in {'model', 'gpu_model', 'data', 'control', 'task', 'rig', 'graph'}:
            roots.append(value)
    rig = getattr(runtime, 'rig', None)
    if rig is not None:
        # The render context contains backend caches. Render output buffers and
        # their delivery clock are the state visible to the policy; caches are
        # recomputed by capture().
        roots.extend((getattr(rig, 'rgb', None), getattr(rig, 'depth', None),
                      getattr(rig, 'valid', None)))
    found, seen_obj, seen_array = [], set(), set()

    def visit(value, path):
        if value is None or id(value) in seen_obj:
            return
        if _array(value):
            if id(value) not in seen_array:
                seen_array.add(id(value))
                found.append((path, value))
            return
        if isinstance(value, np.ndarray) or isinstance(value, (str, bytes, int, float, bool, type)):
            return
        if isinstance(value, dict):
            seen_obj.add(id(value))
            for key, item in value.items():
                visit(item, f'{path}[{key!r}]')
        elif isinstance(value, (list, tuple)):
            seen_obj.add(id(value))
            for index, item in enumerate(value):
                visit(item, f'{path}[{index}]')
        elif path in {'root0', 'root1'}:
            # MuJoCo's native model uses extension-backed slots rather than a
            # Python __dict__. Read its exposed array properties explicitly.
            seen_obj.add(id(value))
            for key in dir(value):
                if key.startswith('_'):
                    continue
                try:
                    item = getattr(value, key)
                except Exception:
                    continue
                if isinstance(item, np.ndarray) or _array(item):
                    visit(item, f'{path}.{key}')
        elif hasattr(value, '__dict__'):
            seen_obj.add(id(value))
            for key, item in vars(value).items():
                # No Python modules, classes, or callbacks are snapshot state.
                if not key.startswith('__'):
                    visit(item, f'{path}.{key}')

    for i, root in enumerate(roots):
        visit(root, f'root{i}')
    return tuple(found)


def _signature(arrays):
    return tuple((path, tuple((value if isinstance(value, np.ndarray) else value.numpy()).shape),
                  str((value if isinstance(value, np.ndarray) else value.numpy()).dtype))
                 for path, value in arrays)


def _physics_profile(runtime):
    """Scalar settings that define a fixed branchable runtime profile."""
    profile = []
    for name in ('worlds', 'control_dt', 'dt', 'substeps', 'arm_speed_rad_s',
                 'solver_iterations', 'device_name'):
        if hasattr(runtime, name):
            profile.append((name, str(getattr(runtime, name))))
    model = getattr(runtime, 'model', None)
    opt = getattr(model, 'opt', None)
    for name in ('timestep', 'integrator', 'cone', 'iterations', 'tolerance',
                 'ls_iterations', 'noslip_iterations', 'noslip_tolerance',
                 'disableflags', 'enableflags'):
        if opt is not None and hasattr(opt, name):
            value = getattr(opt, name)
            if np.isscalar(value):
                profile.append((f'model.opt.{name}', str(value.item() if hasattr(value, 'item') else value)))
    data = getattr(runtime, 'data', None)
    for name in ('nconmax', 'njmax', 'njmax_nnz'):
        if data is not None and hasattr(data, name):
            profile.append((f'data.{name}', str(getattr(data, name))))
    rig = getattr(runtime, 'rig', None)
    if rig is not None:
        for name in ('cameras', 'width', 'height', 'minimum_m', 'maximum_m'):
            if hasattr(rig, name):
                profile.append((f'rig.{name}', repr(getattr(rig, name))))
    # jaw_cap is a mutable controller array and is captured with the other
    # Warp buffers; its values are restored rather than frozen as profile data.
    return tuple(profile)


def _read_array(value):
    return value.copy() if isinstance(value, np.ndarray) else value.numpy().copy()


def _write_array(target, value):
    if isinstance(target, np.ndarray):
        np.copyto(target, value)
    else:
        target.assign(value)


def _rng_capture():
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    try:
        import torch
    except ImportError:
        return state
    state['torch_cpu'] = torch.random.get_rng_state().clone()
    if torch.cuda.is_available():
        state['torch_cuda'] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _rng_restore(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'torch_cpu' in state:
        import torch
        torch.random.set_rng_state(state['torch_cpu'])
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda'])


def capture(runtime, *, application_state=None, rng_generators=None):
    """Capture a full continuation point for one unchanged runtime instance.

    Pass collector/policy state here, including recurrent memory, progress and
    sensor scheduling clocks. Python, NumPy and Torch global RNG streams are
    captured automatically. Runtime arrays are copied to host for portability
    within this process; no Warp array is replaced.
    """
    arrays = _runtime_arrays(runtime)
    attrs = {}
    rig = getattr(runtime, 'rig', None)
    if rig is not None:
        for name in ('frame_id', 'timestamp_s', 'available'):
            attrs[f'rig.{name}'] = deepcopy(getattr(rig, name))
    return RuntimeSnapshot(
        arrays=tuple((path, _read_array(value)) for path, value in arrays),
        attributes=attrs,
        application_state=deepcopy(application_state),
        rng_state={**_rng_capture(), 'generators': _capture_generators(rng_generators)},
        signature=_signature(arrays),
        runtime_identity=id(runtime),
        physics_profile=_physics_profile(runtime),
    )


def _capture_generators(generators):
    result = {}
    for name, generator in (generators or {}).items():
        if isinstance(generator, np.random.Generator):
            result[name] = ('numpy', deepcopy(generator.bit_generator.state))
        elif isinstance(generator, random.Random):
            result[name] = ('python', generator.getstate())
        elif generator.__class__.__module__.startswith('torch') and hasattr(generator, 'get_state'):
            result[name] = ('torch', generator.get_state().clone())
        else:
            raise TypeError(f'Unsupported RNG generator: {name}')
    return result


def _restore_generators(saved, generators):
    generators = generators or {}
    if set(saved) != set(generators):
        raise ValueError('RNG generator names do not match the snapshot')
    for name, (kind, state) in saved.items():
        generator = generators[name]
        if kind == 'numpy' and isinstance(generator, np.random.Generator):
            generator.bit_generator.state = deepcopy(state)
        elif kind == 'python' and isinstance(generator, random.Random):
            generator.setstate(state)
        elif kind == 'torch' and generator.__class__.__module__.startswith('torch'):
            generator.set_state(state)
        else:
            raise TypeError(f'RNG generator type changed: {name}')


def restore(runtime, snapshot, *, rng_generators=None):
    """Restore a snapshot in place; return a copy of caller-owned state."""
    if not isinstance(snapshot, RuntimeSnapshot):
        raise TypeError('snapshot must be a RuntimeSnapshot')
    if id(runtime) != snapshot.runtime_identity:
        raise ValueError('Snapshot belongs to a different runtime instance')
    if _physics_profile(runtime) != snapshot.physics_profile:
        raise ValueError('Snapshot physics profile changed')
    arrays = _runtime_arrays(runtime)
    if _signature(arrays) != snapshot.signature:
        raise ValueError('Snapshot does not match this runtime structure')
    current = dict(arrays)
    for path, saved in snapshot.arrays:
        target = current[path]
        target_shape = target.shape if isinstance(target, np.ndarray) else target.numpy().shape
        if tuple(target_shape) != tuple(saved.shape):
            raise ValueError(f'Snapshot array shape changed at {path}')
        _write_array(target, saved)
    rig = getattr(runtime, 'rig', None)
    if rig is not None:
        for key, value in snapshot.attributes.items():
            owner, name = key.split('.', 1)
            if owner == 'rig':
                setattr(rig, name, deepcopy(value))
    _rng_restore(snapshot.rng_state)
    _restore_generators(snapshot.rng_state['generators'], rng_generators)
    return deepcopy(snapshot.application_state)


@dataclass
class WorldSnapshot:
    """Selected teacher-world continuation state, held on the source device.

    Global packed collision caches are not portable between world counts. They
    are rebuilt by MJWarp forward() before the next physics step consumes them.
    This is a continuation point, not a copy of raw contact diagnostics.
    """
    arrays: dict
    application_state: object
    source_world_ids: tuple
    physics_profile: tuple
    model_sha256: str


# Explicit runtime ownership: never identify a world axis just because its
# length happens to equal runtime.worlds (e.g. seven worlds and seven joints).
_CONTROL_WORLD = ('targets', 'commands', 'previous', 'jaw_cap', 'effort', 'observations')
_CONTROL_SHARED = ('qids', 'dofs', 'actuators', 'obs_order', 'home', 'kp', 'kd', 'limits', 'knee_table')
_TASK_WORLD = ('_hand_hits', '_basket_hits', '_ground_hits', '_finger_hits', '_jaw_hits',
               'detached', 'hand_contact', 'basket_contact', 'ground_contact',
               'bilateral_contact', 'stable_grasp', 'ever_grasped', 'grasp_time',
               'stem_force', 'hand_load', 'finger_load', 'jaw_load', 'palm_load',
               'damage_proxy', 'settle_time', 'success', 'failed')
_TASK_SHARED = ('fruit_geom', 'fruit_body', 'equality_index', 'kind')
_RUNTIME_WORLD = ('_previous_distance', '_distance', '_reward', '_terminated',
                  '_flags', '_all_mask', '_actions')
_RUNTIME_SHARED = ('_initial_targets', '_initial_qpos', '_initial_qvel',
                   '_action_lower', '_action_upper')


def _world_tensor(array):
    import torch
    if isinstance(array, torch.Tensor):
        return array
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    import warp as wp
    return wp.to_torch(array)


def _world_arrays(runtime):
    """Use MJWarp's declared symbolic dimensions, not runtime shape guesses."""
    from dataclasses import fields, is_dataclass
    arrays = {}
    data = runtime.data
    if not is_dataclass(data) or not is_dataclass(data.efc):
        raise ValueError('World transfer requires MJWarp declared Data/Constraint schemas')
    for prefix, owner in (('data', data), ('data.efc', data.efc)):
        declared = {field.name: field for field in fields(owner)}
        for name, value in vars(owner).items():
            if _array(value) and name not in declared:
                raise ValueError(f'Undeclared backend array: {prefix}.{name}')
        for name, field in declared.items():
            dimensions = getattr(field.type, 'shape', ())
            if dimensions and dimensions[0] == 'nworld':
                value = getattr(owner, name)
                tensor = _world_tensor(value)
                if tensor.numel() == 0:
                    continue  # An explicitly disabled backend feature.
                if tensor.shape[0] != runtime.worlds:
                    raise ValueError(f'Invalid declared world axis: {prefix}.{name}')
                arrays[f'{prefix}.{name}'] = tensor
    for prefix, owner, selected, shared, aliases in (
        ('control', runtime.control, _CONTROL_WORLD, _CONTROL_SHARED, ()),
        ('task', runtime.task, _TASK_WORLD, _TASK_SHARED, ('eq_active', '_efc_type', '_efc_id')),
        ('runtime', runtime, _RUNTIME_WORLD, _RUNTIME_SHARED, ())):
        known = set(selected + shared + aliases)
        for name, value in vars(owner).items():
            if _array(value) and name not in known:
                raise ValueError(f'Unclassified runtime array: {prefix}.{name}')
        for name in selected:
            tensor = _world_tensor(getattr(owner, name))
            if not tensor.ndim or tensor.shape[0] != runtime.worlds:
                raise ValueError(f'Invalid declared world axis: {prefix}.{name}')
            arrays[f'{prefix}.{name}'] = tensor
    return arrays


def _world_profile(runtime):
    if getattr(runtime, 'rig', None) is not None:
        raise ValueError('World transfer currently supports the camera-free teacher runtime')
    if not runtime.manifest.get('model_sha256'):
        raise ValueError('World transfer requires a model hash')
    # Sleeping can consume previous global contact caches before rebuilding them.
    # The approved fast teacher profile does not enable it.
    import mujoco
    if runtime.model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
        raise ValueError('World transfer does not support sleeping bodies')
    if not runtime.gpu_model.opt.run_collision_detection:
        raise ValueError('World transfer requires collision-cache regeneration')
    profile = tuple((name, value) for name, value in _physics_profile(runtime) if name != 'worlds')
    capacities = tuple((f'data.{name}', str(getattr(runtime.data, name))) for name in
                       ('njmax', 'nvmax', 'nvmax_pad', 'njmax_pad', 'njmax_nnz'))
    return profile + capacities


def _world_ids(world_ids, worlds):
    import torch
    ids = torch.as_tensor(world_ids)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not 1 <= ids.numel() <= 64:
        raise ValueError('World ids must be a nonempty integer vector with at most 64 entries')
    values = tuple(ids.cpu().tolist())
    if len(set(values)) != len(values) or any(index < 0 or index >= worlds for index in values):
        raise ValueError('World ids must be unique and within the runtime')
    return values


def capture_worlds(runtime, world_ids, *, application_state=None):
    """Gather <=64 selected worlds without copying the factual batch to host.

    Call only between complete FastRuntime control intervals on the shared
    Torch/Warp stream. application_state must already be sliced to these worlds
    by its owner (GRU memory, EpisodeProgress, reset mask, collector tick).
    No global RNG is read or modified; branch noise belongs to the caller.
    Static model/controller parameters must remain unchanged during collection.
    """
    import torch
    ids = _world_ids(world_ids, runtime.worlds)
    profile = _world_profile(runtime)
    arrays = _world_arrays(runtime)
    indices = torch.tensor(ids, dtype=torch.long, device=next(iter(arrays.values())).device)
    return WorldSnapshot(
        arrays={path: value.index_select(0, indices).detach().clone() for path, value in arrays.items()},
        application_state=deepcopy(application_state), source_world_ids=ids,
        physics_profile=profile, model_sha256=runtime.manifest['model_sha256'])


def restore_worlds(runtime, snapshot, world_ids=None):
    """Restore selected worlds in place; preserve every other target world.

    Model, solver, controller, and buffer layouts must match. Raw global contact
    diagnostics are stale until the next step; do not call task.record() before
    that step. Current task signals, observations and warm starts are restored.
    """
    import torch
    if not isinstance(snapshot, WorldSnapshot):
        raise TypeError('snapshot must be a WorldSnapshot')
    ids = _world_ids(range(len(snapshot.source_world_ids)) if world_ids is None else world_ids,
                     runtime.worlds)
    if len(ids) != len(snapshot.source_world_ids):
        raise ValueError('Source and destination world counts must match')
    if snapshot.model_sha256 != runtime.manifest.get('model_sha256') or snapshot.physics_profile != _world_profile(runtime):
        raise ValueError('World transfer model or physics profile mismatch')
    arrays = _world_arrays(runtime)
    if arrays.keys() != snapshot.arrays.keys():
        raise ValueError('World transfer buffer schema mismatch')
    # Validate everything before the first write, including per-world capacity.
    for path, target in arrays.items():
        saved = snapshot.arrays[path]
        if target.shape[1:] != saved.shape[1:] or target.dtype != saved.dtype or target.device != saved.device:
            raise ValueError(f'World transfer buffer layout mismatch: {path}')
    indices = torch.tensor(ids, dtype=torch.long, device=next(iter(arrays.values())).device)
    with torch.no_grad():
        for path, target in arrays.items():
            target.index_copy_(0, indices, snapshot.arrays[path])
    return deepcopy(snapshot.application_state)
