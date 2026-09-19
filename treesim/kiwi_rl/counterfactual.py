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
