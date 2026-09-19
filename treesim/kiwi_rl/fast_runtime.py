"""Small, bounded, GPU-only runtime for the rigid-fruit training scene.

The detailed deformable runtime remains in :mod:`runtime`.  This module is
deliberately narrow: one 50 Hz control step contains four 200 Hz physics
steps, and all action, observation, reward, and termination buffers stay on
the device until the caller asks for an explicit numerical check.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

from .control_warp import WarpSpotControl, _set_gait_targets


_OVERFLOW_BITS = {
    'NEFC': 1 << 0,
    'NJMAX_NNZ': 1 << 1,
    'BROADPHASE': 1 << 2,
    'NARROWPHASE': 1 << 3,
    'CCD': 1 << 4,
    'HFIELD': 1 << 5,
    'CONTACT_MATCH': 1 << 6,
    'NVMAX': 1 << 7,
    'EPA_HORIZON': 1 << 8,
    'ITERATIONS': 1 << 9,
    'LS_ITERATIONS': 1 << 10,
    'TACTILE': 1 << 11,
}

# MJWarp 3.13.0's convex narrowphase uses this constant to size a per-step
# EPA horizon buffer. Doubling the scratch capacity handles larger valid
# polytope boundaries; overflow remains enabled and latched by `_latch_state`.
_EPA_HORIZON_CAPACITY = 48


def _configure_epa_horizon(mjwarp, collision_convex):
    """Increase only the pinned MJWarp 3.13.0 scratch buffer capacity."""
    if getattr(mjwarp, '__version__', None) != '3.13.0':
        return None
    capacity = int(collision_convex.MJ_MAX_EPAHORIZON)
    if capacity < _EPA_HORIZON_CAPACITY:
        collision_convex.MJ_MAX_EPAHORIZON = _EPA_HORIZON_CAPACITY
        capacity = _EPA_HORIZON_CAPACITY
    return capacity


@wp.kernel
def _action_increment(actions: wp.array2d(dtype=float), targets: wp.array2d(dtype=float),
                      lower: wp.array(dtype=float), upper: wp.array(dtype=float),
                      flags: wp.array(dtype=int), max_delta: float):
    world, joint = wp.tid()
    action = actions[world, joint]
    if not wp.isfinite(action):
        wp.atomic_or(flags, world, 4)
        action = 0.0
    action = wp.clamp(action, -1.0, 1.0) * max_delta
    value = targets[world, joint + 12] + action
    targets[world, joint + 12] = wp.clamp(value, lower[joint + 12], upper[joint + 12])


@wp.kernel
def _reward_and_done(xipos: wp.array2d(dtype=wp.vec3), site_xpos: wp.array2d(dtype=wp.vec3),
                     xmat: wp.array2d(dtype=wp.mat33), tcp_site: int, fruit: int, chassis: int, previous: wp.array(dtype=float),
                     reward: wp.array(dtype=float), terminated: wp.array(dtype=wp.uint8),
                     distance: wp.array(dtype=float), mask: wp.array(dtype=wp.uint8),
                     success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8)):
    world = wp.tid()
    if mask[world] == 0:
        return
    delta = site_xpos[world, tcp_site] - xipos[world, fruit]
    d = wp.length(delta)
    distance[world] = d
    reward[world] = previous[world] - d
    previous[world] = d
    up = xmat[world, chassis][2, 2]
    fallen = (xipos[world, chassis][2] < 0.30) or (up < 0.6967067)
    terminated[world] = wp.uint8(fallen or failed[world] != 0 or success[world] != 0)
    if fallen or failed[world] != 0:
        reward[world] -= 1.0
    if success[world] != 0:
        reward[world] += 10.0


@wp.kernel
def _latch_state(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                    overflow: wp.array(dtype=int), flags: wp.array(dtype=int)):
    world, index = wp.tid()
    if index < qpos.shape[1] and not wp.isfinite(qpos[world, index]):
        wp.atomic_or(flags, world, 1)
    if index < qvel.shape[1] and not wp.isfinite(qvel[world, index]):
        wp.atomic_or(flags, world, 1)
    if index == 0 and overflow[world] != 0:
        # Keep the generic failure bit and the backend's actual OverflowType
        # mask. The latter survives a subsequent backend reset.
        wp.atomic_or(flags, world, 2 | (overflow[world] << 3))


@wp.kernel
def _masked_target_reset(mask: wp.array(dtype=wp.uint8), targets: wp.array2d(dtype=float),
                         initial: wp.array(dtype=float)):
    world, joint = wp.tid()
    if mask[world] != 0:
        targets[world, joint] = initial[joint]


@wp.kernel
def _masked_episode_reset(mask: wp.array(dtype=wp.uint8), previous: wp.array2d(dtype=float),
                          distance: wp.array(dtype=float), reward: wp.array(dtype=float),
                          terminated: wp.array(dtype=wp.uint8), flags: wp.array(dtype=int)):
    world, index = wp.tid()
    if mask[world] == 0:
        return
    if index < previous.shape[1]:
        previous[world, index] = 0.0
    if index == 0:
        distance[world] = 0.0
        reward[world] = 0.0
        terminated[world] = wp.uint8(0)


@wp.kernel
def _masked_restore_state(mask: wp.array(dtype=wp.uint8), qpos: wp.array2d(dtype=float),
                          qvel: wp.array2d(dtype=float), initial_qpos: wp.array(dtype=float),
                          initial_qvel: wp.array(dtype=float)):
    world, index = wp.tid()
    if mask[world] != 0:
        if index < qpos.shape[1]:
            qpos[world, index] = initial_qpos[index]
        if index < qvel.shape[1]:
            qvel[world, index] = initial_qvel[index]


@wp.kernel
def _masked_seed_distance(mask: wp.array(dtype=wp.uint8), distance: wp.array(dtype=float),
                          previous: wp.array(dtype=float), reward: wp.array(dtype=float)):
    world = wp.tid()
    if mask[world] != 0:
        previous[world] = distance[world]
        reward[world] = 0.0


class FastRuntime:
    """Bounded rigid-fruit runtime.

    ``step`` accepts CUDA Torch ``[worlds, 7]`` arm increments and returns
    CUDA Torch tensors. Use a shared non-default Torch/Warp stream, as the
    training and benchmark CLIs do. Fruit release and outcomes are evaluated
    on device each physics step; this is an uncalibrated rigid approximation.
    """

    def __init__(self, directory, worlds=64, control_dt=.02, camera=None,
                 resolution=(64, 48), nconmax=128, njmax=512, device='cuda:0',
                 arm_speed_rad_s=2.5, solver_iterations=100, jaw_cap_Nm=1.0, task_profile=None):
        if not isinstance(worlds, int) or not 1 <= worlds <= 4096:
            raise ValueError('worlds must be an integer in [1, 4096]')
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError('control_dt must be finite and positive')
        if int(nconmax) < 1 or int(njmax) < 1:
            raise ValueError('nconmax and njmax must be positive')
        if not np.isfinite(arm_speed_rad_s) or arm_speed_rad_s <= 0:
            raise ValueError('arm_speed_rad_s must be finite and positive')
        if int(solver_iterations) < 1 or int(solver_iterations) > 1000:
            raise ValueError('solver_iterations must be an integer in [1, 1000]')
        if not np.isfinite(jaw_cap_Nm) or jaw_cap_Nm <= 0:
            raise ValueError('jaw_cap_Nm must be finite and positive')
        from .fast_scene import load_fast_scene
        import mujoco
        import mujoco_warp as mw
        wp.init()
        # Do this before any Warp collision kernels compile. The 3.13.0
        # narrowphase sizes its EPA horizon scratch from this module constant.
        from mujoco_warp._src import collision_convex
        self.epa_horizon_capacity = _configure_epa_horizon(mw, collision_convex)
        self.model, initial, self.manifest = load_fast_scene(Path(directory))
        # MJWarp does not implement the legacy disabled-midphase path.  Fast
        # scenes are assembled for the supported native/GPU midphase path.
        if self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE):
            raise ValueError('Re-export the fast scene with supported midphase enabled')
        self.device_name = device
        self.camera = camera
        if self.manifest.get('schema') != 'fast-training-scene/v1':
            raise ValueError('Invalid fast-training scene schema')
        self.worlds, self.control_dt = worlds, float(control_dt)
        self.arm_speed_rad_s = float(arm_speed_rad_s)
        self.solver_iterations = int(solver_iterations)
        self.jaw_cap_Nm = float(jaw_cap_Nm)
        self.model.opt.iterations = self.solver_iterations
        self.dt = float(self.model.opt.timestep)
        self.substeps = round(self.control_dt / self.dt)
        if self.substeps not in (4, 10) or not math.isclose(self.substeps * self.dt, self.control_dt, abs_tol=1e-9):
            raise ValueError('FastRuntime requires four or ten integral physics substeps')
        with wp.ScopedDevice(device):
            self.gpu_model = mw.put_model(self.model)
            self.data = mw.put_data(self.model, initial, nworld=worlds,
                                    nconmax=int(nconmax), njmax=int(njmax))
            self.device = self.data.qpos.device
            self.control = WarpSpotControl(self.model, self.data, self.manifest['robot'])
            if self.jaw_cap_Nm > self.control.contract.limits[-1]:
                raise ValueError(f'jaw_cap_Nm exceeds the actuator limit of {self.control.contract.limits[-1]:g} Nm')
            self.control.set_jaw_caps(np.full(self.worlds, self.jaw_cap_Nm, dtype=np.float32))
            self._initial_targets = wp.array(self.control.contract.targets, dtype=float, device=self.device)
            self._initial_qpos = wp.array(initial.qpos, dtype=float, device=self.device)
            self._initial_qvel = wp.array(initial.qvel, dtype=float, device=self.device)
            lower = np.full(19, -np.inf, dtype=np.float32)
            upper = np.full(19, np.inf, dtype=np.float32)
            joints = np.asarray(self.control.contract.joints, dtype=int)
            limited = np.asarray(self.model.jnt_limited, dtype=bool)[joints]
            ranges = np.asarray(self.model.jnt_range)[joints]
            lower[limited] = ranges[limited, 0]
            upper[limited] = ranges[limited, 1]
            self._action_lower = wp.array(lower, dtype=float, device=self.device)
            self._action_upper = wp.array(upper, dtype=float, device=self.device)
            self._previous_distance = wp.zeros(worlds, dtype=float, device=self.device)
            self._distance = wp.zeros(worlds, dtype=float, device=self.device)
            self._reward = wp.zeros(worlds, dtype=float, device=self.device)
            self._terminated = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._flags = wp.zeros(worlds, dtype=int, device=self.device)
            self._all_mask = wp.ones(worlds, dtype=wp.uint8, device=self.device)
            self._actions = wp.zeros((worlds, 7), dtype=float, device=self.device)
            fruit = self.manifest.get('fruits', self.manifest.get('fruit', []))
            if not fruit:
                raise ValueError('Fast scene must contain at least one fruit')
            self.fruit_body = int(self.model.body(fruit[0]['body']).id)
            robot = self.manifest['robot']
            tcp_site_name = robot.get('tcp_site', 'hand_tcp')
            if tcp_site_name not in [self.model.site(i).name for i in range(self.model.nsite)]:
                tcp_name = robot.get('tcp_body', robot.get('prefix', '') + 'arm_link_fngr')
                tcp_site_name = next((self.model.site(i).name for i in range(self.model.nsite)
                                      if self.model.site(i).bodyid == self.model.body(tcp_name).id), None)
            if tcp_site_name is None:
                tcp_site_name = next((self.model.site(i).name for i in range(self.model.nsite)
                                      if any(token in (self.model.site(i).name or '').lower()
                                             for token in ('tcp', 'hand', 'fngr'))), None)
            if tcp_site_name is None:
                raise ValueError('Fast scene robot must provide tcp_site or a body with a site')
            self.tcp_site = int(self.model.site(tcp_site_name).id)
            self.chassis = self.control.chassis
            from .fast_task import FastHarvestTask
            self.task_profile = task_profile
            self.task = FastHarvestTask(self.model, self.data, self.manifest, task_profile=task_profile)
            self._refresh(mw)
            self._measure_reward()
            self.reset()
            # Capture the complete control interval with persistent device buffers.
            # The action buffer is updated in-place before each launch.
            with wp.ScopedCapture() as capture:
                wp.launch(_action_increment, dim=(self.worlds, 7),
                          inputs=[self._actions, self.control.targets, self._action_lower,
                                  self._action_upper, self._flags,
                                  self.arm_speed_rad_s * self.control_dt], device=self.device)
                for _ in range(self.substeps):
                    self.control.apply()
                    mw.step(self.gpu_model, self.data)
                    self._refresh(mw)
                    self._latch()
                    self.task.record()
                self._measure_reward()
            self.graph = capture.graph
            self.rig = None
            if camera is not None:
                cameras = self.manifest.get('cameras', ())
                if isinstance(cameras, dict):
                    camera_names = set(cameras)
                else:
                    camera_names = {entry.get('name') if isinstance(entry, dict) else str(entry)
                                    for entry in cameras}
                if camera not in camera_names:
                    raise ValueError(f"Camera {camera!r} is not declared in the fast-scene manifest")
                native_camera_names = {self.model.camera(i).name for i in range(self.model.ncam)}
                if camera not in native_camera_names:
                    raise ValueError(f"Camera {camera!r} is not present in the native model") from None
                from .sensors_warp import WarpRGBDRig
                size = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
                self.rig = WarpRGBDRig(self.model, self.data, cameras=(camera,), resolution=size)

    def _refresh(self, mw):
        mw.kinematics(self.gpu_model, self.data)
        mw.com_pos(self.gpu_model, self.data)
        mw.com_vel(self.gpu_model, self.data)
        if hasattr(mw, 'camlight'):
            mw.camlight(self.gpu_model, self.data)

    def _measure_reward(self, mask=None):
        if mask is None:
            mask = self._all_mask
        wp.launch(_reward_and_done, dim=self.worlds,
                  inputs=[self.data.xipos, self.data.site_xpos, self.data.xmat, self.tcp_site, self.fruit_body,
                          self.chassis, self._previous_distance, self._reward,
                          self._terminated, self._distance, mask, self.task.success, self.task.failed], device=self.device)

    def _latch(self):
        wp.launch(_latch_state, dim=(self.worlds, max(self.data.qpos.shape[1], self.data.qvel.shape[1])),
                  inputs=[self.data.qpos, self.data.qvel, self.data.overflow, self._flags], device=self.device)

    def step(self, actions):
        import torch
        if not isinstance(actions, torch.Tensor) or not actions.is_cuda:
            raise ValueError('actions must be a CUDA Torch tensor')
        if tuple(actions.shape) != (self.worlds, 7) or actions.dtype != torch.float32:
            raise ValueError(f'actions must have CUDA float32 shape ({self.worlds}, 7)')
        if actions.device.index is not None and str(self.device).startswith('cuda:') and actions.device.index != int(str(self.device).split(':')[-1]):
            raise ValueError('actions and runtime must use the same CUDA device')
        action_wp = wp.from_torch(actions)
        import mujoco_warp as mw
        with wp.ScopedDevice(self.device):
            wp.copy(self._actions, action_wp)
            wp.capture_launch(self.graph)
        return self.observe(), wp.to_torch(self._reward), wp.to_torch(self._terminated).bool(), {
            'distance_m': wp.to_torch(self._distance),
            'fallen': (wp.to_torch(self.data.xpos)[:,self.chassis,2] < .3) |
                      (wp.to_torch(self.data.xmat)[:,self.chassis,2,2] < .6967067),
            'reached': wp.to_torch(self._distance) < 0.08,
            'release_supported': True,
            **{key: wp.to_torch(value) for key,value in self.task.outputs().items()},
        }

    def observe(self):
        self._refresh(__import__('mujoco_warp'))
        return wp.to_torch(self.control.observe())

    def set_gait_actions(self, actions):
        """Latch a CUDA ``[worlds, 12]`` gait action for the next substep."""
        import torch
        if not isinstance(actions, torch.Tensor) or not actions.is_cuda or actions.dtype != torch.float32:
            raise ValueError('gait actions must be a CUDA float32 Torch tensor')
        if tuple(actions.shape) != (self.worlds, 12):
            raise ValueError(f'gait actions must have shape ({self.worlds}, 12)')
        with wp.ScopedDevice(self.device):
            values = wp.from_torch(actions)
            self.control.previous.assign(values)
            wp.launch(_set_gait_targets, dim=(self.worlds, 12),
                      inputs=[self.control.previous, self.control.home, self.control.targets], device=self.device)

    def pixels(self):
        if self.rig is None:
            raise RuntimeError('FastRuntime was created without a camera')
        import mujoco_warp as mw
        self._refresh(mw)
        self.rig.capture(self.gpu_model, self.data, 0.0)
        return self.rig.tensor(self.camera)

    def reset(self, mask=None):
        import mujoco_warp as mw
        import torch
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.uint8, device=self.device_name)
        elif not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (self.worlds,) or not mask.is_cuda:
            raise ValueError(f'mask must be a CUDA tensor of shape ({self.worlds},)')
        mask_wp = wp.from_torch(mask.to(dtype=torch.uint8))
        with wp.ScopedDevice(self.device):
            mw.reset_data(self.gpu_model, self.data, reset=mask_wp)
            wp.launch(_masked_restore_state, dim=(self.worlds, max(self.data.qpos.shape[1], self.data.qvel.shape[1])),
                      inputs=[mask_wp, self.data.qpos, self.data.qvel, self._initial_qpos, self._initial_qvel], device=self.device)
            wp.launch(_masked_target_reset, dim=(self.worlds, 19),
                      inputs=[mask_wp, self.control.targets, self._initial_targets], device=self.device)
            wp.launch(_masked_episode_reset, dim=(self.worlds, 12),
                      inputs=[mask_wp, self.control.previous, self._previous_distance,
                              self._reward, self._terminated, self._flags], device=self.device)
            if hasattr(self, '_settled_previous'):
                wp.to_torch(self.control.previous)[mask.bool()] = self._settled_previous
            self.task.reset(mask_wp)
            mw.forward(self.gpu_model, self.data)
            self._refresh(mw)
            self._measure_reward(mask_wp)
            wp.launch(_masked_seed_distance, dim=self.worlds,
                      inputs=[mask_wp, self._distance, self._previous_distance, self._reward], device=self.device)
        return self.observe()

    def prepare_settled_reset(self, gait, seconds=4.):
        """Settle once, then reset worlds from the same physical controller state.

        No training rewards or policy history exist during this preparation.
        Existing runtime arrays stay in place for CUDA graph and replay safety.
        """
        import torch
        from .reward_graph import GRAPH_PROFILE
        if self.task_profile != GRAPH_PROFILE or hasattr(self, '_settled_previous'):
            return
        with torch.no_grad():
            zero = torch.zeros((self.worlds, 7), device=self.device_name)
            for _ in range(round(seconds/self.control_dt)):
                self.set_gait_actions(gait(self.observe()))
                self.step(zero)
            self.check()
            if bool(wp.to_torch(self.task.failed).bool().any() | wp.to_torch(self.task.detached).bool().any()):
                raise RuntimeError('Reset settling damaged or detached fruit, or destabilized the robot')
            self._initial_qpos.assign(self.data.qpos.numpy()[0])
            self._initial_qvel.assign(self.data.qvel.numpy()[0])
            self._initial_targets.assign(self.control.targets.numpy()[0])
            self._settled_previous = wp.to_torch(self.control.previous)[0].clone()
            self.reset()

    def check(self):
        flags = np.asarray(self._flags.numpy(), dtype=np.int64)
        if not np.any(flags):
            return {'flags': flags.tolist(), 'flagged_world_count': 0}
        diagnostics = self._diagnostics(flags)
        raise RuntimeError(
            'GPU numerical failure: '
            f"flagged_worlds={diagnostics['flagged_world_count']}; "
            f"flag_bits={diagnostics['flag_bit_counts']}; "
            f"overflow_bits={diagnostics['overflow_bit_counts']}; "
            f"needed={diagnostics['needed']}"
        )

    def _diagnostics(self, flags):
        """Return compact backend overflow and capacity diagnostics.

        ``overflow`` is a bitmask from MJWarp, while ``nacon`` is a batched
        count. The sparse Jacobian row arrays provide a lower bound for the
        non-zero capacity required by the rows that were written.
        """
        # MJWarp may clear data.overflow during reset; use the latched copy
        # first and merge the live value when it is still present.
        overflow = np.asarray(self.data.overflow.numpy(), dtype=np.int64)
        overflow |= flags >> 3
        flag_bits = {
            name: int(np.count_nonzero(flags & bit))
            for name, bit in (("NONFINITE", 1), ("OVERFLOW", 2), ("INVALID_ACTION", 4))
            if np.any(flags & bit)
        }
        overflow_bits = {
            name: int(np.count_nonzero(overflow & bit))
            for name, bit in _OVERFLOW_BITS.items()
            if np.any(overflow & bit)
        }
        needed = {
            'contacts_total': int(np.max(np.asarray(self.data.nacon.numpy(), dtype=np.int64))),
            'contacts_capacity_total': int(self.data.naconmax),
            'constraints_max': int(np.max(np.asarray(self.data.nefc.numpy(), dtype=np.int64))),
            'constraints_capacity': int(self.data.njmax),
            'constraint_nnz_capacity': int(self.data.njmax_nnz),
        }
        try:
            row_adr = np.asarray(self.data.efc.J_rowadr.numpy(), dtype=np.int64)
            row_nnz = np.asarray(self.data.efc.J_rownnz.numpy(), dtype=np.int64)
            nefc = np.asarray(self.data.nefc.numpy(), dtype=np.int64).reshape(-1)
            lower_bounds = [
                int(np.max(row_adr[w, :max(0, min(int(n), row_adr.shape[1]))] +
                          row_nnz[w, :max(0, min(int(n), row_nnz.shape[1]))]))
                for w, n in enumerate(nefc)
                if n > 0
            ]
            if lower_bounds:
                needed['constraint_nnz_lower_bound'] = max(lower_bounds)
        except (AttributeError, IndexError, ValueError):
            pass
        return {
            'flags': flags.tolist(),
            'flagged_world_count': int(np.count_nonzero(flags)),
            'flag_bit_counts': flag_bits,
            'overflow_bit_counts': overflow_bits,
            'needed': needed,
        }
