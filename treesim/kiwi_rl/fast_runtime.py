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
        wp.atomic_or(flags, world, 2)


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
                 resolution=64, nconmax=128, njmax=512, device='cuda:0'):
        if not isinstance(worlds, int) or not 1 <= worlds <= 4096:
            raise ValueError('worlds must be an integer in [1, 4096]')
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError('control_dt must be finite and positive')
        if int(nconmax) < 1 or int(njmax) < 1:
            raise ValueError('nconmax and njmax must be positive')
        from .fast_scene import load_fast_scene
        import mujoco
        import mujoco_warp as mw
        wp.init()
        self.model, initial, self.manifest = load_fast_scene(Path(directory))
        # MJWarp does not implement the legacy disabled-midphase path.  Fast
        # scenes are assembled for the supported native/GPU midphase path.
        if self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE):
            raise ValueError('Re-export the fast scene with supported midphase enabled')
        self.device_name = device
        if self.manifest.get('schema') != 'fast-training-scene/v1':
            raise ValueError('Invalid fast-training scene schema')
        self.worlds, self.control_dt = worlds, float(control_dt)
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
            self.task = FastHarvestTask(self.model, self.data, self.manifest)
            self._refresh(mw)
            self._measure_reward()
            self.reset()
            # Capture the complete control interval with persistent device buffers.
            # The action buffer is updated in-place before each launch.
            with wp.ScopedCapture() as capture:
                wp.launch(_action_increment, dim=(self.worlds, 7),
                          inputs=[self._actions, self.control.targets, self._action_lower,
                                  self._action_upper, self._flags, 2.5 * self.control_dt], device=self.device)
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
                if camera != 'body_camera':
                    raise ValueError("FastRuntime supports only camera='body_camera'")
                from .sensors_warp import WarpRGBDRig
                size = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
                self.rig = WarpRGBDRig(self.model, self.data, cameras=('body_camera',), resolution=size)

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
            raise RuntimeError("FastRuntime was created without camera='body_camera'")
        import mujoco_warp as mw
        self._refresh(mw)
        self.rig.capture(self.gpu_model, self.data, 0.0)
        return self.rig.tensor('body_camera')

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
            self.task.reset(mask_wp)
            mw.forward(self.gpu_model, self.data)
            self._refresh(mw)
            self._measure_reward(mask_wp)
            wp.launch(_masked_seed_distance, dim=self.worlds,
                      inputs=[mask_wp, self._distance, self._previous_distance, self._reward], device=self.device)
        return self.observe()

    def check(self):
        flags = self._flags.numpy()
        if np.any(flags):
            raise RuntimeError(f'GPU numerical failure flags={flags.tolist()}')
        return {'flags': flags.tolist()}
