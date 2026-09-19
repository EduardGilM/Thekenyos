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
from .fast_task import MAX_FRUITS, _pad_ids
from .rewards import (
    W_DAMAGE_PER_UNIT, W_DEPOSIT, W_DETACH_HELD, W_FALL, W_GRASP_STABLE,
    W_LOSS, W_SMOOTH, W_TIME_PER_S,
)

# Latched per-world bits. Nonfinite qpos/qvel abort the job; overflow and a
# nonfinite action reset that world so one solver stall does not kill the batch.
FLAG_NONFINITE = 1
FLAG_OVERFLOW = 2
FLAG_BAD_ACTION = 4
FLAG_RECOVERABLE = FLAG_OVERFLOW | FLAG_BAD_ACTION


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
                     xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                     tcp_site: int, fruit_bodies: wp.array(dtype=int),
                     active_fruit: wp.array(dtype=int), chassis: int,
                     previous_potential: wp.array(dtype=float), shaping_ref: wp.array(dtype=int),
                     previous_damage: wp.array(dtype=float), previous_action: wp.array2d(dtype=float),
                     actions: wp.array2d(dtype=float), episode_time: wp.array(dtype=float),
                     timeout_s: wp.array(dtype=float), guidance: wp.array(dtype=float),
                     goal: wp.array(dtype=int), reward: wp.array(dtype=float),
                     terminated: wp.array(dtype=wp.uint8), timed_out: wp.array(dtype=wp.uint8),
                     distance: wp.array(dtype=float), mask: wp.array(dtype=wp.uint8),
                     success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8),
                     detached: wp.array(dtype=wp.uint8), grasped: wp.array(dtype=wp.uint8),
                     retained_detach: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8),
                     damage_proxy: wp.array(dtype=float),
                     grasp_paid: wp.array(dtype=wp.uint8), detach_paid: wp.array(dtype=wp.uint8),
                     deposit_paid: wp.array2d(dtype=wp.uint8), deposited: wp.array2d(dtype=wp.uint8),
                     loss_paid: wp.array(dtype=wp.uint8),
                     basket_center: wp.vec3, dt: float, gamma_step: float,
                     w_deposit: float, w_grasp: float, w_detach: float, w_loss: float,
                     w_damage: float, w_fall: float, w_time: float, w_smooth: float):
    world = wp.tid()
    if mask[world] == 0:
        return
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    tcp = site_xpos[world, tcp_site]
    fruit_pos = xipos[world, fruit]
    d_tcp = wp.length(tcp - fruit_pos)
    distance[world] = d_tcp
    rotation = xmat[world, chassis]
    basket_world = xpos[world, chassis] + rotation @ basket_center
    d_basket = wp.length(xpos[world, fruit] - basket_world)
    use_basket = 1 if (goal[world] == 0 or detached[world] != 0) else 0
    d_shape = d_basket if use_basket != 0 else d_tcp
    phi = wp.exp(-d_shape / 0.25)
    shaped = float(0.)
    if shaping_ref[world] == use_basket:
        shaped = guidance[world] * 2.0 * (gamma_step * phi - previous_potential[world])
    previous_potential[world] = phi
    shaping_ref[world] = use_basket
    r = shaped
    if grasped[world] != 0 and grasp_paid[world] == 0:
        r = r + w_grasp
        grasp_paid[world] = wp.uint8(1)
    if retained_detach[world] != 0 and detach_paid[world] == 0:
        r = r + w_detach
        detach_paid[world] = wp.uint8(1)
    delta_damage = damage_proxy[world] - previous_damage[world]
    if delta_damage < 0.:
        delta_damage = 0.
    r = r + w_damage * delta_damage
    previous_damage[world] = damage_proxy[world]
    smooth = float(0.)
    for joint in range(7):
        delta_act = actions[world, joint] - previous_action[world, joint]
        smooth = smooth + delta_act * delta_act
        previous_action[world, joint] = actions[world, joint]
    r = r + w_smooth * (smooth / 7.) + w_time * dt
    up = rotation[2, 2]
    fallen = (xipos[world, chassis][2] < 0.30) or (up < 0.6967067)
    if fallen:
        r = r + w_fall
    unpaid_deposit = int(0)
    for fruit_index in range(MAX_FRUITS):
        if deposited[world, fruit_index] != 0 and deposit_paid[world, fruit_index] == 0:
            unpaid_deposit = unpaid_deposit + 1
            deposit_paid[world, fruit_index] = wp.uint8(1)
    if ground_contact[world] != 0 and loss_paid[world] == 0 and unpaid_deposit == 0:
        r = r + w_loss
        loss_paid[world] = wp.uint8(1)
    if unpaid_deposit > 0 and goal[world] != 1:
        r = r + w_deposit * float(unpaid_deposit)
    episode_time[world] = episode_time[world] + dt
    timed_out[world] = wp.uint8(episode_time[world] >= timeout_s[world])
    terminated[world] = wp.uint8(fallen or failed[world] != 0 or success[world] != 0 or timed_out[world] != 0)
    reward[world] = r


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
def _clear_masked_int(mask: wp.array(dtype=wp.uint8), values: wp.array(dtype=int)):
    world = wp.tid()
    if mask[world] != 0:
        values[world] = 0


@wp.kernel
def _masked_target_reset(mask: wp.array(dtype=wp.uint8), targets: wp.array2d(dtype=float),
                         initial: wp.array(dtype=float)):
    world, joint = wp.tid()
    if mask[world] != 0:
        targets[world, joint] = initial[joint]


@wp.kernel
def _masked_episode_reset(mask: wp.array(dtype=wp.uint8), previous: wp.array2d(dtype=float),
                          distance: wp.array(dtype=float), reward: wp.array(dtype=float),
                          terminated: wp.array(dtype=wp.uint8),
                          previous_potential: wp.array(dtype=float), previous_damage: wp.array(dtype=float),
                          previous_action: wp.array2d(dtype=float), episode_time: wp.array(dtype=float),
                          timed_out: wp.array(dtype=wp.uint8), shaping_ref: wp.array(dtype=int)):
    world, index = wp.tid()
    if mask[world] == 0:
        return
    if index < previous.shape[1]:
        previous[world, index] = 0.0
    if index < previous_action.shape[1]:
        previous_action[world, index] = 0.0
    if index == 0:
        distance[world] = 0.0
        reward[world] = 0.0
        terminated[world] = wp.uint8(0)
        previous_potential[world] = 0.0
        previous_damage[world] = 0.0
        episode_time[world] = 0.0
        timed_out[world] = wp.uint8(0)
        shaping_ref[world] = 0


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
                          previous_potential: wp.array(dtype=float), reward: wp.array(dtype=float),
                          goal: wp.array(dtype=int), detached: wp.array(dtype=wp.uint8),
                          shaping_ref: wp.array(dtype=int), basket_distance: wp.array(dtype=float),
                          episode_time: wp.array(dtype=float), timed_out: wp.array(dtype=wp.uint8)):
    world = wp.tid()
    if mask[world] == 0:
        return
    use_basket = 1 if (goal[world] == 0 or detached[world] != 0) else 0
    d_shape = basket_distance[world] if use_basket != 0 else distance[world]
    previous_potential[world] = wp.exp(-d_shape / 0.25)
    shaping_ref[world] = use_basket
    reward[world] = 0.0
    episode_time[world] = 0.0
    timed_out[world] = wp.uint8(0)


@wp.kernel
def _basket_distance(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                     chassis: int, fruit_bodies: wp.array(dtype=int),
                     active_fruit: wp.array(dtype=int), basket_center: wp.vec3,
                     distance: wp.array(dtype=float)):
    world = wp.tid()
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
    distance[world] = wp.length(xpos[world, fruit] - basket_world)


@wp.kernel
def _write_base_commands(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float),
                         allow: wp.array(dtype=wp.uint8), scale: wp.vec3, slew: wp.vec3):
    world, axis = wp.tid()
    if allow[world] == 0:
        dst[world, axis] = 0.0
        return
    scale_i = scale[0] if axis == 0 else (scale[1] if axis == 1 else scale[2])
    slew_i = slew[0] if axis == 0 else (slew[1] if axis == 1 else slew[2])
    desired = wp.clamp(src[world, axis], -1.0, 1.0) * scale_i
    delta = wp.clamp(desired - dst[world, axis], -slew_i, slew_i)
    dst[world, axis] = dst[world, axis] + delta


@wp.kernel
def _configure_skill_reset(mask: wp.array(dtype=wp.uint8), reset_mode: wp.array(dtype=int),
                           qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                           targets: wp.array2d(dtype=float), site_xpos: wp.array2d(dtype=wp.vec3),
                           fruit_qposadr: wp.array(dtype=int), fruit_dofadr: wp.array(dtype=int),
                           tcp_site: int, jaw_qposadr: int, jaw_closed: float, jaw_open: float,
                           equality_index: wp.array(dtype=int),
                           eq_active: wp.array2d(dtype=wp.bool), detached: wp.array(dtype=wp.uint8),
                           grasped: wp.array(dtype=wp.uint8), grasp_paid: wp.array(dtype=wp.uint8),
                           chassis_qposadr: int, approach_offset_m: float,
                           randomize: wp.array(dtype=wp.uint8),
                           layout_dx: wp.array(dtype=float), layout_dy: wp.array(dtype=float)):
    world = wp.tid()
    if mask[world] == 0:
        return
    mode = reset_mode[world]
    fruit_qadr = fruit_qposadr[0]
    fruit_dadr = fruit_dofadr[0]
    target_eq = equality_index[0]
    if mode == 1:
        tcp = site_xpos[world, tcp_site]
        qpos[world, fruit_qadr + 0] = tcp[0]
        qpos[world, fruit_qadr + 1] = tcp[1]
        qpos[world, fruit_qadr + 2] = tcp[2]
        qpos[world, fruit_qadr + 3] = 1.0
        qpos[world, fruit_qadr + 4] = 0.0
        qpos[world, fruit_qadr + 5] = 0.0
        qpos[world, fruit_qadr + 6] = 0.0
        for i in range(6):
            qvel[world, fruit_dadr + i] = 0.0
        if target_eq >= 0 and target_eq < eq_active.shape[1]:
            eq_active[world, target_eq] = False
        detached[world] = wp.uint8(1)
        grasped[world] = wp.uint8(1)
        grasp_paid[world] = wp.uint8(1)
        qpos[world, jaw_qposadr] = jaw_closed
        targets[world, 18] = jaw_closed
    if mode == 2:
        qpos[world, jaw_qposadr] = jaw_open
        targets[world, 18] = jaw_open
    if mode == 3:
        qpos[world, chassis_qposadr + 0] = qpos[world, chassis_qposadr + 0] - approach_offset_m
    if randomize[world] != 0:
        qpos[world, chassis_qposadr + 0] = qpos[world, chassis_qposadr + 0] + layout_dx[world]
        qpos[world, chassis_qposadr + 1] = qpos[world, chassis_qposadr + 1] + layout_dy[world]


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
        if camera is not None:
            from .spot_cameras import GRIPPER_FRAMES, require_mujoco_gripper_cameras
            require_mujoco_gripper_cameras(self.model, self.manifest.get('robot'))
            if camera not in GRIPPER_FRAMES:
                raise ValueError(f'FastRuntime camera must be a RELIC gripper sensor, not {camera!r}')
            self.policy_camera = camera
        else:
            self.policy_camera = None
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
            self._previous_potential = wp.zeros(worlds, dtype=float, device=self.device)
            self._previous_damage = wp.zeros(worlds, dtype=float, device=self.device)
            self._previous_action = wp.zeros((worlds, 7), dtype=float, device=self.device)
            self._episode_time = wp.zeros(worlds, dtype=float, device=self.device)
            self._timeout_s = wp.array(np.full(worlds, 180.0, dtype=np.float32), dtype=float, device=self.device)
            self._guidance = wp.ones(worlds, dtype=float, device=self.device)
            self._shaping_ref = wp.zeros(worlds, dtype=int, device=self.device)
            self._reset_mode = wp.zeros(worlds, dtype=int, device=self.device)
            self._allow_locomotion = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._basket_distance = wp.zeros(worlds, dtype=float, device=self.device)
            self._timed_out = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._base_commands = wp.zeros((worlds, 3), dtype=float, device=self.device)
            self._distance = wp.zeros(worlds, dtype=float, device=self.device)
            self._reward = wp.zeros(worlds, dtype=float, device=self.device)
            self._terminated = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._flags = wp.zeros(worlds, dtype=int, device=self.device)
            self._all_mask = wp.ones(worlds, dtype=wp.uint8, device=self.device)
            self._actions = wp.zeros((worlds, 7), dtype=float, device=self.device)
            self._randomize_layout = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._layout_dx = wp.zeros(worlds, dtype=float, device=self.device)
            self._layout_dy = wp.zeros(worlds, dtype=float, device=self.device)
            fruit = self.manifest.get('fruits', self.manifest.get('fruit', []))
            if not fruit:
                raise ValueError('Fast scene must contain at least one fruit')
            if len(fruit) > MAX_FRUITS:
                raise ValueError(f'FastRuntime supports at most {MAX_FRUITS} independent fruit bodies')
            qposadrs, dofadrs = [], []
            for entry in fruit:
                body = int(self.model.body(entry['body']).id)
                joint = int(self.model.body_jntadr[body])
                qposadrs.append(int(self.model.jnt_qposadr[joint]))
                dofadrs.append(int(self.model.jnt_dofadr[joint]))
            self.fruit_body = int(self.model.body(fruit[0]['body']).id)
            self._fruit_qposadr = qposadrs[0]
            self._fruit_dofadr = dofadrs[0]
            self._fruit_qposadrs = wp.array(_pad_ids(qposadrs, fill=0), dtype=int, device=self.device)
            self._fruit_dofadrs = wp.array(_pad_ids(dofadrs, fill=0), dtype=int, device=self.device)
            chassis_joint = int(self.model.body_jntadr[self.control.chassis])
            self._chassis_qposadr = int(self.model.jnt_qposadr[chassis_joint])
            self._jaw_qposadr = int(self.control.contract.qids[18])
            jaw_range = np.asarray(self.model.jnt_range[self.control.contract.joints[18]], dtype=np.float32)
            self._jaw_closed = float(jaw_range[0] if np.isfinite(jaw_range[0]) else 0.0)
            self._jaw_open = float(jaw_range[1] if np.isfinite(jaw_range[1]) else 0.8)
            from treesim.basket import CENTER
            self._basket_center = wp.vec3(*CENTER)
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
            self.task.goal.assign(np.full(worlds, 2, dtype=np.int32))
            self._gamma_step = 0.9996
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
            if self.policy_camera is not None:
                from .sensors_warp import WarpRGBDRig
                size = (resolution, resolution) if isinstance(resolution, int) else tuple(resolution)
                self.rig = WarpRGBDRig(self.model, self.data, cameras=(self.policy_camera,), resolution=size)

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
                  inputs=[self.data.xipos, self.data.site_xpos, self.data.xpos, self.data.xmat,
                          self.tcp_site, self.task.fruit_body, self.task.active_fruit, self.chassis,
                          self._previous_potential,
                          self._shaping_ref, self._previous_damage, self._previous_action, self._actions,
                          self._episode_time, self._timeout_s, self._guidance, self.task.goal,
                          self._reward, self._terminated, self._timed_out, self._distance, mask,
                          self.task.success, self.task.failed, self.task.detached, self.task.grasped,
                          self.task.retained_detach, self.task.ground_contact, self.task.damage_proxy,
                          self.task.grasp_paid, self.task.detach_paid, self.task.deposit_paid,
                          self.task.deposited, self.task.loss_paid, self._basket_center, self.control_dt,
                          self._gamma_step, W_DEPOSIT, W_GRASP_STABLE, W_DETACH_HELD, W_LOSS,
                          W_DAMAGE_PER_UNIT, W_FALL, W_TIME_PER_S, W_SMOOTH], device=self.device)

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
            'timed_out': wp.to_torch(self._timed_out).bool(),
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

    def set_base_commands(self, commands):
        """Latch tanh-scaled N3 base commands. Zeroed when locomotion is disabled."""
        import torch
        if not isinstance(commands, torch.Tensor) or not commands.is_cuda or commands.dtype != torch.float32:
            raise ValueError('base commands must be a CUDA float32 Torch tensor')
        if tuple(commands.shape) != (self.worlds, 3):
            raise ValueError(f'base commands must have shape ({self.worlds}, 3)')
        with wp.ScopedDevice(self.device):
            wp.copy(self._base_commands, wp.from_torch(commands))
            wp.launch(_write_base_commands, dim=(self.worlds, 3),
                      inputs=[self._base_commands, self.control.commands, self._allow_locomotion,
                              wp.vec3(0.4, 0.3, 0.7), wp.vec3(0.02, 0.02, 0.04)], device=self.device)

    def configure_skills(self, skills):
        """Write per-world goal/reset/timeout buffers. Safe during captured graphs."""
        import numpy as np
        required = ('goal_id', 'reset_mode', 'timeout_s', 'guidance_weight', 'allow_locomotion')
        if any(name not in skills for name in required):
            raise ValueError('configure_skills requires goal, reset, timeout, guidance and locomotion')
        worlds = self.worlds
        def _arr(name, dtype, default=None):
            if name not in skills:
                if default is None:
                    raise ValueError(f'{name} missing')
                return np.full(worlds, default, dtype=dtype)
            value = np.asarray(skills[name])
            if value.shape != (worlds,):
                raise ValueError(f'{name} must have shape ({worlds},)')
            return np.ascontiguousarray(value.astype(dtype, copy=False))
        self.task.goal.assign(_arr('goal_id', np.int32))
        self._reset_mode.assign(_arr('reset_mode', np.int32))
        self._timeout_s.assign(_arr('timeout_s', np.float32))
        self._guidance.assign(_arr('guidance_weight', np.float32))
        self._allow_locomotion.assign(_arr('allow_locomotion', np.uint8))
        self.task.continue_after_success.assign(_arr('continue_after_success', np.uint8, 0))
        required_harvests = np.clip(_arr('required_harvests', np.int32, 1), 1, self.task.fruit_count)
        self.task.required_harvests.assign(required_harvests)
        self._randomize_layout.assign(_arr('randomize_layout', np.uint8, 0))
        self._layout_dx.assign(_arr('layout_dx_m', np.float32, 0.0))
        self._layout_dy.assign(_arr('layout_dy_m', np.float32, 0.0))

    def pixels(self):
        if self.rig is None or self.policy_camera is None:
            raise RuntimeError('FastRuntime was created without a gripper camera')
        import mujoco_warp as mw
        self._refresh(mw)
        self.rig.capture(self.gpu_model, self.data, 0.0)
        return self.rig.tensor(self.policy_camera)

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
                              self._reward, self._terminated, self._previous_potential,
                              self._previous_damage, self._previous_action, self._episode_time,
                              self._timed_out, self._shaping_ref], device=self.device)
            self.task.reset(mask_wp)
            mw.forward(self.gpu_model, self.data)
            self._refresh(mw)
            wp.launch(_configure_skill_reset, dim=self.worlds, inputs=[
                mask_wp, self._reset_mode, self.data.qpos, self.data.qvel, self.control.targets,
                self.data.site_xpos, self._fruit_qposadrs, self._fruit_dofadrs, self.tcp_site,
                self._jaw_qposadr, self._jaw_closed, self._jaw_open, self.task.equality_index,
                self.task.eq_active, self.task.detached, self.task.grasped, self.task.grasp_paid,
                self._chassis_qposadr, 1.0, self._randomize_layout, self._layout_dx, self._layout_dy],
                device=self.device)
            mw.forward(self.gpu_model, self.data)
            self._refresh(mw)
            self._measure_reward(mask_wp)
            wp.launch(_basket_distance, dim=self.worlds,
                      inputs=[self.data.xpos, self.data.xmat, self.chassis, self.task.fruit_body,
                              self.task.active_fruit, self._basket_center, self._basket_distance],
                      device=self.device)
            wp.launch(_masked_seed_distance, dim=self.worlds,
                      inputs=[mask_wp, self._distance, self._previous_potential, self._reward,
                              self.task.goal, self.task.detached, self._shaping_ref,
                              self._basket_distance, self._episode_time, self._timed_out], device=self.device)
        return self.observe()

    def drain_faults(self):
        """Reset overflow/bad-action worlds. Abort only on nonfinite qpos/qvel.

        ``check()`` still raises on any latched flag so tests and the final
        report keep the strict contract. Training calls this instead.
        """
        import torch
        flags = wp.to_torch(self._flags)
        nonfinite = (flags & FLAG_NONFINITE) != 0
        if bool(nonfinite.any()):
            raise RuntimeError(f'GPU numerical failure flags={self._flags.numpy().tolist()}')
        recoverable = (flags & FLAG_RECOVERABLE) != 0
        overflow_worlds = int(((flags & FLAG_OVERFLOW) != 0).sum().item())
        bad_action_worlds = int(((flags & FLAG_BAD_ACTION) != 0).sum().item())
        recovered = int(recoverable.sum().item())
        if recovered:
            mask = recoverable.to(dtype=torch.uint8)
            self.reset(mask)
            mask_wp = wp.from_torch(mask)
            with wp.ScopedDevice(self.device):
                wp.launch(_clear_masked_int, dim=self.worlds,
                          inputs=[mask_wp, self._flags], device=self.device)
                wp.launch(_clear_masked_int, dim=self.worlds,
                          inputs=[mask_wp, self.data.overflow], device=self.device)
        return {
            'recovered_worlds': recovered,
            'overflow_worlds': overflow_worlds,
            'bad_action_worlds': bad_action_worlds,
            'mask': recoverable,
        }

    def check(self):
        flags = self._flags.numpy()
        if np.any(flags):
            raise RuntimeError(f'GPU numerical failure flags={flags.tolist()}')
        return {'flags': flags.tolist()}
