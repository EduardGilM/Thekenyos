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
from .curriculum import (
    EASY_PRESET, HOLD_SWEEP_CLEARANCE_M, HOLD_SWEEP_MARGIN_M,
    apply_easy_hover_cohort,
)
from .fast_task import MAX_FRUITS, _pad_ids
from .rewards import (
    W_DAMAGE_PER_UNIT, W_DEPOSIT, W_DETACH_HELD, W_FALL, W_GRASP_STABLE,
    W_LOSS, W_SMOOTH, W_TIME_PER_S,
)

# Latched per-world bits. Overflow, a nonfinite action, or a single-world
# nonfinite qpos/qvel reset that world so one solver stall does not kill the
# batch. A systemic NaN (too many worlds) still aborts.
FLAG_NONFINITE = 1
FLAG_OVERFLOW = 2
FLAG_BAD_ACTION = 4
FLAG_RECOVERABLE = FLAG_NONFINITE | FLAG_OVERFLOW | FLAG_BAD_ACTION
# Abort only when this fraction of the batch is nonfinite in one drain.
NONFINITE_ABORT_FRACTION = 0.25


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
                     shaping_coef: wp.array(dtype=float),
                     shaping_length: wp.array(dtype=float),
                     goal: wp.array(dtype=int), reward: wp.array(dtype=float),
                     terminated: wp.array(dtype=wp.uint8), timed_out: wp.array(dtype=wp.uint8),
                     distance: wp.array(dtype=float), basket_distance: wp.array(dtype=float),
                     basket_xy: wp.array(dtype=float), mask: wp.array(dtype=wp.uint8),
                     success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8),
                     detached: wp.array(dtype=wp.uint8), grasped: wp.array(dtype=wp.uint8),
                     retained_detach: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8),
                     damage_proxy: wp.array(dtype=float),
                     grasp_paid: wp.array(dtype=wp.uint8), detach_paid: wp.array(dtype=wp.uint8),
                     deposit_paid: wp.array2d(dtype=wp.uint8), deposited: wp.array2d(dtype=wp.uint8),
                     loss_paid: wp.array(dtype=wp.uint8),
                     basket_center: wp.vec3, hover_offset: wp.vec3, dt: float, gamma_step: float,
                     deposit_w: wp.array(dtype=float),
                     fail_w: wp.array(dtype=float), fail_paid: wp.array(dtype=wp.uint8),
                     w_grasp: float, w_detach: float, w_loss: float,
                     w_damage: float, w_fall: float, w_time: float, w_smooth: float,
                     shape_hand_fruit: wp.array(dtype=int)):
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
    fruit_world = xpos[world, fruit]
    diff = fruit_world - basket_world
    d_basket = wp.length(diff)
    basket_distance[world] = d_basket
    basket_xy[world] = wp.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
    use_basket = 1 if (goal[world] == 0 or detached[world] != 0) else 0
    d_shape = d_basket if use_basket != 0 else d_tcp
    length = shaping_length[world]
    if length <= 0.0:
        length = 0.25
    if use_basket != 0 and shape_hand_fruit[0] != 0:
        hover_world = basket_world + rotation @ hover_offset
        d_hover = wp.length(fruit_world - hover_world)
        tcp_diff = tcp - basket_world
        d_hand_xy = wp.sqrt(tcp_diff[0] * tcp_diff[0] + tcp_diff[1] * tcp_diff[1])
        phi = 0.25 * wp.exp(-d_hover / length) + 0.75 * wp.exp(-d_hand_xy / length)
    else:
        phi = wp.exp(-d_shape / length)
    shaped = float(0.)
    if shaping_ref[world] == use_basket:
        shaped = guidance[world] * shaping_coef[world] * (gamma_step * phi - previous_potential[world])
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
        r = r + deposit_w[world] * float(unpaid_deposit)
    episode_time[world] = episode_time[world] + dt
    timed_out[world] = wp.uint8(episode_time[world] >= timeout_s[world])
    terminated[world] = wp.uint8(fallen or failed[world] != 0 or success[world] != 0 or timed_out[world] != 0)
    # Ground dumps already pay W_LOSS and end the episode. Putting the
    # −10000 miss on every spill taught easy33 to flee (basket XY 0.42→1.23 m).
    # Keep the jackpot for a true timeout without a settled deposit.
    if timed_out[world] != 0 and success[world] == 0 and fail_paid[world] == 0:
        r = r + fail_w[world]
        fail_paid[world] = wp.uint8(1)
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
                          episode_time: wp.array(dtype=float), timed_out: wp.array(dtype=wp.uint8),
                          shaping_length: wp.array(dtype=float),
                          site_xpos: wp.array2d(dtype=wp.vec3), tcp_site: int,
                          xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                          chassis: int, basket_center: wp.vec3, hover_offset: wp.vec3,
                          shape_hand_fruit: wp.array(dtype=int),
                          fruit_bodies: wp.array(dtype=int), active_fruit: wp.array(dtype=int)):
    world = wp.tid()
    if mask[world] == 0:
        return
    use_basket = 1 if (goal[world] == 0 or detached[world] != 0) else 0
    d_shape = basket_distance[world] if use_basket != 0 else distance[world]
    length = shaping_length[world]
    if length <= 0.0:
        length = 0.25
    if use_basket != 0 and shape_hand_fruit[0] != 0:
        basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
        hover_world = basket_world + xmat[world, chassis] @ hover_offset
        idx = active_fruit[world]
        if idx < 0 or idx >= MAX_FRUITS:
            idx = 0
        fruit = fruit_bodies[idx]
        d_hover = wp.length(xpos[world, fruit] - hover_world)
        tcp = site_xpos[world, tcp_site]
        d_hand_xy = wp.sqrt((tcp[0] - basket_world[0]) * (tcp[0] - basket_world[0])
                            + (tcp[1] - basket_world[1]) * (tcp[1] - basket_world[1]))
        previous_potential[world] = 0.5 * (wp.exp(-d_hover / length) + wp.exp(-d_hand_xy / length))
    else:
        previous_potential[world] = wp.exp(-d_shape / length)
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
                           xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                           fruit_qposadr: wp.array(dtype=int), fruit_dofadr: wp.array(dtype=int),
                           tcp_site: int, tcp_body: int, grasp_local: wp.vec3, use_pocket: int,
                           jaw_qposadr: int, deposit_jaw: wp.array(dtype=float), jaw_open: float,
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
        if use_pocket != 0:
            pos = xpos[world, tcp_body] + xmat[world, tcp_body] @ grasp_local
        else:
            pos = site_xpos[world, tcp_site]
        qpos[world, fruit_qadr + 0] = pos[0]
        qpos[world, fruit_qadr + 1] = pos[1]
        qpos[world, fruit_qadr + 2] = pos[2]
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
        jaw = deposit_jaw[world]
        qpos[world, jaw_qposadr] = jaw
        targets[world, 18] = jaw
    if mode == 2:
        qpos[world, jaw_qposadr] = jaw_open
        targets[world, 18] = jaw_open
    if mode == 3:
        qpos[world, chassis_qposadr + 0] = qpos[world, chassis_qposadr + 0] - approach_offset_m
    if randomize[world] != 0:
        qpos[world, chassis_qposadr + 0] = qpos[world, chassis_qposadr + 0] + layout_dx[world]
        qpos[world, chassis_qposadr + 1] = qpos[world, chassis_qposadr + 1] + layout_dy[world]


@wp.kernel
def _apply_easy_start(mask: wp.array(dtype=wp.uint8), reset_mode: wp.array(dtype=int),
                      qpos: wp.array2d(dtype=float), targets: wp.array2d(dtype=float),
                      qids: wp.array(dtype=int), start_q: wp.array2d(dtype=float),
                      start_index: wp.array(dtype=int), jaw_qposadr: int,
                      jaw_hold_next: wp.array(dtype=float),
                      jaw_hold: wp.array(dtype=float)):
    world = wp.tid()
    if mask[world] == 0 or reset_mode[world] != 1:
        return
    idx = start_index[world]
    if idx < 0:
        idx = 0
    for joint in range(6):
        qid = qids[joint + 12]
        value = start_q[idx, joint]
        qpos[world, qid] = value
        targets[world, joint + 12] = value
    hold = jaw_hold_next[world]
    jaw_hold[world] = hold
    qpos[world, jaw_qposadr] = hold
    targets[world, 18] = hold


@wp.kernel
def _apply_easy_jaw_hold(mask: wp.array(dtype=wp.uint8), reset_mode: wp.array(dtype=int),
                         qpos: wp.array2d(dtype=float), targets: wp.array2d(dtype=float),
                         jaw_qposadr: int, jaw_hold: wp.array(dtype=float)):
    world = wp.tid()
    if mask[world] == 0 or reset_mode[world] != 1:
        return
    hold = jaw_hold[world]
    qpos[world, jaw_qposadr] = hold
    targets[world, 18] = hold


@wp.func
def _in_release_zone(fruit: wp.vec3, tcp: wp.vec3, basket: wp.vec3,
                     rot: wp.mat33, open_half_xy: wp.vec2,
                     open_xy_m: float, rim_z_m: float, release_at_center: int,
                     release_over_opening: int, open_max_above_rim_m: float) -> int:
    """1 if the scripted jaw should open. Not a weld."""
    if release_over_opening != 0:
        f_local = wp.transpose(rot) @ (fruit - basket)
        t_local = wp.transpose(rot) @ (tcp - basket)
        fruit_ok = wp.abs(f_local[0]) < open_half_xy[0] and wp.abs(f_local[1]) < open_half_xy[1]
        tcp_ok = wp.abs(t_local[0]) < open_half_xy[0] and wp.abs(t_local[1]) < open_half_xy[1]
        if fruit_ok and tcp_ok and f_local[2] < rim_z_m + open_max_above_rim_m:
            return 1
        return 0
    fdx = fruit[0] - basket[0]
    fdy = fruit[1] - basket[1]
    fruit_xy = wp.sqrt(fdx * fdx + fdy * fdy) < open_xy_m
    if release_at_center != 0:
        tdx = tcp[0] - basket[0]
        tdy = tcp[1] - basket[1]
        if fruit_xy and wp.sqrt(tdx * tdx + tdy * tdy) < open_xy_m:
            return 1
        return 0
    dz = fruit[2] - basket[2]
    if fruit_xy and dz > 0.0 and dz < rim_z_m:
        return 1
    return 0


@wp.kernel
def _scripted_jaw_hold(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                       site_xpos: wp.array2d(dtype=wp.vec3), tcp_site: int,
                       chassis: int, fruit_bodies: wp.array(dtype=int),
                       active_fruit: wp.array(dtype=int), basket_center: wp.vec3,
                       targets: wp.array2d(dtype=float), jaw_hold: wp.array(dtype=float),
                       jaw_open: float, open_xy_m: float, rim_z_m: float, max_delta: float,
                       release_at_center: wp.array(dtype=int),
                       release_over_opening: wp.array(dtype=int),
                       open_half_xy: wp.vec2, open_max_above_rim_m: float,
                       actions: wp.array2d(dtype=float)):
    """Overwrite only the jaw increment. The student still moves the arm."""
    world = wp.tid()
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
    fruit_pos = xpos[world, fruit]
    over = _in_release_zone(fruit_pos, site_xpos[world, tcp_site], basket_world,
                            xmat[world, chassis], open_half_xy,
                            open_xy_m, rim_z_m, release_at_center[0],
                            release_over_opening[0], open_max_above_rim_m)
    desired = jaw_open if over == 1 else jaw_hold[world]
    current = targets[world, 18]
    delta = wp.clamp(desired - current, -max_delta, max_delta)
    actions[world, 6] = delta / max_delta


@wp.kernel
def _adapt_scripted_jaw(site_xpos: wp.array2d(dtype=wp.vec3), tcp_site: int,
                        xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                        chassis: int, fruit_bodies: wp.array(dtype=int),
                        active_fruit: wp.array(dtype=int), basket_center: wp.vec3,
                        jaw_hold: wp.array(dtype=float), jaw_open: float, jaw_closed: float,
                        open_xy_m: float, rim_z_m: float, slip_tighten_m: float,
                        max_close_frac: float, release_at_center: wp.array(dtype=int),
                        release_over_opening: wp.array(dtype=int),
                        open_half_xy: wp.vec2, open_max_above_rim_m: float):
    """Tighten the hold if the free fruit is leaving the mouth. Not a weld."""
    world = wp.tid()
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
    fruit_pos = xpos[world, fruit]
    if _in_release_zone(fruit_pos, site_xpos[world, tcp_site], basket_world,
                        xmat[world, chassis], open_half_xy,
                        open_xy_m, rim_z_m, release_at_center[0],
                        release_over_opening[0], open_max_above_rim_m) == 1:
        return
    slip = wp.length(fruit_pos - site_xpos[world, tcp_site])
    if slip <= slip_tighten_m:
        return
    max_hold = jaw_open + max_close_frac * (jaw_closed - jaw_open)
    nxt = jaw_hold[world] + float(0.25) * (jaw_closed - jaw_hold[world])
    lo = wp.min(jaw_hold[world], max_hold)
    hi = wp.max(jaw_hold[world], max_hold)
    jaw_hold[world] = wp.clamp(nxt, lo, hi)


@wp.kernel
def _pin_scripted_jaw(enabled: wp.array(dtype=int),
                      xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                      chassis: int, fruit_bodies: wp.array(dtype=int),
                      active_fruit: wp.array(dtype=int), basket_center: wp.vec3,
                      qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                      targets: wp.array2d(dtype=float), jaw_qposadr: int, jaw_dofadr: int,
                      jaw_hold: wp.array(dtype=float), jaw_open: float, open_xy_m: float,
                      rim_z_m: float, site_xpos: wp.array2d(dtype=wp.vec3), tcp_site: int,
                      release_at_center: wp.array(dtype=int),
                      release_over_opening: wp.array(dtype=int),
                      open_half_xy: wp.vec2, open_max_above_rim_m: float):
    """Kinematic jaw hold/open. The 0.3 N·m PD alone lets the kiwi slip out."""
    if enabled[0] == 0:
        return
    world = wp.tid()
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
    fruit_pos = xpos[world, fruit]
    over = _in_release_zone(fruit_pos, site_xpos[world, tcp_site], basket_world,
                            xmat[world, chassis], open_half_xy,
                            open_xy_m, rim_z_m, release_at_center[0],
                            release_over_opening[0], open_max_above_rim_m)
    desired = jaw_open if over == 1 else jaw_hold[world]
    qpos[world, jaw_qposadr] = desired
    qvel[world, jaw_dofadr] = 0.0
    targets[world, 18] = desired


@wp.kernel
def _privileged_deposit_action(xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33),
                               chassis: int, fruit_bodies: wp.array(dtype=int),
                               active_fruit: wp.array(dtype=int), basket_center: wp.vec3,
                               goal: wp.array(dtype=int), detached: wp.array(dtype=wp.uint8),
                               targets: wp.array2d(dtype=float), hover_q: wp.array(dtype=float),
                               jaw_hold: wp.array(dtype=float), jaw_open: float, open_xy_m: float,
                               rim_z_m: float, max_delta: float,
                               site_xpos: wp.array2d(dtype=wp.vec3), tcp_site: int,
                               release_at_center: wp.array(dtype=int),
                               release_over_opening: wp.array(dtype=int),
                               open_half_xy: wp.vec2, open_max_above_rim_m: float,
                               out_applied: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    active = 1 if (goal[world] == 0 or detached[world] != 0) else 0
    if active == 0:
        out_applied[world, joint] = 0.0
        return
    idx = active_fruit[world]
    if idx < 0 or idx >= MAX_FRUITS:
        idx = 0
    fruit = fruit_bodies[idx]
    basket_world = xpos[world, chassis] + xmat[world, chassis] @ basket_center
    fruit_pos = xpos[world, fruit]
    over = _in_release_zone(fruit_pos, site_xpos[world, tcp_site], basket_world,
                            xmat[world, chassis], open_half_xy,
                            open_xy_m, rim_z_m, release_at_center[0],
                            release_over_opening[0], open_max_above_rim_m)
    if joint < 6:
        if over == 1:
            out_applied[world, joint] = 0.0
            return
        desired = hover_q[joint]
        current = targets[world, joint + 12]
    else:
        desired = jaw_open if over == 1 else jaw_hold[world]
        current = targets[world, 18]
    delta = wp.clamp(desired - current, -max_delta, max_delta)
    out_applied[world, joint] = delta / max_delta


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
            self._shaping_coef = wp.full(worlds, float(EASY_PRESET['default_shaping_coef']),
                                         dtype=float, device=self.device)
            self._shaping_length = wp.full(worlds, float(EASY_PRESET['default_shaping_length_m']),
                                           dtype=float, device=self.device)
            self._deposit_w = wp.full(worlds, float(W_DEPOSIT), dtype=float, device=self.device)
            self._fail_w = wp.zeros(worlds, dtype=float, device=self.device)
            self._shaping_ref = wp.zeros(worlds, dtype=int, device=self.device)
            self._reset_mode = wp.zeros(worlds, dtype=int, device=self.device)
            self._allow_locomotion = wp.zeros(worlds, dtype=wp.uint8, device=self.device)
            self._basket_distance = wp.zeros(worlds, dtype=float, device=self.device)
            self._basket_xy = wp.zeros(worlds, dtype=float, device=self.device)
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
            self._jaw_dofadr = int(self.control.contract.dofs[18])
            from .reach_teacher import jaw_open_closed_q
            self._jaw_open, self._jaw_closed = jaw_open_closed_q(self.model, self._jaw_qposadr)
            self._easy_pin = wp.zeros(1, dtype=int, device=self.device)
            self._shape_hand_fruit = wp.zeros(1, dtype=int, device=self.device)
            self._release_at_center = wp.zeros(1, dtype=int, device=self.device)
            self._release_over_opening = wp.zeros(1, dtype=int, device=self.device)
            from treesim.basket import CENTER, SIZE
            from .reach_teacher import opening_half_xy_m
            self._basket_center = wp.vec3(*CENTER)
            self._open_rim_z_m = float(SIZE[2])
            hx, hy = opening_half_xy_m(inset_m=float(EASY_PRESET['release_opening_inset_m']))
            self._open_half_xy = wp.vec2(float(hx), float(hy))
            # Captured into the CUDA graph: changing this later does not
            # retarget `_pin_scripted_jaw`. 16 cm above the rim is below
            # the 28 cm hover so a hover-high dump is not forced.
            self._open_max_above_rim_m = float(EASY_PRESET['release_max_above_rim_m'])
            if (not np.isfinite(self._open_max_above_rim_m)
                    or not 0.0 <= self._open_max_above_rim_m <= 0.4):
                raise ValueError('release_max_above_rim_m must be finite in [0, 0.4] m')
            # Floor-to-hover: SIZE.z + 28 cm. A 10 cm-over-hole TCP puts the
            # ~0.20 m wrist through the liner; this is the known-safe IK height.
            self._hover_offset = wp.vec3(
                0.0, 0.0, float(SIZE[2] + EASY_PRESET['hover_clearance_m']))
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
            self._tcp_body = int(self.model.site_bodyid[self.tcp_site])
            self._grasp_local = wp.vec3(0.0, 0.0, 0.0)
            self._grasp_local_host = np.zeros(3, dtype=np.float64)
            self._use_pocket = 0
            self.chassis = self.control.chassis
            from .fast_task import FastHarvestTask
            self.task = FastHarvestTask(self.model, self.data, self.manifest)
            self.task.goal.assign(np.full(worlds, 2, dtype=np.int32))
            self._gamma_step = 0.9996
            self._easy = False
            self._open_xy_m = float(EASY_PRESET['open_xy_m'])
            self._easy_far_frac = 0.0
            drop_q, drop_err, start_qs, start_errs = self._solve_easy_poses(initial.qpos)
            from .reach_teacher import hold_close_fracs
            self._hover_q = wp.array(drop_q, dtype=float, device=self.device)
            self._easy_catalog_n = int(start_qs.shape[0])
            hover = np.asarray(drop_q, dtype=np.float32).reshape(1, 6)
            start_qs = np.concatenate([start_qs, hover], axis=0)
            self._hover_start_index = int(start_qs.shape[0] - 1)
            self._easy_start_q = wp.array(start_qs, dtype=float, device=self.device)
            self._easy_start_index = wp.zeros(worlds, dtype=int, device=self.device)
            self._easy_hover_cohort = wp.zeros(worlds, dtype=int, device=self.device)
            self._easy_jaw_hold = wp.zeros(worlds, dtype=float, device=self.device)
            self._easy_jaw_hold_next = wp.zeros(worlds, dtype=float, device=self.device)
            closed_holds = np.full(worlds, self._jaw_closed, dtype=np.float32)
            self._easy_jaw_hold.assign(closed_holds)
            self._easy_jaw_hold_next.assign(closed_holds)
            self._hold_close_fracs = np.asarray(hold_close_fracs(), dtype=np.float64)
            self._chosen_close_frac = float(self._hold_close_fracs[len(self._hold_close_fracs) // 2])
            self._hold_sweep = None
            self.hover_error_m = float(drop_err) if np.isfinite(drop_err) else 1.0
            self.easy_start_error_m = float(np.max(start_errs)) if np.isfinite(start_errs).all() else 1.0
            self._teacher_applied = wp.zeros((worlds, 7), dtype=float, device=self.device)
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
                    wp.launch(_pin_scripted_jaw, dim=self.worlds, inputs=[
                        self._easy_pin, self.data.xpos, self.data.xmat, self.chassis,
                        self.task.fruit_body, self.task.active_fruit, self._basket_center,
                        self.data.qpos, self.data.qvel, self.control.targets,
                        int(self._jaw_qposadr), int(self._jaw_dofadr), self._easy_jaw_hold,
                        float(self._jaw_open), float(self._open_xy_m),
                        float(self._open_rim_z_m), self.data.site_xpos, int(self.tcp_site),
                        self._release_at_center, self._release_over_opening,
                        self._open_half_xy, float(self._open_max_above_rim_m)],
                        device=self.device)
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
                          self._episode_time, self._timeout_s, self._guidance, self._shaping_coef,
                          self._shaping_length, self.task.goal,
                          self._reward, self._terminated, self._timed_out, self._distance,
                          self._basket_distance, self._basket_xy, mask,
                          self.task.success, self.task.failed, self.task.detached, self.task.grasped,
                          self.task.retained_detach, self.task.ground_contact, self.task.damage_proxy,
                          self.task.grasp_paid, self.task.detach_paid, self.task.deposit_paid,
                          self.task.deposited, self.task.loss_paid, self._basket_center,
                          self._hover_offset, self.control_dt,
                          self._gamma_step, self._deposit_w, self._fail_w, self.task.fail_paid,
                          W_GRASP_STABLE, W_DETACH_HELD, W_LOSS,
                          W_DAMAGE_PER_UNIT, W_FALL, W_TIME_PER_S, W_SMOOTH,
                          self._shape_hand_fruit], device=self.device)

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
            if self._easy:
                wp.launch(_adapt_scripted_jaw, dim=self.worlds, inputs=[
                    self.data.site_xpos, int(self.tcp_site), self.data.xpos, self.data.xmat,
                    self.chassis, self.task.fruit_body, self.task.active_fruit,
                    self._basket_center, self._easy_jaw_hold, float(self._jaw_open),
                    float(self._jaw_closed), float(self._open_xy_m),
                    float(self._open_rim_z_m), 0.04, 0.70, self._release_at_center,
                    self._release_over_opening, self._open_half_xy,
                    float(self._open_max_above_rim_m)],
                    device=self.device)
                wp.launch(_scripted_jaw_hold, dim=self.worlds, inputs=[
                    self.data.xpos, self.data.xmat, self.data.site_xpos, int(self.tcp_site),
                    self.chassis, self.task.fruit_body, self.task.active_fruit,
                    self._basket_center, self.control.targets,
                    self._easy_jaw_hold, float(self._jaw_open), float(self._open_xy_m),
                    float(self._open_rim_z_m), float(2.5 * self.control_dt),
                    self._release_at_center, self._release_over_opening,
                    self._open_half_xy, float(self._open_max_above_rim_m),
                    self._actions], device=self.device)
            wp.capture_launch(self.graph)
        return self.observe(), wp.to_torch(self._reward), wp.to_torch(self._terminated).bool(), {
            'distance_m': wp.to_torch(self._distance),
            'basket_distance_m': wp.to_torch(self._basket_distance),
            'basket_xy_m': wp.to_torch(self._basket_xy),
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

    def _solve_easy_poses(self, initial_qpos):
        """CPU IK: high drop pose, then over-opening or outside-crate starts."""
        from .reach_teacher import (
            easy_over_opening_local_m, easy_start_local_m, hover_tcp_world_m,
            random_easy_start_local_m, solve_tcp_hover, tcp_outside_basket,
            tcp_over_opening_above_rim,
        )
        contract = self.control.contract
        qpos = np.asarray(initial_qpos, dtype=np.float64).reshape(-1)
        qids = np.asarray(contract.qids[12:18], dtype=int)
        dofs = np.asarray(contract.dofs[12:18], dtype=int)
        ranges = np.tile(np.array([-np.pi, np.pi], dtype=np.float64), (6, 1))
        limited = np.asarray(self.model.jnt_limited[contract.joints[12:18]], dtype=bool)
        ranges[limited] = np.asarray(self.model.jnt_range[contract.joints[12:18]], dtype=np.float64)[limited]
        q_home = qpos[qids].copy()
        import mujoco
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, data)
        chassis_p = np.asarray(data.xpos[self.chassis], dtype=np.float64)
        chassis_R = np.asarray(data.xmat[self.chassis], dtype=np.float64).reshape(3, 3)
        home_tcp = np.asarray(data.site_xpos[self.tcp_site], dtype=np.float64)
        home_local = chassis_R.T @ (home_tcp - chassis_p)
        drop_target = hover_tcp_world_m(
            chassis_p, chassis_R, EASY_PRESET['hover_clearance_m'])
        drop_q, drop_err = solve_tcp_hover(
            self.model, qpos, self.tcp_site, drop_target, qids, dofs, q_home, ranges)
        if not np.isfinite(drop_q).all() or not np.isfinite(drop_err):
            drop_q, drop_err = q_home.astype(np.float32), 1.0
        n = int(EASY_PRESET['n_start_poses'])
        accept = float(EASY_PRESET['ik_accept_err_m'])
        over_opening = bool(EASY_PRESET.get('start_over_opening'))
        start_qs = []
        start_errs = []
        q_init = drop_q.astype(np.float64) if over_opening else q_home.copy()
        if (over_opening and np.isfinite(drop_q).all() and np.isfinite(drop_err)
                and float(drop_err) <= accept):
            start_qs.append(np.asarray(drop_q, dtype=np.float32).reshape(6))
            start_errs.append(float(drop_err))
        rng = np.random.default_rng(7)
        attempts = 0
        max_attempts = n * 8
        while len(start_qs) < n and attempts < max_attempts:
            attempts += 1
            if over_opening:
                local = easy_over_opening_local_m(None if not start_qs else rng)
                if not tcp_over_opening_above_rim(local):
                    continue
            elif not start_qs:
                local = easy_start_local_m(
                    0.0, home_local,
                    margin_m=EASY_PRESET['start_margin_m'],
                    clearance_m=EASY_PRESET['start_clearance_m'])
                if not tcp_outside_basket(local, margin_m=0.04, above_rim_m=0.0):
                    continue
            else:
                local = random_easy_start_local_m(
                    rng, home_local,
                    margin_m=EASY_PRESET['start_margin_m'],
                    clearance_m=EASY_PRESET['start_clearance_m'])
                if not tcp_outside_basket(local, margin_m=0.04, above_rim_m=0.0):
                    continue
            target = chassis_p + chassis_R @ local
            arm_q, err = solve_tcp_hover(
                self.model, qpos, self.tcp_site, target, qids, dofs, q_init, ranges)
            if not np.isfinite(arm_q).all() or not np.isfinite(err) or float(err) > accept:
                continue
            start_qs.append(arm_q.astype(np.float32))
            start_errs.append(float(err))
            q_init = arm_q.astype(np.float64)
        if not start_qs:
            kind = 'over-opening' if over_opening else 'outside-crate'
            raise ValueError(f'easy start IK found no physics-safe {kind} pose')
        while len(start_qs) < n:
            start_qs.append(start_qs[0])
            start_errs.append(start_errs[0])
        # Static hold sweep stays at 0.40 m so a closer training start cannot
        # knock the fruit into the front wall and poison close-fraction choice.
        sweep_local = easy_start_local_m(
            0.0, home_local, margin_m=HOLD_SWEEP_MARGIN_M,
            clearance_m=HOLD_SWEEP_CLEARANCE_M, side_y_m=0.0)
        sweep_q, sweep_err = solve_tcp_hover(
            self.model, qpos, self.tcp_site, chassis_p + chassis_R @ sweep_local,
            qids, dofs, q_home, ranges)
        if (not np.isfinite(sweep_q).all() or not np.isfinite(sweep_err)
                or float(sweep_err) > accept):
            sweep_q = start_qs[-1]
        self._hold_sweep_q = np.asarray(sweep_q, dtype=np.float32).reshape(6)
        return drop_q.astype(np.float32), float(drop_err), np.stack(start_qs), np.asarray(start_errs)

    def set_easy_progress(self, far_frac, rng):
        """Sample a catalog start pose and a jaw hold close-fraction.

        ``far_frac`` is kept for logs/clips; starts are drawn from the whole
        IK catalog. Over-opening starts only vary the lower/release, not a
        full carry. Fruit stays free.
        """
        frac = float(far_frac)
        if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
            raise ValueError('far_frac must be finite in [0, 1]')
        n = int(getattr(self, '_easy_catalog_n', self._easy_start_q.shape[0]))
        if n < 1:
            raise ValueError('easy start catalog is empty')
        idx = np.asarray(rng.integers(0, n, size=self.worlds), dtype=np.int32)
        cohort = np.asarray(self._easy_hover_cohort.numpy(), dtype=np.int32).reshape(-1)
        idx = apply_easy_hover_cohort(idx, cohort, self._hover_start_index)
        close = float(self._chosen_close_frac)
        jaw_hold = self._jaw_open + close * (self._jaw_closed - self._jaw_open)
        self._easy_start_index.assign(idx)
        self._easy_jaw_hold_next.assign(np.full(self.worlds, jaw_hold, dtype=np.float32))
        self._easy_far_frac = frac
        return {
            'easy_far_frac': frac,
            'easy_start_index_max': int(idx.max()) if idx.size else int(n - 1),
            'easy_start_index_mean': float(idx.mean()) if idx.size else 0.0,
            'easy_hold_close_mean': close,
            'easy_hold_index_mean': close,
            'easy_hover_cohort_worlds': int(np.count_nonzero(cohort)),
        }

    def snapshot_easy_hover(self):
        """Host copy of start indices and the success hover cohort."""
        return {
            'index': np.asarray(self._easy_start_index.numpy(), dtype=np.int32).reshape(-1).copy(),
            'cohort': np.asarray(self._easy_hover_cohort.numpy(), dtype=np.int32).reshape(-1).copy(),
        }

    def restore_easy_hover(self, saved):
        if not saved:
            return
        self._easy_start_index.assign(np.asarray(saved['index'], dtype=np.int32).reshape(-1))
        self._easy_hover_cohort.assign(np.asarray(saved['cohort'], dtype=np.int32).reshape(-1))

    def clear_easy_hover_starts(self, rng):
        """Eval catalog restore. Does not weld fruit. Training cohort is the caller's to restore."""
        self._easy_hover_cohort.assign(np.zeros(self.worlds, dtype=np.int32))
        if hasattr(self, 'task') and hasattr(self.task, 'success'):
            self.task.success.assign(np.zeros(self.worlds, dtype=np.uint8))
        return self.set_easy_progress(0.0, rng)

    def _prefer_hover_after_success(self, mask):
        """A world that just deposited stays in the hover cohort, including later fails.

        Read ``task.success`` before ``task.reset`` clears it. Fruit stays free.
        """
        success = np.asarray(self.task.success.numpy(), dtype=np.uint8).reshape(-1)
        if success.size != self.worlds or not success.any():
            return
        if mask is None:
            hit = success != 0
        else:
            import torch
            if not isinstance(mask, torch.Tensor):
                return
            hit = (success != 0) & (np.asarray(mask.detach().cpu().numpy(), dtype=np.uint8) != 0)
        if not np.any(hit):
            return
        idx = np.asarray(self._easy_start_index.numpy(), dtype=np.int32).reshape(-1)
        cohort = np.asarray(self._easy_hover_cohort.numpy(), dtype=np.int32).reshape(-1)
        cohort[hit] = 1
        idx[hit] = int(self._hover_start_index)
        self._easy_hover_cohort.assign(cohort)
        self._easy_start_index.assign(idx)

    def enable_easy(self, enabled=True, *, shaping_coef=None, open_xy_m=None):
        """Kiwi starts in the jaws; a jaw script holds or opens; RL moves the arm.

        Does not weld fruit, spawn the arm inside the liner, or write fruit
        into the liner. Starts stay outside the crate. Shaping pulls fruit 3D
        and hand XY toward the open hover (rim + 28 cm), not the liner floor,
        so the wrist is not paid to ram the crate. The script opens when both
        XY sit over the opening AABB and the fruit is at most 16 cm above the
        rim. Far starts stay capped so nearby deposits are not erased. Eval
        still sets guidance_weight=0 and must keep teacher_mix at 0. Jaw close
        fractions are a rigid contact sweep, not a calibrated tissue-safe force.
        """
        self._easy = bool(enabled)
        self._easy_pin.assign(np.array([1 if self._easy else 0], dtype=np.int32))
        shape_both = bool(self._easy and EASY_PRESET.get('shape_hand_and_fruit'))
        release_center = bool(self._easy and EASY_PRESET.get('release_at_center'))
        release_opening = bool(self._easy and EASY_PRESET.get('release_over_opening'))
        self._shape_hand_fruit.assign(np.array([1 if shape_both else 0], dtype=np.int32))
        self._release_at_center.assign(np.array([1 if release_center else 0], dtype=np.int32))
        self._release_over_opening.assign(np.array([1 if release_opening else 0], dtype=np.int32))
        if shaping_coef is None:
            coef = EASY_PRESET['shaping_coef'] if self._easy else EASY_PRESET['default_shaping_coef']
        else:
            coef = float(shaping_coef)
        if not np.isfinite(coef) or coef < 0 or coef > 50:
            raise ValueError('shaping_coef must be finite in [0, 50]')
        if open_xy_m is None:
            self._open_xy_m = float(EASY_PRESET['open_xy_m'])
        else:
            self._open_xy_m = float(open_xy_m)
        if not np.isfinite(self._open_xy_m) or not 0 < self._open_xy_m <= 0.5:
            raise ValueError('open_xy_m must be finite in (0, 0.5] m')
        self._shaping_coef.assign(np.full(self.worlds, coef, dtype=np.float32))
        if self._easy:
            length = float(EASY_PRESET['shaping_length_m'])
        else:
            length = float(EASY_PRESET['default_shaping_length_m'])
        if not np.isfinite(length) or not 0.05 <= length <= 2.0:
            raise ValueError('shaping_length_m must be finite in [0.05, 2.0] m')
        self._shaping_length.assign(np.full(self.worlds, length, dtype=np.float32))
        if self._easy:
            deposit = float(EASY_PRESET['deposit_reward'])
        else:
            deposit = float(W_DEPOSIT)
        if not np.isfinite(deposit) or not 1.0 <= deposit <= 10000.0:
            raise ValueError('deposit_reward must be finite in [1, 10000]')
        self._deposit_w.assign(np.full(self.worlds, deposit, dtype=np.float32))
        if self._easy:
            fail = float(EASY_PRESET['fail_reward'])
        else:
            fail = 0.0
        if not np.isfinite(fail) or not -10000.0 <= fail <= 0.0:
            raise ValueError('fail_reward must be finite in [-10000, 0]')
        self._fail_w.assign(np.full(self.worlds, fail, dtype=np.float32))
        if self._easy and self._hold_sweep is None:
            self._hold_sweep = self._run_hold_sweep()
            self._chosen_close_frac = float(self._hold_sweep['chosen_close_frac'])
            hold = self._jaw_open + self._chosen_close_frac * (self._jaw_closed - self._jaw_open)
            holds = np.full(self.worlds, hold, dtype=np.float32)
            self._easy_jaw_hold.assign(holds)
            self._easy_jaw_hold_next.assign(holds)
        if self._easy:
            self._use_pocket = 1
            from .reach_teacher import grasp_local_fallback_m, grasp_local_near_tcp
            tcp_local = np.asarray(grasp_local_fallback_m(self.model, self.tcp_site), dtype=np.float64)
            local = self._hold_sweep.get('grasp_local_m') if self._hold_sweep is not None else None
            if local is None:
                local = tcp_local
            else:
                local = np.asarray(local, dtype=np.float64).reshape(3)
            if not grasp_local_near_tcp(local, tcp_local):
                local = tcp_local
            self._grasp_local_host = np.asarray(local, dtype=np.float64).reshape(3)
            self._grasp_local = wp.vec3(float(self._grasp_local_host[0]),
                                       float(self._grasp_local_host[1]),
                                       float(self._grasp_local_host[2]))
        else:
            self._use_pocket = 0
            self._grasp_local_host = np.zeros(3, dtype=np.float64)
            self._grasp_local = wp.vec3(0.0, 0.0, 0.0)
        return {
            'easy': self._easy,
            'shaping_coef': coef,
            'shaping_length_m': length,
            'deposit_reward': deposit,
            'fail_reward': fail,
            'open_xy_m': self._open_xy_m,
            'open_rim_z_m': self._open_rim_z_m,
            'hover_error_m': float(self.hover_error_m),
            'easy_start_error_m': float(self.easy_start_error_m),
            'n_start_poses': int(getattr(self, '_easy_catalog_n', self._easy_start_q.shape[0])),
            'hover_start_index': int(getattr(self, '_hover_start_index', -1)),
            'n_hold_levels': int(self._hold_close_fracs.shape[0]),
            'hold_close_frac': float(self._chosen_close_frac),
            'hold_sweep_slip_m': None if self._hold_sweep is None else self._hold_sweep.get('chosen_slip_m'),
            'hold_sweep_load_N': None if self._hold_sweep is None else self._hold_sweep.get('chosen_load_N'),
            'hold_sweep_rows': None if self._hold_sweep is None else [
                {
                    'close_frac': float(row['close_frac']),
                    'slip_m': float(row['slip_m']),
                    'max_load_N': float(row['max_load_N']),
                    'jaw_q': None if row.get('jaw_q') is None else float(row['jaw_q']),
                    'retained': bool(row['retained']),
                }
                for row in (self._hold_sweep.get('rows') or [])
            ],
            'grasp_local_m': None if not self._easy else [float(x) for x in self._grasp_local_host],
            'shape_hand_and_fruit': shape_both,
            'release_at_center': release_center,
            'release_over_opening': release_opening,
            'release_opening_inset_m': float(EASY_PRESET['release_opening_inset_m']),
            'release_max_above_rim_m': float(self._open_max_above_rim_m),
            'far_horizon_updates': int(EASY_PRESET['far_horizon_updates']),
            'far_frac_cap': float(EASY_PRESET['far_frac_cap']),
            'hover_clearance_m': float(EASY_PRESET['hover_clearance_m']),
            'shape_to_hover': shape_both,
            'weld': False,
            'scope': ('kiwi starts in the jaws; scripted hold/open under 15 N; '
                      'student arm deposits; fruit stays free; no weld'),
        }

    def _run_hold_sweep(self):
        """One CPU sweep of jaw close-fractions. Not a tissue calibration."""
        from .fast_task import JAW_FORCE_LIMIT_N
        from .reach_teacher import sweep_jaw_hold
        host = np.asarray(self._initial_qpos.numpy(), dtype=np.float64)
        qpos = host[0].copy() if host.ndim == 2 else host.reshape(-1).copy()
        sweep = getattr(self, '_hold_sweep_q', None)
        if sweep is None:
            start_host = np.asarray(self._easy_start_q.numpy(), dtype=np.float64)
            sweep = start_host[-1] if start_host.ndim == 2 else start_host.reshape(6)
        start_q = np.asarray(sweep, dtype=np.float64).reshape(6).copy()
        try:
            return sweep_jaw_hold(
                self.model, qpos, tcp_site=self.tcp_site,
                fruit_qposadr=int(self._fruit_qposadr), fruit_dofadr=int(self._fruit_dofadr),
                jaw_qposadr=int(self._jaw_qposadr),
                arm_qids=self.control.contract.qids[12:18], start_q=start_q,
                jaw_open=self._jaw_open, jaw_closed=self._jaw_closed,
                load_limit_n=float(JAW_FORCE_LIMIT_N),
                fruit_equality=int(self.task.equality_id),
                jaw_actuator=int(self.control.contract.actuators[18]),
                jaw_kp=float(self.control.contract.kp[18]),
                jaw_kd=float(self.control.contract.kd[18]),
                jaw_cap_nm=min(0.3, float(self.control.contract.limits[18])))
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            mid = float(self._hold_close_fracs[len(self._hold_close_fracs) // 2])
            from .reach_teacher import grasp_local_fallback_m
            try:
                local = np.asarray(grasp_local_fallback_m(self.model, self.tcp_site), dtype=np.float64)
            except (ValueError, RuntimeError, TypeError, AttributeError):
                local = None
            return {
                'rows': [],
                'chosen_close_frac': mid,
                'chosen_slip_m': None,
                'chosen_load_N': None,
                'grasp_local_m': local,
                'weld': False,
                'scope': f'hold sweep fallback; mid close-frac ({type(exc).__name__})',
            }

    def privileged_deposit_action(self):
        """Joint increments toward hover, but hold still when fruit XY is over the basket.

        Returns tanh-space applied actions in [-1, 1]. The fruit remains free;
        this is not an oracle demonstration and must stay off during eval.
        Holding the arm over the opening avoids driving the gripper into the
        crate after a carried fruit arrives.
        """
        import torch
        max_delta = 2.5 * self.control_dt
        with wp.ScopedDevice(self.device):
            wp.launch(_privileged_deposit_action, dim=(self.worlds, 7), inputs=[
                self.data.xpos, self.data.xmat, self.chassis, self.task.fruit_body,
                self.task.active_fruit, self._basket_center, self.task.goal,
                self.task.detached, self.control.targets, self._hover_q,
                self._easy_jaw_hold, self._jaw_open, float(self._open_xy_m),
                float(self._open_rim_z_m), float(max_delta),
                self.data.site_xpos, int(self.tcp_site), self._release_at_center,
                self._release_over_opening, self._open_half_xy,
                float(self._open_max_above_rim_m), self._teacher_applied],
                device=self.device)
        applied = wp.to_torch(self._teacher_applied)
        if tuple(applied.shape) != (self.worlds, 7):
            raise RuntimeError('privileged deposit action has the wrong shape')
        return applied

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
            if self._easy:
                self._prefer_hover_after_success(mask)
            self.task.reset(mask_wp)
            mw.forward(self.gpu_model, self.data)
            self._refresh(mw)
            if self._easy:
                wp.launch(_apply_easy_start, dim=self.worlds, inputs=[
                    mask_wp, self._reset_mode, self.data.qpos, self.control.targets,
                    self.control.qids, self._easy_start_q, self._easy_start_index,
                    self._jaw_qposadr, self._easy_jaw_hold_next, self._easy_jaw_hold],
                    device=self.device)
                mw.forward(self.gpu_model, self.data)
                self._refresh(mw)
            wp.launch(_configure_skill_reset, dim=self.worlds, inputs=[
                mask_wp, self._reset_mode, self.data.qpos, self.data.qvel, self.control.targets,
                self.data.site_xpos, self.data.xpos, self.data.xmat,
                self._fruit_qposadrs, self._fruit_dofadrs, self.tcp_site, self._tcp_body,
                self._grasp_local, int(self._use_pocket),
                self._jaw_qposadr, self._easy_jaw_hold, self._jaw_open, self.task.equality_index,
                self.task.eq_active, self.task.detached, self.task.grasped, self.task.grasp_paid,
                self._chassis_qposadr, 1.0, self._randomize_layout, self._layout_dx, self._layout_dy],
                device=self.device)
            if self._easy:
                wp.launch(_apply_easy_jaw_hold, dim=self.worlds, inputs=[
                    mask_wp, self._reset_mode, self.data.qpos, self.control.targets,
                    self._jaw_qposadr, self._easy_jaw_hold],
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
                              self._basket_distance, self._episode_time, self._timed_out,
                              self._shaping_length, self.data.site_xpos, int(self.tcp_site),
                              self.data.xpos, self.data.xmat, self.chassis, self._basket_center,
                              self._hover_offset, self._shape_hand_fruit,
                              self.task.fruit_body, self.task.active_fruit], device=self.device)
        return self.observe()

    def drain_faults(self):
        """Reset overflow, bad-action, and sparse nonfinite worlds.

        ``check()`` still raises on any latched flag so tests and the final
        report keep the strict contract. Training calls this instead. Abort
        only when nonfinite worlds exceed ``NONFINITE_ABORT_FRACTION``.
        """
        import torch
        flags = wp.to_torch(self._flags)
        nonfinite_worlds = int(((flags & FLAG_NONFINITE) != 0).sum().item())
        overflow_worlds = int(((flags & FLAG_OVERFLOW) != 0).sum().item())
        bad_action_worlds = int(((flags & FLAG_BAD_ACTION) != 0).sum().item())
        if nonfinite_worlds > self.worlds * NONFINITE_ABORT_FRACTION:
            hit = torch.nonzero((flags & FLAG_NONFINITE) != 0, as_tuple=False).flatten()
            raise RuntimeError(
                f'GPU nonfinite worlds={nonfinite_worlds}/{self.worlds} '
                f'indices={hit[:16].tolist()}')
        recoverable = (flags & FLAG_RECOVERABLE) != 0
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
            'nonfinite_worlds': nonfinite_worlds,
            'mask': recoverable,
        }

    def check(self):
        flags = self._flags.numpy()
        if np.any(flags):
            raise RuntimeError(f'GPU numerical failure flags={flags.tolist()}')
        return {'flags': flags.tolist()}
