"""Privileged scripted reach teacher for the existing deformable model."""

from __future__ import annotations

import numpy as np


def hover_tcp_local_m(clearance_m=0.12):
    """Chassis-frame TCP *drop* target above the open basket rim.

    Used by the privileged teacher, not as the episode start. Start poses use
    ``easy_start_local_m`` so the arm is spawned outside the crate.
    """
    from treesim.basket import CENTER, SIZE
    clearance = float(clearance_m)
    if not np.isfinite(clearance) or not 0 < clearance <= 0.5:
        raise ValueError('hover clearance must be finite in (0, 0.5] m')
    local = np.asarray(CENTER, dtype=np.float64) + np.array(
        [0.0, 0.0, float(SIZE[2]) + clearance], dtype=np.float64)
    if not np.isfinite(local).all():
        raise ValueError('hover TCP local frame must be finite')
    return local


def hover_tcp_world_m(chassis_xpos, chassis_xmat, clearance_m=0.12):
    """World TCP drop target from chassis pose and the basket-hover local offset."""
    xpos = np.asarray(chassis_xpos, dtype=np.float64).reshape(3)
    xmat = np.asarray(chassis_xmat, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(xpos).all() or not np.isfinite(xmat).all():
        raise ValueError('chassis pose must be finite')
    return xpos + xmat @ hover_tcp_local_m(clearance_m)


def easy_airdrop_world_m(chassis_xpos, chassis_xmat, *, above_rim_m=None):
    """World spawn of a free fruit over the basket opening, above the rim.

    Independent of TCP so a 40 cm outside-crate arm start does not drop the
    fruit on the ground. XY is the chassis-frame basket centre; Z is the rim
    plus ``airdrop_above_rim_m``. Not a weld and not a liner teleport.
    """
    from treesim.basket import CENTER, SIZE
    from treesim.kiwi_rl.curriculum import EASY_PRESET
    xpos = np.asarray(chassis_xpos, dtype=np.float64).reshape(3)
    xmat = np.asarray(chassis_xmat, dtype=np.float64).reshape(3, 3)
    above = float(above_rim_m if above_rim_m is not None else EASY_PRESET.get('airdrop_above_rim_m', 0.06))
    if not np.isfinite(above) or not 0 < above <= 0.3:
        raise ValueError('airdrop_above_rim_m must be finite in (0, 0.3] m')
    if not np.isfinite(xpos).all() or not np.isfinite(xmat).all():
        raise ValueError('airdrop pose inputs must be finite')
    local = np.asarray(CENTER, dtype=np.float64) + np.array(
        [0.0, 0.0, float(SIZE[2]) + above], dtype=np.float64)
    pos = xpos + xmat @ local
    if not np.isfinite(pos).all():
        raise ValueError('airdrop world position must be finite')
    return pos


def basket_chassis_aabb_m():
    """Axis-aligned crate bounds in the chassis frame, floor to open rim.

    ``SIZE`` is the outer box (length, width, wall height), not a box centered
    on ``CENTER``. The floor sits at ``CENTER.z``; the rim is ``CENTER.z + SIZE.z``.
    """
    from treesim.basket import CENTER, SIZE, WALL
    center = np.asarray(CENTER, dtype=np.float64).reshape(3)
    size = np.asarray(SIZE, dtype=np.float64).reshape(3)
    wall = float(WALL)
    if not np.isfinite(center).all() or not np.isfinite(size).all() or not np.isfinite(wall):
        raise ValueError('basket geometry must be finite')
    lo = np.array([center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, center[2] - wall / 2.0],
                  dtype=np.float64)
    hi = np.array([center[0] + size[0] / 2.0, center[1] + size[1] / 2.0, center[2] + size[2]],
                  dtype=np.float64)
    return lo, hi


def tcp_outside_basket(local, *, margin_m=0.08, above_rim_m=0.0):
    """True if a chassis-frame TCP is beside the crate, not over the opening.

    Hovering above the open top does not count: that pose puts the wrist
    through the liner. ``above_rim_m`` only raises the required Z once XY
    is already outside the walls.
    """
    point = np.asarray(local, dtype=np.float64).reshape(3)
    margin = float(margin_m)
    above = float(above_rim_m)
    if not np.isfinite(point).all() or not np.isfinite(margin) or not np.isfinite(above):
        raise ValueError('TCP/basket checks must be finite')
    if margin < 0 or above < 0:
        raise ValueError('margin and above-rim must be >= 0')
    lo, hi = basket_chassis_aabb_m()
    xy_inside = (lo[0] - margin < point[0] < hi[0] + margin
                 and lo[1] - margin < point[1] < hi[1] + margin)
    if xy_inside:
        return False
    return point[2] >= hi[2] + above


def push_tcp_outside_basket(local, *, margin_m=0.12):
    """Push a chassis-frame TCP out of the crate toward the arm mount (+X)."""
    point = np.asarray(local, dtype=np.float64).reshape(3).copy()
    margin = float(margin_m)
    if not np.isfinite(point).all() or not np.isfinite(margin) or margin <= 0:
        raise ValueError('push_tcp_outside_basket inputs must be finite, margin > 0')
    lo, hi = basket_chassis_aabb_m()
    xy_inside = (lo[0] - margin < point[0] < hi[0] + margin
                 and lo[1] - margin < point[1] < hi[1] + margin)
    if xy_inside:
        point[0] = hi[0] + margin
    if point[2] < hi[2]:
        point[2] = hi[2]
    return point


def offset_grasp_local(tcp_local, pocket_local, *, min_m=0.0, max_m=0.05, prefer_m=0.0):
    """Keep a grasp offset in the jaw opening, never past the TCP tip.

    Pad collision centres can sit in front of ``hand_tcp``. Placing fruit there
    drops it. When the pad midpoint is at or past the tip, stay at the TCP or
    take ``prefer_m`` toward the palm. Pure kinematics; fruit stays a free body.
    """
    tcp = np.asarray(tcp_local, dtype=np.float64).reshape(3)
    pocket = np.asarray(pocket_local, dtype=np.float64).reshape(3)
    if not np.isfinite(tcp).all() or not np.isfinite(pocket).all():
        raise ValueError('grasp offset inputs must be finite')
    min_d, max_d, prefer = float(min_m), float(max_m), float(prefer_m)
    if not np.isfinite([min_d, max_d, prefer]).all() or not 0.0 <= min_d <= prefer <= max_d <= 0.12:
        raise ValueError('grasp offset bounds must be ordered in [0, 0.12] m')
    tcp_n = float(np.linalg.norm(tcp))
    if float(np.linalg.norm(pocket)) > tcp_n + 1e-9:
        pocket = tcp.copy()
    delta = pocket - tcp
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        if prefer <= 1e-9:
            return tcp.copy()
        direction = (-tcp / tcp_n) if tcp_n > 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return tcp + direction * prefer
    return tcp + (delta / dist) * min(max(dist, min_d), max_d)


def axial_mouth_local(tcp_local, pocket_local=None, *, inset_m=0.02, max_inset_m=0.05):
    """Fruit COM on the TCP axis, inset into the opening, never past the teeth.

    Lateral pad-geom offsets are discarded so the kiwi is not spawned beside
    the jaws. Inset is capped at 5 cm so this is not a knuckle spawn.
    """
    tcp = np.asarray(tcp_local, dtype=np.float64).reshape(3)
    inset = float(inset_m)
    max_in = float(max_inset_m)
    if tcp.shape != (3,) or not np.isfinite(tcp).all():
        raise ValueError('TCP local frame must be a finite 3-vector')
    n = float(np.linalg.norm(tcp))
    if n < 1e-9:
        raise ValueError('TCP local frame must be nonzero')
    if not np.isfinite([inset, max_in]).all() or not 0.0 <= inset <= max_in <= 0.12:
        raise ValueError('mouth inset must be ordered in [0, 0.12] m')
    axis = tcp / n
    along = n - inset
    if pocket_local is not None:
        pocket = np.asarray(pocket_local, dtype=np.float64).reshape(3)
        if pocket.shape != (3,) or not np.isfinite(pocket).all():
            raise ValueError('pocket local frame must be a finite 3-vector')
        along = float(np.dot(pocket, axis))
        along = min(along, n - inset)
        along = max(along, n - max_in)
    return axis * along


def _joint_actuator_id(model, qposadr):
    """Actuator transmitting to the joint that owns ``qposadr``, or None."""
    qposadr = int(qposadr)
    joint_id = None
    for index in range(int(model.njnt)):
        if int(model.jnt_qposadr[index]) == qposadr:
            joint_id = int(index)
            break
    if joint_id is None:
        return None
    for index in range(int(model.nu)):
        if int(model.actuator_trnid[index, 0]) == joint_id:
            return int(index)
    return None


def _apply_jaw_close_ctrl(model, data, jaw_act, jaw_qposadr, hold, *, cap_nm=0.3):
    """Command the jaw toward ``hold`` without rewriting qpos each step.

    Affine-bias actuators get a position target. Motors get a PD torque clipped
    to ``cap_nm`` (the native CPU jaw cap in ``NativeSpotControl.apply``, not a
    tissue-safe force). If no actuator is found, set qpos once as a fallback.
    """
    import mujoco
    target = float(hold)
    cap = float(cap_nm)
    if not np.isfinite(target) or not np.isfinite(cap) or cap <= 0:
        raise ValueError('jaw hold and torque cap must be finite and positive')
    if jaw_act is None:
        data.qpos[int(jaw_qposadr)] = target
        return 'qpos'
    act = int(jaw_act)
    if int(model.actuator_biastype[act]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
        data.ctrl[act] = target
        return 'position'
    joint_id = None
    for index in range(int(model.njnt)):
        if int(model.jnt_qposadr[index]) == int(jaw_qposadr):
            joint_id = int(index)
            break
    q = float(data.qpos[int(jaw_qposadr)])
    v = 0.0 if joint_id is None else float(data.qvel[int(model.jnt_dofadr[joint_id])])
    gain = float(model.actuator_gainprm[act, 0])
    if not np.isfinite(gain) or abs(gain) < 1e-9:
        gain = 2.0
    data.ctrl[act] = float(np.clip(gain * (target - q) - 0.04 * v, -cap, cap))
    return 'motor'


def _pad_geom_ids(model):
    """Collision geoms on the moving finger vs the fixed jaw. Visuals are skipped."""
    jaw, finger = [], []
    for index in range(int(model.ngeom)):
        if int(model.geom_contype[index]) == 0 and int(model.geom_conaffinity[index]) == 0:
            continue
        name = (model.geom(index).name or '').lower()
        body = (model.body(int(model.geom_bodyid[index])).name or '').lower()
        label = f'{name} {body}'
        if any(token in label for token in ('fngr', 'finger')):
            finger.append(int(index))
        elif 'jaw' in label:
            jaw.append(int(index))
    return tuple(jaw), tuple(finger)


def _closest_geom_ids(data, ids, origin, n):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ranked = sorted(
        ids, key=lambda index: float(np.linalg.norm(
            np.asarray(data.geom_xpos[int(index)], dtype=np.float64).reshape(3) - origin)))
    keep = max(1, min(int(n), len(ranked)))
    return tuple(int(index) for index in ranked[:keep])


def _hand_link_ids(model):
    jaw_body = finger_body = None
    for index in range(int(model.nbody)):
        name = (model.body(index).name or '').lower()
        if finger_body is None and any(token in name for token in ('fngr', 'finger')):
            finger_body = int(index)
        elif jaw_body is None and 'jaw' in name:
            jaw_body = int(index)
    return jaw_body, finger_body


def grasp_local_near_tcp(local, tcp_local, *, max_offset_m=0.08):
    """True if a body-frame pocket sits near the TCP, not at the knuckle origin.

    Pad collision centres are ~15 cm from the jaw-body origin. A 12 cm
    origin-norm gate therefore rejects the real pocket and falls back to the
    TCP tip. Compare against the TCP site local pose instead.
    """
    pocket = np.asarray(local, dtype=np.float64).reshape(3)
    tcp = np.asarray(tcp_local, dtype=np.float64).reshape(3)
    max_off = float(max_offset_m)
    if pocket.shape != (3,) or tcp.shape != (3,):
        raise ValueError('grasp local frames must be length-3')
    if not np.isfinite(max_off) or not 0.0 < max_off <= 0.12:
        raise ValueError('max_offset_m must be finite in (0, 0.12] m')
    if not np.isfinite(pocket).all() or not np.isfinite(tcp).all():
        return False
    return float(np.linalg.norm(pocket - tcp)) <= max_off


def grasp_local_fallback_m(model, tcp_site, *, inset_m=0.0):
    """TCP in the site body frame. Spot's tool centre already sits between the pads."""
    inset = float(inset_m)
    if not np.isfinite(inset) or not 0.0 <= inset <= 0.08:
        raise ValueError('grasp inset must be finite in [0, 0.08] m')
    if not isinstance(tcp_site, (int, np.integer)) or int(tcp_site) < 0:
        raise ValueError('tcp_site must be a non-negative integer')
    tcp_local = np.asarray(model.site_pos[int(tcp_site)], dtype=np.float64).reshape(3)
    if not np.isfinite(tcp_local).all():
        raise ValueError('TCP site local position must be finite')
    if inset <= 1e-9:
        return tcp_local
    return offset_grasp_local(tcp_local, tcp_local, prefer_m=inset, min_m=0.0, max_m=max(inset, 0.05))


def grasp_local_in_body_m(model, data, tcp_site, *, inset_m=0.02):
    """Grasp pocket in the TCP site's body frame, from pad collision geoms.

    Body origins of the jaw/finger links sit at the knuckle, ~12 cm behind the
    pads. Using those midpoints pulled fruit out of the grasp. Pad geom centres
    can sit a few centimetres past ``hand_tcp``; those are clamped back to the
    opening and given a 2 cm mouth inset so the COM is not on the teeth.
    """
    body = int(model.site_bodyid[int(tcp_site)])
    origin = np.asarray(data.xpos[body], dtype=np.float64).reshape(3)
    rot = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    tcp = np.asarray(data.site_xpos[int(tcp_site)], dtype=np.float64).reshape(3)
    if not np.isfinite(origin).all() or not np.isfinite(rot).all() or not np.isfinite(tcp).all():
        raise ValueError('hand pose for the grasp pocket must be finite')
    tcp_local = rot.T @ (tcp - origin)
    jaw_geoms, finger_geoms = _pad_geom_ids(model)
    pocket_local = None
    if jaw_geoms and finger_geoms:
        ids = _closest_geom_ids(data, jaw_geoms, tcp, 2) + _closest_geom_ids(data, finger_geoms, tcp, 2)
        mid = np.mean([np.asarray(data.geom_xpos[index], dtype=np.float64).reshape(3) for index in ids], axis=0)
        if np.isfinite(mid).all():
            pocket_local = rot.T @ (mid - origin)
    return axial_mouth_local(tcp_local, pocket_local, inset_m=float(inset_m))


def grasp_pocket_world_m(model, data, tcp_site, *, inset_m=0.02):
    """World COM for a free fruit sitting between the pads. Not a weld."""
    body = int(model.site_bodyid[int(tcp_site)])
    origin = np.asarray(data.xpos[body], dtype=np.float64).reshape(3)
    rot = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    local = grasp_local_in_body_m(model, data, tcp_site, inset_m=inset_m)
    pocket = origin + rot @ local
    if not np.isfinite(pocket).all():
        raise ValueError('grasp pocket must be finite')
    return pocket


def hold_close_fracs(n_levels=None, close_min=None, close_max=None):
    """Jaw close fractions for the rigid hold sweep. 0=open, 1=fully closed.

    Contact force is a result of this position command, not a measured tissue
    load. The 15 N jaw-limit in the evaluator is an engineering gate.
    """
    from treesim.kiwi_rl.curriculum import EASY_PRESET
    n = int(EASY_PRESET['n_hold_levels'] if n_levels is None else n_levels)
    lo = float(EASY_PRESET['hold_close_min'] if close_min is None else close_min)
    hi = float(EASY_PRESET['hold_close_max'] if close_max is None else close_max)
    if n < 2 or n > 32:
        raise ValueError('n_hold_levels must be an integer in [2, 32]')
    if not np.isfinite(lo) or not np.isfinite(hi) or not 0.0 <= lo <= hi <= 1.0:
        raise ValueError('hold close range must be finite in [0, 1] with min<=max')
    return np.linspace(lo, hi, n, dtype=np.float64)


def jaw_hold_q(close_frac, jaw_open, jaw_closed):
    """Interpolate the jaw target. Fruit stays a free body; this is not a weld."""
    frac = float(close_frac)
    opened = float(jaw_open)
    closed = float(jaw_closed)
    if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError('close_frac must be finite in [0, 1]')
    if not np.isfinite(opened) or not np.isfinite(closed):
        raise ValueError('jaw limits must be finite')
    return float(opened + frac * (closed - opened))


def select_hold_close(rows, *, slip_ok_m=0.04, load_limit_n=15.0):
    """Pick a close-fraction that retains under the 15 N engineering gate.

    If several static holds keep the fruit without crossing the load gate, take
    one step tighter than the lightest keeper so a moving carry is less likely
    to drop it. If nothing retains, take the tightest close still under 15 N
    rather than the lightest slip that still dumps. Do not slam past 15 N.
    ``max_load_N`` is a rigid-sim contact result, not a tissue-safe force.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('hold sweep rows must be a non-empty sequence')
    slip_ok = float(slip_ok_m)
    load_limit = float(load_limit_n)
    if not np.isfinite(slip_ok) or slip_ok <= 0 or not np.isfinite(load_limit) or load_limit <= 0:
        raise ValueError('slip_ok_m and load_limit_n must be finite and positive')
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError('each hold sweep row must be a dict')
        try:
            close = float(row['close_frac'])
            slip = float(row['slip_m'])
            load = float(row['max_load_N'])
            retained = bool(row['retained'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('hold sweep row must have close_frac, slip_m, max_load_N, retained') from exc
        if not np.isfinite([close, slip, load]).all() or not 0.0 <= close <= 1.0 or slip < 0 or load < 0:
            raise ValueError('hold sweep row values must be finite and physically ranged')
        cleaned.append({'close_frac': close, 'slip_m': slip, 'max_load_N': load, 'retained': retained})
    viable = [row for row in cleaned if row['retained'] and row['slip_m'] <= slip_ok
              and row['max_load_N'] <= load_limit]
    if viable:
        ordered = sorted(viable, key=lambda row: row['close_frac'])
        idx = 1 if len(ordered) > 1 else 0
        return ordered[idx]
    under = [row for row in cleaned if row['max_load_N'] <= load_limit]
    if under:
        return max(under, key=lambda row: (row['close_frac'], -row['slip_m']))
    return min(cleaned, key=lambda row: (row['slip_m'], row['max_load_N'], row['close_frac']))


def sweep_jaw_hold(model, qpos, *, tcp_site, fruit_qposadr, fruit_dofadr, jaw_qposadr,
                   arm_qids, start_q, jaw_open, jaw_closed, close_fracs=None,
                   hold_s=0.4, slip_ok_m=0.04, load_limit_n=15.0, fruit_equality=None):
    """CPU hold sweep: close-fraction vs slip and hand contact load.

    Fruit stays a free body. The pocket is the open-mouth axial COM, then each
    close-fraction shuts the jaw around that same point. The chosen command is
    not a weld and not a calibrated kiwi-safe force. Uses a 5 ms CPU timestep
    so a 0.4 s hold does not expand into tens of thousands of native substeps.
    """
    import mujoco
    from treesim.kiwi_rl.fast_task import JAW_FORCE_LIMIT_N
    q0 = np.asarray(qpos, dtype=np.float64).reshape(-1)
    start = np.asarray(start_q, dtype=np.float64).reshape(6)
    arm_qids = np.asarray(arm_qids, dtype=int).reshape(6)
    fracs = np.asarray(hold_close_fracs() if close_fracs is None else close_fracs, dtype=np.float64)
    hold_time = float(hold_s)
    slip_ok = float(slip_ok_m)
    load_limit = float(load_limit_n if load_limit_n is not None else JAW_FORCE_LIMIT_N)
    if not np.isfinite(hold_time) or not 0.05 <= hold_time <= 2.0:
        raise ValueError('hold_s must be finite in [0.05, 2] s')
    if not np.isfinite(q0).all() or not np.isfinite(start).all() or not np.isfinite(fracs).all():
        raise ValueError('sweep inputs must be finite')
    if q0.shape[0] != int(model.nq):
        raise ValueError('sweep qpos must be one native configuration, not a batched world stack')
    saved_dt = float(model.opt.timestep)
    model.opt.timestep = 0.005
    local = None
    try:
        steps = max(1, min(80, int(round(hold_time / float(model.opt.timestep)))))
        hand_geoms = _hand_geom_ids(model)
        jaw_act = _joint_actuator_id(model, jaw_qposadr)
        fruit_geoms = _fruit_geom_ids(model, fruit_qposadr)
        data = mujoco.MjData(model)
        eq = None if fruit_equality is None else int(fruit_equality)
        data.qpos[:] = q0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        if eq is not None and 0 <= eq < int(data.eq_active.shape[0]):
            data.eq_active[eq] = 0
        data.qpos[arm_qids] = start
        data.qpos[int(jaw_qposadr)] = jaw_open
        if jaw_act is not None:
            data.ctrl[jaw_act] = jaw_open
        mujoco.mj_forward(model, data)
        local = grasp_local_in_body_m(model, data, tcp_site)
        body = int(model.site_bodyid[int(tcp_site)])
        origin = np.asarray(data.xpos[body], dtype=np.float64).reshape(3)
        rot = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
        pocket0 = origin + rot @ local
        rows = []
        for frac in fracs:
            hold = jaw_hold_q(frac, jaw_open, jaw_closed)
            data.qpos[:] = q0
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            if eq is not None and 0 <= eq < int(data.eq_active.shape[0]):
                data.eq_active[eq] = 0
            data.qpos[arm_qids] = start
            data.qpos[int(jaw_qposadr)] = jaw_open
            if jaw_act is not None:
                data.ctrl[jaw_act] = jaw_open
            data.qpos[int(fruit_qposadr):int(fruit_qposadr) + 3] = pocket0
            data.qpos[int(fruit_qposadr) + 3:int(fruit_qposadr) + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[int(fruit_dofadr):int(fruit_dofadr) + 6] = 0.0
            mujoco.mj_forward(model, data)
            max_load = 0.0
            for _ in range(steps):
                data.qpos[arm_qids] = start
                _apply_jaw_close_ctrl(model, data, jaw_act, jaw_qposadr, hold)
                if eq is not None and 0 <= eq < int(data.eq_active.shape[0]):
                    data.eq_active[eq] = 0
                mujoco.mj_step(model, data)
                max_load = max(max_load, _hand_fruit_contact_load_n(
                    model, data, hand_geoms, fruit_geoms))
            carry_steps = max(1, min(40, steps // 2))
            carry_q = start.copy()
            carry_q[0] = start[0] + 0.10
            for step in range(carry_steps):
                alpha = float(step + 1) / float(carry_steps)
                data.qpos[arm_qids] = (1.0 - alpha) * start + alpha * carry_q
                _apply_jaw_close_ctrl(model, data, jaw_act, jaw_qposadr, hold)  # hold during carry screen
                if eq is not None and 0 <= eq < int(data.eq_active.shape[0]):
                    data.eq_active[eq] = 0
                mujoco.mj_step(model, data)
                max_load = max(max_load, _hand_fruit_contact_load_n(
                    model, data, hand_geoms, fruit_geoms))
            mujoco.mj_forward(model, data)
            fruit = np.asarray(data.qpos[int(fruit_qposadr):int(fruit_qposadr) + 3], dtype=np.float64)
            slip = float(np.linalg.norm(fruit - pocket0))
            if not np.isfinite(slip) or not np.isfinite(max_load):
                slip, max_load = float('inf'), float('inf')
            rows.append({
                'close_frac': float(frac),
                'slip_m': slip if np.isfinite(slip) else 1.0,
                'max_load_N': max_load if np.isfinite(max_load) else 1.0e6,
                'retained': bool(np.isfinite(slip) and slip <= slip_ok),
            })
        chosen = select_hold_close(rows, slip_ok_m=slip_ok, load_limit_n=load_limit)
    finally:
        model.opt.timestep = saved_dt
    if local is None:
        local = grasp_local_fallback_m(model, tcp_site)
    return {
        'rows': rows,
        'chosen_close_frac': float(chosen['close_frac']),
        'chosen_slip_m': float(chosen['slip_m']),
        'chosen_load_N': float(chosen['max_load_N']),
        'grasp_local_m': np.asarray(local, dtype=np.float64).reshape(3),
        'load_limit_N': load_limit,
        'slip_ok_m': slip_ok,
        'weld': False,
        'scope': 'rigid jaw-close sweep in the pad pocket; not a tissue-safe force',
    }


def _fruit_geom_ids(model, fruit_qposadr):
    """Collision geoms on the free fruit body that owns ``fruit_qposadr``."""
    qposadr = int(fruit_qposadr)
    joint_id = None
    for index in range(int(model.njnt)):
        if int(model.jnt_qposadr[index]) == qposadr:
            joint_id = int(index)
            break
    if joint_id is None:
        return tuple()
    body = int(model.jnt_bodyid[joint_id])
    ids = []
    for index in range(int(model.ngeom)):
        if int(model.geom_bodyid[index]) != body:
            continue
        if int(model.geom_contype[index]) == 0 and int(model.geom_conaffinity[index]) == 0:
            continue
        ids.append(int(index))
    return tuple(ids)


def _hand_geom_ids(model):
    tokens = ('jaw', 'fngr', 'finger', 'hand', 'pad', 'grip')
    ids = []
    for index in range(int(model.ngeom)):
        name = (model.geom(index).name or '').lower()
        if any(token in name for token in tokens):
            ids.append(int(index))
    return tuple(ids)


def _hand_fruit_contact_load_n(model, data, hand_geoms, fruit_geoms):
    """Normal load from hand/fruit contacts only. Jaw-vs-finger is ignored."""
    import mujoco
    if not hand_geoms or not fruit_geoms:
        return 0.0
    hand = set(int(g) for g in hand_geoms)
    fruit = set(int(g) for g in fruit_geoms)
    load = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not (pair & hand) or not (pair & fruit):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if np.isfinite(force[0]):
            load += abs(float(force[0]))
    return load


def _hand_contact_load_n(model, data, hand_geoms):
    """All hand contacts, including jaw-finger. Prefer ``_hand_fruit_contact_load_n``."""
    import mujoco
    if not hand_geoms:
        return 0.0
    hand = set(int(g) for g in hand_geoms)
    load = 0.0
    force = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) not in hand and int(contact.geom2) not in hand:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        if np.isfinite(force[0]):
            load += abs(float(force[0]))
    return load


def random_easy_start_local_m(rng, home_local=None, *, margin_m=None, clearance_m=None):
    """Random chassis-frame TCP outside the crate in a bounded IK box.

    Rejects poses over the opening. Does not sample inside the liner. ``rng``
    must be a NumPy Generator.
    """
    from treesim.kiwi_rl.curriculum import EASY_PRESET
    if rng is None or not hasattr(rng, 'uniform'):
        raise TypeError('rng must be a NumPy Generator')
    margin = float(EASY_PRESET['start_margin_m'] if margin_m is None else margin_m)
    clearance = float(EASY_PRESET['start_clearance_m'] if clearance_m is None else clearance_m)
    x_span = float(EASY_PRESET['start_x_span_m'])
    y_span = float(EASY_PRESET['start_y_span_m'])
    z_span = float(EASY_PRESET['start_z_span_m'])
    if not np.isfinite([margin, clearance, x_span, y_span, z_span]).all():
        raise ValueError('random start spans must be finite')
    if not 0.05 <= margin <= 0.5 or not 0.05 <= clearance <= 0.5:
        raise ValueError('start margin/clearance must be in [0.05, 0.5] m')
    if not 0.05 <= x_span <= 0.6 or not 0.0 <= y_span <= 0.2 or not 0.0 <= z_span <= 0.2:
        raise ValueError('random start spans are outside the physics-safe box')
    lo, hi = basket_chassis_aabb_m()
    x = float(rng.uniform(hi[0] + margin, hi[0] + margin + x_span))
    y = float(rng.uniform(-y_span, y_span))
    z = float(rng.uniform(hi[2] + clearance, hi[2] + clearance + z_span))
    local = push_tcp_outside_basket(np.array([x, y, z], dtype=np.float64), margin_m=margin)
    local[2] = max(float(local[2]), float(hi[2] + clearance))
    if home_local is not None:
        home = np.asarray(home_local, dtype=np.float64).reshape(3)
        if not np.isfinite(home).all():
            raise ValueError('home_local must be finite')
    if not tcp_outside_basket(local, margin_m=0.04, above_rim_m=0.0):
        raise ValueError('random easy start TCP still intersects the crate volume')
    return local


def easy_start_local_m(frac=0.0, home_local=None, *, margin_m=0.40, clearance_m=0.28):
    """Chassis-frame TCP start: frac 0 = clear of the crate, 1 = toward home.

    The near pose is on the robot side of the front wall with enough margin that
    the wrist is not spawned through the liner. A 12 cm margin still clips.
    Fruit is not placed here; the caller puts a free body in the pad pocket.
    """
    from treesim.basket import CENTER, SIZE
    frac = float(frac)
    margin = float(margin_m)
    clearance = float(clearance_m)
    if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError('start frac must be finite in [0, 1]')
    if not np.isfinite(margin) or not 0.05 <= margin <= 0.5:
        raise ValueError('start margin must be finite in [0.05, 0.5] m')
    if not np.isfinite(clearance) or not 0.05 <= clearance <= 0.5:
        raise ValueError('start clearance must be finite in [0.05, 0.5] m')
    lo, hi = basket_chassis_aabb_m()
    near = np.array([
        hi[0] + margin,
        float(CENTER[1]),
        float(CENTER[2] + SIZE[2] + clearance),
    ], dtype=np.float64)
    if home_local is None:
        far = near + np.array([0.28, 0.0, 0.04], dtype=np.float64)
    else:
        far = np.asarray(home_local, dtype=np.float64).reshape(3)
        if not np.isfinite(far).all():
            raise ValueError('home_local must be finite')
        far = push_tcp_outside_basket(far, margin_m=margin)
    local = (1.0 - frac) * near + frac * far
    local = push_tcp_outside_basket(local, margin_m=margin)
    local[2] = max(float(local[2]), float(hi[2] + clearance))
    if not tcp_outside_basket(local, margin_m=0.04, above_rim_m=0.0):
        raise ValueError('easy start TCP still intersects the crate volume')
    return local


def solve_tcp_hover(model, qpos, site_id, target_world, joint_qposadr, joint_dofadr,
                    q_init, ranges, *, damping=.05, max_step=.1, steps=80, tol_m=0.02):
    """CPU DLS that moves one site toward a world point. Fruit stays a free body."""
    import mujoco
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    target_world = np.asarray(target_world, dtype=np.float64).reshape(3)
    joint_qposadr = np.asarray(joint_qposadr, dtype=int).reshape(-1)
    joint_dofadr = np.asarray(joint_dofadr, dtype=int).reshape(-1)
    q_init = np.asarray(q_init, dtype=np.float64).reshape(-1)
    ranges = np.asarray(ranges, dtype=np.float64)
    n_arm = joint_qposadr.shape[0]
    if n_arm != 6 or joint_dofadr.shape[0] != 6 or q_init.shape != (6,) or ranges.shape != (6, 2):
        raise ValueError('hover IK uses the six arm joints, not the jaw')
    if not isinstance(site_id, (int, np.integer)) or int(site_id) < 0:
        raise ValueError('site_id must be a non-negative integer')
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 200:
        raise ValueError('steps must be an integer in [1, 200]')
    if not np.isfinite(target_world).all() or not np.isfinite(qpos).all() or not np.isfinite(q_init).all():
        raise ValueError('hover IK inputs must be finite')
    if not np.isfinite(tol_m) or tol_m <= 0:
        raise ValueError('tol_m must be finite and positive')
    data = mujoco.MjData(model)
    q = qpos.copy()
    commands = q_init.copy()
    q[joint_qposadr] = commands
    error_norm = np.inf
    for _ in range(steps):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        hand = np.asarray(data.site_xpos[int(site_id)], dtype=np.float64)
        error = target_world - hand
        error_norm = float(np.linalg.norm(error))
        if error_norm <= float(tol_m):
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, int(site_id))
        delta = bounded_damped_least_squares(
            jacp[:, joint_dofadr], error, commands,
            ranges[:, 0], ranges[:, 1], damping, max_step)
        commands = np.clip(commands + delta.astype(np.float64), ranges[:, 0], ranges[:, 1])
        q[joint_qposadr] = commands
    return commands.astype(np.float32), float(error_norm)


def damped_least_squares(J, error, damping=.05, max_step=.1):
    """Return a bounded joint increment for a Cartesian position error."""
    J, error = np.asarray(J, dtype=np.float64), np.asarray(error, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != 3 or error.shape != (3,):
        raise ValueError('J must be [3,n] and error must be [3]')
    if not np.isfinite(J).all() or not np.isfinite(error).all() or not np.isfinite(damping) or not 0 < damping or not np.isfinite(max_step) or not 0 < max_step <= .1:
        raise ValueError('DLS inputs must be finite and damping must be positive')
    step = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(3), error)
    if not np.isfinite(step).all():
        raise ValueError('DLS produced a nonfinite step')
    return np.clip(step, -max_step, max_step).astype(np.float32)


def bounded_damped_least_squares(J, error, current, lower, upper,
                                 damping=.05, max_step=.1):
    """DLS increment with one active-limit pass and joint bounds.

    A joint at a bound is removed when its unconstrained increment points
    farther outside that bound. The solve is then recomputed once using the
    remaining columns, so a saturated elbow does not consume the Cartesian
    correction intended for the wrist.
    """
    J = np.asarray(J, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != 3 or current.shape != (J.shape[1],):
        raise ValueError('invalid bounded DLS shapes')
    if lower.shape != current.shape or upper.shape != current.shape:
        raise ValueError('joint bounds must match DLS columns')
    if not np.isfinite(np.r_[J.ravel(), error, current, lower, upper]).all() or np.any(lower > upper):
        raise ValueError('bounded DLS inputs must be finite and ordered')
    if np.any(current < lower - 1e-6) or np.any(current > upper + 1e-6):
        raise ValueError('current joint target is outside limits')
    raw = damped_least_squares(J, error, damping, max_step).astype(np.float64)
    at_lower = current <= lower + 1e-6
    at_upper = current >= upper - 1e-6
    blocked = (at_lower & (raw < 0)) | (at_upper & (raw > 0))
    free = ~blocked
    if np.any(blocked) and np.any(free):
        raw[free] = damped_least_squares(J[:, free], error, damping, max_step).astype(np.float64)
        raw[blocked] = 0.
    raw = np.clip(raw, lower - current, upper - current)
    return np.clip(raw, -max_step, max_step).astype(np.float32)


class PrivilegedReachTeacher:
    """Jacobian teacher using the runtime's native MuJoCo model only."""

    def __init__(self, runtime, fruit_index=0, damping=.05, max_step=.1):
        import mujoco
        self.runtime = runtime
        self.model = runtime.model
        self.data = mujoco.MjData(self.model)
        self.damping, self.max_step = float(damping), float(max_step)
        if not np.isfinite(self.damping) or not 0 < self.damping or not np.isfinite(self.max_step) or not 0 < self.max_step <= .1:
            raise ValueError('invalid teacher limits')
        attachments = runtime.manifest.get('attachments', ())
        if not 0 <= fruit_index < len(attachments):
            raise ValueError('fruit_index is outside scene attachments')
        self.flex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, attachments[fruit_index]['flex'])
        if self.flex_id < 0:
            raise ValueError('Missing fruit flex')
        self.site_id = self.model.site('hand_tcp').id
        contract = runtime.control.contract
        self.joints, self.dofs = np.asarray(contract.joints)[12:19], np.asarray(contract.dofs)[12:19]
        self.ranges = np.tile(np.array([-np.pi, np.pi], dtype=np.float32), (7, 1))
        limited = np.asarray(self.model.jnt_limited[self.joints], dtype=bool)
        self.ranges[limited] = self.model.jnt_range[self.joints][limited]

    def step(self):
        import mujoco
        qpos = np.asarray(self.runtime.data.qpos.numpy())
        commands = self.runtime.control.targets.numpy()[:, 12:19].copy()
        vertices = self.runtime.data.flexvert_xpos.numpy()
        worlds = qpos.shape[0]
        targets, distances = np.empty((worlds, 7), np.float32), np.empty(worlds, np.float32)
        start = int(self.model.flex_vertadr[self.flex_id])
        count = int(self.model.flex_vertnum[self.flex_id])
        for world in range(worlds):
            self.data.qpos[:] = qpos[world]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            fruit = vertices[world, start:start + count].mean(axis=0)
            hand = np.asarray(self.data.site_xpos[self.site_id])
            error = fruit - hand
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            delta = bounded_damped_least_squares(
                jacp[:, self.dofs[:6]], error, commands[world, :6],
                self.ranges[:6, 0], self.ranges[:6, 1], self.damping, self.max_step)
            target = commands[world].copy()
            target[:6] = commands[world, :6] + delta
            target = np.clip(target, commands[world] - self.max_step, commands[world] + self.max_step)
            target = np.clip(target, self.ranges[:, 0], self.ranges[:, 1])
            targets[world], distances[world] = target, np.linalg.norm(error)
        return {'targets': targets, 'target_distance_m': distances,
                'privileged': True, 'scope': 'scripted privileged reach teacher; not trained'}
