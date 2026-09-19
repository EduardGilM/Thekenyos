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
    above = float(EASY_PRESET['airdrop_above_rim_m'] if above_rim_m is None else above_rim_m)
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


def easy_start_local_m(frac=0.0, home_local=None, *, margin_m=0.40, clearance_m=0.28):
    """Chassis-frame TCP start: frac 0 = clear of the crate, 1 = toward home.

    The near pose is on the robot side of the front wall with enough margin that
    the wrist is not spawned through the liner. A 12 cm margin still clips.
    Fruit is not placed here; the caller puts a free body at the solved TCP
    with the jaw closed.
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
