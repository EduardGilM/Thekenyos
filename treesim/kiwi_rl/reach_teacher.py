"""Privileged scripted reach teacher for the existing deformable model."""

from __future__ import annotations

import numpy as np


def hover_tcp_local_m(clearance_m=0.12):
    """Chassis-frame TCP target above the open basket rim.

    This is a privileged reset/teacher pose, not a liner teleport. Fruit still
    has to fall under gravity and settle. ``clearance_m`` is extra +Z above
    ``CENTER + (0, 0, SIZE_z)``, the same top used by the deposit oracle.
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
    """World TCP target from chassis pose and the basket-hover local offset."""
    xpos = np.asarray(chassis_xpos, dtype=np.float64).reshape(3)
    xmat = np.asarray(chassis_xmat, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(xpos).all() or not np.isfinite(xmat).all():
        raise ValueError('chassis pose must be finite')
    return xpos + xmat @ hover_tcp_local_m(clearance_m)


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
