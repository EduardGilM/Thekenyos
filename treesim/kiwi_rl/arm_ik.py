"""Spot arm IK side-model (newton.ik DLS). JP-ONLY (needs newton + URDF).

Mirrors treesim/picker.py::ArmIK exactly, but for the Spot arm instead
of the Franka: a private base-fixed side model built from the external
RELIC URDF (same file treesim/spot.py loads, never vendored), solved
with Levenberg-Marquardt + analytic Jacobians, warm-started, max 32
iterations, halve-and-retry once on failure.

Arm chain (from spot_with_arm.urdf @ pinned commit):
sh0(z) @ (0.292,0,0.188) -> sh1(y) @ +0.3385x -> el0(y) @ +0.4033x+0.075z
-> el1(x) -> wr0(y) -> wr1(x) -> f1x(y, jaw). 7 revolute joints; the
f1x jaw angle is part of the solution (gripper Pinch mapping TBD).

Conventions: positions in chassis frame, xyzw quaternions, hand +Z as
the approach axis (same as picker.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1",
              "arm_wr0", "arm_wr1", "arm_f1x"]
ARM_MOUNT = (0.292, 0.0, 0.188)
MAX_ITERS = 32
POS_TOL_M = 0.005
ROT_TOL_DEG = 5.0


def _require_newton():
    try:
        import newton
        import warp as wp
        import newton.ik as ik
    except ImportError as e:
        raise ImportError("arm IK stack required (newton)") from e
    return newton, wp, ik


class SpotArmIK:
    """DLS IK on a private Spot-arm side model (chassis frame)."""

    def __init__(self, relic_path: str, arm_home: dict):
        newton, wp, ik = _require_newton()
        self._wp = wp
        urdf = (Path(relic_path).resolve() / "source/relic/relic/assets/spot"
                / "spot_with_arm.urdf")
        if not urdf.is_file():
            raise FileNotFoundError(f"missing {urdf} (external checkout)")
        b = newton.ModelBuilder()
        # Isolate the arm: rebuild only the arm chain on a fixed base at the
        # chassis mount. Body labels keep their URDF names for TCP lookup.
        b.add_urdf(str(urdf),
                   xform=wp.transform(p=wp.vec3(*ARM_MOUNT),
                                      q=wp.quat_identity()),
                   floating=False, enable_self_collisions=False)
        names = [l.rsplit("/", 1)[-1] for l in b.joint_label]
        self._arm_q = []
        for n in ARM_JOINTS:
            if n not in arm_home:
                raise ValueError(f"home missing joint {n}")
            j = names.index(n)
            b.joint_q[b.joint_q_start[j]] = float(arm_home[n])
            self._arm_q.append(int(b.joint_q_start[j]))
        self.model = b.finalize()
        labels = [l.rsplit("/", 1)[-1] for l in self.model.body_label]
        for cand in ("arm_jaw", "arm_link_f1x", "arm_f1x"):
            hits = [i for i, l in enumerate(labels) if l == cand]
            if hits:
                self.tcp = hits[0]
                break
        else:
            raise ValueError(f"no TCP link in {labels}")
        self.q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3()], dtype=wp.vec3))
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.tcp, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)],
                                      dtype=wp.vec4),
            weight=0.15)
        lim = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper, weight=10.0)
        self.solver = ik.IKSolver(model=self.model, n_problems=1,
                                  objectives=[self.pos_obj, self.rot_obj, lim],
                                  lambda_initial=0.1,
                                  jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self._state = self.model.state()

    def solve(self, target_base: np.ndarray, approach_base: np.ndarray,
              q_init: np.ndarray, iters: int = MAX_ITERS):
        """IK for TCP position + approach axis. Returns (q_arm(7), err_m).

        On convergence failure halves the request once (same policy as the
        spec §4.3); persistent failure returns the current solution with
        its measured error so the caller can gate RECOVER.
        """
        newton, wp = _require_newton()[0], self._wp
        qh = self.q.numpy()
        qh[0, self._arm_q] = np.asarray(q_init, dtype=float)
        self.q.assign(qh)
        # Seed TCP via FK so halve-and-retry has a cartesian anchor.
        self.model.joint_q.assign(qh[0])
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self._state)
        tcp0 = self._state.body_q.numpy()[self.tcp, :3].copy()
        want = np.asarray(target_base, dtype=float)
        for attempt in range(2):
            if attempt == 0:
                self.q.assign(qh)  # seed pose; attempt 1 warm-starts instead
            tgt = want if attempt == 0 else 0.5 * (tcp0 + want)
            z = np.asarray(approach_base, dtype=float)
            z = z / (np.linalg.norm(z) + 1e-9)
            up = np.array([0.0, 0.0, 1.0])
            y = np.cross(z, up)
            if np.linalg.norm(y) < 1e-6:
                y = np.array([0.0, 1.0, 0.0])
            y /= np.linalg.norm(y)
            x = np.cross(y, z)
            quat = wp.quat_from_matrix(wp.mat33f(x[0], y[0], z[0],
                                                 x[1], y[1], z[1],
                                                 x[2], y[2], z[2]))
            self.pos_obj.set_target_position(0, wp.vec3(*map(float, tgt)))
            self.rot_obj.set_target_rotation(
                0, wp.vec4(quat[0], quat[1], quat[2], quat[3]))
            self.solver.step(self.q, self.q, iterations=iters)
            qs = self.q.numpy()[0]
            self.model.joint_q.assign(qs)
            newton.eval_fk(self.model, self.model.joint_q,
                           self.model.joint_qd, self._state)
            tcp = self._state.body_q.numpy()[self.tcp, :3]
            err = float(np.linalg.norm(tcp - tgt))
            if err <= POS_TOL_M or attempt == 1:
                return qs[self._arm_q].copy(), err
        raise RuntimeError("unreachable")  # pragma: no cover
