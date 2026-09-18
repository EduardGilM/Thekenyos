"""RELIC Spot policy adapter for Newton/MuJoCo-Warp.

Assets, constants and weights stay in an external, pinned RELIC checkout:
https://github.com/rai-opensource/relic (noncommercial research license).
The floating base moves only through dynamics and foot contact.
"""
from pathlib import Path
import runpy

import numpy as np
import newton
import warp as wp

from .builder import _qrot, _qconj

LEGS = [f"{leg}_{axis}" for axis in ("hx", "hy", "kn")
        for leg in ("fl", "fr", "hl", "hr")]
ARM = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]
# Confirmed against RAI judo/tasks/spot/spot_constants.py: arm first at each depth.
# Judo's deployed ONNX is byte-identical to RELIC's export.
OBS_JOINTS = ARM[:1] + LEGS[:4] + ARM[1:2] + LEGS[4:8] + ARM[2:3] + LEGS[8:] + ARM[3:]


def build_robot(builder, params):
    asset = Path(params.relic_path).resolve() / "source/relic/relic/assets/spot"
    constants = runpy.run_path(str(asset / "constants.py"))
    nb, nj, ns = builder.body_count, builder.joint_count, builder.shape_count
    builder.add_urdf(str(asset / "spot_with_arm.urdf"), floating=True,
                     xform=wp.transform(wp.vec3(*params.position, float(params.base_z)),
                                        wp.quat_from_axis_angle(wp.vec3(0, 0, 1), params.yaw)),
                     enable_self_collisions=True, joint_ordering="dfs")
    # PhysX trains with convex collision hulls, not triangle-mesh self contacts.
    builder.approximate_meshes(method="convex_hull", raise_on_failure=True,
                              shape_indices=[s for s in range(ns, builder.shape_count)
                                             if builder.shape_type[s] == newton.GeoType.MESH
                                             and builder.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES])
    # Some RELIC lower-leg collision meshes are planar. Give only those hulls
    # 1 mm thickness so native MuJoCo can compile them; preserve body inertia.
    for s in range(ns, builder.shape_count):
        if builder.shape_type[s] not in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH) or not builder.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES:
            continue
        mesh = builder.shape_source[s]
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        _, singular, axes = np.linalg.svd(vertices-vertices.mean(axis=0), full_matrices=False)
        if singular[-1] < 1e-6*max(singular[0], 1e-9):
            from scipy.spatial import ConvexHull
            points = np.concatenate([vertices+.0005*axes[-1], vertices-.0005*axes[-1]])
            hull = ConvexHull(points)
            builder.shape_source[s] = newton.Mesh(points.astype(np.float32), hull.simplices.astype(np.int32).flatten())
    for s in range(ns, builder.shape_count):
        name = builder.body_label[builder.shape_body[s]].rsplit("/", 1)[-1]
        yellow = name == "body" or "uleg" in name or name in ("arm_link_sh0", "arm_link_sh1")
        builder.shape_color[s] = (0.96, 0.73, 0.04) if yellow else (0.22, 0.24, 0.27)
    joints = {}
    for j in range(nj, builder.joint_count):
        name = builder.joint_label[j].rsplit("/", 1)[-1]
        if name not in constants["SPOT_DEFAULT_JOINT_POS"]:
            continue
        joints[name] = j
        qs, ds = builder.joint_q_start[j], builder.joint_qd_start[j]
        builder.joint_q[qs] = constants["SPOT_DEFAULT_JOINT_POS"][name]
        builder.joint_target_ke[ds] = builder.joint_target_kd[ds] = 0.0
        builder.joint_damping[ds] = builder.joint_friction[ds] = 0.0
        builder.joint_armature[ds] = constants["ARM_ARMATURE"][ARM.index(name)] if name in ARM else 0.0
    if set(joints) != set(OBS_JOINTS):
        raise ValueError(f"Unexpected Spot joints: {joints}")
    chassis = next(i for i in range(nb, builder.body_count)
                   if builder.body_label[i].rsplit("/", 1)[-1] == "body")
    basket = None
    if params.basket:
        from .basket import add_basket
        spawn = wp.transform(wp.vec3(*params.position, float(params.base_z)),
                             wp.quat_from_axis_angle(wp.vec3(0, 0, 1), params.yaw))
        basket = add_basket(builder, chassis, params.basket_mass, params.payload_mass, params.payload_seed, spawn)
    return dict(kind="spot", nbody=builder.body_count-nb, body_start=nb,
                chassis=chassis, basket=basket,
                joints=joints, asset=str(asset), constants=constants)


def finalize_maps(model, maps):
    maps = maps.copy()
    maps["q"] = {n: int(model.joint_q_start.numpy()[j]) for n, j in maps["joints"].items()}
    maps["dof"] = {n: int(model.joint_qd_start.numpy()[j]) for n, j in maps["joints"].items()}
    return maps


@wp.kernel
def _pd(q: wp.array(dtype=float), qd: wp.array(dtype=float),
        qids: wp.array(dtype=int), dofs: wp.array(dtype=int),
        target: wp.array(dtype=float), kp: wp.array(dtype=float), kd: wp.array(dtype=float),
        limits: wp.array(dtype=float), knee_table: wp.array2d(dtype=float),
        force: wp.array(dtype=float)):
    i = wp.tid()
    angle, speed = q[qids[i]], qd[dofs[i]]
    effort = kp[i] * (target[i] - angle) - kd[i] * speed
    limit = limits[i]
    if i >= 8 and i < 12:
        # RELIC's remotized knee angle-dependent output-torque limit.
        limit = knee_table[0, 2]
        for k in range(1, knee_table.shape[0]):
            if angle >= knee_table[k, 0]:
                limit = knee_table[k, 2]
            elif angle >= knee_table[k - 1, 0]:
                t = (angle - knee_table[k - 1, 0]) / (knee_table[k, 0] - knee_table[k - 1, 0])
                limit = (1.0-t)*knee_table[k-1, 2] + t*knee_table[k, 2]
        effort = wp.clamp(effort, -limit, limit)
        lower = -96.9972 * wp.clamp(1.0 + speed / 15.0, 0.0, 1.0)
        upper = 96.9972 * wp.clamp(1.0 - speed / 14.0, 0.0, 1.0)
        effort = wp.clamp(effort, lower, upper)
    else:
        effort = wp.clamp(effort, -limit, limit)
    force[dofs[i]] = effort


class SpotController:
    def __init__(self, sim):
        import onnxruntime as ort
        self.sim = sim
        self.data = sim.tree.robot_data
        c = self.data["constants"]
        self.home = c["SPOT_DEFAULT_JOINT_POS"]
        self.names = LEGS + ARM
        self.qids = np.array([self.data["q"][n] for n in self.names])
        self.dofs = np.array([self.data["dof"][n] for n in self.names])
        self.obs_q = [self.data["q"][n] for n in OBS_JOINTS]
        self.obs_dof = [self.data["dof"][n] for n in OBS_JOINTS]
        self.obs_home = np.array([self.home[n] for n in OBS_JOINTS])
        self.targets = np.array([self.home[n] for n in self.names], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(Path(self.data["asset"]) / "pretrained/policy.onnx"),
                                          sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.policy.get_inputs()[0].name
        self.device = sim.model.device
        self.target = wp.array(self.targets, device=self.device)
        self.buffers = [wp.array(self.qids, dtype=int, device=self.device),
                        wp.array(self.dofs, dtype=int, device=self.device), self.target,
                        wp.array([60.0]*12 + list(c["ARM_STIFFNESS"]), dtype=float, device=self.device),
                        wp.array([1.5]*12 + list(c["ARM_DAMPING"]), dtype=float, device=self.device),
                        wp.array([45.0]*8+[113.24]*4+list(c["ARM_EFFORT_LIMIT"]), dtype=float, device=self.device),
                        wp.array(np.asarray(c["JOINT_PARAMETER_LOOKUP_TABLE"], dtype=np.float32), device=self.device)]
        sim.robot_controller = self

    def update(self, command):
        state = self.sim.state_0
        pose = state.body_q.numpy()[self.data["chassis"]]
        velocity = state.body_qd.numpy()[self.data["chassis"]]
        inv = _qconj(pose[3:])
        # Newton linear velocity is measured at COM; Isaac's root_lin_vel_b is too.
        obs = np.concatenate([
            _qrot(inv, velocity[:3]), _qrot(inv, velocity[3:]),
            _qrot(inv, np.array([0., 0., -1.])), command,
            self.targets[12:], np.zeros(12), [0., 0., 0.55],
            state.joint_q.numpy()[self.obs_q] - self.obs_home,
            state.joint_qd.numpy()[self.obs_dof], self.last_action,
        ]).astype(np.float32)
        if obs.shape != (84,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid Spot observation")
        action = self.policy.run(None, {self.input_name: obs[None]})[0][0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid Spot policy output")
        self.last_action = action
        self.targets[:12] = np.array([self.home[n] for n in LEGS]) + 0.2 * action
        self.target.assign(self.targets)

    def apply(self, state, control):
        wp.launch(_pd, dim=19, inputs=[state.joint_q, state.joint_qd, *self.buffers, control.joint_f],
                  device=self.device)
