import numpy as np
import warp as wp


NONFINITE_STATE = 65536
INVALID_ELEMENT = 131072


def contact_flex_ids(geom, flex):
    geom, flex = np.asarray(geom), np.asarray(flex)
    if geom.shape != (2,) or flex.shape != (2,):
        raise ValueError('Contact sides must have two entries')
    return np.where(geom >= 0, -1, flex)


@wp.kernel
def _latch_solver(overflow: wp.array(dtype=int), iterations: wp.array(dtype=int),
                  flags: wp.array(dtype=int), peak_iterations: wp.array(dtype=int)):
    world = wp.tid()
    wp.atomic_or(flags, world, overflow[world])
    wp.atomic_max(peak_iterations, world, iterations[world])


@wp.kernel
def _check_state(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
                 nq: int, nv: int, flags: wp.array(dtype=int)):
    world, index = wp.tid()
    if index < nq:
        if not wp.isfinite(q[world, index]):
            wp.atomic_or(flags, world, 65536)
    if index < nv:
        if not wp.isfinite(v[world, index]):
            wp.atomic_or(flags, world, 65536)


@wp.kernel
def _check_elements(vertices: wp.array2d(dtype=wp.vec3), elements: wp.array(dtype=wp.vec4i),
                    inverse_rest_volume: wp.array(dtype=float), flags: wp.array(dtype=int),
                    minimum_ratio: wp.array(dtype=float)):
    world, element = wp.tid()
    ids = elements[element]
    a = vertices[world, ids[0]]
    b = vertices[world, ids[1]] - a
    c = vertices[world, ids[2]] - a
    d = vertices[world, ids[3]] - a
    determinant = wp.determinant(wp.mat33(b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]))
    ratio = determinant / 6.0 * inverse_rest_volume[element]
    wp.atomic_min(minimum_ratio, world, ratio)
    if not wp.isfinite(ratio) or ratio <= 0.1:
        wp.atomic_or(flags, world, 131072)


def tetrahedra(model):
    indices = []
    for f in range(model.nflex):
        if model.flex_dim[f] != 3:
            raise ValueError('Expected volumetric fruit flexes')
        start, count = model.flex_elemdataadr[f], model.flex_elemnum[f]
        indices.extend(model.flex_elem[start:start + 4 * count].reshape(-1, 4) + model.flex_vertadr[f])
    return np.asarray(indices, dtype=np.int32).reshape(-1, 4)


class DeformableMonitor:
    def __init__(self, model, reference_data, gpu_data):
        self.data = gpu_data
        indices = tetrahedra(model)
        tetra = reference_data.flexvert_xpos[indices]
        volumes = np.linalg.det(tetra[:, 1:] - tetra[:, :1]) / 6
        if not len(volumes) or not np.isfinite(volumes).all() or np.any(volumes == 0):
            raise ValueError('Invalid rest tetrahedra')
        self.worlds, self.nq = gpu_data.qpos.shape
        self.nv = gpu_data.qvel.shape[1]
        device = gpu_data.qpos.device
        self.elements = wp.array(indices, dtype=wp.vec4i, device=device)
        self.inverse_volumes = wp.array(1 / volumes, dtype=float, device=device)
        self.flags = wp.zeros(self.worlds, dtype=int, device=device)
        self.peak_iterations = wp.zeros(self.worlds, dtype=int, device=device)
        self.minimum_ratio = wp.ones(self.worlds, dtype=float, device=device)

    def reset(self):
        self.flags.zero_()
        self.peak_iterations.zero_()
        self.minimum_ratio.fill_(1.)

    def record(self):
        device = self.data.qpos.device
        wp.launch(_latch_solver, dim=self.worlds,
                  inputs=[self.data.overflow, self.data.solver_niter, self.flags, self.peak_iterations], device=device)
        wp.launch(_check_state, dim=(self.worlds, max(self.nq, self.nv)),
                  inputs=[self.data.qpos, self.data.qvel, self.nq, self.nv, self.flags], device=device)
        wp.launch(_check_elements, dim=(self.worlds, len(self.elements)),
                  inputs=[self.data.flexvert_xpos, self.elements, self.inverse_volumes, self.flags, self.minimum_ratio],
                  device=device)

    def report(self):
        import mujoco_warp as mjw
        flags = self.flags.numpy()
        names = []
        for value in flags:
            items = [item.name for item in mjw.OverflowType if item.value and item.name != 'ALL' and int(value) & item.value]
            if int(value) & NONFINITE_STATE:
                items.append('NONFINITE_STATE')
            if int(value) & INVALID_ELEMENT:
                items.append('INVALID_ELEMENT')
            names.append(items)
        return dict(flags=flags.tolist(), failures=names,
                    minimum_volume_ratio=self.minimum_ratio.numpy().tolist(),
                    peak_solver_iterations=self.peak_iterations.numpy().tolist())

    def check(self):
        result = self.report()
        if any(result['flags']):
            raise RuntimeError(f'GPU numerical failure: {result}')
        return result


@wp.kernel
def _flex_contact_loads(nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                        flex: wp.array(dtype=wp.vec2i), world_id: wp.array(dtype=int),
                        dimensions: wp.array(dtype=int), address: wp.array2d(dtype=int),
                        efc_force: wp.array2d(dtype=float), nefc: wp.array(dtype=int),
                        kind: wp.array(dtype=int), pyramidal: int, steps: wp.array(dtype=wp.int64),
                        loads: wp.array3d(dtype=float), first_ground: wp.array2d(dtype=wp.int64),
                        flags: wp.array(dtype=int)):
    contact = wp.tid()
    if contact >= nacon[0]:
        return
    gs, fs = geom[contact], flex[contact]
    world = world_id[contact]
    if world < 0 or world >= loads.shape[0]:
        wp.atomic_or(flags, 0, 262144)
        return
    if gs[0] >= 0 and gs[1] >= 0:
        return
    rows = 1
    if pyramidal != 0 and dimensions[contact] > 1:
        rows = 2 * (dimensions[contact] - 1)
    if rows > address.shape[1]:
        wp.atomic_or(flags, world, 262144)
        return
    normal = float(0.)
    for i in range(rows):
        row = address[contact, i]
        if row >= 0:
            if row >= nefc[world]:
                wp.atomic_or(flags, world, 262144)
                return
            normal += efc_force[world, row]
    if not wp.isfinite(normal):
        wp.atomic_or(flags, world, 65536)
        return
    for side in range(2):
        fruit = fs[side]
        if gs[side] < 0 and fruit >= 0:
            other = gs[1 - side]
            if fruit >= loads.shape[1] or other >= kind.shape[0]:
                wp.atomic_or(flags, world, 262144)
                return
            if other >= 0:
                group = kind[other]
                if group >= 0 and group < 3:
                    wp.atomic_add(loads, world, fruit, group, wp.abs(normal))
                if group == 3 and normal > .01:
                    wp.atomic_min(first_ground, world, fruit, steps[world] + wp.int64(1))


@wp.kernel
def _peak_contact_loads(loads: wp.array3d(dtype=float), peaks: wp.array3d(dtype=float)):
    world, fruit, side = wp.tid()
    wp.atomic_max(peaks, world, fruit, side, loads[world, fruit, side])


@wp.kernel
def _advance_contact_clock(steps: wp.array(dtype=wp.int64)):
    world = wp.tid()
    steps[world] += wp.int64(1)


def contact_geom_groups(model):
    import mujoco
    groups = np.full(model.ngeom, -1, dtype=np.int32)
    for g in range(model.ngeom):
        name = model.geom(g).name or ''
        for index, token in enumerate(('arm_link_jaw_collision', 'arm_link_fngr_collision', 'arm_link_wr1_collision')):
            if token in name:
                groups[g] = index
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            groups[g] = 3
    return groups


class FlexContactObserver:
    def __init__(self, model, data):
        import mujoco
        self.data = data
        self.worlds = data.qpos.shape[0]
        self.fruits = model.nflex
        self.pyramidal = int(model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL)
        groups = contact_geom_groups(model)
        device = data.qpos.device
        self.kind = wp.array(groups, dtype=int, device=device)
        self.loads = wp.zeros((self.worlds, self.fruits, 3), device=device)
        self.peaks = wp.zeros_like(self.loads)
        self.first_ground = wp.full((self.worlds, self.fruits), 9223372036854775807, dtype=wp.int64, device=device)
        self.steps = wp.zeros(self.worlds, dtype=wp.int64, device=device)
        self.flags = wp.zeros(self.worlds, dtype=int, device=device)

    def reset(self):
        self.loads.zero_()
        self.peaks.zero_()
        self.first_ground.fill_(9223372036854775807)
        self.steps.zero_()
        self.flags.zero_()

    def record(self):
        d, device = self.data, self.data.qpos.device
        self.loads.zero_()
        wp.launch(_flex_contact_loads, dim=d.naconmax,
            inputs=[d.nacon, d.contact.geom, d.contact.flex, d.contact.worldid, d.contact.dim,
                    d.contact.efc_address, d.efc.force, d.nefc, self.kind, self.pyramidal, self.steps,
                    self.loads, self.first_ground, self.flags], device=device)
        wp.launch(_peak_contact_loads, dim=(self.worlds, self.fruits, 3),
                  inputs=[self.loads, self.peaks], device=device)
        wp.launch(_advance_contact_clock, dim=self.worlds, inputs=[self.steps], device=device)

    def report(self):
        ground = self.first_ground.numpy()
        return dict(jaw_palm_load_N=self.loads.numpy().tolist(), peak_jaw_palm_load_N=self.peaks.numpy().tolist(),
                    first_ground_step=np.where(ground == np.iinfo(np.int64).max, -1, ground).tolist(),
                    steps=self.steps.numpy().tolist(), flags=self.flags.numpy().tolist())

    def check(self):
        result = self.report()
        if any(result['flags']):
            raise RuntimeError(f'Invalid flex contact forces: {result}')
        return result


class NativeFlexContactObserver:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.kind = contact_geom_groups(model)
        self.loads = np.zeros((1, model.nflex, 3))
        self.peaks = np.zeros_like(self.loads)
        self.first_ground = np.full((1, model.nflex), -1, dtype=np.int64)
        self.steps = 0
        self.force = np.zeros(6)

    def record(self):
        import mujoco
        self.steps += 1
        self.loads.fill(0.)
        for index, contact in enumerate(self.data.contact):
            gs, fs = contact.geom, contact.flex
            if gs[0] >= 0 and gs[1] >= 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, self.force)
            normal = float(self.force[0])
            if not np.isfinite(normal):
                raise RuntimeError('Nonfinite native contact force')
            for side in (0, 1):
                fruit, other = int(fs[side]), int(gs[1 - side])
                if gs[side] < 0 and fruit >= 0 and other >= 0:
                    group = self.kind[other]
                    if 0 <= group < 3:
                        self.loads[0, fruit, group] += abs(normal)
                    if group == 3 and normal > .01 and self.first_ground[0, fruit] < 0:
                        self.first_ground[0, fruit] = self.steps
        np.maximum(self.peaks, self.loads, out=self.peaks)

    def check(self):
        return dict(jaw_palm_load_N=self.loads.tolist(), peak_jaw_palm_load_N=self.peaks.tolist(),
                    first_ground_step=self.first_ground.tolist(), steps=[self.steps], flags=[0])
