"""Device-side outcome evaluator for the bounded rigid-fruit task."""

from __future__ import annotations

import numpy as np
import warp as wp

from treesim.basket import CENTER, SIZE, WALL
from treesim.native_kiwi import RADII_M


DETACH_FORCE_N = 8.0  # Engineering approximation; not a calibrated stem threshold.
SETTLE_SPEED_M_S = .05
SETTLE_JITTER_SPEED_M_S = .10
SETTLE_TIME_S = .5
JAW_FORCE_LIMIT_N = 15.0
# The coarse 5 ms rigid solver settles the 36 mm-radius fruit 7–10 mm into
# the simplified liner. Basket contact is still mandatory; this tolerance
# only prevents that numerical penetration from resetting the dwell forever.
CONTAINMENT_TOL_M = .012
MAX_FRUITS = 5
FORCE_CHECKS = {
    'source': 'MJWarp efc.force normal constraint rows',
    'detachment_threshold_N': DETACH_FORCE_N,
    'threshold_status': 'engineering approximation; not calibrated',
}


def _pad_ids(values, fill=-1):
    padded = [fill] * MAX_FRUITS
    for index, value in enumerate(values):
        padded[index] = int(value)
    return padded


@wp.kernel
def _clear_contacts(hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
                     hand_load: wp.array(dtype=float)):
    world = wp.tid()
    hand_hits[world] = 0
    basket_hits[world] = 0
    ground_hits[world] = 0
    hand_load[world] = 0.


@wp.kernel
def _contact_pass(
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
    dim: wp.array(dtype=int), address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int), fruit_count: int, fruit_geom: wp.array(dtype=int),
    active_fruit: wp.array(dtype=int), deposited: wp.array2d(dtype=wp.uint8), kind: wp.array(dtype=int),
    hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
    hand_load: wp.array(dtype=float), rows_per_contact: int,
):
    contact = wp.tid()
    if contact >= nacon[0]:
        return
    pair = geom[contact]
    fruit_index = int(-1)
    for index in range(MAX_FRUITS):
        geom_id = fruit_geom[index]
        matched = int(0)
        if index < fruit_count and geom_id >= 0:
            if pair[0] == geom_id or pair[1] == geom_id:
                matched = int(1)
        if fruit_index < 0 and matched != 0:
            fruit_index = int(index)
    if fruit_index < 0:
        return
    world = worldid[contact]
    if world < 0 or world >= hand_hits.shape[0]:
        return
    fruit = fruit_geom[fruit_index]
    other = pair[1] if pair[0] == fruit else pair[0]
    group = kind[other] if other >= 0 and other < kind.shape[0] else -1
    load = float(0.)
    rows = int(1)
    if rows_per_contact > 1 and dim[contact] > 1:
        rows = int(2 * (dim[contact] - 1))
    if rows > address.shape[1]:
        rows = int(address.shape[1])
    for row_index in range(rows):
        row = address[contact, row_index]
        if row >= 0 and row < nefc[world]:
            load += wp.abs(efc_force[world, row])
    is_active = fruit_index == active_fruit[world]
    if group == 1 and is_active:
        wp.atomic_add(hand_hits, world, 1)
        wp.atomic_add(hand_load, world, load)
    elif group == 2 and is_active:
        wp.atomic_add(basket_hits, world, 1)
    elif group == 3 and deposited[world, fruit_index] == 0:
        wp.atomic_add(ground_hits, world, 1)


@wp.kernel
def _record(
    nefc: wp.array(dtype=int), efc_force: wp.array2d(dtype=float), efc_type: wp.array2d(dtype=int), efc_id: wp.array2d(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3), xipos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), cvel: wp.array2d(dtype=wp.spatial_vector),
    eq_active: wp.array2d(dtype=wp.bool), fruit_count: int, fruit_body: wp.array(dtype=int),
    fruit_roots: wp.array(dtype=int), equality_index: wp.array(dtype=int),
    active_fruit: wp.array(dtype=int), deposited: wp.array2d(dtype=wp.uint8),
    harvested: wp.array(dtype=int), required_harvests: wp.array(dtype=int),
    continue_after_success: wp.array(dtype=wp.uint8),
    hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
    hand_load: wp.array(dtype=float), dt: float, detached: wp.array(dtype=wp.uint8),
    hand_contact: wp.array(dtype=wp.uint8), basket_contact: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8),
    stem_force: wp.array(dtype=float), damage_proxy: wp.array(dtype=float), settle_time: wp.array(dtype=float),
    success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8), chassis: int, fruit_radius: wp.vec3,
    basket_center: wp.vec3, basket_size: wp.vec3, wall: float, chassis_root: int,
    grasped: wp.array(dtype=wp.uint8), grasp_time: wp.array(dtype=float),
    retain_time: wp.array(dtype=float), retained_detach: wp.array(dtype=wp.uint8),
    grasp_paid: wp.array(dtype=wp.uint8), detach_paid: wp.array(dtype=wp.uint8),
    goal: wp.array(dtype=int),
):
    world = wp.tid()
    if success[world] != 0 or failed[world] != 0:
        return
    idx = active_fruit[world]
    if idx < 0 or idx >= fruit_count:
        idx = 0
    fruit_body_id = fruit_body[idx]
    target_eq = equality_index[idx]
    fruit_root = fruit_roots[idx]
    jaws = hand_load[world]
    hand_contact[world] = wp.uint8(hand_hits[world] > 0)
    basket_contact[world] = wp.uint8(basket_hits[world] > 0)
    ground_contact[world] = wp.uint8(ground_contact[world] != 0 or ground_hits[world] > 0)
    # DEPOSIT_ONLY fruit starts detached; rigid jaw/hand overlap while it falls
    # past the gripper is not a grasp overload and must not end the episode.
    if jaws > JAW_FORCE_LIMIT_N and goal[world] != 0:
        damage_proxy[world] = wp.max(damage_proxy[world], (jaws - JAW_FORCE_LIMIT_N) / JAW_FORCE_LIMIT_N)

    stem_squared = float(0.)
    for row in range(efc_type.shape[1]):
        if row >= nefc[world]:
            continue
        if efc_type[world, row] == 0 and efc_id[world, row] == target_eq:
            force = efc_force[world, row]
            stem_squared += force * force
    stem = wp.sqrt(stem_squared)
    stem_force[world] = stem
    if detached[world] == 0 and stem > DETACH_FORCE_N:
        detached[world] = wp.uint8(1)
        if target_eq >= 0 and target_eq < eq_active.shape[1]:
            eq_active[world, target_eq] = False

    # A fully contained ellipsoid, expressed in the chassis frame.
    delta = xpos[world, fruit_body_id] - xpos[world, chassis]
    chassis_rotation = xmat[world, chassis]
    local = wp.transpose(chassis_rotation) @ delta
    relative_rotation = wp.transpose(chassis_rotation) @ xmat[world, fruit_body_id]
    extent = wp.vec3(
        wp.sqrt((relative_rotation[0][0] * fruit_radius[0]) * (relative_rotation[0][0] * fruit_radius[0]) + (relative_rotation[0][1] * fruit_radius[1]) * (relative_rotation[0][1] * fruit_radius[1]) + (relative_rotation[0][2] * fruit_radius[2]) * (relative_rotation[0][2] * fruit_radius[2])),
        wp.sqrt((relative_rotation[1][0] * fruit_radius[0]) * (relative_rotation[1][0] * fruit_radius[0]) + (relative_rotation[1][1] * fruit_radius[1]) * (relative_rotation[1][1] * fruit_radius[1]) + (relative_rotation[1][2] * fruit_radius[2]) * (relative_rotation[1][2] * fruit_radius[2])),
        wp.sqrt((relative_rotation[2][0] * fruit_radius[0]) * (relative_rotation[2][0] * fruit_radius[0]) + (relative_rotation[2][1] * fruit_radius[1]) * (relative_rotation[2][1] * fruit_radius[1]) + (relative_rotation[2][2] * fruit_radius[2]) * (relative_rotation[2][2] * fruit_radius[2])),
    )
    inside = (wp.abs(local[0] - basket_center[0]) + extent[0] <
              basket_size[0] / 2. - wall + CONTAINMENT_TOL_M and
              wp.abs(local[1] - basket_center[1]) + extent[1] <
              basket_size[1] / 2. - wall + CONTAINMENT_TOL_M and
              local[2] - extent[2] >= basket_center[2] + wall / 2. - CONTAINMENT_TOL_M and
              local[2] + extent[2] < basket_center[2] + basket_size[2])
    fruit_angular = wp.vec3(cvel[world, fruit_body_id][0], cvel[world, fruit_body_id][1], cvel[world, fruit_body_id][2])
    fruit_velocity = wp.vec3(cvel[world, fruit_body_id][3], cvel[world, fruit_body_id][4], cvel[world, fruit_body_id][5])
    fruit_velocity += wp.cross(fruit_angular, xipos[world, fruit_body_id] - subtree_com[world, fruit_root])
    chassis_angular = wp.vec3(cvel[world, chassis][0], cvel[world, chassis][1], cvel[world, chassis][2])
    chassis_velocity = wp.vec3(cvel[world, chassis][3], cvel[world, chassis][4], cvel[world, chassis][5])
    chassis_velocity += wp.cross(chassis_angular, xipos[world, fruit_body_id] - subtree_com[world, chassis_root])
    relative_speed = wp.length(fruit_velocity - chassis_velocity)
    contained_contact = (
        detached[world] != 0 and hand_contact[world] == 0
        and inside and basket_contact[world] != 0)
    if contained_contact and relative_speed < SETTLE_SPEED_M_S:
        settle_time[world] += dt
    elif contained_contact and relative_speed < SETTLE_JITTER_SPEED_M_S:
        # Preserve a policy-rate stable dwell across one coarse-solver
        # velocity spike, but never accumulate time above the physical gate.
        settle_time[world] = wp.max(0.0, settle_time[world] - dt)
    else:
        settle_time[world] = 0.
    up = chassis_rotation[2, 2]
    fallen = xpos[world, chassis][2] < .30 or up < .6967067
    jaw_overload = (goal[world] != 0) and (jaws > JAW_FORCE_LIMIT_N)
    if ground_contact[world] != 0 or fallen or jaw_overload or damage_proxy[world] > .05:
        failed[world] = wp.uint8(1)
    elif settle_time[world] >= SETTLE_TIME_S and detached[world] != 0 and hand_contact[world] == 0:
        if deposited[world, idx] == 0:
            deposited[world, idx] = wp.uint8(1)
            harvested[world] = harvested[world] + 1
        if continue_after_success[world] != 0 and harvested[world] < required_harvests[world]:
            next_i = int(-1)
            for candidate in range(MAX_FRUITS):
                free = int(0)
                if candidate < fruit_count and deposited[world, candidate] == 0:
                    free = int(1)
                if next_i < 0 and free != 0:
                    next_i = int(candidate)
            if next_i >= 0:
                active_fruit[world] = next_i
                settle_time[world] = 0.
                grasped[world] = wp.uint8(0)
                grasp_time[world] = 0.
                retain_time[world] = 0.
                retained_detach[world] = wp.uint8(0)
                grasp_paid[world] = wp.uint8(0)
                detach_paid[world] = wp.uint8(0)
                detached[world] = wp.uint8(0)
            else:
                success[world] = wp.uint8(1)
        else:
            success[world] = wp.uint8(1)


@wp.kernel
def _apply_goal(goal: wp.array(dtype=int), dt: float, detached: wp.array(dtype=wp.uint8),
                hand_contact: wp.array(dtype=wp.uint8), grasped: wp.array(dtype=wp.uint8),
                grasp_time: wp.array(dtype=float), retain_time: wp.array(dtype=float),
                retained_detach: wp.array(dtype=wp.uint8), success: wp.array(dtype=wp.uint8),
                failed: wp.array(dtype=wp.uint8)):
    world = wp.tid()
    if success[world] != 0 or failed[world] != 0:
        return
    if detached[world] == 0:
        if hand_contact[world] != 0:
            grasp_time[world] = grasp_time[world] + dt
            if grasp_time[world] >= 0.12:
                grasped[world] = wp.uint8(1)
        else:
            grasp_time[world] = 0.
    if detached[world] != 0 and grasped[world] != 0 and hand_contact[world] != 0:
        retain_time[world] = retain_time[world] + dt
        if retain_time[world] >= 0.2:
            retained_detach[world] = wp.uint8(1)
    # DETACH_ONLY succeeds on retained physical separation; deposit/harvest use settle.
    if goal[world] == 1 and retained_detach[world] != 0:
        success[world] = wp.uint8(1)


@wp.kernel
def _reset(mask: wp.array(dtype=wp.uint8), detached: wp.array(dtype=wp.uint8), hand_contact: wp.array(dtype=wp.uint8),
           basket_contact: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8), stem_force: wp.array(dtype=float),
           hand_load: wp.array(dtype=float), damage_proxy: wp.array(dtype=float), settle_time: wp.array(dtype=float),
           success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8), eq_active: wp.array2d(dtype=wp.bool),
           fruit_count: int, equality_index: wp.array(dtype=int), active_fruit: wp.array(dtype=int),
           deposited: wp.array2d(dtype=wp.uint8), harvested: wp.array(dtype=int),
           deposit_paid: wp.array2d(dtype=wp.uint8), grasped: wp.array(dtype=wp.uint8), grasp_time: wp.array(dtype=float),
           retain_time: wp.array(dtype=float), retained_detach: wp.array(dtype=wp.uint8),
           grasp_paid: wp.array(dtype=wp.uint8), detach_paid: wp.array(dtype=wp.uint8),
           loss_paid: wp.array(dtype=wp.uint8), fail_paid: wp.array(dtype=wp.uint8)):
    world = wp.tid()
    if mask[world] != 0:
        detached[world] = wp.uint8(0)
        hand_contact[world] = wp.uint8(0)
        basket_contact[world] = wp.uint8(0)
        ground_contact[world] = wp.uint8(0)
        stem_force[world] = 0.
        hand_load[world] = 0.
        damage_proxy[world] = 0.
        settle_time[world] = 0.
        success[world] = wp.uint8(0)
        failed[world] = wp.uint8(0)
        grasped[world] = wp.uint8(0)
        grasp_time[world] = 0.
        retain_time[world] = 0.
        retained_detach[world] = wp.uint8(0)
        grasp_paid[world] = wp.uint8(0)
        detach_paid[world] = wp.uint8(0)
        loss_paid[world] = wp.uint8(0)
        fail_paid[world] = wp.uint8(0)
        active_fruit[world] = 0
        harvested[world] = 0
        for index in range(MAX_FRUITS):
            deposited[world, index] = wp.uint8(0)
            deposit_paid[world, index] = wp.uint8(0)
            if index < fruit_count:
                target_eq = equality_index[index]
                if target_eq >= 0 and target_eq < eq_active.shape[1]:
                    eq_active[world, target_eq] = True


class FastHarvestTask:
    """Per-world evaluator; it never supplies actions or actor observations."""

    FORCE_CHECKS = FORCE_CHECKS
    MAX_FRUITS = MAX_FRUITS

    def __init__(self, model, data, manifest):
        fruits = manifest.get('fruits', [])
        if not fruits:
            raise ValueError('Fast scene manifest must contain fruits')
        if len(fruits) > MAX_FRUITS:
            raise ValueError(f'FastHarvestTask supports at most {MAX_FRUITS} independent fruit bodies')
        self.model, self.data, self.manifest = model, data, manifest
        self.worlds = int(data.qpos.shape[0])
        self.device = data.qpos.device
        self.fruit_count = len(fruits)
        self.chassis = int(model.body(manifest['robot']['chassis']).id)
        geom_ids = [int(model.geom(fruit['geom']).id) for fruit in fruits]
        body_ids = [int(model.body(fruit['body']).id) for fruit in fruits]
        eq_ids = [int(model.equality(fruit['equality']).id) for fruit in fruits]
        root_ids = [int(model.body_rootid[body]) for body in body_ids]
        self.fruit_geom = wp.array(_pad_ids(geom_ids), dtype=int, device=self.device)
        self.fruit_body = wp.array(_pad_ids(body_ids), dtype=int, device=self.device)
        self.fruit_roots = wp.array(_pad_ids(root_ids, fill=0), dtype=int, device=self.device)
        self.equality_id = eq_ids[0]
        self.equality_index = wp.array(_pad_ids(eq_ids), dtype=int, device=self.device)
        groups = np.full(model.ngeom, -1, dtype=np.int32)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ''
            body = int(model.geom_bodyid[geom_id])
            if model.body_rootid[body] == model.body_rootid[self.chassis]:
                groups[geom_id] = 1
            if name.startswith(('basket_', 'basket_floor', 'basket_liner')):
                groups[geom_id] = 2
            elif model.geom_type[geom_id] == 0:  # plane
                groups[geom_id] = 3
        self.kind = wp.array(groups, dtype=int, device=self.device)
        self._hand_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._basket_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._ground_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.detached = wp.zeros(self.worlds, dtype=wp.uint8, device=self.device)
        self.hand_contact = wp.zeros_like(self.detached)
        self.basket_contact = wp.zeros_like(self.detached)
        self.ground_contact = wp.zeros_like(self.detached)
        self.stem_force = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.hand_load = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.damage_proxy = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.settle_time = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.success = wp.zeros_like(self.detached)
        self.failed = wp.zeros_like(self.detached)
        self.grasped = wp.zeros_like(self.detached)
        self.grasp_time = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.retain_time = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.retained_detach = wp.zeros_like(self.detached)
        self.grasp_paid = wp.zeros_like(self.detached)
        self.detach_paid = wp.zeros_like(self.detached)
        self.deposit_paid = wp.zeros((self.worlds, MAX_FRUITS), dtype=wp.uint8, device=self.device)
        self.loss_paid = wp.zeros_like(self.detached)
        self.fail_paid = wp.zeros_like(self.detached)
        self.deposited = wp.zeros((self.worlds, MAX_FRUITS), dtype=wp.uint8, device=self.device)
        self.active_fruit = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.harvested = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.required_harvests = wp.ones(self.worlds, dtype=int, device=self.device)
        self.continue_after_success = wp.zeros(self.worlds, dtype=wp.uint8, device=self.device)
        self.goal = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.eq_active = getattr(data, 'eq_active', None)
        if self.eq_active is None or not hasattr(data, 'efc') or not hasattr(data.efc, 'type') or not hasattr(data.efc, 'id'):
            raise ValueError('MJWarp data must expose per-world eq_active and efc.type/id for physical release')
        self._efc_type = data.efc.type
        self._efc_id = data.efc.id
        self.rows_per_contact = 1
        try:
            import mujoco
            self.rows_per_contact = 1 if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL else 4
        except (ImportError, AttributeError):
            pass

    def record(self):
        wp.launch(_clear_contacts, dim=self.worlds,
                  inputs=[self._hand_hits, self._basket_hits, self._ground_hits, self.hand_load], device=self.device)
        wp.launch(_contact_pass, dim=self.data.contact.geom.shape[0], inputs=[
            self.data.nacon, self.data.contact.geom, self.data.contact.worldid, self.data.contact.dim,
            self.data.contact.efc_address, self.data.efc.force, self.data.nefc, self.fruit_count,
            self.fruit_geom, self.active_fruit, self.deposited, self.kind,
            self._hand_hits, self._basket_hits, self._ground_hits, self.hand_load, self.rows_per_contact], device=self.device)
        wp.launch(_record, dim=self.worlds, inputs=[
            self.data.nefc, self.data.efc.force, self._efc_type, self._efc_id, self.data.xpos, self.data.xipos,
            self.data.subtree_com, self.data.xmat, self.data.cvel, self.eq_active,
            self.fruit_count, self.fruit_body, self.fruit_roots, self.equality_index,
            self.active_fruit, self.deposited, self.harvested, self.required_harvests,
            self.continue_after_success,
            self._hand_hits, self._basket_hits, self._ground_hits,
            self.hand_load, float(self.model.opt.timestep), self.detached, self.hand_contact,
            self.basket_contact, self.ground_contact, self.stem_force, self.damage_proxy, self.settle_time,
            self.success, self.failed, self.chassis, wp.vec3(*RADII_M), wp.vec3(*CENTER),
            wp.vec3(*SIZE), float(WALL),
            int(self.model.body_rootid[self.chassis]),
            self.grasped, self.grasp_time, self.retain_time, self.retained_detach,
            self.grasp_paid, self.detach_paid, self.goal], device=self.device)
        wp.launch(_apply_goal, dim=self.worlds, inputs=[
            self.goal, float(self.model.opt.timestep), self.detached, self.hand_contact, self.grasped,
            self.grasp_time, self.retain_time, self.retained_detach, self.success, self.failed],
            device=self.device)
        return self.outputs()

    def reset(self, mask=None):
        if mask is None:
            mask = wp.ones(self.worlds, dtype=wp.uint8, device=self.device)
        else:
            # FastRuntime callers commonly hold episode masks as CUDA Torch tensors.
            if hasattr(mask, 'is_cuda'):
                if not mask.is_cuda or tuple(mask.shape) != (self.worlds,):
                    raise ValueError(f'mask must be a CUDA tensor of shape ({self.worlds},)')
                mask = wp.from_torch(mask.to(dtype=__import__('torch').uint8))
            elif not hasattr(mask, 'dtype'):
                raise ValueError('mask must be a Warp array or CUDA Torch tensor')
        wp.launch(_reset, dim=self.worlds, inputs=[mask, self.detached, self.hand_contact, self.basket_contact,
                   self.ground_contact, self.stem_force, self.hand_load, self.damage_proxy, self.settle_time,
                   self.success, self.failed, self.eq_active, self.fruit_count, self.equality_index,
                   self.active_fruit, self.deposited, self.harvested, self.deposit_paid, self.grasped, self.grasp_time,
                   self.retain_time, self.retained_detach, self.grasp_paid, self.detach_paid,
                   self.loss_paid, self.fail_paid], device=self.device)

    def outputs(self):
        return {'detached': self.detached, 'success': self.success, 'failed': self.failed,
                'damage_proxy': self.damage_proxy, 'grasped': self.grasped,
                'retained_detach': self.retained_detach, 'hand_contact': self.hand_contact,
                'ground_contact': self.ground_contact, 'harvested': self.harvested,
                'required_harvests': self.required_harvests, 'active_fruit': self.active_fruit,
                'hand_load_N': self.hand_load}
