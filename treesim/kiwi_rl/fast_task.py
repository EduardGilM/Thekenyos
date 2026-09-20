"""Device-side outcome evaluator for the bounded rigid-fruit task."""

from __future__ import annotations

import numpy as np
import warp as wp

from treesim.basket import CENTER, SIZE, WALL
from treesim.native_kiwi import RADII_M


DETACH_FORCE_N = 8.0  # Engineering approximation; not a calibrated stem threshold.
SETTLE_SPEED_M_S = .05
SETTLE_TIME_S = .5
JAW_FORCE_LIMIT_N = 15.0
GRASP_DWELL_S = .1
# Contact groups.  Pad groups are assigned from the geom's owning body below;
# they must not be inferred from a geom name or a distance threshold.
HAND_GROUP = 1
BASKET_GROUP = 2
GROUND_GROUP = 3
FNGR_GROUP = 4
JAW_GROUP = 5
FORCE_CHECKS = {
    'source': 'MJWarp efc.force normal constraint rows',
    'detachment_threshold_N': DETACH_FORCE_N,
    'threshold_status': 'engineering approximation; not calibrated',
}


@wp.kernel
def _clear_contacts(hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
                     finger_hits: wp.array(dtype=int), jaw_hits: wp.array(dtype=int), hand_load: wp.array(dtype=float),
                     finger_load: wp.array(dtype=float), jaw_load: wp.array(dtype=float), palm_load: wp.array(dtype=float)):
    world = wp.tid()
    hand_hits[world] = 0
    basket_hits[world] = 0
    ground_hits[world] = 0
    finger_hits[world] = 0
    jaw_hits[world] = 0
    hand_load[world] = 0.
    finger_load[world] = 0.
    jaw_load[world] = 0.
    palm_load[world] = 0.


@wp.kernel
def _contact_pass(
    nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
    dim: wp.array(dtype=int), address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float),
    nefc: wp.array(dtype=int), fruit_geom: wp.array(dtype=int), kind: wp.array(dtype=int),
    hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
    finger_hits: wp.array(dtype=int), jaw_hits: wp.array(dtype=int), hand_load: wp.array(dtype=float),
    finger_load: wp.array(dtype=float), jaw_load: wp.array(dtype=float), palm_load: wp.array(dtype=float),
    rows_per_contact: int,
):
    contact = wp.tid()
    if contact >= nacon[0]:
        return
    pair = geom[contact]
    fruit = fruit_geom[0]
    if pair[0] != fruit and pair[1] != fruit:
        return
    world = worldid[contact]
    if world < 0 or world >= hand_hits.shape[0]:
        return
    other = pair[1] if pair[0] == fruit else pair[0]
    group = kind[other] if other >= 0 and other < kind.shape[0] else -1
    load = float(0.)
    rows = 1
    if rows_per_contact > 1 and dim[contact] > 1:
        rows = 2 * (dim[contact] - 1)
    if rows > address.shape[1]:
        rows = address.shape[1]
    for row_index in range(rows):
        row = address[contact, row_index]
        if row >= 0 and row < nefc[world]:
            load += wp.abs(efc_force[world, row])
    if group == HAND_GROUP:
        wp.atomic_add(hand_hits, world, 1)
        wp.atomic_add(hand_load, world, load)
        wp.atomic_add(palm_load, world, load)
    elif group == FNGR_GROUP or group == JAW_GROUP:
        wp.atomic_add(hand_hits, world, 1)
        wp.atomic_add(hand_load, world, load)
        if group == FNGR_GROUP:
            wp.atomic_add(finger_load, world, load)
            if load > 0.2:
                wp.atomic_add(finger_hits, world, 1)
        elif group == JAW_GROUP:
            wp.atomic_add(jaw_load, world, load)
            if load > 0.2:
                wp.atomic_add(jaw_hits, world, 1)
    elif group == 2:
        wp.atomic_add(basket_hits, world, 1)
    elif group == 3:
        wp.atomic_add(ground_hits, world, 1)


@wp.kernel
def _record(
    nefc: wp.array(dtype=int), efc_force: wp.array2d(dtype=float), efc_type: wp.array2d(dtype=int), efc_id: wp.array2d(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3), xipos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), cvel: wp.array2d(dtype=wp.spatial_vector),
    eq_active: wp.array2d(dtype=wp.bool), fruit_body: wp.array(dtype=int), equality_index: wp.array(dtype=int),
    hand_hits: wp.array(dtype=int), basket_hits: wp.array(dtype=int), ground_hits: wp.array(dtype=int),
    finger_load: wp.array(dtype=float), jaw_load: wp.array(dtype=float), palm_load: wp.array(dtype=float),
    dt: float, detached: wp.array(dtype=wp.uint8),
    hand_contact: wp.array(dtype=wp.uint8), basket_contact: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8),
    bilateral_contact: wp.array(dtype=wp.uint8), stable_grasp: wp.array(dtype=wp.uint8), ever_grasped: wp.array(dtype=wp.uint8),
    grasp_time: wp.array(dtype=float), finger_hits: wp.array(dtype=int), jaw_hits: wp.array(dtype=int),
    stem_force: wp.array(dtype=float), damage_proxy: wp.array(dtype=float), settle_time: wp.array(dtype=float),
    success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8), chassis: int, fruit_radius: wp.vec3,
    basket_center: wp.vec3, basket_size: wp.vec3, wall: float, fruit_root: int, chassis_root: int,
    settle_seconds: float, ground_is_failure: int, held_at_detach: wp.array(dtype=wp.uint8),
    detach_requires_hold: int,
):
    world = wp.tid()
    if success[world] != 0 or failed[world] != 0:
        return
    fruit_body_id = fruit_body[0]
    target_eq = equality_index[0]
    jaws = wp.max(finger_load[world], jaw_load[world])
    nonpad_load = palm_load[world]
    maximum_load = wp.max(jaws, nonpad_load)
    hand_contact[world] = wp.uint8(hand_hits[world] > 0)
    bilateral_contact[world] = wp.uint8(finger_hits[world] > 0 and jaw_hits[world] > 0)
    if bilateral_contact[world] != 0:
        grasp_time[world] += dt
    else:
        grasp_time[world] = 0.
    stable_grasp[world] = wp.uint8(grasp_time[world] >= GRASP_DWELL_S)
    if stable_grasp[world] != 0:
        ever_grasped[world] = wp.uint8(1)
    basket_contact[world] = wp.uint8(basket_hits[world] > 0)
    ground_contact[world] = wp.uint8(ground_contact[world] != 0 or ground_hits[world] > 0)
    if maximum_load > JAW_FORCE_LIMIT_N:
        damage_proxy[world] = wp.max(damage_proxy[world], (maximum_load - JAW_FORCE_LIMIT_N) / JAW_FORCE_LIMIT_N)

    stem_squared = float(0.)
    target_eq = equality_index[0]
    for row in range(efc_type.shape[1]):
        if row >= nefc[world]:
            continue
        if efc_type[world, row] == 0 and efc_id[world, row] == target_eq:
            force = efc_force[world, row]
            stem_squared += force * force
    stem = wp.sqrt(stem_squared)
    stem_force[world] = stem
    if detached[world] == 0 and stem > DETACH_FORCE_N and (detach_requires_hold == 0 or stable_grasp[world] != 0):
        detached[world] = wp.uint8(1)
        held_at_detach[world] = wp.uint8(stable_grasp[world] != 0 and
            maximum_load <= JAW_FORCE_LIMIT_N and damage_proxy[world] <= .05)
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
    inside = (wp.abs(local[0] - basket_center[0]) + extent[0] < basket_size[0] / 2. - wall and
              wp.abs(local[1] - basket_center[1]) + extent[1] < basket_size[1] / 2. - wall and
              local[2] - extent[2] >= basket_center[2] + wall / 2. - .002 and
              local[2] + extent[2] < basket_center[2] + basket_size[2])
    fruit_angular = wp.vec3(cvel[world, fruit_body_id][0], cvel[world, fruit_body_id][1], cvel[world, fruit_body_id][2])
    fruit_velocity = wp.vec3(cvel[world, fruit_body_id][3], cvel[world, fruit_body_id][4], cvel[world, fruit_body_id][5])
    fruit_velocity += wp.cross(fruit_angular, xipos[world, fruit_body_id] - subtree_com[world, fruit_root])
    chassis_angular = wp.vec3(cvel[world, chassis][0], cvel[world, chassis][1], cvel[world, chassis][2])
    chassis_velocity = wp.vec3(cvel[world, chassis][3], cvel[world, chassis][4], cvel[world, chassis][5])
    chassis_velocity += wp.cross(chassis_angular, xipos[world, fruit_body_id] - subtree_com[world, chassis_root])
    relative_speed = wp.length(fruit_velocity - chassis_velocity)
    if detached[world] != 0 and hand_contact[world] == 0 and inside and basket_contact[world] != 0 and relative_speed < SETTLE_SPEED_M_S:
        settle_time[world] += dt
    else:
        settle_time[world] = 0.
    up = chassis_rotation[2, 2]
    fallen = xpos[world, chassis][2] < .30 or up < .6967067
    if (ground_contact[world] != 0 and ground_is_failure != 0) or fallen or maximum_load > JAW_FORCE_LIMIT_N or damage_proxy[world] > .05:
        failed[world] = wp.uint8(1)
    elif settle_time[world] >= settle_seconds and detached[world] != 0 and hand_contact[world] == 0:
        success[world] = wp.uint8(1)


@wp.kernel
def _reset(mask: wp.array(dtype=wp.uint8), detached: wp.array(dtype=wp.uint8), hand_contact: wp.array(dtype=wp.uint8),
           basket_contact: wp.array(dtype=wp.uint8), ground_contact: wp.array(dtype=wp.uint8),
           bilateral_contact: wp.array(dtype=wp.uint8), stable_grasp: wp.array(dtype=wp.uint8), ever_grasped: wp.array(dtype=wp.uint8),
           grasp_time: wp.array(dtype=float), hand_load: wp.array(dtype=float), finger_load: wp.array(dtype=float),
           jaw_load: wp.array(dtype=float), palm_load: wp.array(dtype=float),
           damage_proxy: wp.array(dtype=float), settle_time: wp.array(dtype=float),
           stem_force: wp.array(dtype=float),
           success: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8), eq_active: wp.array2d(dtype=wp.bool),
           target_eq: int, held_at_detach: wp.array(dtype=wp.uint8)):
    world = wp.tid()
    if mask[world] != 0:
        detached[world] = wp.uint8(0)
        hand_contact[world] = wp.uint8(0)
        basket_contact[world] = wp.uint8(0)
        ground_contact[world] = wp.uint8(0)
        bilateral_contact[world] = wp.uint8(0)
        stable_grasp[world] = wp.uint8(0)
        ever_grasped[world] = wp.uint8(0)
        held_at_detach[world] = wp.uint8(0)
        grasp_time[world] = 0.
        stem_force[world] = 0.
        hand_load[world] = 0.
        finger_load[world] = 0.
        jaw_load[world] = 0.
        palm_load[world] = 0.
        damage_proxy[world] = 0.
        settle_time[world] = 0.
        success[world] = wp.uint8(0)
        failed[world] = wp.uint8(0)
        if target_eq >= 0 and target_eq < eq_active.shape[1]:
            eq_active[world, target_eq] = True


class FastHarvestTask:
    """Per-world evaluator; it never supplies actions or actor observations."""

    FORCE_CHECKS = FORCE_CHECKS

    def __init__(self, model, data, manifest, *, task_profile=None, detach_requires_hold=False):
        from .reward_graph import GRAPH_PROFILES
        # Deployment/demo option: the stem only yields to a secure grasp, so a knocked fruit
        # swings on its stem instead of being torn off and flung.
        self.detach_requires_hold = bool(detach_requires_hold)
        if task_profile not in (None, *GRAPH_PROFILES):
            raise ValueError('Unknown physical task profile')
        self.settle_seconds = 2. if task_profile in GRAPH_PROFILES else SETTLE_TIME_S
        self.ground_is_failure = task_profile not in GRAPH_PROFILES
        fruits = manifest.get('fruits', [])
        if not fruits:
            raise ValueError('Fast scene manifest must contain fruits')
        self.model, self.data, self.manifest = model, data, manifest
        self.worlds = int(data.qpos.shape[0])
        self.device = data.qpos.device
        # One TARGET fruit at a time; the target can be switched with retarget().
        self.fruits = fruits
        self.chassis = int(model.body(manifest['robot']['chassis']).id)
        fruit = fruits[0]
        self.fruit_geom = wp.array([int(model.geom(fruit['geom']).id)], dtype=int, device=self.device)
        self.fruit_body = wp.array([int(model.body(fruit['body']).id)], dtype=int, device=self.device)
        self.equality_id = int(model.equality(fruit['equality']).id)
        self.equality_index = wp.array([self.equality_id], dtype=int, device=self.device)
        self.target_index = 0
        groups = np.full(model.ngeom, -1, dtype=np.int32)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ''
            body = int(model.geom_bodyid[geom_id])
            if model.body_rootid[body] == model.body_rootid[self.chassis]:
                # Use the owning body: collision geom names are not stable
                # across the exported Spot assets.
                body_name = model.body(body).name or ''
                if body_name.endswith('arm_link_fngr'):
                    groups[geom_id] = FNGR_GROUP
                elif body_name.endswith('arm_link_jaw'):
                    groups[geom_id] = JAW_GROUP
                else:
                    groups[geom_id] = HAND_GROUP
            if name.startswith(('basket_', 'basket_floor', 'basket_liner')):
                groups[geom_id] = 2
            elif model.geom_type[geom_id] == 0:  # plane
                groups[geom_id] = 3
        self.kind = wp.array(groups, dtype=int, device=self.device)
        self._hand_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._basket_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._ground_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._finger_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self._jaw_hits = wp.zeros(self.worlds, dtype=int, device=self.device)
        self.detached = wp.zeros(self.worlds, dtype=wp.uint8, device=self.device)
        self.hand_contact = wp.zeros_like(self.detached)
        self.basket_contact = wp.zeros_like(self.detached)
        self.ground_contact = wp.zeros_like(self.detached)
        self.bilateral_contact = wp.zeros_like(self.detached)
        self.stable_grasp = wp.zeros_like(self.detached)
        self.ever_grasped = wp.zeros_like(self.detached)
        self.held_at_detach = wp.zeros_like(self.detached)
        self.grasp_time = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.stem_force = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.hand_load = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.finger_load = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.jaw_load = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.palm_load = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.damage_proxy = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.settle_time = wp.zeros(self.worlds, dtype=float, device=self.device)
        self.success = wp.zeros_like(self.detached)
        self.failed = wp.zeros_like(self.detached)
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

    def retarget(self, index):
        """Point the evaluator at another fruit of the scene (arrays are read by the captured graph)."""
        fruit = self.fruits[index]
        self.fruit_geom.assign(np.array([int(self.model.geom(fruit['geom']).id)], dtype=np.int32))
        self.fruit_body.assign(np.array([int(self.model.body(fruit['body']).id)], dtype=np.int32))
        self.equality_id = int(self.model.equality(fruit['equality']).id)
        self.equality_index.assign(np.array([self.equality_id], dtype=np.int32))
        self.target_index = index
        # Fresh per-target state: the new fruit has not been touched yet.
        self.reset()

    def record(self):
        wp.launch(_clear_contacts, dim=self.worlds,
                  inputs=[self._hand_hits, self._basket_hits, self._ground_hits, self._finger_hits, self._jaw_hits,
                          self.hand_load, self.finger_load, self.jaw_load, self.palm_load], device=self.device)
        wp.launch(_contact_pass, dim=self.data.contact.geom.shape[0], inputs=[
            self.data.nacon, self.data.contact.geom, self.data.contact.worldid, self.data.contact.dim,
            self.data.contact.efc_address, self.data.efc.force, self.data.nefc, self.fruit_geom, self.kind,
            self._hand_hits, self._basket_hits, self._ground_hits, self._finger_hits, self._jaw_hits,
            self.hand_load, self.finger_load, self.jaw_load, self.palm_load, self.rows_per_contact], device=self.device)
        wp.launch(_record, dim=self.worlds, inputs=[
            self.data.nefc, self.data.efc.force, self._efc_type, self._efc_id, self.data.xpos, self.data.xipos,
            self.data.subtree_com, self.data.xmat, self.data.cvel, self.eq_active,
            self.fruit_body, self.equality_index, self._hand_hits, self._basket_hits, self._ground_hits,
            self.finger_load, self.jaw_load, self.palm_load, float(self.model.opt.timestep), self.detached, self.hand_contact,
            self.basket_contact, self.ground_contact, self.bilateral_contact, self.stable_grasp, self.ever_grasped,
            self.grasp_time, self._finger_hits, self._jaw_hits, self.stem_force, self.damage_proxy, self.settle_time,
            self.success, self.failed, self.chassis, wp.vec3(*RADII_M), wp.vec3(*CENTER),
            wp.vec3(*SIZE), float(WALL),
            int(self.model.body_rootid[self.model.body(self.manifest['fruits'][0]['body']).id]),
            int(self.model.body_rootid[self.chassis]), self.settle_seconds, int(self.ground_is_failure),
            self.held_at_detach, int(self.detach_requires_hold)], device=self.device)
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
                   self.ground_contact, self.bilateral_contact, self.stable_grasp, self.ever_grasped, self.grasp_time,
                   self.hand_load, self.finger_load, self.jaw_load, self.palm_load, self.damage_proxy,
                   self.settle_time, self.stem_force,
                   self.success, self.failed, self.eq_active, self.equality_id, self.held_at_detach], device=self.device)

    def outputs(self):
        return {'detached': self.detached, 'success': self.success, 'failed': self.failed,
                'damage_proxy': self.damage_proxy, 'hand_contact': self.hand_contact,
                'bilateral_contact': self.bilateral_contact, 'stable_grasp': self.stable_grasp,
                'ever_grasped': self.ever_grasped, 'finger_load_N': self.finger_load,
                'jaw_load_N': self.jaw_load, 'palm_load_N': self.palm_load, 'hand_load_N': self.hand_load}
