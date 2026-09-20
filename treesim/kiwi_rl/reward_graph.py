"""Temporal physical grades; engineering tolerances, never action commands."""
import numpy as np
import torch

LEGACY_GRAPH_PROFILE = 'graph-harvest/v1'
CONTINUOUS_GRAPH_PROFILE = 'graph-harvest/v2'
GRAPH_PROFILE = 'graph-harvest/v3'
SOFT_GRAPH_PROFILE = 'graph-harvest/v4'
CONTROL_GRAPH_PROFILE = 'graph-harvest/v5'
SEQUENCE_PROFILE = 'sequence-harvest/v2'
PREREQUISITE_GRAPH_PROFILES = (GRAPH_PROFILE, SOFT_GRAPH_PROFILE, CONTROL_GRAPH_PROFILE)
REGRESSION_WEIGHTS = dict(position=1., grip=1.5, extract=2., carry=2., deposit=2., complete=2.)
CONTINUOUS_GRAPH_PROFILES = (CONTINUOUS_GRAPH_PROFILE, *PREREQUISITE_GRAPH_PROFILES)
GRAPH_PROFILES = (LEGACY_GRAPH_PROFILE, *CONTINUOUS_GRAPH_PROFILES, SEQUENCE_PROFILE)
STAGE_NAMES = ('position', 'grip', 'extract', 'carry', 'deposit', 'complete')
GRAPH_CONSTANTS = dict(approach_radius_m=.5, approach_outer_m=1.5,
    enclosure_tolerance_m=.008, enclosure_core_radius_fraction=.25, insertion_range_m=.15, slip_enter_m_s=.05,
    slip_exit_m_s=.08, grip_dwell_s=.1, settle_seconds=2., release_clearance_m=.04,
    closure_error_range_m=.04, minimum_aperture_m=.005, jaw_load_limit_N=15.,
    stem_detach_force_N=8., damage_limit=.05, carry_range_m=1.)


def approach_score(distance):
    return ((1.5 - distance) / 1.).clamp(0., 1.)


def ellipsoid_extent(rotation, radii):
    return ((rotation * radii).square().sum(-1)).sqrt()


def enclosure_features(center, extent, finger, jaw, *, continuous=False):
    """Bounds are actual collision supports in the wrist frame [world,2,3].

    Opposing collision supports define the open aperture. An open gripper does
    not contain the entire ellipsoid: side/tip grasps leave fruit outside its
    walls. Require its central core (quarter of each projected radius) inside
    that aperture; this rules out mere tangent overlap while permitting the
    independently measured bilateral side grasp. Both the core fraction and
    8 mm coarse-surface tolerance are explicit engineering assumptions.
    """
    lower = torch.maximum(finger[:, 0], jaw[:, 0])
    upper = torch.minimum(finger[:, 1], jaw[:, 1])
    # Fixed jaw lies below the moving finger in the authored wrist frame.
    lower = lower.clone(); upper = upper.clone()
    lower[:, 2] = jaw[:, 1, 2]
    upper[:, 2] = finger[:, 0, 2]
    gap = (upper[:, 2] - lower[:, 2]).clamp_min(0)
    core = GRAPH_CONSTANTS['enclosure_core_radius_fraction'] * extent
    violations = torch.maximum(lower + core - center, center + core - upper)
    tolerance = GRAPH_CONSTANTS['enclosure_tolerance_m']
    error = (violations - tolerance).clamp_min(0).norm(dim=-1)
    enclosed = (violations <= tolerance).all(-1) & (gap > GRAPH_CONSTANTS['minimum_aperture_m'])
    insertion = (1 - error / GRAPH_CONSTANTS['insertion_range_m']).clamp(0, 1)
    if continuous:
        # No dead zone: every reduction in distance to the valid jaw region
        # improves the grade, including from outside the former 15 cm window.
        aperture_error = (GRAPH_CONSTANTS['minimum_aperture_m'] - gap).clamp_min(0)
        insertion = 1 / (1 + (error + aperture_error) / .15)
    diameter = 2 * extent[:, 2]
    # Quality peaks at the fruit diameter; compression never earns more credit.
    closure = (1 - (gap - diameter).abs() / GRAPH_CONSTANTS['closure_error_range_m']).clamp(0, 1) * enclosed
    return enclosed, insertion, closure


def basket_frame_errors(local, extent):
    """Per-axis horizontal overshoot beyond the open footprint, and fruit height above the rim."""
    from treesim.basket import CENTER, SIZE, WALL
    center = local.new_tensor(CENTER); size = local.new_tensor(SIZE)
    horizontal_error = ((local[:, :2]-center[:2]).abs()+extent[:, :2]-(size[:2]/2-WALL)).clamp_min(0)
    rim_z = center[2] + size[2]
    return horizontal_error, local[:, 2] - rim_z


def release_position_distance(local, extent):
    """Distance to the valid center region above the open basket, not its base."""
    horizontal_error, height = basket_frame_errors(local, extent)
    target = extent[:, 2] + GRAPH_CONSTANTS['release_clearance_m']
    return torch.cat((horizontal_error, (height-target)[:, None]), -1).norm(dim=-1)


class CollisionGeometry:
    """Cache authored mesh support vertices once; transform on device per step."""
    def __init__(self, runtime):
        import mujoco
        m = runtime.model
        self.hand = int(m.site(runtime.tcp_site).bodyid)
        # The TCP can be owned by another hand body; use the fixed wrist frame.
        self.hand = next(i for i in range(m.nbody) if m.body(i).name.endswith('arm_link_wr1'))
        self.parts = []
        for suffix in ('arm_link_fngr', 'arm_link_jaw'):
            body = next(i for i in range(m.nbody) if m.body(i).name.endswith(suffix))
            vertices = []
            for g in np.where(m.geom_bodyid == body)[0]:
                if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError('Graph jaw geometry requires actual collision meshes')
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                rot = np.empty(9); mujoco.mju_quat2Mat(rot, m.geom_quat[g])
                vertices.append(v @ rot.reshape(3, 3).T + m.geom_pos[g])
            if not vertices:
                raise ValueError('Graph requires both jaw collision meshes')
            # All vertices retained: a rotated AABB would overestimate supports.
            self.parts.append((body, torch.tensor(np.concatenate(vertices), dtype=torch.float32,
                                                  device=runtime.device_name)))
        geom = int(m.geom(runtime.manifest['fruits'][getattr(runtime.task, 'target_index', 0)]['geom']).id)
        self.radii = torch.tensor(m.geom_size[geom], dtype=torch.float32, device=runtime.device_name)

    def signals(self, runtime):
        import warp as wp
        p = wp.to_torch(runtime.data.xpos); r = wp.to_torch(runtime.data.xmat)
        inv = r[:, self.hand].transpose(1, 2)
        fruit = runtime.fruit_body
        local = (inv @ (p[:, fruit] - p[:, self.hand])[..., None]).squeeze(-1)
        extent = ellipsoid_extent(inv @ r[:, fruit], self.radii)
        bounds = []
        for body, vertices in self.parts:
            points = (inv @ r[:, body]) @ vertices.T
            points += (inv @ (p[:, body] - p[:, self.hand])[..., None])
            bounds.append(torch.stack((points.amin(-1), points.amax(-1)), dim=1))
        enclosed, insertion, closure = enclosure_features(local, extent, *bounds,
            continuous=runtime.task_profile in (*CONTINUOUS_GRAPH_PROFILES, SEQUENCE_PROFILE))
        cvel = wp.to_torch(runtime.data.cvel)
        com = wp.to_torch(runtime.data.subtree_com)
        point = wp.to_torch(runtime.data.xipos)[:, fruit]
        def velocity(body):
            root = int(runtime.model.body_rootid[body])
            return cvel[:, body, 3:] + torch.cross(cvel[:, body, :3], point-com[:, root], dim=-1)
        slip = (velocity(fruit)-velocity(self.hand)).norm(dim=-1)
        bi = r[:, runtime.chassis].transpose(1, 2)
        basket_local = (bi @ (p[:, fruit]-p[:, runtime.chassis])[..., None]).squeeze(-1)
        be = ellipsoid_extent(bi @ r[:, fruit], self.radii)
        release_error = release_position_distance(basket_local, be)
        horizontal, height = basket_frame_errors(basket_local, be)
        result = dict(enclosed=enclosed, insertion=insertion, closure=closure, slip=slip,
                      release_distance=release_error, basket_horizontal=horizontal.norm(dim=-1),
                      basket_height=height)
        if runtime.task_profile == SEQUENCE_PROFILE:
            from .surface_distance import JawSurfaceDistance
            if not hasattr(self, 'surface'):
                self.surface = JawSurfaceDistance(runtime)
            result.update(self.surface.signals(runtime))
            # The old AABB aperture score is never used by the new curriculum.
            result['closure'] = 1 / (1 + (result['finger_gap'] + result['jaw_gap']) / .04)
            # Seating depth: fruit centre along the wrist's reach axis relative to the
            # TCP. Positive = out toward the fingertips. Tip grasps do not survive the
            # post-detachment swing; seated grasps (near zero) do.
            tcp_x = float(runtime.manifest['robot'].get('tcp_offset_m', (0.195, 0., 0.005))[0])
            result['grasp_depth'] = local[:, 0] - tcp_x
        return result


def graph_step(progress, now):
    """Update explicit dwell/hysteresis and current-state grade, including regressions."""
    c = GRAPH_CONSTANTS
    if progress.reward_profile in PREREQUISITE_GRAPH_PROFILES:
        return prerequisite_graph_step(progress, now)
    safe = ~now['failed'] & (now['max_load'] <= c['jaw_load_limit_N']) & (now['damage'] <= c['damage_limit'])
    threshold = torch.where(progress.graph_grip, c['slip_exit_m_s'], c['slip_enter_m_s'])
    loaded = now['enclosed'] & now['bilateral'] & safe & (now['slip'] < threshold)
    progress.graph_dwell = torch.where(loaded, progress.graph_dwell + progress.control_dt, 0.)
    progress.graph_grip = loaded & now['holding'] & (progress.graph_dwell >= c['grip_dwell_s'])
    progress.graph_enclosed = now['enclosed'].clone()
    progress.graph_slip = now['slip'].clone()
    progress.graph_detached |= now['detached']
    detached = progress.graph_detached
    approach = approach_score(now['distance'])
    insertion = torch.where(approach >= 1., now['insertion'], 0.)
    score = approach + insertion
    if progress.reward_profile in CONTINUOUS_GRAPH_PROFILES:
        score = 2 * now['insertion']
    grip_quality = torch.minimum((progress.graph_dwell / c['grip_dwell_s']).clamp(0, 1), now['closure'])
    if progress.reward_profile in CONTINUOUS_GRAPH_PROFILES:
        # Jaw positioning can improve before loaded contact; sustained safe
        # contact earns the remainder. Empty closure earns nothing.
        grip_quality = .25 * now['closure'] + .75 * grip_quality
    score = torch.where(now['enclosed'], 2. + grip_quality, score)
    score = torch.where(progress.graph_grip, 3. + .5*(now['stem_force']/c['stem_detach_force_N']).clamp(0, 1), score)
    carry = (1-now['release_distance']/c['carry_range_m']).clamp(0, 1)
    if progress.reward_profile in CONTINUOUS_GRAPH_PROFILES:
        carry = 1 / (1 + now['release_distance']/c['carry_range_m'])
    score = torch.where(detached, torch.where(progress.graph_grip, 4.+carry, 3.6), score)
    settling = detached & ~now['touching'] & (now['settle_time'] > 0)
    score = torch.where(settling, 5.+(now['settle_time']/c['settle_seconds']).clamp(0, 1), score)
    score = torch.where(now['success'], 6., score)
    score = torch.where(safe, score, 0.)
    progress.graph_ground_drop = detached & now['ground_contact'] & ~progress.graph_grip & safe & ~now['success']
    progress.graph_score = score
    progress.graph_stage = torch.where(detached & ~progress.graph_grip & ~settling, 3, score.floor().long())
    # Additive bands telescope exactly to the score, including regressions.
    # This permits an honest per-stage reward decomposition in telemetry.
    edges = (0., 2., 3., 3.6, 5., 6., 6.)
    progress.graph_components = {
        name: (score - edges[i]).clamp(0., edges[i+1]-edges[i])
        for i, name in enumerate(STAGE_NAMES)}
    return score


def prerequisite_graph_step(p, now):
    """Physical prerequisite graph. Events never substitute for maintained control."""
    c = GRAPH_CONSTANTS
    safe = ~now['failed'] & (now['max_load'] <= c['jaw_load_limit_N']) & (now['damage'] <= c['damage_limit'])
    was_gripped = p.graph_grip.clone()
    was_detached = p.graph_detached.clone()
    threshold = torch.where(was_gripped, c['slip_exit_m_s'], c['slip_enter_m_s'])
    loaded = now['enclosed'] & now['bilateral'] & safe & (now['slip'] < threshold)
    p.graph_dwell = torch.where(loaded, p.graph_dwell + p.control_dt, 0.)
    p.graph_grip = loaded & now['holding'] & (p.graph_dwell >= c['grip_dwell_s'])
    p.graph_detached |= now['detached']
    detached = p.graph_detached
    new_detach = detached & ~was_detached
    # Both sides of the observed detachment interval must have secure control.
    p.graph_valid_extract |= new_detach & was_gripped & p.graph_grip
    p.graph_invalid_extract |= new_detach & ~p.graph_valid_extract
    releasing = (detached & p.graph_valid_extract & was_gripped & ~p.graph_grip & safe
                 & (p.previous['release_distance'] <= c['release_clearance_m']))
    p.graph_valid_release |= releasing
    p.graph_valid_release &= safe & ~now['ground_contact'] & ~p.graph_grip
    held = detached & p.graph_valid_extract & p.graph_grip
    settling = p.graph_valid_release & ~now['touching'] & (now['settle_time'] > 0)
    p.graph_success = now['success'] & settling & p.graph_valid_extract & safe
    p.graph_ground_drop = detached & now['ground_contact'] & ~p.graph_success
    p.graph_enclosed = now['enclosed'].clone()
    p.graph_slip = now['slip'].clone()
    attached = ~detached
    fit = .25*now['closure'] + .75*torch.minimum((p.graph_dwell/c['grip_dwell_s']).clamp(0,1),now['closure'])
    score = 2*now['insertion']
    score = torch.where(now['enclosed'], 2.+fit, score)
    score = torch.where(p.graph_grip, 4.+(now['stem_force']/c['stem_detach_force_N']).clamp(0,1),score)
    # Detached but uncontrolled fruit has no approach/grip/extraction credit.
    score = torch.where(detached, 0., score)
    carry = 1/(1+now['release_distance']/c['carry_range_m'])
    score = torch.where(held, 6.+2*carry, score)
    score = torch.where(p.graph_valid_release, 8., score)
    score = torch.where(settling, 10.+(now['settle_time']/c['settle_seconds']).clamp(0,1),score)
    score = torch.where(p.graph_success, 12., score)
    score = torch.where(safe & ~p.graph_ground_drop & ~p.graph_invalid_extract,score,0.)
    p.graph_score = score
    stage = torch.zeros_like(p.graph_stage)
    stage = torch.where(attached & now['enclosed'],1,stage)
    stage = torch.where(attached & p.graph_grip,2,stage)
    stage = torch.where(held,3,stage)
    stage = torch.where((held & (now['release_distance'] <= c['release_clearance_m'])) | p.graph_valid_release,4,stage)
    stage = torch.where(settling,5,stage)
    stage = torch.where(p.graph_success,6,stage)
    p.graph_stage = torch.where(safe & ~p.graph_ground_drop & ~p.graph_invalid_extract,stage,0)
    p.graph_position_dwell = torch.where(attached & now['enclosed'] & safe,p.graph_position_dwell+p.control_dt,0.)
    p.graph_completed['position'] |= p.graph_position_dwell >= c['grip_dwell_s']
    p.graph_completed['grip'] |= attached & p.graph_grip
    p.graph_completed['extract'] |= new_detach & p.graph_valid_extract
    p.graph_completed['carry'] |= held & (now['release_distance'] <= c['release_clearance_m'])
    p.graph_completed['deposit'] |= p.graph_success
    p.graph_completed['complete'] |= p.graph_success
    p.graph_components = {name:(score-2*i).clamp(0.,2.) for i,name in enumerate(STAGE_NAMES)}
    return score
