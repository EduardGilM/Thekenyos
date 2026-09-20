"""Nonpenetrating distances using the simulator's own convex support maps."""
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.collision_core import geom_collision_pair_from_types
from mujoco_warp._src.collision_gjk import gjk_phase
wp.set_module_options({'enable_backward': False})


@wp.kernel
def _distances(geom_type: wp.array(dtype=int), geom_dataid: wp.array2d(dtype=int),
               geom_size: wp.array2d(dtype=wp.vec3), mesh_vertadr: wp.array(dtype=int),
               mesh_vertnum: wp.array(dtype=int), mesh_graphadr: wp.array(dtype=int),
               mesh_vert: wp.array(dtype=wp.vec3), mesh_graph: wp.array(dtype=int),
               mesh_polynum: wp.array(dtype=int), mesh_polyadr: wp.array(dtype=int),
               mesh_polynormal: wp.array(dtype=wp.vec3), mesh_polyvertadr: wp.array(dtype=int),
               mesh_polyvertnum: wp.array(dtype=int), mesh_polyvert: wp.array(dtype=int),
               mesh_polymapadr: wp.array(dtype=int), mesh_polymapnum: wp.array(dtype=int),
               mesh_polymap: wp.array(dtype=int), geom_xpos: wp.array2d(dtype=wp.vec3),
               geom_xmat: wp.array2d(dtype=wp.mat33), pairs: wp.array(dtype=wp.vec2i),
               gaps: wp.array2d(dtype=float), witnesses: wp.array2d(dtype=wp.vec3)):
    world, index = wp.tid()
    pair = pairs[index]
    t1, t2 = geom_type[pair[0]], geom_type[pair[1]]
    g1, g2 = geom_collision_pair_from_types(
        geom_dataid, geom_size, mesh_vertadr, mesh_vertnum,
        mesh_graphadr, mesh_vert, mesh_graph, mesh_polynum, mesh_polyadr,
        mesh_polynormal, mesh_polyvertadr, mesh_polyvertnum, mesh_polyvert,
        mesh_polymapadr, mesh_polymapnum, mesh_polymap,
        geom_xpos, geom_xmat, t1, t2, pair, world)
    # Match the backend narrowphase: small clearances lose accuracy if support
    # subtraction is performed in absolute orchard coordinates.
    origin = g1.pos
    g1.pos = wp.vec3(0.)
    g2.pos = g2.pos - origin
    penetration, distance, count, x1, x2, simplex, a, b = gjk_phase(
        1.e-6, 100., 64, g1, g2, t1, t2, g1.pos, g2.pos)
    # No reward for compression. EPA depth is unnecessary for this measurement;
    # actual contact loads, damage and slip determine a valid grasp.
    gaps[world, index] = 0. if penetration else wp.max(distance, 0.)
    witnesses[world, index] = x1 + origin


class JawSurfaceDistance:
    def __init__(self, runtime):
        m = runtime.model
        fruit = int(m.geom(runtime.manifest['fruits'][getattr(runtime.task, 'target_index', 0)]['geom']).id)
        pairs, self.groups = [], []
        for suffix in ('arm_link_fngr', 'arm_link_jaw'):
            body = next(i for i in range(m.nbody) if m.body(i).name.endswith(suffix))
            ids = [int(g) for g in np.where(m.geom_bodyid == body)[0]
                   if m.geom_contype[g] or m.geom_conaffinity[g]]
            if not ids:
                raise ValueError('Both physical jaw collision surfaces are required')
            self.groups.append(slice(len(pairs), len(pairs) + len(ids)))
            pairs.extend((g, fruit) for g in ids)
        self.pairs = wp.array(pairs, dtype=wp.vec2i, device=runtime.device_name)
        self.gaps = wp.zeros((runtime.worlds, len(pairs)), dtype=float, device=runtime.device_name)
        self.witnesses = wp.zeros((runtime.worlds, len(pairs)), dtype=wp.vec3, device=runtime.device_name)

    def signals(self, runtime):
        arrays = [getattr(runtime.gpu_model, name) for name in (
            'geom_type', 'geom_dataid', 'geom_size', 'mesh_vertadr', 'mesh_vertnum',
            'mesh_graphadr', 'mesh_vert', 'mesh_graph', 'mesh_polynum', 'mesh_polyadr',
            'mesh_polynormal', 'mesh_polyvertadr', 'mesh_polyvertnum', 'mesh_polyvert',
            'mesh_polymapadr', 'mesh_polymapnum', 'mesh_polymap')]
        wp.launch(_distances, dim=self.gaps.shape,
                  inputs=[*arrays, runtime.data.geom_xpos, runtime.data.geom_xmat,
                          self.pairs, self.gaps, self.witnesses],
                  device=runtime.device_name)
        gaps, points = wp.to_torch(self.gaps), wp.to_torch(self.witnesses)
        center = wp.to_torch(runtime.data.xpos)[:, runtime.fruit_body]
        distances, directions = [], []
        for group in self.groups:
            distance, index = gaps[:, group].min(-1)
            point = points[:, group].gather(1, index[:, None, None].expand(-1, 1, 3)).squeeze(1)
            distances.append(distance.clone())
            directions.append(torch.nn.functional.normalize(point - center, dim=-1))
        return dict(finger_gap=distances[0], jaw_gap=distances[1],
                    opposition=(-(directions[0] * directions[1]).sum(-1)).clamp(-1, 1))
