"""Free grape-cluster dynamics: per-fruit peduncle beam forces with an
irreversible break state, plus a whole-cluster contact-force diagnostic.

Peduncle stiffness derives from the assumed Young's modulus in
treesim/grape_material (engineering assumption, not measured).  Abscission is
a sampled tensile force proxy (DETACH_RANGE); the contact metric applies a
single-berry rupture threshold to whole-cluster contact force as a proxy.
"""
import numpy as np
import warp as wp

from . import grape_material
from .fruit import AppleField
from .kiwi import contact_loads


@wp.kernel
def peduncle_force(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
                   com: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                   fruit: wp.array(dtype=int), parent: wp.array(dtype=int),
                   offset: wp.array(dtype=wp.vec3), axis: wp.array(dtype=wp.vec3),
                   half: wp.array(dtype=float), length: wp.array(dtype=float),
                   inertia: wp.array(dtype=float),
                   ka: wp.array(dtype=float), kb: wp.array(dtype=float),
                   kr: wp.array(dtype=float),
                   strength: wp.array(dtype=float), broken: wp.array(dtype=int),
                   load: wp.array(dtype=float), dt: float,
                   force: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if broken[i] != 0:
        load[i] = 0.
        return
    a, b = fruit[i], parent[i]
    ac = wp.transform_point(q[a], com[a])
    bc = wp.transform_point(q[b], com[b])
    site = wp.transform_point(q[a], wp.vec3(0., 0., half[i]))
    anchor = wp.transform_point(q[b], offset[i])
    up = wp.transform_vector(q[b], axis[i])
    target = anchor - length[i] * up
    ra, rb = site-ac, anchor-bc
    va = wp.spatial_top(qd[a])+wp.cross(wp.spatial_bottom(qd[a]), ra)
    vb = wp.spatial_top(qd[b])+wp.cross(wp.spatial_bottom(qd[b]), rb)
    vb = vb - length[i]*wp.cross(wp.spatial_bottom(qd[b]), up)
    error, velocity = site-target, va-vb
    axial, speed = wp.dot(error, up), wp.dot(velocity, up)
    # Backward-Euler spring response, with critical damping. Effective axial
    # mass uses fruit translation; small-angle beam approximation at the site.
    ca = 2.*wp.sqrt(ka[i]*mass[a])
    fa = -(ka[i]*(axial+dt*speed)+ca*speed)/(1.+dt*ca/mass[a]+dt*dt*ka[i]/mass[a])
    transverse = error-axial*up
    vt = velocity-speed*up
    # The transverse response acts through the offset site: the cluster rotates
    # as well as translates, so the effective mass is 1/(1/m + half^2/I), much
    # smaller than m.  Using m here (the kiwi kernel's scalar) leaves a stiff
    # peduncle under-damped at this dt and the offset spring pumps rotation.
    meff = 1./(1./mass[a] + half[i]*half[i]/inertia[i])
    cb = 2.*wp.sqrt(kb[i]*meff)
    ft = -(kb[i]*(transverse+dt*vt)+cb*vt)/(1.+dt*cb/meff+dt*dt*kb[i]/meff)
    f = fa*up+ft
    load[i] = wp.length(f)
    # Actual spring load catches contact-induced pulls and canopy acceleration.
    if load[i] > strength[i]:
        broken[i] = 1
        return
    fruit_up = wp.transform_vector(q[a], wp.vec3(0., 0., 1.))
    # Alignment torque plus critical rotational damping: stocky peduncles give
    # kr ~5 N m/rad on a ~1e-3 kg m^2 cluster (75x the kiwi stem), so the
    # undamped kiwi torque law is an unstable torsional oscillator here.
    cr = 2.*wp.sqrt(kr[i]*inertia[i])
    torque = kr[i]*wp.cross(fruit_up, up) - cr*(wp.spatial_bottom(qd[a])-wp.spatial_bottom(qd[b]))
    force[a] = force[a]+wp.spatial_vector(f, wp.cross(ra, f)+torque)
    # Include the cantilever end-force couple in the parent reaction.
    wp.atomic_add(force, b, wp.spatial_vector(-f, wp.cross(site-bc, -f)-torque))


class GrapeField(AppleField):
    """Clusters held by a per-fruit peduncle beam; a load past the sampled
    detach strength breaks the peduncle (irreversible) and the cluster falls.
    ``cut(i)`` severs a peduncle directly — the scripted harvest API."""
    def __init__(self, tree, dt):
        super().__init__(tree)
        self.dt = dt
        q = tree.state_pair()[0].body_q.numpy()
        directions = []
        for b in tree.apple_data['parent_body']:
            inv = wp.quat_inverse(wp.quat(*map(float, q[b, 3:])))
            directions.append(list(wp.quat_rotate(inv, wp.vec3(0., 0., 1.))))
        self.axis = wp.array(directions, dtype=wp.vec3, device=self.dev)
        gd = tree.grape_data
        ped_l = np.asarray(gd['peduncle_length'], dtype=np.float32)
        ped_d = np.asarray(gd['peduncle_diameter'], dtype=np.float32)
        ka, kb, kr = grape_material.peduncle_stiffness(ped_d, ped_l)
        radii = np.asarray(gd['radii'], dtype=np.float32)
        mass0 = tree.model.body_mass.numpy()[tree.apple_data['apple_body']]
        # transverse-axis ellipsoid inertia (1/5)m(rx^2+rz^2); drives the
        # rotation-coupled effective mass and rotational damping in the kernel
        inertia = 0.2*mass0*(radii[:, 0]**2 + radii[:, 2]**2)
        self.inertia = wp.array(inertia.astype(np.float32), device=self.dev)
        self.length = wp.array(ped_l, device=self.dev)
        self.ka = wp.array(ka.astype(np.float32), device=self.dev)
        self.kb = wp.array(kb.astype(np.float32), device=self.dev)
        self.kr = wp.array(kr.astype(np.float32), device=self.dev)
        half = np.asarray(tree.apple_data['hang_drop']) - ped_l
        self.half = wp.array(half.astype(np.float32), device=self.dev)
        # A harvest-ready attached cluster must support its own weight. This
        # conditioning is an engineering assumption, not a source distribution.
        mass = mass0
        self.detach_force = np.maximum(self.detach_force, 1.25*mass*9.81)
        self.strength = wp.array(self.detach_force.astype(np.float32), device=self.dev)

    def hold(self, i, hand_body):
        raise RuntimeError('Cluster grasping requires physical pad contact; no grip-assist spring')

    def apply(self, state):
        if self.n == 0:
            return
        wp.launch(peduncle_force, dim=self.n,
            inputs=[state.body_q, state.body_qd, self.model.body_com, self.model.body_mass,
                    self.apple_body, self.parent_body, self.offset, self.axis, self.half,
                    self.length, self.inertia, self.ka, self.kb, self.kr,
                    self.strength, self._flag, self._tension, self.dt, state.body_f],
            device=self.dev)

    def update(self, state):
        flags = self._flag.numpy()[:self.n].astype(bool)
        count = int(np.count_nonzero(flags & ~self.detached))
        self.detached[:] = flags
        self.broken_count = int(flags.sum())
        return count

    def cut(self, i: int) -> bool:
        """Sever cluster ``i``'s peduncle (scripted harvest).  Returns True if
        the cluster was still attached.  Writes the device flag in place."""
        i = int(i)
        if i < 0 or i >= self.n or self.detached[i]:
            return False
        self._flag_host[i] = 1
        self._flag.assign(self._flag_host)
        self.detached[i] = True
        self.broken_count = int(self.detached.sum())
        return True

    def attached_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.detached)


@wp.kernel
def _peak_step(load: wp.array(dtype=float), peak: wp.array(dtype=float)):
    i = wp.tid()
    peak[i] = wp.max(peak[i], load[i])


class ClusterContact:
    """Per-cluster normal contact-force accumulation and a rupture-risk flag.

    Applies the single-berry rupture threshold (BERRY_RUPTURE_FORCE_MIN) to the
    whole-cluster contact force — a proxy, not a measured cluster limit.  The
    rigid collision ellipsoid does not deform; this is diagnostics only.
    """
    def __init__(self, tree):
        m = tree.model
        self.bodies = list(tree.apple_bodies)
        radii = np.zeros(m.body_count, dtype=np.float32)
        sb, scale = m.shape_body.numpy(), m.shape_scale.numpy()
        for body in self.bodies:
            ids = np.flatnonzero(sb == body)
            radii[body] = float(np.min(scale[ids[0]]))   # first shape = collision ellipsoid
        self.radius = wp.array(radii, device=m.device)
        self.load = wp.zeros(m.body_count, device=m.device)
        self.peak = wp.zeros(m.body_count, device=m.device)
        self.risk = np.zeros(len(self.bodies), dtype=bool)
        self.model = m

    def apply(self, contacts, dt):
        self.load.zero_()
        wp.launch(contact_loads, dim=contacts.rigid_contact_max,
                  inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0,
                          contacts.rigid_contact_shape1, self.model.shape_body,
                          contacts.force, contacts.rigid_contact_normal,
                          self.radius, self.load], device=self.model.device)
        wp.launch(_peak_step, dim=self.model.body_count,
                  inputs=[self.load, self.peak], device=self.model.device)

    def update(self):
        self.risk = self.peak.numpy()[self.bodies] > grape_material.BERRY_RUPTURE_FORCE_MIN

    def metrics(self):
        peak = self.peak.numpy()[self.bodies].tolist()
        return dict(model='single-berry rupture threshold applied to whole-cluster contact; proxy',
                    body_ids=self.bodies, peak_force_N=peak,
                    rupture_threshold_N=grape_material.BERRY_RUPTURE_FORCE_MIN,
                    rupture_risk=[bool(p > grape_material.BERRY_RUPTURE_FORCE_MIN)
                                  for p in peak])


def build_clusters(b, skeleton, seg_to_body, config, placements, apple_rng,
                   apple_jids, apple_bodies, colors, ap_parent, ap_offset,
                   ap_drop, ap_detach, meta):
    """Append one rigid free-body cluster per placement to ``b`` (a
    ModelBuilder), appending to the same lists the apple path fills so
    ``apple_data`` works unchanged.  First shape = colliding ellipsoid
    carrying all the mass; rachis + berries are massless visual shapes.
    ``meta`` collects per-cluster peduncle_diameter / peduncle_length."""
    from .builder import _qrot, _qconj, _wv, _wq
    from .grape_material import FRUIT_FRICTION, RESTITUTION, DETACH_RANGE
    vis = b.ShapeConfig(density=0.0, mu=0.5, collision_group=0,
                        has_shape_collision=False, has_particle_collision=False)
    for ap in placements:
        volume = 4.0*np.pi*np.prod(ap.radii)/3.0
        acfg = b.ShapeConfig(density=float(ap.mass/volume), mu=FRUIT_FRICTION,
                             restitution=RESTITUTION, collision_group=1, is_visible=False)
        pseg = skeleton[ap.parent_seg]
        pbody = int(seg_to_body[ap.parent_seg])
        drop = ap.peduncle_length + float(ap.radii[2])
        hang = ap.attach + np.array([0.0, 0.0, -drop])
        n = len(apple_bodies)
        abody = b.add_link(
            xform=wp.transform(p=_wv(hang), q=wp.quat_identity()),
            label=f"apple{n}")
        b.add_shape_ellipsoid(abody, rx=float(ap.radii[0]), ry=float(ap.radii[1]),
                              rz=float(ap.radii[2]), cfg=acfg, color=ap.color)
        # rachis: thin vertical capsule through the cluster axis, visual only
        b.add_shape_capsule(
            abody, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            radius=0.003, half_height=float(ap.radii[2]),
            cfg=vis, color=(0.18, 0.14, 0.08))
        b.add_shape_capsule(abody,
            xform=wp.transform(p=wp.vec3(0., 0., float(ap.radii[2] + ap.peduncle_length / 2)),
                               q=wp.quat_identity()),
            radius=ap.peduncle_diameter / 2, half_height=ap.peduncle_length / 2,
            cfg=vis, color=(.19, .25, .065))
        for j, off in enumerate(ap.berries):
            tone = .82 + .36 * ((j * 37 % 101) / 100)
            color = tuple(float(min(1., c * tone)) for c in ap.color)
            b.add_shape_sphere(
                abody, xform=wp.transform(p=_wv(off), q=wp.quat_identity()),
                radius=ap.berry_radius, cfg=vis, color=color)
        ajid = b.add_joint_free(child=abody)
        b.add_articulation([ajid], label=f"apple{n}")
        apple_jids.append(ajid)
        apple_bodies.append(abody)
        colors.append(ap.color)
        off = _qrot(_qconj(pseg.frame), ap.attach - pseg.start)
        ap_parent.append(pbody)
        ap_offset.append([float(off[0]), float(off[1]), float(off[2])])
        ap_drop.append(float(drop))
        ap_detach.append(float(apple_rng.uniform(*DETACH_RANGE)))
        meta["peduncle_diameter"].append(float(ap.peduncle_diameter))
        meta["peduncle_length"].append(float(ap.peduncle_length))
        meta["radii"].append([float(ap.radii[0]), float(ap.radii[1]), float(ap.radii[2])])
