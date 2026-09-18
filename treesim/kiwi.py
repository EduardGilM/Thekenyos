"""Free kiwi dynamics with stem-site beam forces and irreversible break state.

Stem stiffness derives from He2024 Hayward. Abscission is a sampled force
proxy (Mu2020 pooled measurements); angle-conditioned fracture and torsion
are not calibrated. No artificial hand attachment is used for kiwi.
"""
import numpy as np
import warp as wp
from .kiwi_material import STEM_LENGTH, STEM_AXIAL, STEM_BENDING, STEM_YOUNG, STEM_I
from .fruit import AppleField


@wp.kernel
def stem_force(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
               com: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
               fruit: wp.array(dtype=int), parent: wp.array(dtype=int),
               offset: wp.array(dtype=wp.vec3), axis: wp.array(dtype=wp.vec3),
               half: wp.array(dtype=float), strength: wp.array(dtype=float),
               broken: wp.array(dtype=int), load: wp.array(dtype=float),
               dt: float, ka: float, kb: float, kr: float,
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
    target = anchor - 0.04384 * up
    ra, rb = site-ac, anchor-bc
    va = wp.spatial_top(qd[a])+wp.cross(wp.spatial_bottom(qd[a]), ra)
    vb = wp.spatial_top(qd[b])+wp.cross(wp.spatial_bottom(qd[b]), rb)
    vb = vb - 0.04384*wp.cross(wp.spatial_bottom(qd[b]), up)
    error, velocity = site-target, va-vb
    axial, speed = wp.dot(error, up), wp.dot(velocity, up)
    # Backward-Euler spring response, with critical damping. Effective axial
    # mass uses fruit translation; small-angle beam approximation at the site.
    ca = 2.*wp.sqrt(ka*mass[a])
    cb = 2.*wp.sqrt(kb*mass[a])
    fa = -(ka*(axial+dt*speed)+ca*speed)/(1.+dt*ca/mass[a]+dt*dt*ka/mass[a])
    transverse = error-axial*up
    vt = velocity-speed*up
    ft = -(kb*(transverse+dt*vt)+cb*vt)/(1.+dt*cb/mass[a]+dt*dt*kb/mass[a])
    f = fa*up+ft
    load[i] = wp.length(f)
    # Actual spring load catches contact-induced pulls and canopy acceleration.
    if load[i] > strength[i]:
        broken[i] = 1
        return
    fruit_up = wp.transform_vector(q[a], wp.vec3(0., 0., 1.))
    torque = kr*wp.cross(fruit_up, up)
    force[a] = force[a]+wp.spatial_vector(f, wp.cross(ra, f)+torque)
    # Include the cantilever end-force couple in the parent reaction.
    wp.atomic_add(force, b, wp.spatial_vector(-f, wp.cross(site-bc, -f)-torque))


class KiwiField(AppleField):
    def __init__(self, tree, dt):
        super().__init__(tree)
        self.dt = dt
        q = tree.state_pair()[0].body_q.numpy()
        directions = []
        for b in tree.apple_data['parent_body']:
            inv = wp.quat_inverse(wp.quat(*map(float,q[b,3:])))
            directions.append(list(wp.quat_rotate(inv, wp.vec3(0.,0.,1.))))
        self.axis = wp.array(directions, dtype=wp.vec3, device=self.dev)
        half = np.asarray(tree.apple_data['hang_drop'])-STEM_LENGTH
        self.half = wp.array(half.astype(np.float32), device=self.dev)
        # A harvest-ready attached fruit must support its own weight. This
        # conditioning is an engineering assumption, not a source distribution.
        mass = tree.model.body_mass.numpy()[tree.apple_data['apple_body']]
        self.detach_force = np.maximum(self.detach_force, 1.25*mass*9.81)
        self.strength = wp.array(self.detach_force.astype(np.float32), device=self.dev)

    def hold(self, i, hand_body):
        raise RuntimeError('Kiwi grasping requires physical pad contact; no grip-assist spring')

    def apply(self, state):
        wp.launch(stem_force, dim=self.n,
            inputs=[state.body_q, state.body_qd, self.model.body_com, self.model.body_mass,
                    self.apple_body, self.parent_body, self.offset, self.axis, self.half,
                    self.strength, self._flag, self._tension, self.dt,
                    float(STEM_AXIAL), float(STEM_BENDING), float(STEM_YOUNG*STEM_I/STEM_LENGTH), state.body_f],
            device=self.dev)

    def update(self, state):
        flags = self._flag.numpy()[:self.n].astype(bool)
        count = int(np.count_nonzero(flags & ~self.detached))
        self.detached[:] = flags
        self.broken_count = int(flags.sum())
        return count


@wp.kernel
def contact_loads(count: wp.array(dtype=int), shape0: wp.array(dtype=int),
                  shape1: wp.array(dtype=int), bodies: wp.array(dtype=int),
                  forces: wp.array(dtype=wp.spatial_vector), normal: wp.array(dtype=wp.vec3),
                  radius: wp.array(dtype=float), load: wp.array(dtype=float)):
    i = wp.tid()
    if i >= count[0]:
        return
    f = wp.abs(wp.dot(wp.spatial_top(forces[i]), normal[i]))
    a, b = bodies[shape0[i]], bodies[shape1[i]]
    if a >= 0:
        if radius[a] > 0.:
            wp.atomic_add(load, a, f)
    if b >= 0:
        if radius[b] > 0.:
            wp.atomic_add(load, b, f)


@wp.kernel
def damage_step(radius: wp.array(dtype=float), load: wp.array(dtype=float),
                dt: float, peak: wp.array(dtype=float), damage: wp.array(dtype=float),
                peak_strain: wp.array(dtype=float), peak_pressure: wp.array(dtype=float)):
    i = wp.tid()
    r, f = radius[i], load[i]
    if r <= 0. or f <= 0.:
        return
    # Hertz sphere/rigid-plane estimate, one equivalent patch per fruit.
    # Xuxiang tissue response transferred to Hayward is explicitly a proxy.
    effective_e = 1570000. / (1.-.4*.4)
    a = wp.pow(3.*f*r/(4.*effective_e), 1./3.)
    indentation = a*a/r
    strain = indentation/(2.*r)
    pressure = 3.*f/(2.*wp.pi*a*a)
    peak[i] = wp.max(peak[i], f)
    peak_strain[i] = wp.max(peak_strain[i], strain)
    peak_pressure[i] = wp.max(peak_pressure[i], pressure)
    excess = wp.max(strain/.05-1., 0.)+wp.max(pressure/260000.-1., 0.)
    # 1-second accumulation scale is assumed, not a measured damage law.
    damage[i] = wp.min(1., damage[i]+dt*excess)


class ContactDamage:
    """Persistent per-fruit contact diagnostics and a negative damage reward.

    This does not deform rigid GPU collision geometry or predict real bruises.
    The native flex bench independently checks actual material deformation.
    """
    def __init__(self, tree):
        m = tree.model
        self.bodies = list(tree.apple_bodies)
        if tree.robot_data and tree.robot_data.get('basket'):
            self.bodies += tree.robot_data['basket']['fruit_bodies']
        radii = np.zeros(m.body_count, dtype=np.float32)
        sb, scale = m.shape_body.numpy(), m.shape_scale.numpy()
        for body in self.bodies:
            ids = np.flatnonzero(sb == body)
            radii[body] = float(np.min(scale[ids[0]]))
        self.radius = wp.array(radii, device=m.device)
        self.load, self.peak, self.damage, self.strain, self.pressure = [wp.zeros(m.body_count, device=m.device) for _ in range(5)]
        self.previous = 0.
        self.reward = 0.
        self.model = m

    def apply(self, contacts, dt):
        self.load.zero_()
        wp.launch(contact_loads, dim=contacts.rigid_contact_max,
                  inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0,
                          contacts.rigid_contact_shape1, self.model.shape_body, contacts.force,
                          contacts.rigid_contact_normal, self.radius, self.load], device=self.model.device)
        wp.launch(damage_step, dim=self.model.body_count,
                  inputs=[self.radius, self.load, dt, self.peak, self.damage, self.strain, self.pressure], device=self.model.device)

    def update(self):
        total = float(self.damage.numpy()[self.bodies].sum())
        self.reward = -(total-self.previous)
        self.previous = total

    def metrics(self):
        def values(a):
            return a.numpy()[self.bodies].tolist()
        return dict(model='uncalibrated Hertz/Xuxiang damage proxy; rigid GPU fruit',
                    body_ids=self.bodies, damage=values(self.damage), peak_force_N=values(self.peak),
                    peak_strain=values(self.strain), peak_pressure_Pa=values(self.pressure),
                    cumulative_damage_penalty=-self.previous)
