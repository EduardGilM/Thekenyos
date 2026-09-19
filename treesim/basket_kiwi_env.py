from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from .assisted_kiwi_env import AssistedKiwiEnv, AssistedTask, PREFIX, can_capture
from .basket import CENTER, SIZE, WALL


SCOPE = 'Multi-kiwi assisted pick and free basket deposit; frozen RELIC; ideal sensing; not physical grasp or damage validation'
RADII = np.array([.03, .03, .04])
CONTAINMENT_TOLERANCE_M = .0002


@dataclass(frozen=True)
class BasketTask(AssistedTask):
    time_limit_s: float = 120.
    picks: int = 6
    start_phase: str = 'pick'
    settle_time_s: float = .5

    def __post_init__(self):
        super().__post_init__()
        if type(self.picks) is not int or not 1 <= self.picks <= 6 or self.start_phase not in ('pick', 'carry', 'release'):
            raise ValueError('Invalid basket curriculum or fruit count')
        if not np.isfinite(self.settle_time_s) or not .5 <= self.settle_time_s <= 2.:
            raise ValueError('Basket settling dwell must be between .5 and 2 seconds')


def fruit_in_basket(center, rotation, base, base_rotation):
    local = np.asarray(base_rotation).T@(np.asarray(center)-base)-CENTER
    relative_rotation = np.asarray(base_rotation).T@np.asarray(rotation).reshape(3, 3)
    extent = np.sqrt(relative_rotation**2@RADII**2)
    return bool(np.all(np.abs(local[:2])+extent[:2] <= SIZE[:2]/2-WALL+CONTAINMENT_TOLERANCE_M)
                and local[2]-extent[2] >= WALL/2-CONTAINMENT_TOLERANCE_M
                and local[2]+extent[2] <= SIZE[2]+CONTAINMENT_TOLERANCE_M)


def add_existing_basket(model, relic):
    import newton
    from PIL import Image
    from .config import TreeConfig
    from .spot import build_robot

    with tempfile.TemporaryDirectory(prefix='basket-kiwi-') as directory:
        path = Path(directory)/'base.xml'
        mujoco.mj_saveLastXML(str(path), model)
        root = ET.parse(path).getroot()
        cfg = TreeConfig.compliant('pergola').robot
        cfg.relic_path, cfg.position, cfg.yaw, cfg.base_z = str(relic), (0., 0.), 0., .6
        cfg.fixed_base, cfg.basket, cfg.payload_mass = False, True, 0.
        builder = newton.ModelBuilder()
        maps = build_robot(builder, cfg)
        source = builder.finalize(device='cpu')
        path = Path(directory)/'basket.xml'
        newton.solvers.SolverMuJoCo(source, use_mujoco_cpu=True, use_mujoco_contacts=True, save_to_mjcf=str(path))
        basket_body = ET.parse(path).getroot().find(f'.//body[@name="{PREFIX}body"]')
        body = root.find(f'.//body[@name="{PREFIX}body"]')
        body.remove(body.find('inertial'))
        body.insert(0, basket_body.find('inertial'))
        for geom in basket_body.findall('geom'):
            if geom.get('name', '').startswith(('basket_floor', 'basket_liner')):
                geom.set('group', '0' if geom.get('name', '').startswith('basket_floor') else '3')
                geom.set('rgba', '.065 .075 .085 1')
                geom.set('condim', '6')
                geom.set('priority', '1')
                body.append(geom)
        assets, files = root.find('asset'), {}
        transforms, scales = source.shape_transform.numpy(), source.shape_scale.numpy()
        colors, types = source.shape_color.numpy(), source.shape_type.numpy()
        basket = maps['basket']
        for i in range(basket['first_shape']+5, basket['shape_end']):
            t = transforms[i]
            attrs = dict(name=f'basket_visual_{i}', pos=' '.join(map(str, t[:3])), quat=' '.join(map(str, t[[6, 3, 4, 5]])),
                         contype='0', conaffinity='0', group='2', mass='0', rgba=' '.join(map(str, [*colors[i], 1.])))
            if types[i] == newton.GeoType.BOX:
                attrs.update(type='box', size=' '.join(map(str, scales[i])))
            else:
                mesh, name = source.shape_source[i], f'basket_mesh_{i}'
                element = ET.SubElement(assets, 'mesh', name=name, inertia='shell', vertex=' '.join(map(str, np.asarray(mesh.vertices).ravel())),
                                        face=' '.join(map(str, np.asarray(mesh.indices).ravel())))
                attrs.update(type='mesh', mesh=name)
                if getattr(mesh, 'texture', None) is not None:
                    image = BytesIO()
                    Image.fromarray(np.asarray(mesh.texture)).save(image, format='PNG')
                    files[f'{name}.png'] = image.getvalue()
                    element.set('texcoord', ' '.join(map(str, np.asarray(mesh.uvs).ravel())))
                    ET.SubElement(assets, 'texture', name=name, type='2d', file=f'{name}.png')
                    ET.SubElement(assets, 'material', name=name, texture=name)
                    attrs['material'] = name
            ET.SubElement(body, 'geom', **attrs)
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=files)


class BasketKiwiEnv(AssistedKiwiEnv):
    def __init__(self, relic, *, task=None, render_mode=None, guidance_weight=1.):
        if not np.isfinite(guidance_weight) or not 0 <= guidance_weight <= 4:
            raise ValueError('Guidance weight must be finite in [0, 4]')
        self.guidance_weight = guidance_weight
        super().__init__(relic, task=task or BasketTask(), render_mode=render_mode)
        self.base_observation_space = self.observation_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (99,), np.float32)
        self.harvest_active = False

    def _build(self):
        super()._build()
        original = self.model
        self.model = add_existing_basket(original, self.relic)
        for kind, count in (('body', original.nbody), ('joint', original.njnt), ('site', original.nsite), ('equality', original.neq)):
            for i in range(count):
                if getattr(self.model, kind)(i).name != getattr(original, kind)(i).name:
                    raise RuntimeError('Basket conversion changed the robot or fruit indexing')
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.model.site_sameframe[self.hand_sites] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
        self.basket_mass_added = float(self.model.body_mass[self.chassis]-original.body_mass[self.chassis])
        if not np.isclose(self.basket_mass_added, 1.2, atol=1e-5):
            raise RuntimeError('Basket mass was not preserved')
        self.basket_geoms = {i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith(('basket_floor', 'basket_liner'))}
        self.fruit_geoms = {self.model.geom(f'assisted_kiwi_{i}').id: i for i in range(6)}
        self.ground_geoms = set(np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        self.arm_geoms = {i for i in range(self.model.ngeom) if self.model.body(self.model.geom_bodyid[i]).name.startswith(PREFIX+'arm_') and self.model.geom_contype[i]}
        self.contact_force = np.zeros(6)
        self.avoidance_jacobian = np.zeros((3, self.model.nv))
        self.closest_points = np.zeros(6)

    def _base_velocity(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.chassis, velocity, 0)
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        return np.r_[rotation.T@velocity[3:], rotation.T@velocity[:3]]

    def _initial_arm(self, position):
        from scipy.optimize import least_squares
        self.scratch.qpos[:] = self.data.qpos
        lower, upper = self.joint_limits[12:18].T
        if self.preparing_start:
            position = position+np.array([.15, 0., -.07])
        guesses = ([0., -1.4, 1.6, 0., 1.1, 0.], [0., -1.9, 1., 0., .9, 1.5],
                   [0., -.8, 1.3, 0., .8, 0.], [2.8, -1.3, 1.3, 0., 1., 0.],
                   [-2.3, -1.3, 1.3, 0., 1., 0.], [1.5, -1.3, 1.3, 0., 1., 0.])
        for guess in guesses:
            def residual(q):
                self.scratch.qpos[self.qids[12:18]] = q
                mujoco.mj_kinematics(self.model, self.scratch)
                return np.r_[self.scratch.site_xpos[self.tcp]-position, .0005*(q-guess)]
            result = least_squares(residual, np.clip(guess, lower+1e-5, upper-1e-5), bounds=(lower, upper), max_nfev=180)
            error = np.linalg.norm(residual(result.x)[:3])
            mujoco.mj_forward(self.model, self.scratch)
            overlap = max((max(0., -float(c.dist)) for c in self.scratch.contact[:self.scratch.ncon]
                           if (int(c.geom[0]) in self.arm_geoms and int(c.geom[1]) in self.basket_geoms)
                           or (int(c.geom[1]) in self.arm_geoms and int(c.geom[0]) in self.basket_geoms)), default=0.)
            if error < .01 and overlap < 1e-5:
                self.targets[12:18] = result.x
                self.data.qpos[self.qids[12:18]] = result.x
                mujoco.mj_forward(self.model, self.data)
                return
        raise RuntimeError(f'No collision-free reset arm pose for {position}')

    def reset(self, *, seed=None, options=None):
        self.harvest_active = False
        self.phase = 'approach'
        self.picked = np.zeros(6, dtype=bool)
        self.released = np.zeros(6, dtype=bool)
        self.deposited = np.zeros(6, dtype=bool)
        self.spilled = np.zeros(6, dtype=bool)
        self.settle_times = np.zeros(6)
        self.events = []
        self.path_m = 0.
        self.max_arm_basket_overlap = 0.
        self.preparing_start = True
        super().reset(seed=seed, options=options)
        self.preparing_start = False
        if self.task.start_phase != 'pick':
            position = self.drop_goal() if self.task.start_phase == 'release' else self.target.copy()
            self._initial_arm(position+([0., 0., .105] if self.task.start_phase == 'release' else [-.06, 0., 0.]))
            q = self.model.jnt_qposadr[self.model.joint(f'assisted_kiwi_{self.target_index}').id]
            self.data.qpos[q:q+3] = position
            self.data.qpos[self.qids[-1]] = self.targets[-1] = 0.
            mujoco.mj_forward(self.model, self.data)
            self._capture()
            self.picked[self.target_index] = True
            self.phase = 'carry'
            self.hold_ticks = int(np.ceil(self.task.hold_time_s/.02))
        self.initial_base = self.data.xpos[self.chassis].copy()
        self.previous_xy = self.initial_base[:2].copy()
        self.previous_distance = self.best_distance = self._distance()
        self.harvest_active = True
        return self._observation(), self._info()

    def drop_goal(self):
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        return self.data.xpos[self.chassis]+rotation@(CENTER+[0., 0., SIZE[2]+.12])

    def _observation(self):
        space = self.observation_space
        self.observation_space = self.base_observation_space
        try:
            base = super()._observation()
        finally:
            self.observation_space = space
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        extra = np.r_[rotation.T@(self.drop_goal()-self.data.site_xpos[self.tcp]),
                      rotation.T@(self.drop_goal()-self.data.xpos[self.target_body]),
                      [self.phase == name for name in ('approach', 'carry', 'settle')],
                      self.deposited.astype(float), (~self.picked).astype(float)]
        return np.r_[base, extra].astype(np.float32)

    def _info(self):
        info = super()._info()
        success = bool(self.deposited.sum() >= self.task.picks and not self.failure
                       and np.all(self.settle_times[self.deposited] >= self.task.settle_time_s))
        info.update(scope=SCOPE, success=success, phase=self.phase,
                    outcome=self.failure or ('basket_complete' if success else 'timeout' if self.steps >= np.ceil(self.task.time_limit_s*10) else 'running'),
                    deposited_count=int(self.deposited.sum()), retained_count=int(np.count_nonzero(self.deposited & ~self.spilled)),
                    picked_count=int(self.picked.sum()),
                    deposited_ids=np.flatnonzero(self.deposited).tolist(), spilled_ids=np.flatnonzero(self.spilled).tolist(),
                    settle_times_s=self.settle_times.tolist(), events=list(self.events),
                    basket_distance_m=float(np.linalg.norm(self.data.xpos[self.target_body]-self.drop_goal())),
                    base_path_m=self.path_m, max_arm_basket_overlap_m=self.max_arm_basket_overlap,
                    start_phase=self.task.start_phase)
        return info

    def _physics_step(self):
        super()._physics_step()
        if not self.harvest_active or self.failure:
            return
        contacts, neighbors = set(), [set() for _ in range(6)]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            a, b = map(int, contact.geom)
            if (a in self.arm_geoms and b in self.basket_geoms) or (b in self.arm_geoms and a in self.basket_geoms):
                self.max_arm_basket_overlap = max(self.max_arm_basket_overlap, -float(contact.dist))
                if contact.dist < -.001:
                    self.failure = self.failure or 'arm_basket_collision'
            for fruit_geom, other in ((a, b), (b, a)):
                if fruit_geom not in self.fruit_geoms:
                    continue
                i = self.fruit_geoms[fruit_geom]
                if not self.picked[i]:
                    continue
                mujoco.mj_contactForce(self.model, self.data, contact_index, self.contact_force)
                if self.contact_force[0] <= .01:
                    continue
                if other in self.ground_geoms:
                    self.failure = self.failure or 'dropped'
                if other in self.basket_geoms:
                    contacts.add(i)
                elif other in self.fruit_geoms and self.released[i] and self.released[self.fruit_geoms[other]]:
                    neighbors[i].add(self.fruit_geoms[other])
        for _ in range(6):
            contacts.update(i for i in range(6) if neighbors[i] & contacts)
        rotation = self.data.xmat[self.chassis].reshape(3, 3)
        base = self.data.xpos[self.chassis]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.chassis, velocity, 0)
        for i in np.flatnonzero(self.released):
            body = self.fruit_bodies[i]
            position = self.data.xpos[body]
            local = rotation.T@(position-base)-CENTER
            if self.deposited[i] and (np.any(np.abs(local[:2]) > SIZE[:2]/2+.04) or local[2] < -.045):
                self.spilled[i] = True
                self.failure = self.failure or 'spilled'
            fruit_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, fruit_velocity, 0)
            basket_velocity = velocity[3:]+np.cross(velocity[:3], position-self.data.xipos[self.chassis])
            inside = fruit_in_basket(position, self.data.xmat[body], base, rotation)
            stable = (inside and i in contacts and not self.data.eq_active[self.grips[i]]
                      and np.linalg.norm(fruit_velocity[3:]-basket_velocity) < .05
                      and np.linalg.norm(fruit_velocity[:3]-velocity[:3]) < 1.)
            self.settle_times[i] = self.settle_times[i]+1/self.task.physics_hz if stable else 0.
            if not self.deposited[i] and self.settle_times[i] >= self.task.settle_time_s and not self.failure:
                self.deposited[i] = True
                self.events.append(dict(event='deposited', fruit=int(i), time_s=float(self.data.time-self.start_time)))

    def _avoid_basket(self, velocity, nullspace):
        for arm in sorted(self.arm_geoms):
            for basket in sorted(self.basket_geoms):
                distance = mujoco.mj_geomDistance(self.model, self.data, arm, basket, .04, self.closest_points)
                if distance >= .04 or abs(distance) < 1e-8:
                    continue
                normal = (self.closest_points[:3]-self.closest_points[3:])/distance
                mujoco.mj_jac(self.model, self.data, self.avoidance_jacobian, None,
                             self.closest_points[:3], int(self.model.geom_bodyid[arm]))
                gradient = normal@self.avoidance_jacobian[:, self.dofs[12:18]]
                minimum_speed = 4*(.02-distance)
                speed = float(gradient@velocity)
                if speed >= minimum_speed:
                    continue
                direction = nullspace@gradient
                denominator = float(gradient@direction)
                if denominator < 1e-5:
                    direction = gradient
                    denominator = float(gradient@gradient)
                if denominator > 1e-8:
                    velocity += direction*min((minimum_speed-speed)/denominator, 4.)
        return velocity

    def _next_target(self):
        remaining = np.flatnonzero(~self.picked)
        i = int(remaining[np.argmin(np.linalg.norm(self.fruit_positions[remaining, :2]-self.data.xpos[self.chassis, :2], axis=1))])
        self.target_index, self.target_body = i, int(self.fruit_bodies[i])
        self.target = self.fruit_positions[i].copy()
        self.phase, self.captured, self.hold_ticks = 'approach', False, 0
        self.previous_distance = self.best_distance = self._distance()
        self.events.append(dict(event='next_target', fruit=i, time_s=float(self.data.time-self.start_time)))

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset before stepping a finished episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Action must be seven finite values in [-1, 1]')
        phase_before, count_before = self.phase, int(self.deposited.sum())
        spilled_before = int(self.spilled.sum())
        distance_before = self._distance() if self.phase == 'approach' else np.linalg.norm(self.data.xpos[self.target_body]-self.drop_goal())
        captured_before = self.captured
        self.events = []
        for _ in range(5):
            rotation = self.data.xmat[self.chassis].reshape(3, 3)
            mujoco.mj_jacSite(self.model, self.data, self.jacobian, None, self.tcp)
            jac = self.jacobian[:, self.dofs[12:18]]
            inverse = jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(3), np.eye(3))
            q = self.data.qpos[self.qids[12:18]]
            avoid = np.maximum(self.joint_limits[12:18, 0]+.3-q, 0.)-np.maximum(q-self.joint_limits[12:18, 1]+.3, 0.)
            nullspace = np.eye(6)-inverse@jac
            qd = inverse@(rotation@(action[3:6]*.35))+nullspace@(2*avoid)
            qd = self._avoid_basket(qd, nullspace)
            self.targets[12:18] = np.clip(self.targets[12:18]+.02*np.clip(qd, -.8, .8), self.joint_limits[12:18, 0], self.joint_limits[12:18, 1])
            self.targets[-1] = 0. if action[6] > .2 else -1.
            self._gait_tick(action[:3]*[.4, .25, .7])
            for _ in range(self.task.physics_hz//50):
                self._physics_step()
                if self.failure:
                    break
            if self.failure:
                break
            distance = self._distance()
            if self.phase == 'approach' and distance <= self.task.capture_radius_m and can_capture(
                    distance, action[6] > .2, self.data.qpos[self.qids[-1]], np.linalg.norm(self._base_velocity()[:3])):
                self._capture()
                self.picked[self.target_index] = True
                self.phase = 'carry'
                self.events.append(dict(event='captured', fruit=self.target_index, time_s=float(self.data.time-self.start_time)))
            if self.captured:
                self.hold_ticks += 1
                if distance > self.task.capture_radius_m+.05:
                    self.failure = 'retention_failed'
                if action[6] <= .2 and self.data.qpos[self.qids[-1]] < -.65:
                    self.data.eq_active[self.grips[self.target_index]] = False
                    self.captured = False
                    self.released[self.target_index] = True
                    self.phase = 'settle'
                    self.events.append(dict(event='released', fruit=self.target_index, time_s=float(self.data.time-self.start_time)))
                    if self.hold_ticks*.02 < self.task.hold_time_s:
                        self.failure = 'premature_release'
        self.steps += 1
        self.best_distance = min(self.best_distance, self._distance())
        xy = self.data.xpos[self.chassis, :2].copy()
        self.path_m += float(np.linalg.norm(xy-self.previous_xy))
        self.previous_xy = xy
        distance = self._distance() if self.phase == 'approach' else float(np.linalg.norm(self.data.xpos[self.target_body]-self.drop_goal()))
        terms = dict(progress=self.guidance_weight*8*(distance_before-distance) if self.phase == phase_before and self.phase != 'settle' else 0.,
                     time=-.015, smooth=-.005*float(np.sum((action-self.previous_action)**2)),
                     capture=self.guidance_weight*2. if self.captured and not captured_before and not self.failure else 0.,
                     deposit=15.*(int(self.deposited.sum())-count_before) if not self.failure else 0.,
                     spill=-15.*(int(self.spilled.sum())-spilled_before), failure=-10. if self.failure else 0.)
        self.previous_action = action.copy()
        if self.deposited[self.target_index] and self.deposited.sum() < self.task.picks and not self.failure:
            self._next_target()
        info = self._info()
        terminated = bool(self.failure or info['success'])
        truncated = bool(not terminated and self.steps >= np.ceil(self.task.time_limit_s*10))
        self.done = terminated or truncated
        terms['success'] = 10. if info['success'] else 0.
        info['reward_terms'] = terms
        return self._observation(), float(sum(terms.values())), terminated, truncated, info


gym.register('Thekenyos/BasketKiwi-v0', entry_point='treesim.basket_kiwi_env:BasketKiwiEnv')
