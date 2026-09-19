import numpy as np
import warp as wp

from .control import NativeSpotControl


@wp.kernel
def _motor_pd(q: wp.array2d(dtype=float), qd: wp.array2d(dtype=float),
              qids: wp.array(dtype=int), dofs: wp.array(dtype=int), actuators: wp.array(dtype=int),
              target: wp.array2d(dtype=float), kp: wp.array(dtype=float), kd: wp.array(dtype=float),
              limits: wp.array(dtype=float), knee: wp.array2d(dtype=float), jaw_cap: wp.array(dtype=float),
              control: wp.array2d(dtype=float), applied: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    angle, speed = q[world, qids[joint]], qd[world, dofs[joint]]
    effort = kp[joint] * (target[world, joint] - angle) - kd[joint] * speed
    limit = limits[joint]
    if joint == 18:
        limit = wp.clamp(jaw_cap[world], 0., limit)
    if joint >= 8 and joint < 12:
        limit = knee[0, 2]
        for k in range(1, knee.shape[0]):
            if angle >= knee[k, 0]:
                limit = knee[k, 2]
            elif angle >= knee[k - 1, 0]:
                fraction = (angle - knee[k - 1, 0]) / (knee[k, 0] - knee[k - 1, 0])
                limit = (1. - fraction) * knee[k - 1, 2] + fraction * knee[k, 2]
        effort = wp.clamp(effort, -limit, limit)
        lower = -96.9972 * wp.clamp(1. + speed / 15., 0., 1.)
        upper = 96.9972 * wp.clamp(1. - speed / 14., 0., 1.)
        effort = wp.clamp(effort, lower, upper)
    else:
        effort = wp.clamp(effort, -limit, limit)
    control[world, actuators[joint]] = effort
    applied[world, joint] = effort


@wp.kernel
def _set_gait_targets(actions: wp.array2d(dtype=float), home: wp.array(dtype=float), target: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    target[world, joint] = home[joint] + .2 * actions[world, joint]


@wp.kernel
def _measure_r84(q: wp.array2d(dtype=float), qd: wp.array2d(dtype=float), cvel: wp.array2d(dtype=wp.spatial_vector),
                  xipos: wp.array2d(dtype=wp.vec3), subtree_com: wp.array2d(dtype=wp.vec3),
                  xmat: wp.array2d(dtype=wp.mat33), chassis: int, tree_root: int,
                  qids: wp.array(dtype=int), dofs: wp.array(dtype=int), order: wp.array(dtype=int),
                  home: wp.array(dtype=float), targets: wp.array2d(dtype=float), command: wp.array2d(dtype=float),
                  previous: wp.array2d(dtype=float), obs: wp.array2d(dtype=float)):
    world = wp.tid()
    velocity = cvel[world, chassis]
    angular = wp.vec3(velocity[0], velocity[1], velocity[2])
    linear = wp.vec3(velocity[3], velocity[4], velocity[5])
    linear += wp.cross(angular, xipos[world, chassis] - subtree_com[world, tree_root])
    inverse = wp.transpose(xmat[world, chassis])
    linear, angular = inverse @ linear, inverse @ angular
    gravity = inverse @ wp.vec3(0., 0., -1.)
    for i in range(3):
        obs[world, i] = linear[i]
        obs[world, i + 3] = angular[i]
        obs[world, i + 6] = gravity[i]
        obs[world, i + 9] = command[world, i]
        obs[world, i + 31] = 0.
    obs[world, 33] = .55
    for i in range(7):
        obs[world, i + 12] = targets[world, i + 12]
    for i in range(12):
        obs[world, i + 19] = 0.
        obs[world, i + 72] = previous[world, i]
    for i in range(19):
        joint = order[i]
        obs[world, i + 34] = q[world, qids[joint]] - home[joint]
        obs[world, i + 53] = qd[world, dofs[joint]]


class WarpSpotControl:
    def __init__(self, model, data, robot):
        self.contract = NativeSpotControl(model, robot)
        self.data, self.worlds = data, data.qpos.shape[0]
        self.device = data.qpos.device
        self.chassis = self.contract.chassis
        self.tree_root = int(model.body_rootid[self.chassis])
        for name in ('qids', 'dofs', 'actuators', 'obs_order'):
            setattr(self, name, wp.array(getattr(self.contract, name), dtype=int, device=self.device))
        for name in ('home', 'kp', 'kd', 'limits', 'knee_table'):
            setattr(self, name, wp.array(getattr(self.contract, name), dtype=float, device=self.device))
        self.targets = wp.array(np.tile(self.contract.targets, (self.worlds, 1)), dtype=float, device=self.device)
        self.commands = wp.zeros((self.worlds, 3), device=self.device)
        self.previous = wp.zeros((self.worlds, 12), device=self.device)
        self.jaw_cap = wp.full(self.worlds, min(.3, float(self.contract.limits[-1])), device=self.device)
        self.effort = wp.zeros((self.worlds, 19), device=self.device)
        self.observations = wp.zeros((self.worlds, 84), device=self.device)

    def set_targets(self, targets):
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != (self.worlds, 19) or not np.isfinite(targets).all():
            raise ValueError('Expected finite batched joint targets')
        self.targets.assign(targets)

    def set_commands(self, commands):
        commands = np.asarray(commands, dtype=np.float32)
        if commands.shape != (self.worlds, 3) or not np.isfinite(commands).all():
            raise ValueError('Expected finite batched velocity commands')
        self.commands.assign(commands)

    def set_jaw_caps(self, caps):
        caps = np.asarray(caps, dtype=np.float32)
        if caps.shape != (self.worlds,) or not np.isfinite(caps).all() or np.any(caps < 0) or np.any(caps > self.contract.limits[-1]):
            raise ValueError('Invalid jaw effort caps')
        self.jaw_cap.assign(caps)

    def set_gait_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.worlds, 12) or not np.isfinite(actions).all():
            raise ValueError('Expected finite batched gait actions')
        self.previous.assign(actions)
        wp.launch(_set_gait_targets, dim=(self.worlds, 12), inputs=[self.previous, self.home, self.targets], device=self.device)

    def apply(self):
        wp.launch(_motor_pd, dim=(self.worlds, 19), inputs=[self.data.qpos, self.data.qvel, self.qids, self.dofs,
            self.actuators, self.targets, self.kp, self.kd, self.limits, self.knee_table, self.jaw_cap,
            self.data.ctrl, self.effort], device=self.device)

    def observe(self):
        d = self.data
        wp.launch(_measure_r84, dim=self.worlds, inputs=[d.qpos, d.qvel, d.cvel, d.xipos, d.subtree_com,
            d.xmat, self.chassis, self.tree_root, self.qids, self.dofs, self.obs_order, self.home,
            self.targets, self.commands, self.previous, self.observations], device=self.device)
        return self.observations
