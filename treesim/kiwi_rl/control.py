import numpy as np


def body_com_velocity(model, data, body):
    rotation = data.xmat[body].reshape(3, 3)
    angular = data.cvel[body, :3]
    offset = data.xipos[body] - data.subtree_com[model.body_rootid[body]]
    linear = data.cvel[body, 3:] + np.cross(angular, offset)
    return rotation.T @ linear, rotation.T @ angular


def load_gait_artifact(path):
    import hashlib
    import json
    from pathlib import Path
    from .models_torch import load_trainable_relic_actor
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if not metadata.get('passed') or metadata.get('device') != 'cpu' or metadata.get('samples', 0) < 10000:
        raise ValueError('Use a verified CPU RELIC import artifact')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('exported_sha256'):
        raise ValueError('Gait artifact checksum mismatch')
    return load_trainable_relic_actor(path)


def gait_cpu_inference(actor, observations):
    import torch
    observations = torch.as_tensor(observations, dtype=torch.float32, device='cpu')
    if observations.ndim != 2 or observations.shape[1] != 84 or not len(observations) or not torch.isfinite(observations).all():
        raise ValueError('Expected finite nonempty [batch, 84] gait observations')
    if next(actor.parameters()).device.type != 'cpu':
        raise ValueError('This gait precision profile requires a CPU actor')
    batch = len(observations)
    padding = (-batch) % 16
    padded = torch.cat((observations, observations.new_zeros((padding, 84)))) if padding else observations
    with torch.no_grad():
        actions = torch.cat([actor(chunk) for chunk in padded.split(16)])[:batch]
    if actions.shape != (batch, 12) or not torch.isfinite(actions).all():
        raise RuntimeError('Invalid live gait output')
    return actions.numpy()


class NativeSpotControl:
    def __init__(self, model, robot, gait=None):
        self.model, self.robot, self.gait = model, robot, gait
        self.names = robot['legs'] + robot['arm']
        if len(self.names) != 19 or len(set(self.names)) != 19 or len(robot['observation_joints']) != 19:
            raise ValueError('Expected the 19-joint RELIC contract')
        self.joints = np.array([model.joint(robot['prefix'] + name).id for name in self.names])
        self.qids, self.dofs = model.jnt_qposadr[self.joints], model.jnt_dofadr[self.joints]
        self.actuators = np.array([model.actuator(name).id for name in self.names])
        self.obs_order = np.array([self.names.index(name) for name in robot['observation_joints']])
        self.home = np.array([robot['home_position_rad'][name] for name in self.names], dtype=np.float32)
        self.targets = np.array([robot['initial_position_rad'][name] for name in self.names], dtype=np.float32)
        self.kp, self.kd = np.asarray(robot['kp']), np.asarray(robot['kd'])
        self.limits = np.asarray(robot['controller_torque_limit_Nm'])
        self.knee_table = np.asarray(robot['knee_lookup'])
        if any(value.shape != (19,) for value in (self.kp, self.kd, self.limits)):
            raise ValueError('Invalid actuator contract shape')
        if not np.isfinite(np.r_[self.kp, self.kd, self.limits]).all() or np.any(self.limits <= 0):
            raise ValueError('Invalid actuator contract values')
        if self.knee_table.ndim != 2 or self.knee_table.shape[1] < 3 or len(self.knee_table) < 2:
            raise ValueError('Invalid knee torque lookup')
        if not np.isfinite(self.knee_table).all() or np.any(np.diff(self.knee_table[:, 0]) <= 0):
            raise ValueError('Knee lookup must be finite with strictly increasing angles')
        self.chassis = model.body(robot['chassis']).id
        self.last_action = np.zeros(12, dtype=np.float32)

    def observe(self, data, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Expected a finite velocity command')
        linear, angular = body_com_velocity(self.model, data, self.chassis)
        rotation = data.xmat[self.chassis].reshape(3, 3)
        obs = np.concatenate((linear, angular, rotation.T @ [0., 0., -1.], command,
            self.targets[12:], np.zeros(12), [0., 0., .55],
            data.qpos[self.qids[self.obs_order]] - self.home[self.obs_order],
            data.qvel[self.dofs[self.obs_order]], self.last_action)).astype(np.float32)
        if obs.shape != (84,) or not np.isfinite(obs).all():
            raise RuntimeError('Invalid measured R84 observation')
        return obs

    def set_gait_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected a finite 12-joint gait action')
        self.last_action[:] = action
        self.targets[:12] = self.home[:12] + .2 * action

    def update_gait(self, data, command):
        if self.gait is None:
            raise RuntimeError('No trained gait actor connected')
        self.set_gait_action(gait_cpu_inference(self.gait, self.observe(data, command)[None])[0])

    def apply(self, data, jaw_cap_Nm=.3):
        if not np.isfinite(jaw_cap_Nm) or not 0 <= jaw_cap_Nm <= self.limits[-1]:
            raise ValueError('Jaw effort exceeds imported controller limits')
        q, speed = data.qpos[self.qids], data.qvel[self.dofs]
        if not np.isfinite(np.r_[q, speed, self.targets]).all():
            raise RuntimeError('Nonfinite actuator state or target')
        limits = self.limits.copy()
        limits[8:12] = np.interp(q[8:12], self.knee_table[:, 0], self.knee_table[:, 2])
        limits[-1] = jaw_cap_Nm
        effort = np.clip(self.kp * (self.targets - q) - self.kd * speed, -limits, limits)
        lower = -96.9972 * np.clip(1. + speed[8:12] / 15., 0., 1.)
        upper = 96.9972 * np.clip(1. - speed[8:12] / 14., 0., 1.)
        effort[8:12] = np.clip(effort[8:12], lower, upper)
        data.ctrl[self.actuators] = effort
        return effort
