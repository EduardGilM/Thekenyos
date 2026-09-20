"""GPU-batched privileged PPO policy and its simulator-state observations."""

import hashlib
import json
import math
from pathlib import Path

import torch

from treesim.basket import CENTER


PRIVILEGED_DIM = 32  # Relative positions/orientations, fruit velocity, and task signals.
TEACHER_SCHEMA = 'fast-harvest-privileged-r84/v1'


def validate_teacher_report(report_path, checkpoint_path, *, training_model_sha256,
                            evaluation_model_sha256, minimum_episodes=32):
    """Require matched teacher weights and adequate unguided held-out evidence."""
    report = json.loads(Path(report_path).read_text())
    checkpoint_sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    evaluation = report.get('evaluation', {})
    success = evaluation.get('success', 0.)
    episodes = evaluation.get('episodes', 0)
    if (report.get('role') != 'teacher' or report.get('schema') != TEACHER_SCHEMA or
            report.get('teacher_checkpoint_sha256') != checkpoint_sha or
            report.get('training_model_sha256') != training_model_sha256 or
            report.get('evaluation_model_sha256') != evaluation_model_sha256 or
            report.get('guidance_enabled') is not False or
            report.get('numerical_failures') != 0 or
            not isinstance(episodes, int) or episodes < minimum_episodes or
            not isinstance(success, (int, float)) or not math.isfinite(success) or
            not .5 <= success <= 1. or training_model_sha256 == evaluation_model_sha256):
        raise ValueError('Teacher report does not meet the matched unguided held-out success gate')
    return report


def privileged_observation(runtime, r84=None, *, progress=None):
    """Build a differentiable-free GPU tensor from true state and R84 proprioception."""
    import warp as wp
    if r84 is None:
        r84 = runtime.observe()
    device = r84.device
    xpos = wp.to_torch(runtime.data.xpos)
    xipos = wp.to_torch(runtime.data.xipos)
    xmat = wp.to_torch(runtime.data.xmat)
    site_xpos = wp.to_torch(runtime.data.site_xpos)
    cvel = wp.to_torch(runtime.data.cvel)
    chassis_rotation = xmat[:, runtime.chassis]
    inverse = chassis_rotation.transpose(1, 2)
    tcp_to_fruit = (inverse @ (xpos[:, runtime.fruit_body] - site_xpos[:, runtime.tcp_site])[..., None]).squeeze(-1)
    center = torch.as_tensor(CENTER, dtype=xpos.dtype, device=device)
    basket_world = xpos[:, runtime.chassis] + (chassis_rotation @ center.expand(runtime.worlds, 3)[..., None]).squeeze(-1)
    fruit_to_basket = (inverse @ (basket_world - xpos[:, runtime.fruit_body])[..., None]).squeeze(-1)
    fruit_orientation = inverse @ xmat[:, runtime.fruit_body]
    hand_body = int(runtime.model.site(runtime.tcp_site).bodyid)
    hand_orientation = inverse @ xmat[:, hand_body]
    fruit_ang = cvel[:, runtime.fruit_body, :3]
    fruit_velocity = cvel[:, runtime.fruit_body, 3:] + torch.cross(
        fruit_ang, xipos[:, runtime.fruit_body] - wp.to_torch(runtime.data.subtree_com)[:, int(runtime.model.body_rootid[runtime.fruit_body])], dim=-1)
    chassis_ang = cvel[:, runtime.chassis, :3]
    chassis_velocity = cvel[:, runtime.chassis, 3:] + torch.cross(
        chassis_ang, xipos[:, runtime.fruit_body] - wp.to_torch(runtime.data.subtree_com)[:, int(runtime.model.body_rootid[runtime.chassis])], dim=-1)
    relative_velocity = (inverse @ (fruit_velocity - chassis_velocity)[..., None]).squeeze(-1)
    task = runtime.task
    contact = torch.stack((
        wp.to_torch(task.hand_contact).bool(), wp.to_torch(task.bilateral_contact).bool(),
        wp.to_torch(task.stable_grasp).bool(), wp.to_torch(task.ever_grasped).bool(),
        wp.to_torch(task.detached).bool(), wp.to_torch(task.hand_load).clamp(0., 15.) / 15.,
        wp.to_torch(task.finger_load).clamp(0., 15.) / 15.,
        wp.to_torch(task.jaw_load).clamp(0., 15.) / 15.,
        wp.to_torch(task.palm_load).clamp(0., 15.) / 15.,
        wp.to_torch(task.stem_force).clamp(0., 100.) / 100.,
        wp.to_torch(task.damage_proxy).clamp(0., 1.),
    ), dim=-1).to(dtype=r84.dtype)
    state = torch.cat((tcp_to_fruit, fruit_to_basket, fruit_orientation[:, :, :2].flatten(1),
        hand_orientation[:, :, :2].flatten(1), relative_velocity, contact), dim=-1)
    if tuple(state.shape) != (runtime.worlds, PRIVILEGED_DIM):
        raise RuntimeError('Invalid privileged teacher observation')
    from .reward_graph import SEQUENCE_PROFILE
    if getattr(runtime, 'task_profile', None) == SEQUENCE_PROFILE:
        if progress is None or progress.reward_profile != SEQUENCE_PROFILE:
            raise ValueError('Sequence teacher requires matching episode history')
        state = torch.cat((state, progress.observation()), -1)
    return state


def build_privileged_policy(*, sequence=False):
    """Return the same recurrent actor/critic interface as the RGB-D policy."""
    nn = torch.nn
    input_dim = PRIVILEGED_DIM
    if sequence:
        from .sequence_curriculum import SEQUENCE_OBSERVATION_DIM
        input_dim += SEQUENCE_OBSERVATION_DIM

    class PrivilegedPolicy(nn.Module):
        observation_kind = 'privileged'

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim + 84, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU())
            self.belief = nn.GRUCell(128, 64)
            self.mean = nn.Linear(64, 7)
            self.value = nn.Linear(64, 1)
            self.logstd = nn.Parameter(torch.full((7,), -1.6))

        def forward(self, privileged, r84, memory):
            if privileged.ndim != 2 or privileged.shape[-1] != input_dim:
                raise ValueError(f'Expected privileged observations [batch, {input_dim}]')
            if r84.shape != (privileged.shape[0], 84) or memory.shape != (privileged.shape[0], 64):
                raise ValueError('Privileged policy received incompatible R84 or recurrent state')
            memory = self.belief(self.encoder(torch.cat((privileged, r84), dim=-1)), memory)
            return self.mean(memory), self.logstd.clamp(-5., 1.), self.value(memory).squeeze(-1), memory

    return PrivilegedPolicy()
