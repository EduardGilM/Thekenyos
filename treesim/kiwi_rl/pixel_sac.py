"""Pixel SAC for the sparse wrist-camera lane.

Shared conv encoder between actor and critic. Actor-loss gradients do not
enter the encoder. DrQ random shifts of ±4 px are applied independently to
the two critic-target samples, including replayed demonstrations.
"""

from __future__ import annotations

import numpy as np

from .pixel_obs import CAMERA_CHANNELS, FRAME_STACK, VECTOR_DIM

SHIFT_PAD = 4


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError('training stack required (requirements-train.txt)') from e
    return torch


def random_shift(images, pad=SHIFT_PAD, generator=None):
    """Independent integer shifts in [-pad, pad] after replicate padding."""
    torch = _torch()
    if pad < 0 or images.ndim != 4:
        raise ValueError('random_shift expects [batch, channels, height, width] and pad >= 0')
    if pad == 0:
        return images
    batch, _, height, width = images.shape
    padded = torch.nn.functional.pad(images, (pad, pad, pad, pad), mode='replicate')
    dy = torch.randint(0, 2 * pad + 1, (batch,), generator=generator, device=images.device)
    dx = torch.randint(0, 2 * pad + 1, (batch,), generator=generator, device=images.device)
    shifted = []
    for i in range(batch):
        shifted.append(padded[i, :, dy[i]:dy[i] + height, dx[i]:dx[i] + width])
    return torch.stack(shifted, 0)


def build_wrist_encoder(height, width, features=256):
    torch = _torch()
    nn = torch.nn
    layers = []
    channels = CAMERA_CHANNELS * FRAME_STACK
    spatial = min(height, width)
    widths = (32, 64, 64, 128) if spatial >= 32 else (32, 64)
    for width_c in widths:
        layers.extend((nn.Conv2d(channels, width_c, 3, 2, 1), nn.GroupNorm(8, width_c), nn.SiLU()))
        channels = width_c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                         nn.Linear(channels * 16, features), nn.LayerNorm(features), nn.SiLU())


class PixelSAC:
    """Actor and twin critics sharing a visual encoder; critic is not privileged."""

    def __init__(self, height, width, action_dim, features=64, hidden=64):
        torch = _torch()
        nn = torch.nn
        if min(height, width) < 8 or action_dim < 1:
            raise ValueError('Invalid pixel SAC shapes')
        self.encoder = build_wrist_encoder(height, width, features)
        self.actor = nn.Sequential(nn.Linear(features + VECTOR_DIM, hidden), nn.SiLU(),
                                   nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.q1 = nn.Sequential(nn.Linear(features + VECTOR_DIM + action_dim, hidden), nn.SiLU(),
                                nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(features + VECTOR_DIM + action_dim, hidden), nn.SiLU(),
                                nn.Linear(hidden, 1))
        self.action_dim = action_dim

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.actor.parameters()) + [self.log_std] + list(
            self.q1.parameters()) + list(self.q2.parameters())

    def encode(self, camera, *, detach):
        features = self.encoder(camera)
        return features.detach() if detach else features

    def actor_forward(self, camera, vector):
        torch = _torch()
        features = self.encode(camera, detach=True)
        mu = self.actor(torch.cat((features, vector), dim=-1))
        return mu, self.log_std.clamp(-5., 1.)

    def q(self, camera, vector, action, *, shift=False, detach_encoder=False, generator=None):
        torch = _torch()
        images = random_shift(camera, generator=generator) if shift else camera
        features = self.encode(images, detach=detach_encoder)
        inp = torch.cat((features, vector, action), dim=-1)
        return self.q1(inp).squeeze(-1), self.q2(inp).squeeze(-1)

    def actor_loss(self, camera, vector):
        torch = _torch()
        mu, _log_std = self.actor_forward(camera, vector)
        action = torch.tanh(mu)
        q1, q2 = self.q(camera, vector, action, shift=False, detach_encoder=True)
        return -torch.min(q1, q2).mean()

    def critic_target_samples(self, camera, vector, action, generator=None):
        """Two independently shifted Q estimates (DrQ). Applies to demo replay too."""
        torch = _torch()
        gen_a = generator
        gen_b = None
        if generator is not None:
            gen_b = torch.Generator(device=camera.device)
            gen_b.manual_seed(int(generator.initial_seed()) + 1)
        first = self.q(camera, vector, action, shift=True, generator=gen_a)
        second = self.q(camera, vector, action, shift=True, generator=gen_b)
        return first, second
