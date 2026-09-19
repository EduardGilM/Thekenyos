"""Torch actors/critics (TK-RL-003 v3.0, §3). JP-ONLY at runtime.

Requires torch + torchvision (see requirements-train.txt) and, for G1,
the external RELIC checkout. Nothing in this module imports at package
import time: every entry point imports torch lazily so numpy-only unit
tests keep passing on machines without the training stack.

Layouts (exact, see schemas.py):
- V3:  5ch RGB-D [B,5,240,320] -> z [B,256] (ResNet-18-GN from scratch)
- N3:  488 -> mu [B,3] + logits [B,2], GRU-256
- M3:  382 -> mu [B,8] + logits [B,3], GRU-256
- G1:  84 -> mu [B,12], MLP 84-512-256-128-12 ELU (RELIC layout)
- C_*: 463 -> V [B,1], DeepSets fruit pool + MLP
"""

from __future__ import annotations

import numpy as np

from . import schemas as S

GRU_DIM = 256
LOGSTD_MIN, LOGSTD_MAX = -5.0, 1.0


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("training stack required (requirements-train.txt)") from e
    return torch


def groupnorm8(channels: int):
    """GroupNorm-8 usable as torchvision resnet norm_layer."""
    torch = _torch()
    return torch.nn.GroupNorm(8, channels)


def build_v3():
    """ResNet-18 backbone, 5-channel input, spatial 4x4 pool -> z[256]."""
    torch = _torch()
    import torchvision.models as M
    net = M.resnet18(weights=None, norm_layer=lambda c: torch.nn.GroupNorm(8, c))
    net.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    net.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))
    net.fc = torch.nn.Identity()
    proj = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(8192, 256),
        torch.nn.LayerNorm(256),
        torch.nn.SiLU(),
    )
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")

    class V3(torch.nn.Module):
        def forward(self, x):
            return proj(net(x))
    return V3()


def build_map_encoder():
    """Map CNN [B,3,64,64] -> [B,128] (spec §3.1)."""
    torch = _torch()
    nn = torch.nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(8, cout), nn.SiLU())

    return nn.Sequential(
        nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.SiLU(),
        block(32, 64), block(64, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, 128), nn.SiLU(),
    )


class RecurrentHead:
    """N3/M3 container: fusion MLP + GRU + continuous/event heads."""

    def __init__(self, input_dim: int, cont_dim: int, event_dim: int,
                 sigma_init: float, event_bias: tuple):
        torch = _torch()
        self.mod = torch.nn.ModuleDict({
            "fuse": torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256), torch.nn.SiLU()),
            "gru": torch.nn.GRU(256, GRU_DIM, 1, batch_first=False),
            "post": torch.nn.Sequential(
                torch.nn.Linear(GRU_DIM, 128), torch.nn.SiLU()),
            "mu": torch.nn.Linear(128, cont_dim),
            "event": torch.nn.Linear(128, event_dim),
        })
        self.logstd = torch.nn.Parameter(
            torch.full((cont_dim,), float(np.log(sigma_init))))
        with torch.no_grad():
            self.mod["event"].bias.copy_(
                torch.tensor(event_bias, dtype=torch.float32))

    def parameters(self):
        torch = _torch()
        return list(self.mod.parameters()) + [self.logstd]

    def forward(self, x, h):
        torch = _torch()
        f = self.mod["fuse"](x).unsqueeze(0)
        y, h2 = self.mod["gru"](f, h)
        p = self.mod["post"](y.squeeze(0))
        return self.mod["mu"](p), self.mod["event"](p), h2

    def clamped_logstd(self):
        torch = _torch()
        return torch.clamp(self.logstd, LOGSTD_MIN, LOGSTD_MAX)


def build_n3() -> RecurrentHead:
    return RecurrentHead(S.N3_INPUT_DIM, S.N3_CONT_DIM, S.N3_EVENT_DIM,
                         sigma_init=0.3, event_bias=(2.0, 0.0))


def build_m3() -> RecurrentHead:
    return RecurrentHead(S.M3_INPUT_DIM, S.M3_CONT_DIM, S.M3_EVENT_DIM,
                         sigma_init=0.2, event_bias=(2.0, 0.0, 0.0))


def build_g1_mlp():
    """Fresh MLP with the exact RELIC layout (for weight import)."""
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(84, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )


def load_relic_jit_actor(policy_pt: str):
    """Load the TorchScript actor from the external RELIC checkout.

    policy.pt is a compiled export (no optimizer state): usable for
    inference and as the G1 starting point, not as a training resume.
    """
    torch = _torch()
    mod = torch.jit.load(policy_pt, map_location="cpu")
    mod.eval()
    return mod


def jit_actor_forward(mod, obs: np.ndarray) -> np.ndarray:
    """Run a loaded JIT actor on (N,84) float32; returns (N,12)."""
    torch = _torch()
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != S.R84_DIM or not np.isfinite(x).all():
        raise ValueError(f"obs must be finite (N,{S.R84_DIM})")
    with torch.no_grad():
        y = mod(torch.from_numpy(x))
    return np.asarray(y, dtype=np.float32)


def build_critic():
    """Critic 463 -> V with DeepSets fruit pool (spec §3.4)."""
    torch = _torch()
    nn = torch.nn

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fruit = nn.Sequential(
                nn.Linear(S.FRUIT_TRUTH_DIM, 64), nn.ELU(),
                nn.Linear(64, 128), nn.ELU())
            self.value = nn.Sequential(
                nn.Linear(S.CRITIC_INPUT_DIM, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 1))

        def forward(self, base, fruit, fruit_valid):
            f = self.fruit(fruit)                      # (B,F,128)
            mask = fruit_valid.unsqueeze(-1)           # (B,F,1)
            masked = f * mask
            mean = masked.sum(1) / mask.sum(1).clamp_min(1.0)
            mx = masked.masked_fill(~fruit_valid.bool().unsqueeze(-1),
                                    -1e9).amax(1)
            mx = torch.where(mask.sum(1) > 0, mx, torch.zeros_like(mx))
            pool = torch.cat([mean, mx], -1)           # (B,256)
            return self.value(torch.cat([base, pool], -1)).squeeze(-1)
    return Critic()
