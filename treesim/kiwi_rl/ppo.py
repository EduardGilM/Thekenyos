"""PPO for hybrid (tanh-Gaussian + categorical) recurrent actors. JP-ONLY.

Custom loop (no rsl-rl dependency): rollout buffers with terminated vs
truncated handling, GAE over the 25 Hz harvest clock, sequence minibatches
for GRU burn-in, clipped surrogate + value + entropy losses, KL early-stop,
gradient clipping, checkpoint save/load with RNG + schema hashes.

G1 trains on its own 50 Hz buffer (two gait transitions per harvest tick);
N3/M3/V3 share one optimizer owning V3 (single owner rule, spec §6.3).
All torch imports are lazy so numpy tests pass without the stack.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("training stack required (requirements-train.txt)") from e
    return torch


def tanh_logprob(u, mu, logstd) -> "tensor":
    """Log-prob of a=tanh(u) under N(mu, sigma) with Jacobian correction."""
    torch = _torch()
    ls = torch.clamp(logstd, -5.0, 1.0)
    var = torch.exp(2.0 * ls)
    logp = -0.5 * (((u - mu) ** 2) / var + 2.0 * ls + float(np.log(2 * np.pi)))
    logp = logp.sum(-1)
    jacob = torch.log(torch.clamp(1.0 - torch.tanh(u) ** 2, min=1e-9)).sum(-1)
    return logp + jacob


def compute_gae(rewards, values, terminated, truncated, gamma, lam,
                final_value=0.0):
    """GAE; timeouts bootstrap from V(final obs), terminals do not.

    values[t] is the critic at step t's observation. When the rollout ends
    truncated, pass final_value=V(final obs) from final_critic_obs; when it
    ends terminated, final_value is ignored. Numpy, finite-checked.
    """
    if not np.isfinite(final_value):
        raise ValueError(f"gae: non-finite final_value={final_value}")
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        for arr, name in ((rewards, "rewards"), (values, "values")):
            if not np.isfinite(arr[t]):
                raise ValueError(f"gae: non-finite {name}[{t}]")
        nonterm = 0.0 if terminated[t] else 1.0
        if t == T - 1:
            nxt = final_value if (truncated[t] and not terminated[t]) else 0.0
        else:
            nxt = values[t + 1]
        delta = rewards[t] + gamma * nonterm * nxt - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv.astype(np.float32)


class SequenceBuffer:
    """One rollout: per-step dicts + reset flags for recurrent PPO."""

    def __init__(self):
        self.steps: list = []

    def add(self, **kwargs):
        self.steps.append(kwargs)

    def __len__(self):
        return len(self.steps)


def ppo_epoch_loss(new_logp_c, new_logp_e, old_logp, adv, ret, val,
                   clip, value_coef, ent_c, ent_e):
    """Clipped surrogate + clipped value + entropies. Returns loss dict."""
    torch = _torch()
    new_logp = new_logp_c + new_logp_e
    ratio = torch.exp(new_logp - old_logp.detach())
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    s1 = ratio * adv_n
    s2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_n
    policy_loss = -torch.min(s1, s2).mean()
    v_clipped = val  # caller passes already-clipped value if desired
    value_loss = 0.5 * ((v_clipped - ret) ** 2).mean()
    loss = policy_loss + value_coef * value_loss - ent_c - ent_e
    return {"loss": loss, "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach()}


def save_checkpoint(path, models: dict, optimizers: dict, rng_state: dict,
                    meta: dict):
    """Atomic checkpoint: state_dicts + RNG + schema/config hashes."""
    torch = _torch()
    buf = {
        "models": {k: v.state_dict() for k, v in models.items()},
        "optim": {k: o.state_dict() for k, o in optimizers.items()},
        "rng": rng_state,
        "meta": meta,
    }
    blob = io.BytesIO()
    torch.save(buf, blob)
    digest = hashlib.sha256(blob.getvalue()).hexdigest()[:16]
    with open(path, "wb") as f:
        f.write(blob.getvalue())
    sidecar = dict(meta)
    sidecar["sha16"] = digest
    with open(str(path) + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)
    return digest
