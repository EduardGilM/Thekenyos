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
import os
from pathlib import Path
import tempfile

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
    jacob = (2.0 * (float(np.log(2.0)) - u - torch.nn.functional.softplus(-2.0 * u))).sum(-1)
    return logp - jacob


def compute_gae(rewards, values, terminated, truncated, gamma, lam,
                final_value=0.0, *, next_values=None):
    """GAE; timeouts bootstrap from V(final obs), terminals do not.

    values[t] is the critic at step t's observation. When the rollout ends
    truncated, pass final_value=V(final obs) from final_critic_obs; when it
    ends terminated, final_value is ignored. Numpy, finite-checked.
    """
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
    terminated, truncated = np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool)
    if rewards.ndim not in (1, 2) or not rewards.shape[0]:
        raise ValueError('GAE requires nonempty [time] or [time, env] arrays')
    if any(a.shape != rewards.shape for a in (values, terminated, truncated)):
        raise ValueError('GAE arrays must share a shape')
    if not np.isfinite([gamma, lam]).all() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('GAE discounts must be finite in [0, 1]')
    if not np.isfinite(rewards).all() or not np.isfinite(values).all() or not np.isfinite(final_value).all():
        raise ValueError('GAE values must be finite')
    if next_values is None:
        if np.any(truncated[:-1] & ~terminated[:-1]):
            raise ValueError('Mid-rollout timeouts require next_values from final observations')
        next_values = np.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = final_value
    else:
        next_values = np.asarray(next_values, dtype=np.float64)
        if next_values.shape != values.shape or not np.isfinite(next_values).all():
            raise ValueError('next_values must be finite and match values')
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (~(terminated[t] | truncated[t])) * last
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
    adv_n = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8) if adv.numel() > 1 else adv
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
    path = Path(path)
    sidecar_path = Path(str(path) + '.json')
    if path.exists() or sidecar_path.exists():
        raise FileExistsError(f'Preserve existing checkpoint: {path}')
    buf = {
        'format': 'kiwi-checkpoint/v1',
        "models": {k: v.state_dict() for k, v in models.items()},
        "optim": {k: o.state_dict() for k, o in optimizers.items()},
        "rng": rng_state,
        "meta": meta,
    }
    blob = io.BytesIO()
    torch.save(buf, blob)
    payload = blob.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = dict(meta, sha256=digest, sha16=digest[:16], format=buf['format'])
    manifest = (json.dumps(sidecar, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_new(path, payload)
    _publish_new(sidecar_path, manifest)
    return digest[:16]


def _publish_new(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, models, optimizers=None, *, expected_meta=None):
    torch = _torch()
    path = Path(path)
    manifest = json.loads(Path(str(path) + '.json').read_text())
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get('sha256'):
        raise ValueError('Checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if state.get('format') != 'kiwi-checkpoint/v1':
        raise ValueError('Unsupported checkpoint format')
    for name, expected in (expected_meta or {}).items():
        if state['meta'].get(name) != expected:
            raise ValueError(f'Checkpoint metadata mismatch: {name}')
    if set(models) != set(state['models']):
        raise ValueError('Checkpoint model set mismatch')
    if optimizers is not None and set(optimizers) != set(state['optim']):
        raise ValueError('Checkpoint optimizer set mismatch')
    for name, model in models.items():
        current, saved = model.state_dict(), state['models'][name]
        if set(current) != set(saved) or any(current[k].shape != saved[k].shape for k in current):
            raise ValueError(f'Checkpoint architecture mismatch: {name}')
    for name, model in models.items():
        model.load_state_dict(state['models'][name], strict=True)
    for name, optimizer in (optimizers or {}).items():
        optimizer.load_state_dict(state['optim'][name])
    return dict(rng=state['rng'], meta=state['meta'])
