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


def _action_dim_mask(u, dim_mask):
    """Broadcast a [dim] or matching mask onto the last axis of ``u``."""
    torch = _torch()
    if dim_mask is None:
        return None
    mask = dim_mask if torch.is_tensor(dim_mask) else torch.as_tensor(dim_mask, device=u.device)
    mask = mask.to(dtype=u.dtype, device=u.device)
    if mask.shape != u.shape[-1:] and mask.shape != u.shape:
        raise ValueError('dim_mask must be [action_dim] or match the pre-tanh sample')
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('dim_mask must be finite in [0, 1]')
    return mask


def tanh_logprob(u, mu, logstd, dim_mask=None) -> "tensor":
    """Log-prob of a=tanh(u) under N(mu, sigma) with Jacobian correction.

    ``dim_mask`` zeros idle action dimensions (speedrun: N3 while the chassis
    is held). Unmasked behaviour matches the transformed-Gaussian reference.
    """
    torch = _torch()
    ls = torch.clamp(logstd, -5.0, 1.0)
    var = torch.exp(2.0 * ls)
    logp = -0.5 * (((u - mu) ** 2) / var + 2.0 * ls + float(np.log(2 * np.pi)))
    jacob = 2.0 * (float(np.log(2.0)) - u - torch.nn.functional.softplus(-2.0 * u))
    total = logp - jacob
    mask = _action_dim_mask(u, dim_mask)
    if mask is None:
        return total.sum(-1)
    return (total * mask).sum(-1)


def gaussian_entropy(logstd, dim_mask=None):
    """Analytic differential entropy of N(μ, σ), nats. Peaked σ can make H < 0."""
    torch = _torch()
    ls = torch.clamp(logstd, -5.0, 1.0)
    entropy = 0.5 * (1.0 + float(np.log(2.0 * np.pi))) + ls
    mask = _action_dim_mask(entropy, dim_mask)
    if mask is None:
        return entropy.sum(-1)
    return (entropy * mask).sum(-1)


def tanh_gaussian_entropy(logstd, *, raw, mu, dim_mask=None):
    """Monte-Carlo differential entropy of a=tanh(u), u~N(μ, σ).

    This is −log π(a) for a freshly drawn pre-tanh sample. It is not Shannon
    entropy of a discrete action and is allowed to be negative when the
    squashed Gaussian is peaked (small σ).
    """
    return -tanh_logprob(raw, mu, logstd, dim_mask=dim_mask)


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


def compute_gae_torch(rewards, values, next_values, terminated, truncated, gamma, lam):
    torch = _torch()
    if rewards.ndim != 2 or not rewards.shape[0] or any(x.shape != rewards.shape for x in (values, next_values, terminated, truncated)):
        raise ValueError('GAE tensors must share nonempty [time, env] shape')
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError('GAE boundary masks must be boolean')
    if not np.isfinite([gamma, lam]).all() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('Invalid GAE discounts')
    if not all(torch.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError('Nonfinite GAE tensors')
    with torch.no_grad():
        advantages = torch.zeros_like(values)
        carry = torch.zeros_like(values[0])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
            carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
            advantages[t] = carry
    return advantages


def sample_hybrid(output, head, *, deterministic=False, generator=None):
    torch = _torch()
    if head not in ('n', 'm'):
        raise ValueError('Unknown policy head')
    mu, logstd, logits = output[f'{head}_mu'], output[f'{head}_logstd'].clamp(-5., 1.), output[f'{head}_events']
    if mu.ndim != 2 or logits.ndim != 2 or mu.shape[0] != logits.shape[0] or not len(mu):
        raise ValueError('Expected nonempty batched hybrid outputs')
    if not all(torch.isfinite(value).all() for value in (mu, logstd, logits)):
        raise RuntimeError('Nonfinite policy distribution')
    with torch.no_grad():
        raw = mu if deterministic else mu + logstd.exp() * torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        event = logits.argmax(-1) if deterministic else torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
        logp = tanh_logprob(raw, mu, logstd) + logits.log_softmax(-1).gather(-1, event[:, None]).squeeze(-1)
    return dict(action=raw.detach().tanh(), raw=raw.detach(), event=event.detach(), log_prob=logp.detach())


def unroll_policy(model, observations, initial_memory, resets, burn_in=0):
    from contextlib import nullcontext
    torch = _torch()
    mutable = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm,
               torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    if any(isinstance(module, mutable) and module.training for module in model.modules()):
        raise ValueError('Use evaluation mode for batch-normalized or dropout policies during recurrent PPO')
    if resets.ndim != 2 or resets.dtype != torch.bool or not 0 <= burn_in < len(resets):
        raise ValueError('Invalid recurrent sequence boundaries')
    if not observations or any(value.shape[:2] != resets.shape for value in observations.values()):
        raise ValueError('Observations must share [time, env] sequence dimensions')
    memory = initial_memory.detach()
    outputs = []
    for t in range(len(resets)):
        with torch.no_grad() if t < burn_in else nullcontext():
            output = model(**{name: value[t] for name, value in observations.items()}, memory=memory, reset=resets[t])
        memory = output['memory']
        if t >= burn_in:
            outputs.append(output)
    return {name: torch.stack([output[name] for output in outputs]) for name in outputs[0]}


def hybrid_actor_loss(output, head, samples, *, clip=.2, entropy_coef=.001):
    torch = _torch()
    mask = samples['active']
    if mask.dtype != torch.bool or mask.shape != output[f'{head}_mu'].shape[:-1]:
        raise ValueError('Actor mask must match sequence dimensions')
    if not np.isfinite([clip, entropy_coef]).all() or not 0 < clip < 1 or entropy_coef < 0:
        raise ValueError('Invalid PPO coefficients')
    count = int(mask.sum().item())
    if not count:
        return dict(loss=output[f'{head}_mu'].new_zeros(()), samples=0, kl=0., entropy=0.)
    mu, logits = output[f'{head}_mu'][mask], output[f'{head}_events'][mask]
    std = output[f'{head}_logstd']
    std = std[:, None, :].expand(*mask.shape, std.shape[-1])[mask]
    raw, event = samples['raw'][mask].detach(), samples['event'][mask].detach()
    new_logp = tanh_logprob(raw, mu, std) + logits.log_softmax(-1).gather(-1, event[:, None]).squeeze(-1)
    logratio = new_logp - samples['log_prob'][mask].detach()
    ratio = logratio.exp()
    advantage = samples['advantage'][mask].detach()
    if count > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1. - clip, 1. + clip) * advantage)
    draw = mu + std.exp() * torch.randn_like(mu)
    entropy = -tanh_logprob(draw, mu, std) - (logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    loss = -surrogate.mean() - entropy_coef * entropy.mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite recurrent PPO loss')
    return dict(loss=loss, samples=count, kl=float(((ratio - 1.) - logratio).mean().detach().clamp_min(0.)),
                entropy=float(entropy.mean().detach()))


def recurrent_actor_step(model, optimizer, batch, *, clip=.2, entropy_coef=.001, target_kl=.02, max_grad_norm=.5):
    torch = _torch()
    if batch.get('kind') != 'factual_on_policy':
        raise ValueError('Counterfactual or off-policy branches are not PPO samples')
    if not np.isfinite([target_kl, max_grad_norm]).all() or target_kl <= 0 or max_grad_norm <= 0:
        raise ValueError('Invalid optimizer safeguards')
    burn_in = batch.get('burn_in', 0)
    optimizer.zero_grad(set_to_none=True)
    outputs = unroll_policy(model, batch['observations'], batch['initial_memory'], batch['resets'], burn_in)
    losses = {head: hybrid_actor_loss(outputs, head, {key: value[burn_in:] for key, value in batch[head].items()},
                                    clip=clip, entropy_coef=entropy_coef) for head in ('n', 'm')}
    kl = max(loss['kl'] for loss in losses.values())
    samples = sum(loss['samples'] for loss in losses.values())
    if not samples or kl > target_kl:
        return dict(updated=False, samples=samples, kl=kl, reason='no active actions' if not samples else 'KL limit')
    loss = losses['n']['loss'] + losses['m']['loss']
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return dict(updated=True, samples=samples, kl=kl, loss=float(loss.detach()), grad_norm=float(norm))


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
                    meta: dict, *, replace=False):
    """Atomic checkpoint: state_dicts + RNG + schema/config hashes."""
    torch = _torch()
    path = Path(path)
    sidecar_path = Path(str(path) + '.json')
    if replace:
        for existing in (path, sidecar_path):
            if existing.exists():
                existing.unlink()
    elif path.exists() or sidecar_path.exists():
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
