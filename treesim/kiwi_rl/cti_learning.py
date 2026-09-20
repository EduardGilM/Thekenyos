"""V-trace auxiliary learning from every valid conditional-Gaussian branch.

Uses IMPALA's clipped importance recursion (Espeholt et al., 2018). Stored
pre-action GRU states are detached: this is a truncated recurrent approximation,
not backpropagation through the branch or a reconstruction of current-policy
history. Proposal densities must be conditional on the actual sampling history.
"""
from copy import deepcopy
import math
import time

import torch
import torch.nn.functional as F


def normal_log_prob(raw, mean, logstd):
    """Unsquashed density; the common tanh Jacobian cancels in importance ratios."""
    return (-.5 * ((raw - mean) * (-logstd).exp()).square() - logstd -
            .5 * math.log(2 * math.pi)).sum(-1)


@torch.no_grad()
def vtrace_targets(rewards, values, bootstrap, log_rhos, discounts, mask):
    """Return detached (value targets, policy advantages), all [time, world].

    ``discounts`` is gamma for nonterminal transitions, zero for terminals.
    ``bootstrap`` is the value immediately after each world's last valid step,
    including truncations. Masks must be valid prefixes, with no reset episode
    concatenated in the same column. Padded data is ignored, even if nonfinite.
    """
    if rewards.ndim != 2 or any(x.shape != rewards.shape for x in (values, log_rhos, discounts, mask)):
        raise ValueError('V-trace inputs must share [time, world] shape')
    if bootstrap.shape != rewards.shape[1:]:
        raise ValueError('V-trace bootstrap must have one value per world')
    mask = mask.bool()
    if bool((mask[1:] & ~mask[:-1]).any()):
        raise ValueError('V-trace masks must be valid prefixes; split reset episodes')
    for x in (rewards, values, log_rhos, discounts):
        if not bool(torch.isfinite(x[mask]).all()):
            raise ValueError('Nonfinite valid V-trace inputs')
    if bool(((discounts[mask] < 0) | (discounts[mask] > 1)).any()):
        raise ValueError('V-trace discounts must be in [0, 1]')
    # A terminal needs no bootstrap, so even its absent/nonfinite final value is
    # harmless. A nonterminal valid endpoint must have a finite bootstrap.
    next_valid = torch.cat((mask[1:], torch.zeros_like(mask[:1])))
    needs_bootstrap = (mask & ~next_valid & (discounts != 0)).any(0)
    if not bool(torch.isfinite(bootstrap[needs_bootstrap]).all()):
        raise ValueError('Nonfinite nonterminal bootstrap')
    bootstrap = torch.where(needs_bootstrap, bootstrap, 0.)
    vs = torch.zeros_like(values)
    pg = torch.zeros_like(values)
    next_value, next_target = bootstrap, bootstrap
    for tick in reversed(range(len(rewards))):
        valid = mask[tick]
        value = torch.where(valid, values[tick], 0.)
        reward = torch.where(valid, rewards[tick], 0.)
        discount = torch.where(valid, discounts[tick], 0.)
        # exp(min(log rho,0)) is exactly min(rho,1), without overflow.
        weight = torch.where(valid, log_rhos[tick], -torch.inf).clamp_max(0).exp()
        td = reward + discount * next_value - value
        target = value + weight * td + discount * weight * (next_target - next_value)
        advantage = weight * (reward + discount * next_target - value)
        vs[tick] = torch.where(valid, target, 0.)
        pg[tick] = torch.where(valid, advantage, 0.)
        next_value = torch.where(valid, value, bootstrap)
        next_target = torch.where(valid, target, bootstrap)
    return vs, pg


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _anchors(policy, rows):
    """Freeze factual anchor states, accepting collector rows or explicit memory."""
    if not rows:
        raise ValueError('Factual anchors must not be empty')
    fixed = []
    memory = rows[0].get('initial_memory', rows[0].get('memory'))
    if memory is None:
        raise ValueError('Anchor rows require initial_memory or stored memory')
    with torch.no_grad():
        for row in rows:
            memory = row.get('memory', memory).detach()
            if 'reset' in row:
                memory = memory * (~row['reset'])[:, None]
            mask = row.get('mask', torch.ones(len(memory), dtype=torch.bool, device=memory.device)).bool()
            # Invalid branch padding is never fed into the model.
            priv = torch.where(mask[:, None], row['privileged'], 0.).detach()
            obs = torch.where(mask[:, None], row['r84'], 0.).detach()
            memory = torch.where(mask[:, None], memory, 0.)
            mean, logstd, _, next_memory = policy(priv, obs, memory)
            if bool(mask.any()):
                fixed.append((priv[mask], obs[mask], memory[mask], mean[mask].detach(),
                              logstd.expand_as(mean)[mask].detach()))
            memory = next_memory.detach()
    if not fixed or not _finite(fixed):
        raise ValueError('Factual anchor states and policy outputs must be finite and nonempty')
    return fixed


@torch.no_grad()
def _anchor_kl(policy, fixed):
    total, count = 0., 0
    for priv, obs, memory, old_mean, old_logstd in fixed:
        mean, logstd, _, _ = policy(priv, obs, memory)
        kl = (logstd-old_logstd + .5*((old_logstd-logstd).mul(2).exp() +
              (old_mean-mean).square()*(-2*logstd).exp()) - .5).sum(-1)
        total = total + kl.sum()
        count += len(mean)
    return float((total/count).item())


def update_branch_cti(policy, optimizer, rows, bootstrap, coef=.1, anchor_rows=None,
                      gamma=.999, epochs=2, minibatch_worlds=64):
    """Train actor and critic on valid branches with clipped V-trace targets.

    The separate supplied optimizer never touches PPO's Adam state. Targets and
    advantages are frozen before all epochs/attempts. Gaussian anchor KL is the
    mean KL(old || new), capped at .01. Three attempts use the original LR,
    one quarter, then one sixteenth; each retry restores parameters and Adam.
    Accepted steps restore the configured LR for the next call.
    """
    started = time.perf_counter()
    if (not math.isfinite(coef) or coef < 0 or not 0 < gamma <= 1 or
            not isinstance(epochs, int) or epochs < 1 or
            not isinstance(minibatch_worlds, int) or minibatch_worlds < 1):
        raise ValueError('Invalid CTI learning limits')
    rows = list(rows)
    metrics = dict(valid_transitions=0, physical_failures=0, effective_weight=0.,
        importance_underflow=0, actor_loss=0., value_loss=0., updated=False, epochs=0,
        kl=0., rejected_update=False, nonfinite_update=False, optimizer_steps=0,
        attempted_optimizer_steps=0, learning_seconds=0., attempts=0, lr_scale=0.)
    def finish():
        metrics['learning_seconds'] = time.perf_counter()-started
        return metrics
    if not rows:
        return finish()
    mask = torch.stack([r['mask'].detach().bool() for r in rows])
    valid_count = int(mask.sum().item())
    metrics['valid_transitions'] = valid_count
    metrics['physical_failures'] = sum(int((r.get('physical_failure',
        r.get('failed', torch.zeros_like(r['mask']))).bool() & r['mask'].bool()).sum().item()) for r in rows)
    if not valid_count or not coef:
        return finish()
    keys = ('privileged', 'r84', 'memory', 'raw_action', 'behavior_log_prob', 'reward', 'terminated', 'truncated')
    batch = {key: torch.stack([r[key].detach() for r in rows]) for key in keys}
    if any(not _finite(batch[key][mask]) for key in keys):
        metrics.update(rejected_update=True, nonfinite_update=True)
        return finish()
    worlds = mask.shape[1]
    if bool((mask[1:] & (~mask[:-1] | batch['terminated'][:-1].bool() | batch['truncated'][:-1].bool())).any()):
        raise ValueError('A branch column must end after termination/truncation; split episodes')
    # Prefix masks permit a different final state/bootstrap for every world.
    discounts = gamma * (~batch['terminated'].bool()).to(batch['reward'].dtype)
    values = torch.zeros_like(batch['reward'])
    target_logp = torch.zeros_like(values)
    with torch.no_grad():
        for start in range(0, worlds, minibatch_worlds):
            select = mask[:, start:start+minibatch_worlds]
            if not bool(select.any()):
                continue
            inputs = [batch[k][:, start:start+minibatch_worlds][select] for k in ('privileged', 'r84', 'memory')]
            mean, logstd, value, _ = policy(*inputs)
            values[:, start:start+minibatch_worlds][select] = value
            target_logp[:, start:start+minibatch_worlds][select] = normal_log_prob(
                batch['raw_action'][:, start:start+minibatch_worlds][select], mean, logstd)
    log_rhos = target_logp - batch['behavior_log_prob']
    if not _finite(values[mask]) or not _finite(log_rhos[mask]):
        metrics.update(rejected_update=True, nonfinite_update=True)
        return finish()
    targets, advantages = vtrace_targets(batch['reward'], values, bootstrap.detach(), log_rhos, discounts, mask)
    weights = log_rhos[mask].clamp_max(0).exp()
    metrics['effective_weight'] = float(weights.mean().item())
    metrics['importance_underflow'] = int((weights == 0).sum().item())
    # Scale only, preserving the sign of failed or below-baseline outcomes.
    advantages = advantages / advantages[mask].std(unbiased=False).clamp_min(1.)
    fixed = _anchors(policy, list(anchor_rows) if anchor_rows is not None else rows)
    policy_before, optimizer_before = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    # Keep whole time sequences together; only the world order is shuffled.
    orders = [torch.randperm(worlds, device=mask.device) for _ in range(epochs)]
    for attempt in range(3):
        if attempt:
            policy.load_state_dict(policy_before)
            optimizer.load_state_dict(deepcopy(optimizer_before))
        scale = .25**attempt
        for group, lr in zip(optimizer.param_groups, initial_lrs):
            group['lr'] = lr*scale
        steps, count, actor_sum, value_sum = 0, 0, 0., 0.
        finite = True
        metrics['attempts'] = attempt+1
        for order in orders:
            for start in range(0, worlds, minibatch_worlds):
                ids = order[start:start+minibatch_worlds]
                selected = mask[:, ids]
                samples = int(selected.sum().item())
                if not samples:
                    continue
                inputs = [batch[k][:, ids][selected] for k in ('privileged', 'r84', 'memory')]
                mean, logstd, value, _ = policy(*inputs)
                logp = normal_log_prob(batch['raw_action'][:, ids][selected], mean, logstd)
                actor_loss = -(logp*advantages[:, ids][selected]).mean()
                value_loss = F.smooth_l1_loss(value, targets[:, ids][selected])
                # Match PPO: exploration is measured after bounded controls.
                from .ppo import tanh_logprob
                draw = mean + logstd.exp()*torch.randn_like(mean)
                entropy = (-tanh_logprob(draw, mean, logstd) * log_rhos[:,ids][selected].clamp_max(0).exp()).mean()
                loss = coef*(actor_loss+.5*value_loss-.001*entropy)
                optimizer.zero_grad(set_to_none=True)
                if not bool(torch.isfinite(loss)):
                    finite = False
                    break
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                if not bool(torch.isfinite(norm)):
                    finite = False
                    break
                optimizer.step()
                steps += 1
                metrics['attempted_optimizer_steps'] += 1
                actor_sum += float(actor_loss.detach())*samples
                value_sum += float(value_loss.detach())*samples
                count += samples
            if not finite:
                break
        finite = finite and _finite(policy.state_dict()) and _finite(optimizer.state_dict())
        kl = _anchor_kl(policy, fixed) if finite else float('nan')
        finite = finite and math.isfinite(kl)
        metrics['kl'] = max(0., kl) if math.isfinite(kl) else None
        metrics['nonfinite_update'] |= not finite
        if finite and kl <= .01:
            for group, lr in zip(optimizer.param_groups, initial_lrs):
                group['lr'] = lr
            metrics.update(updated=steps > 0, epochs=epochs, optimizer_steps=steps,
                actor_loss=actor_sum/max(1,count), value_loss=value_sum/max(1,count), lr_scale=scale)
            optimizer.zero_grad(set_to_none=True)
            return finish()
    policy.load_state_dict(policy_before)
    optimizer.load_state_dict(deepcopy(optimizer_before))
    optimizer.zero_grad(set_to_none=True)
    metrics['rejected_update'] = True
    # 'kl' is the last rejected attempted KL; retained parameters are unchanged.
    return finish()
