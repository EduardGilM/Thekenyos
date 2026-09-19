"""Teacher-only retained-progress search; its examples never enter PPO rows."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import time

import torch

from .counterfactual import capture, restore
from .harvest_training import EpisodeProgress, signals, HARVEST_GAMMA
from .fast_teacher import privileged_observation


def _progress_over(candidate, reference, approach_gain=.015):
    """Current retained outcomes only: historical contact is not a grasp."""
    return ((candidate['success'] & ~reference['success']) |
            (candidate['holding'] & ~reference['holding']) |
            (candidate['holding'] & candidate['detached'] & ~reference['detached']) |
            (~candidate['detached'] & ~reference['detached'] &
             (candidate['distance'] <= reference['distance'] - approach_gain)) |
            (candidate['holding'] & candidate['detached'] & reference['holding'] &
             reference['detached'] &
             (candidate['basket_distance'] <= reference['basket_distance'] - approach_gain)))


def _choose_winners(outcomes, root, *, minimum_gain=.1, approach_gain=.015):
    """Require a safe, retained gain over both the matched baseline and root."""
    baseline = outcomes[0]
    gains, reasons = [], []
    for index, result in enumerate(outcomes):
        reason = [[] for _ in range(len(result['score']))]
        checks = {
            'baseline': torch.full_like(result['success'], index == 0),
            'failure': result['failed'] | result['numerical'],
            'unheld_detachment_or_lost_fruit': result['lost_fruit'] & ~result['success'],
            'incomplete_retention': ~result['retained'] & ~result['success'],
            'no_retained_progress': ~(_progress_over(result, root, approach_gain) &
                                      _progress_over(result, baseline, approach_gain)),
            'no_root_potential_gain': (result['retained_potential'] <= EpisodeProgress.potential(root)) &
                                      ~result['success'],
            'insufficient_score_gain': result['score'] - baseline['score'] < minimum_gain,
            'nonfinite_score': ~torch.isfinite(result['score']) | ~torch.isfinite(baseline['score']),
            'unmatched_numerical_baseline': baseline['numerical'],
            'inactive': ~result['triggered'],
        }
        eligible = torch.ones_like(result['success'])
        for name, reject in checks.items():
            eligible &= ~reject
            for world in reject.nonzero(as_tuple=False).flatten().tolist():
                reason[world].append(name)
        gains.append(torch.where(eligible, result['score'] - baseline['score'], -torch.inf))
        reasons.append(reason)
    best_gain, winner = torch.stack(gains).max(dim=0)
    return torch.where(torch.isfinite(best_gain), winner, -1), reasons


class SelectiveCTI:
    """Bounded skill search on an independent runtime, followed by confirmation."""
    def __init__(self, runtime, gait, *, stall_seconds=4., max_episode_seconds=30.,
                 minimum_gain=.1, perturb_steps=None, root_spacing=20,
                 horizon_seconds=4., intervention_seconds=2., retention_seconds=.5,
                 approach_gain=.015):
        if runtime.worlds > 64:
            raise ValueError('CTI runtime must use at most 64 worlds')
        values = (stall_seconds, max_episode_seconds, horizon_seconds,
                  intervention_seconds, retention_seconds, approach_gain)
        if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isfinite(minimum_gain) or minimum_gain < 0:
            raise ValueError('invalid CTI limits or improvement threshold')
        self.runtime, self.gait, self.rounds = runtime, gait, 0
        self.stall_seconds, self.max_episode_seconds = stall_seconds, max_episode_seconds
        self.minimum_gain, self.approach_gain = minimum_gain, approach_gain
        self.horizon_steps = max(1, round(horizon_seconds / runtime.control_dt))
        self.perturb_steps = (max(1, round(intervention_seconds / runtime.control_dt))
                              if perturb_steps is None else int(perturb_steps))
        self.retention_steps = max(1, round(retention_seconds / runtime.control_dt))
        self.root_spacing = int(root_spacing)
        # Short explicit configurations are for tests. Production defaults leave
        # two seconds of original-policy continuation after intervention.
        if self.perturb_steps < 1 or self.root_spacing < 1 or self.horizon_steps < self.perturb_steps + self.retention_steps:
            raise ValueError('horizon must include intervention and a retention continuation')

    def _snapshot(self, memory, progress, task_return):
        return capture(self.runtime, application_state={
            'memory': memory, 'progress': progress, 'task_return': task_return})

    def _restore_app(self, snapshot):
        app = restore(self.runtime, snapshot)
        return app['memory'], app['progress'], app['task_return']

    @staticmethod
    def _event_mask(rt, progress, now, previous_holding):
        return (now['failed'] | (now['detached'] & ~now['holding']) |
                (previous_holding & ~now['holding']) |
                (progress.stale >= max(1, progress.stall_steps - 20)))

    def _act(self, policy, memory, *, explore=False, intervention=None):
        rt = self.runtime
        obs = rt.observe().detach().clone()
        priv = privileged_observation(rt, obs).detach().clone()
        memory_in = memory.detach().clone()
        mean, logstd, _, next_memory = policy(priv, obs, memory)
        raw = mean + logstd.exp() * torch.randn_like(mean) if explore else mean
        action = raw.tanh()
        if intervention is not None:
            kind, alternative = intervention
            if kind == 'hold_arm_close_jaw':
                action[:, :6] = 0.  # Hold joint targets; this does not freeze physics.
            elif kind == 'coherent_alternative':
                action[:, :6] = alternative[:, :6]
            action[:, 6] = 1.  # Positive velocity closes the jaw from -1 toward 0.
        rt.set_gait_actions(self.gait(obs).detach())
        return obs, priv, memory_in, action, next_memory.detach()

    def _branch(self, policy, root, trigger, intervention, noise_seed=None):
        rt = self.runtime
        memory, progress, _ = self._restore_app(root)
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
        initial = signals(rt)
        active = trigger.clone()
        result = {key: torch.zeros_like(initial['distance']) for key in
                  ('score', 'task_return', 'leaf', 'retained_potential', 'distance',
                   'basket_distance', 'hold_fraction', 'steps')}
        result.update({key: torch.zeros_like(trigger) for key in
                       ('holding', 'detached', 'success', 'failed', 'stalled', 'truncated',
                        'horizon', 'numerical', 'lost_fruit', 'retained')})
        result['triggered'] = trigger.clone()
        window, segment = deque(maxlen=self.retention_steps), []
        transitions = 0
        for tick in range(self.horizon_steps):
            if not bool(active.any()):
                break
            obs, priv, memory_in, action, memory = self._act(
                policy, memory, explore=True,
                intervention=intervention if tick < self.perturb_steps else None)
            if tick < self.perturb_steps:
                segment.append(dict(privileged=priv.clone(), r84=obs.clone(),
                                    memory=memory_in.clone(), action=action.detach().clone(),
                                    mask=active.clone()))
            _, _, done, _ = rt.step(action)
            transitions += rt.worlds
            try:
                rt.check()
            except RuntimeError as error:
                if 'GPU numerical failure' not in str(error):
                    raise
                result['numerical'] |= active
                result['score'][active] = -torch.inf
                result['numerical_error'] = str(error)
                break
            now = signals(rt)
            _, terminated, truncated, stalled = progress.step(now, done)
            result['task_return'][active] += HARVEST_GAMMA ** tick * progress.task_reward[active]
            result['lost_fruit'] |= active & ~now['success'] & now['detached'] & ~now['holding']
            window.append({key: value.clone() for key, value in now.items()})
            finish = active & (terminated | truncated | (tick + 1 == self.horizon_steps))
            if bool(finish.any()):
                potential = torch.stack([EpisodeProgress.potential(s) for s in window]).amin(dim=0)
                result['retained_potential'][finish] = potential[finish]
                leaf = torch.where(terminated, 0., HARVEST_GAMMA ** (tick + 1) * potential)
                result['leaf'][finish] = leaf[finish]
                result['score'][finish] = (result['task_return'] + leaf)[finish]
                for key in ('distance', 'basket_distance'):
                    result[key][finish] = torch.stack([s[key] for s in window]).amax(dim=0)[finish]
                holds = torch.stack([s['holding'] for s in window])
                result['holding'][finish] = holds.all(dim=0)[finish]
                result['hold_fraction'][finish] = holds.float().mean(dim=0)[finish]
                result['detached'][finish] = torch.stack([s['detached'] for s in window]).all(dim=0)[finish]
                result['success'][finish] = now['success'][finish]
                result['failed'][finish] = (now['failed'] | (done & ~now['success']))[finish]
                result['stalled'][finish] = stalled[finish]
                result['truncated'][finish] = truncated[finish]
                result['horizon'][finish] = (~terminated & ~truncated)[finish]
                continuation = min(max(1, round(1. / rt.control_dt)),
                                   self.horizon_steps - self.perturb_steps)
                result['retained'][finish] = (len(window) == self.retention_steps and
                                             tick + 1 >= self.perturb_steps + continuation)
                result['steps'][finish] = tick + 1
            active &= ~finish
            # Ended worlds must not accumulate physics failures during another
            # world's continuation. Their stored outcomes and labels are frozen.
            if bool(finish.any()) and bool(active.any()):
                rt.reset(finish.to(dtype=torch.uint8))
                progress.reset(finish, signals(rt))
                memory[finish] = 0
        return result, segment, transitions

    @staticmethod
    def _records(outcomes, reasons, names, pass_name):
        records = []
        for candidate, outcome in enumerate(outcomes):
            for world in range(len(outcome['score'])):
                row = dict(pass_name=pass_name, candidate=candidate, skill=names[candidate],
                           world=world, eligible=not reasons[candidate][world],
                           rejection_reasons=reasons[candidate][world])
                for key, value in outcome.items():
                    item = value[world].item() if isinstance(value, torch.Tensor) else value
                    row[key] = None if isinstance(item, float) and not math.isfinite(item) else item
                records.append(row)
        return records

    def _episode(self, policy, root, trigger_mask, prefix_return, candidates):
        # Prefix is identical in matched branches; root-relative discount keeps
        # gains comparable across early and late decisions.
        del prefix_return
        self._restore_app(root)
        initial = signals(self.runtime)
        outcomes, transitions = [], 0
        for candidate in candidates:
            outcome, _, count = self._branch(policy, root, trigger_mask, candidate)
            outcomes.append(outcome)
            transitions += count
        provisional, reasons = _choose_winners(outcomes, initial, minimum_gain=self.minimum_gain,
                                               approach_gain=self.approach_gain)
        names = ['policy' if item is None else item[0] for item in candidates]
        records = self._records(outcomes, reasons, names, 'search')
        confirmed = torch.full_like(provisional, -1)
        examples = []
        if bool((provisional >= 0).any()):
            # A second policy-noise draw is matched within its own pair. Never
            # compare a candidate under one noise seed to another seed's baseline.
            seed = 91009 + self.rounds
            baseline, _, count = self._branch(policy, root, trigger_mask, None, seed)
            transitions += count
            confirmation = [baseline]
            segments = [[]]
            for index, candidate in enumerate(candidates[1:], start=1):
                selected = trigger_mask & (provisional == index)
                if bool(selected.any()):
                    outcome, segment, count = self._branch(policy, root, selected, candidate, seed)
                    transitions += count
                else:
                    outcome, segment = deepcopy(baseline), []
                    outcome['triggered'].zero_()
                confirmation.append(outcome)
                segments.append(segment)
            repeated, repeated_reasons = _choose_winners(
                confirmation, initial, minimum_gain=self.minimum_gain, approach_gain=self.approach_gain)
            confirmed = torch.where(repeated == provisional, provisional, -1)
            records.extend(self._records(confirmation, repeated_reasons, names, 'confirmation'))
            for index, segment in enumerate(segments):
                for row in segment:
                    mask = row['mask'] & (confirmed == index)
                    if bool(mask.any()):
                        examples.append({**{key: value.detach().clone() for key, value in row.items()},
                                         'mask': mask.clone(),
                                         'success_mask': confirmation[index]['success'].clone()})
        for row in records:
            row['round'] = self.rounds
            row['provisional_candidate'] = int(provisional[row['world']].item())
            row['selected_candidate'] = int(confirmed[row['world']].item())
        return examples, dict(transitions=transitions, candidates=len(candidates),
                              selected_worlds=int((confirmed >= 0).sum().item()),
                              provisional_worlds=int((provisional >= 0).sum().item()),
                              branch_records=records)

    def run(self, policy):
        """Search before a pilot event and restore its endpoint, including RNG."""
        rt = self.runtime
        started = time.perf_counter()
        self.rounds += 1
        device = rt.device_name
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        metrics = dict(version=2, transitions=0, candidate_branches=0, selected_worlds=0,
                       pilot_events=0, seconds=0., branch_records=[])
        examples = []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            rt.reset()
            memory = torch.zeros(rt.worlds, 64, device=device)
            progress = EpisodeProgress(signals(rt),
                stall_steps=max(1, round(self.stall_seconds / rt.control_dt)),
                max_steps=max(1, round(self.max_episode_seconds / rt.control_dt)), guidance=0.)
            task_return = torch.zeros(rt.worlds, device=device)
            roots = deque(maxlen=6)
            previous_holding = signals(rt)['holding']
            endpoint = None
            try:
                for tick in range(progress.max_steps):
                    if tick % self.root_spacing == 0:
                        roots.append((tick, self._snapshot(memory, progress, task_return)))
                    _, _, _, action, memory = self._act(policy, memory, explore=True)
                    _, _, done, _ = rt.step(action)
                    rt.check()
                    now = signals(rt)
                    _, terminated, truncated, _ = progress.step(now, done)
                    task_return += HARVEST_GAMMA ** (progress.age - 1).float() * progress.task_reward
                    metrics['transitions'] += rt.worlds
                    event = self._event_mask(rt, progress, now, previous_holding)
                    exploratory = self.rounds % 4 == 0 and tick == 40 and not bool(event.any())
                    if exploratory:
                        event[(self.rounds // 4 - 1) % rt.worlds] = True
                    if bool(event.any()) and roots:
                        endpoint = self._snapshot(memory, progress, task_return)
                        root_tick, root = roots[0]
                        metrics.update(pilot_events=int(event.sum().item()), pilot_ticks=tick + 1,
                                       root_count=len(roots), root_lookback_ticks=tick + 1 - root_tick,
                                       exploratory_roots=int(exploratory))
                        generator = torch.Generator(device=device).manual_seed(1307 + self.rounds)
                        alternative = (torch.randn((rt.worlds, 7), generator=generator, device=device) * .5).clamp(-1., 1.)
                        candidates = [None, ('close_jaw', None), ('hold_arm_close_jaw', None),
                                      ('coherent_alternative', alternative)]
                        examples, branch = self._episode(policy, root, event,
                            root.application_state['task_return'], candidates)
                        metrics['branch_records'] = branch.pop('branch_records')
                        metrics.update({f'branch_{key}': value for key, value in branch.items()})
                        metrics['candidate_branches'] = len(candidates)
                        metrics['selected_worlds'] = branch['selected_worlds']
                        break
                    ending = terminated | truncated
                    if bool(ending.any()):
                        rt.reset(ending.to(dtype=torch.uint8))
                        progress.reset(ending, signals(rt))
                        memory[ending] = 0
                        task_return[ending] = 0
                        roots.clear()
                    previous_holding = now['holding'].clone()
                if endpoint is None:
                    endpoint = self._snapshot(memory, progress, task_return)
            finally:
                if endpoint is not None:
                    self._restore_app(endpoint)
        metrics['seconds'] = time.perf_counter() - started
        metrics['transitions'] += int(metrics.get('branch_transitions', 0))
        metrics['successful_repairs'] = (int(torch.stack([
            row['success_mask'] & row['mask'] for row in examples]).any(dim=0).sum().item()) if examples else 0)
        metrics['partial_repairs'] = metrics['selected_worlds'] - metrics['successful_repairs']
        return examples, metrics


def update_cti(policy, optimizer, examples, coef=.1, anchor_rows=None):
    """Apply auxiliary action regression only if factual Gaussian KL stays <= .01."""
    if not math.isfinite(coef) or coef < 0:
        raise ValueError('coef must be nonnegative')
    rows = list(examples)
    selected_worlds = (int(torch.stack([row['mask'] for row in rows]).any(dim=0).sum().item()) if rows else 0)
    target_actions = sum(int(row['mask'].sum().item()) for row in rows)
    metrics = dict(loss=0., selected_worlds=selected_worlds, target_actions=target_actions,
                   updated=False, kl=0., rejected_update=False, nonfinite_update=False)
    if not rows or not target_actions or not coef:
        return metrics
    for row in rows:
        selected = row['action'][row['mask']]
        if not torch.isfinite(selected).all() or (selected.abs() > 1).any():
            raise ValueError('selected CTI actions must be finite and bounded')
    anchors = rows if anchor_rows is None else list(anchor_rows)
    if not anchors:
        raise ValueError('factual anchors must not be empty')
    policy_before, optimizer_before = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
    fixed = []
    with torch.no_grad():
        memory = anchors[0].get('initial_memory', anchors[0].get('memory'))
        if memory is None:
            raise ValueError('anchor rows need initial_memory or explicit memory')
        for row in anchors:
            memory = row.get('memory', memory).detach().clone()
            if 'reset' in row:
                memory *= (~row['reset'])[:, None]
            priv, obs = row['privileged'].detach().clone(), row['r84'].detach().clone()
            mean, logstd, _, next_memory = policy(priv, obs, memory)
            fixed.append((priv, obs, memory, mean.detach().clone(), logstd.detach().clone()))
            memory = next_memory.detach()
    optimizer.zero_grad(set_to_none=True)
    total = None
    for row in rows:
        mean, _, _, _ = policy(row['privileged'], row['r84'], row['memory'].detach())
        loss = (mean.tanh()[row['mask']] - row['action'][row['mask']]).square().mean(dim=-1).sum()
        total = loss if total is None else total + loss
    loss = total / target_actions
    loss_value = float(loss.detach().item())
    metrics['nonfinite_update'] = not math.isfinite(loss_value)
    metrics['loss'] = loss_value if math.isfinite(loss_value) else 0.
    (coef * loss).backward()
    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    if torch.isfinite(grad):
        optimizer.step()
        with torch.no_grad():
            kls = []
            for priv, obs, memory, old_mean, old_logstd in fixed:
                mean, logstd, _, _ = policy(priv, obs, memory)
                kl = (logstd - old_logstd +
                      (old_logstd.exp().square() + (old_mean - mean).square()) /
                      (2 * logstd.exp().square()) - .5).sum(dim=-1)
                kls.append(kl)
            kl = torch.cat(kls).mean()
        kl_value = float(kl.item())
        finite_state = all(bool(torch.isfinite(value).all()) for value in policy.state_dict().values())
        metrics['nonfinite_update'] |= not math.isfinite(kl_value) or not finite_state
        # Zero is only a logging placeholder when nonfinite_update is true.
        metrics['kl'] = max(0., kl_value) if math.isfinite(kl_value) else 0.
        accepted = not metrics['nonfinite_update'] and kl_value <= .01
    else:
        metrics['nonfinite_update'], accepted = True, False
    if not accepted:
        policy.load_state_dict(policy_before)
        optimizer.load_state_dict(optimizer_before)
        optimizer.zero_grad(set_to_none=True)
        metrics['rejected_update'] = True
    else:
        metrics['updated'] = True
    return metrics
