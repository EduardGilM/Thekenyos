"""Teacher-only retained-progress search; its examples never enter PPO rows."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import time

import torch

from .counterfactual import capture, restore, restore_worlds, replay_position_errors
from .harvest_training import EpisodeProgress, signals, HARVEST_GAMMA
from .fast_teacher import privileged_observation

_EXPLORATION_SCALES = (.5, 1., 2.)
_EXPLORATION_BLOCKS = (8, 32, 100)


def sample_policy_sequences(count, steps, *, device, seed):
    """Sample coherent latent residuals from local through broad policy scales."""
    candidates = []
    for index in range(count):
        scale = _EXPLORATION_SCALES[index % len(_EXPLORATION_SCALES)]
        block = _EXPLORATION_BLOCKS[(index // len(_EXPLORATION_SCALES)) % len(_EXPLORATION_BLOCKS)]
        candidate_seed = int(seed + index * 7919)
        generator = torch.Generator(device=device).manual_seed(candidate_seed)
        knot_count = math.ceil(steps / block)
        knots = torch.randn((knot_count, 7), generator=generator, device=device)
        residual = knots.repeat_interleave(block, dim=0)[:steps]
        candidates.append(dict(label=f'policy-residual-{index + 1}', residual=residual,
                               noise_seed=candidate_seed, scale=scale,
                               block_steps=block, std_floor=.25 if scale == 2. else 0.))
    return candidates


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
            'factual_replay_mismatch': baseline.get('replay_invalid', torch.zeros_like(result['success'])),
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
                 approach_gain=.015, alternatives=12):
        if runtime.worlds > 64:
            raise ValueError('CTI runtime must use at most 64 worlds')
        values = (stall_seconds, max_episode_seconds, horizon_seconds,
                  intervention_seconds, retention_seconds, approach_gain)
        if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isfinite(minimum_gain) or minimum_gain < 0:
            raise ValueError('invalid CTI limits or improvement threshold')
        self.runtime, self.gait, self.rounds = runtime, gait, 0
        self.stall_seconds, self.max_episode_seconds = stall_seconds, max_episode_seconds
        self.minimum_gain, self.approach_gain = minimum_gain, approach_gain
        self.alternatives = int(alternatives)
        if self.alternatives < 1:
            raise ValueError('at least one CTI alternative is required')
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

    def _act(self, policy, memory, *, explore=False, noise=None, noise_scale=1., std_floor=0.):
        rt = self.runtime
        obs = rt.observe().detach().clone()
        priv = privileged_observation(rt, obs).detach().clone()
        memory_in = memory.detach().clone()
        mean, logstd, _, next_memory = policy(priv, obs, memory)
        if explore:
            sampled_noise = torch.randn_like(mean)  # Keep continuation RNG aligned across candidates.
            residual = sampled_noise if noise is None else noise
            std = logstd.exp().clamp_min(std_floor) if std_floor else logstd.exp()
            raw = mean + std * residual * noise_scale
        else:
            raw = mean
        action = raw.tanh()
        rt.set_gait_actions(self.gait(obs).detach())
        return obs, priv, memory_in, action, next_memory.detach()

    def _branch(self, policy, root, trigger, candidate, noise_seed=None):
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
                        'horizon', 'numerical', 'lost_fruit', 'retained', 'replay_invalid')})
        result['triggered'] = trigger.clone()
        result['replay_steps'] = torch.zeros_like(initial['distance'])
        window, segment = deque(maxlen=self.retention_steps), []
        transitions = 0
        for tick in range(self.horizon_steps):
            if not bool(active.any()):
                break
            factual = getattr(self, 'factual', ())
            noise = (candidate['residual'][tick] if candidate is not None and
                     tick < self.perturb_steps else None)
            obs, priv, memory_in, action, memory = self._act(
                policy, memory, explore=True, noise=noise,
                noise_scale=candidate['scale'] if noise is not None else 1.,
                std_floor=candidate['std_floor'] if noise is not None else 0.)
            recorded = factual[tick] if candidate is None and tick < len(factual) else None
            if recorded is not None:
                action = recorded['action'].clone()
                rt.set_gait_actions(recorded['gait'])
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
            if recorded is not None:
                import warp as wp
                errors = replay_position_errors(rt, recorded)
                checked = active & recorded['active']
                first = tick == 0
                mismatch = ((errors['base_m'] > (1e-5 if first else .002)) |
                            (errors['fruit_m'] > (1e-5 if first else .01)) |
                            (errors['robot_rad'] > (1e-4 if first else .01)))
                if first:
                    mismatch |= errors['fruit_orientation_rad'] > .001
                mismatch |= ~torch.isfinite(torch.stack(list(errors.values()))).all(dim=0)
                mismatch |= (wp.to_torch(rt.control.targets)-recorded['targets']).abs().amax(dim=1) > 1e-6
                # Warm contact trajectories drift even after full same-runtime restore.
                # Check errors in physical units, with exact discrete task outcomes.
                for key, value in now.items():
                    if value.dtype == torch.bool:
                        mismatch |= value != recorded['signals'][key]
                result['replay_invalid'] |= checked & mismatch
                for key, error in errors.items():
                    name = 'replay_' + key
                    result[name] = torch.maximum(result.get(name, torch.zeros_like(error)),
                                                 torch.where(checked, error, 0.))
                result['replay_steps'] += checked.float()
            _, terminated, truncated, stalled = progress.step(now, done)
            if recorded is not None:
                result['replay_invalid'] |= active & ((terminated != recorded['terminated']) |
                    (truncated != recorded['truncated']) | (progress.task_reward != recorded['task_reward']))
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
            # A PPO reset of OTHER worlds also runs mw.forward on every world.
            # Reproduce that solver refresh even when our selected mask is zero.
            source_reset = tick < len(factual) and factual[tick].get('source_reset', False)
            if (bool(finish.any()) or source_reset) and bool(active.any()):
                rt.reset(finish.to(dtype=torch.uint8))
                progress.reset(finish, signals(rt))
                memory[finish] = 0
        return result, segment, transitions

    @staticmethod
    def _records(outcomes, reasons, names, pass_name):
        records = []
        for candidate, outcome in enumerate(outcomes):
            for world in range(len(outcome['score'])):
                gain = float((outcome['score'][world] - outcomes[0]['score'][world]).item())
                row = dict(score_delta_vs_factual=gain if math.isfinite(gain) else None,
                           pass_name=pass_name, candidate=candidate, skill=names[candidate],
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
            outcome, _, count = self._branch(policy, root, trigger_mask, candidate, 71009+self.rounds)
            outcomes.append(outcome)
            transitions += count
        provisional, reasons = _choose_winners(outcomes, initial, minimum_gain=self.minimum_gain,
                                               approach_gain=self.approach_gain)
        names = ['factual-policy'] + [item['label'] for item in candidates[1:]]
        records = self._records(outcomes, reasons, names, 'search')
        for record in records:
            candidate_index = record['candidate']
            if candidate_index and record['world'] == 0:
                candidate = candidates[candidate_index]
                record.update(exploration='piecewise-gaussian/v1',
                    noise_seed=candidate['noise_seed'], noise_scale=candidate['scale'],
                    noise_std_floor=candidate['std_floor'], block_steps=candidate['block_steps'],
                    intervention_steps=self.perturb_steps)
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
                              replay_rejected_worlds=int(outcomes[0].get('replay_invalid', torch.zeros_like(trigger_mask)).sum().item()),
                              replay_checked_steps=int(outcomes[0].get('replay_steps', torch.zeros_like(trigger_mask)).sum().item()),
                              provisional_worlds=int((provisional >= 0).sum().item()),
                              branch_records=records)

    def run(self, policy, batch):
        """Search only a recorded PPO batch using its frozen collection policy."""
        if batch is None:
            raise ValueError('CTI requires an actual PPO decision batch')
        rt = self.runtime
        started = time.perf_counter()
        self.rounds += 1
        devices = [torch.device(rt.device_name).index or 0] if torch.device(rt.device_name).type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            frozen = deepcopy(policy).eval()
            frozen.load_state_dict(batch.policy_state)
            app = restore_worlds(rt, batch.snapshot)
            # The queue captures pre-action GRU/progress from the factual collector.
            app['progress'].guidance = 0.
            root = capture(rt, application_state=dict(memory=app['memory'],
                progress=app['progress'], task_return=torch.zeros(rt.worlds,device=rt.device_name)))
            self.factual = batch.factual
            trigger = torch.ones(rt.worlds,dtype=torch.bool,device=rt.device_name)
            candidates = [None] + sample_policy_sequences(self.alternatives, self.perturb_steps,
                device=rt.device_name, seed=1307+self.rounds)
            try:
                examples, branch = self._episode(frozen,root,trigger,None,candidates)
            finally:
                self.factual = ()
            for record in branch['branch_records']:
                record.update(source='ppo',source_policy_version=batch.policy_version,
                              **batch.sources[record['world']])
        selected=branch['selected_worlds']
        successful=(int(torch.stack([r['success_mask'] & r['mask'] for r in examples]).any(dim=0).sum()) if examples else 0)
        metrics=dict(version=5,source='ppo',seconds=time.perf_counter()-started,
                     transitions=branch['transitions'],candidate_branches=len(candidates),
                     pilot_events=rt.worlds,roots_searched=rt.worlds,
                     alternatives_compared=rt.worlds*(len(candidates)-1),
                     selected_worlds=selected,successful_repairs=successful,
                     partial_repairs=selected-successful,source_policy_version=batch.policy_version,
                     branch_records=branch.pop('branch_records'))
        metrics.update({f'branch_{key}':value for key,value in branch.items()})
        return examples, metrics


def update_cti(policy, optimizer, examples, coef=.1, anchor_rows=None):
    """Apply auxiliary action regression only if factual Gaussian KL stays <= .01."""
    if not math.isfinite(coef) or coef < 0:
        raise ValueError('coef must be nonnegative')
    rows = list(examples)
    selected_worlds = (int(torch.stack([row['mask'] for row in rows]).any(dim=0).sum().item()) if rows else 0)
    target_actions = sum(int(row['mask'].sum().item()) for row in rows)
    metrics = dict(loss=0., selected_worlds=selected_worlds, target_actions=target_actions,
                   updated=False, kl=0., target_error_before=None, target_error_after=None,
                   rejected_update=False, nonfinite_update=False,
                   no_target_improvement=False)
    if not rows or not target_actions or not coef:
        return metrics
    for row in rows:
        selected = row['action'][row['mask']]
        if not torch.isfinite(selected).all() or (selected.abs() > 1).any():
            raise ValueError('selected CTI actions must be finite and bounded')
    anchors = rows if anchor_rows is None else list(anchor_rows)
    if not anchors:
        raise ValueError('factual anchors must not be empty')

    def target_error():
        total_error = None
        with torch.no_grad():
            for row in rows:
                mean, _, _, _ = policy(row['privileged'], row['r84'], row['memory'].detach())
                error = (mean.tanh()[row['mask']] - row['action'][row['mask']]).square().mean(dim=-1).sum()
                total_error = error if total_error is None else total_error + error
        return float((total_error / target_actions).item())

    before_error = target_error()
    metrics['target_error_before'] = before_error if math.isfinite(before_error) else None
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
        attempted_error = target_error()
        finite_state = all(bool(torch.isfinite(value).all()) for value in policy.state_dict().values())
        metrics['nonfinite_update'] |= (not math.isfinite(kl_value) or not finite_state or
                                        not math.isfinite(attempted_error))
        # Zero is only a logging placeholder when nonfinite_update is true.
        metrics['kl'] = max(0., kl_value) if math.isfinite(kl_value) else 0.
        required_improvement = max(abs(before_error) * 1e-4, 1e-8)
        improved_target = (math.isfinite(before_error) and math.isfinite(attempted_error) and
                           before_error - attempted_error > required_improvement)
        metrics['no_target_improvement'] = not improved_target
        accepted = (not metrics['nonfinite_update'] and kl_value <= .01 and
                    improved_target)
    else:
        metrics['nonfinite_update'], accepted = True, False
    if not accepted:
        policy.load_state_dict(policy_before)
        optimizer.load_state_dict(optimizer_before)
        optimizer.zero_grad(set_to_none=True)
        metrics['rejected_update'] = True
    else:
        metrics['updated'] = True
    # Measure the policy state that will be retained. A rejected, nonfinite or
    # excessive-KL step must report the restored pre-update target error.
    final_error = attempted_error if accepted else target_error()
    metrics['target_error_after'] = final_error if math.isfinite(final_error) else None
    return metrics
