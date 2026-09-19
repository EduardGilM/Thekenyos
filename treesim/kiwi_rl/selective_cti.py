"""Small, teacher-only matched counterfactual action search.

This process-local auxiliary stream never creates PPO rollout rows. Its result
is a masked action-regression example for the existing privileged teacher.
"""

from __future__ import annotations

import time
import math

import torch

from .counterfactual import capture, restore
from .harvest_training import EpisodeProgress, signals, HARVEST_GAMMA
from .fast_teacher import privileged_observation


def _choose_winners(scores, resolved, physical_failure, success, *, minimum_gain=0.5):
    """Return per-world candidate indices; -1 means no verified improvement."""
    scores = torch.as_tensor(scores)
    resolved = torch.as_tensor(resolved, dtype=torch.bool, device=scores.device)
    physical_failure = torch.as_tensor(physical_failure, dtype=torch.bool, device=scores.device)
    success = torch.as_tensor(success, dtype=torch.bool, device=scores.device)
    if (scores.ndim != 2 or scores.shape != resolved.shape or scores.shape != physical_failure.shape
            or scores.shape != success.shape):
        raise ValueError('candidate arrays must have matching [candidate, world] shapes')
    baseline = scores[0]
    baseline_success = success[0]
    baseline_resolved = resolved[0]
    categorical_gain = (success & ~baseline_success[None, :]) | (
        ~physical_failure & ~success & physical_failure[0][None, :])
    eligible = (resolved & baseline_resolved[None, :] & ~physical_failure &
                categorical_gain & torch.isfinite(scores))
    eligible[0] = False
    gains = scores - baseline[None, :]
    gains[~eligible] = -torch.inf
    best_gain, winner = gains.max(dim=0)
    return torch.where(best_gain >= minimum_gain, winner, torch.full_like(winner, -1))


class SelectiveCTI:
    """Run bounded matched branches on a small, independent FastRuntime."""

    def __init__(self, runtime, gait, *, stall_seconds=4., max_episode_seconds=30.,
                 minimum_gain=0.5, perturb_steps=10, root_spacing=20):
        if runtime.worlds > 64:
            raise ValueError('CTI runtime must use at most 64 worlds')
        if stall_seconds <= 0 or max_episode_seconds <= 0 or minimum_gain < 0:
            raise ValueError('invalid CTI episode limits or improvement threshold')
        self.runtime, self.gait = runtime, gait
        self.rounds = 0
        self.stall_seconds, self.max_episode_seconds = stall_seconds, max_episode_seconds
        self.minimum_gain, self.perturb_steps = float(minimum_gain), int(perturb_steps)
        self.root_spacing = int(root_spacing)
        if self.perturb_steps < 1 or self.root_spacing < 1:
            raise ValueError('perturb_steps and root_spacing must be positive')

    def _app(self, memory, progress, task_return):
        return {'memory': memory, 'progress': progress, 'task_return': task_return}

    def _snapshot(self, memory, progress, task_return):
        return capture(self.runtime, application_state=self._app(memory, progress, task_return))

    def _restore_app(self, snapshot):
        app = restore(self.runtime, snapshot)
        return app['memory'], app['progress'], app['task_return']

    @staticmethod
    def _event_mask(rt, progress, now, previous_holding):
        failed = now['failed']
        unheld_detach = now['detached'] & ~now['holding']
        lost_hold = previous_holding & ~now['holding']
        nearing_stall = progress.stale >= max(1, progress.stall_steps - 20)
        return failed | unheld_detach | lost_hold | nearing_stall

    def _act(self, policy, memory, *, explore=False, delta=None):
        rt = self.runtime
        obs = rt.observe().detach().clone()
        priv = privileged_observation(rt, obs).detach()
        memory_in = memory.detach().clone()
        mean, logstd, _, next_memory = policy(priv, obs, memory)
        raw = mean + logstd.exp() * torch.randn_like(mean) if explore else mean
        action = raw.tanh()
        if delta is not None:
            action = (action + delta).clamp(-1., 1.)
        rt.set_gait_actions(self.gait(obs).detach())
        return obs, priv, memory_in, action, next_memory.detach()

    def _episode(self, policy, root, trigger_mask, prefix_return, candidates):
        """Replay baseline and fixed first-ten-step action perturbations."""
        rt = self.runtime
        count, worlds = len(candidates), rt.worlds
        scores = torch.full((count, worlds), float('nan'), device=rt.device_name)
        resolved = torch.zeros((count, worlds), dtype=torch.bool, device=rt.device_name)
        physical_failure = torch.zeros_like(resolved)
        success = torch.zeros_like(resolved)
        candidate_segments = [[] for _ in range(count)]
        transitions = 0
        for candidate_id, delta in enumerate(candidates):
            memory, progress, task_return = self._restore_app(root)
            active = trigger_mask.clone()
            branch_return = prefix_return.clone()
            step_index = 0
            while bool(active.any()) and bool((progress.age[active] < progress.max_steps).any()):
                obs, priv, memory_in, action, memory = self._act(
                    policy, memory, explore=True,
                    delta=delta if step_index < self.perturb_steps else None)
                if step_index < self.perturb_steps:
                    candidate_segments[candidate_id].append((
                        priv.clone(), obs.clone(), memory_in.clone(), action.detach().clone(), active.clone()))
                _, _, done, _ = rt.step(action)
                rt.check()  # Inspect numerical latches before any world reset.
                now = signals(rt)
                reward, terminated, truncated, stalled = progress.step(now, done)
                del reward
                increment = progress.task_reward.detach()
                branch_return += (HARVEST_GAMMA ** (progress.age - 1).float()) * increment
                transitions += worlds
                ended_all = terminated | truncated
                ended = active & ended_all
                actual = active & terminated
                if bool(actual.any()):
                    scores[candidate_id, actual] = branch_return[actual]
                    resolved[candidate_id, actual] = True
                    physical_failure[candidate_id, actual] = (done & ~now['success'])[actual]
                    success[candidate_id, actual] = now['success'][actual]
                if bool(ended_all.any()):
                    rt.reset(ended_all.to(dtype=torch.uint8))
                    progress.reset(ended_all, signals(rt))
                    memory[ended_all] = 0
                    active[ended] = False
                step_index += 1
        winners = _choose_winners(scores, resolved, physical_failure, success,
                                  minimum_gain=self.minimum_gain)
        if not candidate_segments[0]:
            return [], {'transitions': transitions, 'selected_worlds': 0,
                        'candidates': count, 'resolved_worlds': int(resolved.sum().item())}
        mask = winners >= 0
        if bool(mask.any()):
            examples = []
            for step in range(min(self.perturb_steps, max(map(len, candidate_segments)))):
                available = torch.zeros(worlds, dtype=torch.bool, device=rt.device_name)
                priv = torch.zeros((worlds, 32), device=rt.device_name)
                obs = torch.zeros((worlds, 84), device=rt.device_name)
                memory = torch.zeros((worlds, 64), device=rt.device_name)
                action = torch.zeros((worlds, 7), device=rt.device_name)
                for candidate_id, segment in enumerate(candidate_segments):
                    if step >= len(segment):
                        continue
                    p, o, m, a, active_at_step = segment[step]
                    selected = (winners == candidate_id) & active_at_step
                    if bool(selected.any()):
                        priv[selected], obs[selected] = p[selected], o[selected]
                        memory[selected], action[selected] = m[selected], a[selected]
                        available |= selected
                if bool(available.any()):
                    winner_success = torch.zeros(worlds, dtype=torch.bool, device=rt.device_name)
                    ids = available.nonzero(as_tuple=False).flatten()
                    winner_success[ids] = success[winners[ids], ids]
                    examples.append(dict(privileged=priv.detach().clone(), r84=obs.detach().clone(),
                                         memory=memory.detach().clone(), action=action.detach().clone(),
                                         mask=available.detach().clone(),
                                         success_mask=winner_success.detach().clone()))
        else:
            examples = []
        return examples, {'transitions': transitions, 'selected_worlds': int(mask.sum().item()),
                          'candidates': count, 'resolved_worlds': int(resolved.sum().item())}

    def run(self, policy):
        """Search from a factual event root and restore the pilot endpoint exactly."""
        rt = self.runtime
        started = time.perf_counter()
        self.rounds += 1
        device = rt.device_name
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        metrics = {'transitions': 0, 'candidate_branches': 0, 'selected_worlds': 0,
                   'pilot_events': 0, 'seconds': 0.}
        examples = []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            rt.reset()
            memory = torch.zeros(rt.worlds, 64, device=device)
            progress = EpisodeProgress(signals(rt),
                stall_steps=round(self.stall_seconds / rt.control_dt),
                max_steps=round(self.max_episode_seconds / rt.control_dt), guidance=0.)
            task_return = torch.zeros(rt.worlds, device=device)
            roots = []
            previous_holding = signals(rt)['holding']
            endpoint = None
            try:
                for tick in range(progress.max_steps):
                    if tick % self.root_spacing == 0:
                        roots.append(self._snapshot(memory, progress, task_return))
                        roots = roots[-2:]
                    _, _, _, action, memory = self._act(policy, memory, explore=True)
                    _, _, done, _ = rt.step(action)
                    rt.check()
                    now = signals(rt)
                    _, terminated, truncated, stalled = progress.step(now, done)
                    task_return += (HARVEST_GAMMA ** (progress.age - 1).float()) * progress.task_reward
                    metrics['transitions'] += rt.worlds
                    event = self._event_mask(rt, progress, now, previous_holding)
                    exploratory = self.rounds % 4 == 0 and tick == 40
                    if exploratory and not bool(event.any()):
                        event = torch.zeros_like(event)
                        event[(self.rounds // 4 - 1) % rt.worlds] = True
                    if bool(event.any()) and roots:
                        endpoint = self._snapshot(memory, progress, task_return)
                        metrics['pilot_events'] = int(event.sum().item())
                        metrics['pilot_ticks'] = tick + 1
                        metrics['root_count'] = len(roots)
                        metrics['exploratory_roots'] = int(exploratory and not bool(
                            self._event_mask(rt, progress, now, previous_holding).any()))
                        root = roots[-2] if len(roots) >= 2 else roots[-1]
                        trigger = event.clone()
                        prefix = root.application_state['task_return'].to(device)
                        # Three bounded, coherent joint perturbations; same delta
                        # is held for the first ten control steps of each branch.
                        generator = torch.Generator(device=device).manual_seed(1307 + self.rounds)
                        deltas = [torch.zeros((rt.worlds, 7), device=device)]
                        deltas.extend((torch.randn((rt.worlds, 7), generator=generator,
                                                   device=device) * .18).clamp(-.3, .3)
                                      for _ in range(3))
                        branch_examples, branch_metrics = self._episode(
                            policy, root, trigger, prefix, deltas)
                        metrics.update({f'branch_{key}': value for key, value in branch_metrics.items()})
                        metrics['candidate_branches'] = len(deltas)
                        metrics['selected_worlds'] = branch_metrics['selected_worlds']
                        examples.extend(branch_examples)
                        break
                    if bool((terminated | truncated).any()):
                        ending = terminated | truncated
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
        if examples:
            repaired = torch.stack([row['success_mask'] & row['mask'] for row in examples]).any(dim=0)
            metrics['successful_repairs'] = int(repaired.sum().item())
        else:
            metrics['successful_repairs'] = 0
        metrics['failure_avoidance'] = metrics['selected_worlds'] - metrics['successful_repairs']
        return examples, metrics


def update_cti(policy, optimizer, examples, coef=0.1):
    """Apply masked auxiliary MSE from CTI examples, separately from PPO rows."""
    if not math.isfinite(coef) or coef < 0:
        raise ValueError('coef must be nonnegative')
    rows = list(examples)
    selected_worlds = (int(torch.stack([row['mask'] for row in rows]).any(dim=0).sum().item())
                       if rows else 0)
    target_actions = sum(int(row['mask'].sum().item()) for row in rows)
    if not rows or target_actions == 0 or coef == 0:
        return {'loss': 0., 'selected_worlds': selected_worlds,
                'target_actions': target_actions, 'updated': False}
    optimizer.zero_grad(set_to_none=True)
    total = None
    for row in rows:
        if (not torch.isfinite(row['action'][row['mask']]).all() or
                (row['action'][row['mask']].abs() > 1).any()):
            raise ValueError('selected CTI actions must be finite and bounded')
        mean, _, _, _ = policy(row['privileged'], row['r84'], row['memory'].detach())
        per_world = (mean.tanh() - row['action']).square().mean(dim=-1)
        loss = per_world[row['mask']].sum()
        total = loss if total is None else total + loss
    loss = total / target_actions
    (coef * loss).backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'loss': float(loss.detach().item()), 'selected_worlds': selected_worlds,
            'target_actions': target_actions, 'updated': True}
