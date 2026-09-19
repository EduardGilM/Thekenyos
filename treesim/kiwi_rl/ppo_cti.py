"""Bounded factual decision segments collected from PPO rollouts."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field


def _clone(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return deepcopy(value)


def _policy_snapshot(policy):
    return {key: value.detach().cpu().clone()
            for key, value in policy.state_dict().items()}


def choose_worlds(distance, count, cursor):
    """Mix nearest-fruit worlds with a deterministic rotating sample."""
    worlds = int(distance.numel())
    count = min(max(0, int(count)), worlds)
    near_count = min((count + 1) // 2, worlds)
    near = distance.flatten().topk(near_count, largest=False).indices.tolist() if near_count else []
    chosen = set(near)
    rotating = []
    for offset in range(worlds):
        world = (cursor + offset) % worlds
        if world not in chosen:
            rotating.append(world)
            chosen.add(world)
            if len(chosen) == count:
                break
    selected = near + rotating[:count - len(near)]
    return selected, (cursor + max(1, count - len(near))) % max(1, worlds)


@dataclass
class DecisionBatch:
    snapshot: object
    source_world_ids: tuple
    source_episode_ids: object
    source_tick: int
    policy_version: int
    policy_state: dict
    initial_memory: object
    initial_progress: object
    rng_state: dict = field(default_factory=dict)
    priority: float = 0.
    factual_steps: list = field(default_factory=list)
    age_updates: int = 0

    @property
    def sources(self):
        episode_ids = self.source_episode_ids.detach().cpu().tolist()
        return [dict(source_world=world, source_episode=int(episode_ids[index]),
                     source_tick=self.source_tick, policy_version=self.policy_version)
                for index, world in enumerate(self.source_world_ids)]

    @property
    def factual(self):
        return self.factual_steps


class PPODecisionQueue:
    """Keep at most eight compact, factual roots for the matched CTI search."""

    def __init__(self, worlds=None, *, worlds_per_batch=16, max_batches=8, max_age=8,
                 segment_steps=64):
        if worlds is not None:
            worlds_per_batch = worlds
        if worlds_per_batch < 1 or max_batches < 1 or max_age < 1 or segment_steps < 1:
            raise ValueError('CTI queue limits must be positive')
        self.worlds_per_batch = worlds_per_batch
        self.max_batches = max_batches
        self.max_age = max_age
        self.segment_steps = segment_steps
        self.cursor = 0
        self.batches = deque()
        self.total_collected_roots = 0
        self.total_dropped_roots = 0
        self.total_expired_roots = 0
        self.total_popped_roots = 0

    def begin(self, runtime, collector, policy, policy_version):
        """Capture selected worlds before their first policy forward pass."""
        import torch
        from .counterfactual import capture_worlds

        from .harvest_training import signals
        ids, self.cursor = choose_worlds(signals(runtime)['distance'], self.worlds_per_batch, self.cursor)
        worlds = torch.as_tensor(ids, dtype=torch.long, device=runtime.device_name)
        progress = collector.progress
        progress_state = {name: _clone(_slice(getattr(progress, name), worlds)) for name in
                          ('age', 'stale', 'closest', 'best_basket', 'best_force',
                           'ever_grasp', 'ever_detached', 'previous')}
        from .harvest_training import EpisodeProgress
        selected_initial = {key: value.index_select(0, worlds).clone()
                            for key, value in progress.previous.items()}
        sliced_progress = EpisodeProgress(selected_initial, stall_steps=progress.stall_steps,
            max_steps=progress.max_steps, guidance=progress.guidance, gamma=progress.gamma)
        for name, value in progress_state.items():
            setattr(sliced_progress, name, value)
        rng_state = dict(torch_cpu=torch.random.get_rng_state().clone())
        if torch.cuda.is_available():
            rng_state['torch_cuda'] = [state.clone() for state in torch.cuda.get_rng_state_all()]
        application = dict(memory=_clone(collector.memory.index_select(0, worlds)),
                           reset_mask=_clone(collector.reset_mask.index_select(0, worlds)),
                           episode_ids=_clone(collector.episode_ids.index_select(0, worlds)),
                           tick=collector.tick, progress=sliced_progress, rng_state=rng_state)
        snapshot = capture_worlds(runtime, ids, application_state=application)
        policy_state = _policy_snapshot(policy)
        distance = signals(runtime)['distance'].index_select(0, worlds)
        priority = float((1. - distance / .25).clamp(0., 1.).max().item())
        return DecisionBatch(snapshot, tuple(ids), application['episode_ids'], collector.tick,
                             int(policy_version), policy_state, application['memory'],
                             sliced_progress, rng_state=rng_state, priority=priority)

    def record(self, batch, *, obs, raw_action, action, gait_action, next_obs,
               runtime, signals, reward, task_reward, shaping_reward,
               terminated, truncated, stalled, episode_ids, active_mask, noise=None):
        """Record one factual step before any ending world's reset is applied."""
        import torch
        ids = torch.as_tensor(batch.source_world_ids, dtype=torch.long, device=runtime.device_name)
        active = active_mask.index_select(0, ids)
        if not bool(active.any()):
            return
        def selected(value):
            if isinstance(value, dict):
                return {key: selected(item) for key, item in value.items()}
            if isinstance(value, torch.Tensor):
                return _clone(value.index_select(0, ids))
            return value
        qpos = runtime.data.qpos
        qvel = runtime.data.qvel
        control_targets = runtime.control.targets
        control_previous = runtime.control.previous
        import warp as wp
        if not isinstance(qpos, torch.Tensor):
            qpos, qvel = wp.to_torch(qpos), wp.to_torch(qvel)
        if not isinstance(control_targets, torch.Tensor):
            control_targets, control_previous = wp.to_torch(control_targets), wp.to_torch(control_previous)
        ended = terminated | truncated
        record = dict(valid=_clone(active), active=_clone(active), episode_ids=selected(episode_ids),
                      obs=selected(obs), raw_action=selected(raw_action), action=selected(action),
                      noise=selected(noise),
                      gait_action=selected(gait_action), gait=selected(gait_action), next_obs=selected(next_obs),
                      qpos=selected(qpos), qvel=selected(qvel),
                      control_targets=selected(control_targets), targets=selected(control_targets),
                      control_previous=selected(control_previous),
                      task_signals=selected(signals), signals=selected(signals), reward=selected(reward),
                      task_reward=selected(task_reward), shaping_reward=selected(shaping_reward),
                      terminated=selected(terminated), truncated=selected(truncated),
                      ending=selected(ended), stalled=selected(stalled))
        batch.factual_steps.append(record)
        event = signals['touching'] | signals['grasp'] | signals['detached'] | signals['success'] | signals['failed'] | stalled
        event = event.index_select(0, ids) & active
        if bool(event.any()):
            batch.priority = max(batch.priority, 2. + float(event.sum().item()) / len(ids))

    def finish(self, batch, *, minimum_steps=8):
        useful = any(bool(step['valid'].any()) for step in batch.factual_steps)
        if not useful or len(batch.factual_steps) < minimum_steps:
            return False
        self.batches.append(batch)
        while len(self.batches) > self.max_batches:
            dropped = self.batches.popleft()
            self.total_dropped_roots += len(dropped.source_world_ids)
        self.total_collected_roots += len(batch.source_world_ids)
        return True

    def pop_ready(self, update_index, *, minimum_steps=8):
        """Return the highest-priority recent batch, dropping expired roots."""
        for batch in self.batches:
            batch.age_updates = max(0, int(update_index) - batch.policy_version)
        retained = [batch for batch in self.batches
                    if batch.age_updates <= self.max_age and len(batch.factual_steps) >= minimum_steps]
        retained_ids = {id(batch) for batch in retained}
        self.total_expired_roots += sum(len(batch.source_world_ids) for batch in self.batches
                                        if id(batch) not in retained_ids)
        self.batches = deque(retained)
        if not self.batches:
            return None
        best = max(range(len(self.batches)),
                   key=lambda index: (self.batches[index].priority, -self.batches[index].age_updates))
        self.batches.rotate(-best)
        self.total_popped_roots += len(self.batches[0].source_world_ids)
        return self.batches.popleft()

    def pop(self, current_policy_version, *, minimum_steps=8):
        return self.pop_ready(current_policy_version, minimum_steps=minimum_steps)

    def metrics(self):
        return dict(queue_batches=len(self.batches), queue_capacity=self.max_batches,
                    queue_roots=sum(len(batch.source_world_ids) for batch in self.batches),
                    total_collected_roots=self.total_collected_roots,
                    total_dropped_roots=self.total_dropped_roots,
                    total_expired_roots=self.total_expired_roots,
                    total_popped_roots=self.total_popped_roots)

    def __len__(self):
        return len(self.batches)


def _slice(value, indices):
    import torch
    if isinstance(value, dict):
        return {key: _slice(item, indices) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.index_select(0, indices)
    return value
