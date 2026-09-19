"""Parallel, iterative PPO-rooted search with all-branch off-policy experience.

Each proposal is a coherent bias, not a commanded action. Fresh Gaussian noise
is sampled at EVERY step; its conditional density is stored for V-trace.
"""
from copy import deepcopy
import math
import time

import torch

from .counterfactual import restore_worlds, replay_position_errors
from .fast_teacher import privileged_observation
from .harvest_training import signals, GRAPH_PROFILE


def expand_application(application, mapping):
    """Expand known world tensors, including every temporal graph tensor."""
    def expand(value):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, mapping).clone()
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return deepcopy(value)
    progress = deepcopy(application['progress'])
    for name, value in vars(progress).items():
        if isinstance(value, (torch.Tensor, dict)):
            setattr(progress, name, expand(value))
    return expand(application['memory']), progress


def initial_plans(roots, alternatives, knots, *, generator, device):
    plans = torch.randn((alternatives+1, roots, knots, 7), generator=generator, device=device)
    scales = plans.new_tensor([.25, .5, 1., 2.])
    plans *= scales[torch.arange(alternatives+1, device=device) % 4, None, None, None]
    plans[:2].zero_()
    return plans


def refine_plans(plans, scores, valid, *, generator):
    """Fit each root's elite biases; retain broad mutations and the best plan."""
    result = torch.empty_like(plans)
    result[:2].zero_()
    candidates = plans.shape[0]-1
    for root in range(plans.shape[1]):
        usable = valid[1:, root] & torch.isfinite(scores[1:, root])
        ids = usable.nonzero().flatten()+1
        if not len(ids):
            result[1:, root] = plans[1:, root]
            continue
        elite_ids = ids[scores[ids, root].topk(min(4, len(ids))).indices]
        elite = plans[elite_ids, root]
        center, spread = elite.mean(0), elite.std(0, unbiased=False).clamp_min(.25)
        mutation = torch.randn((candidates, *center.shape), generator=generator, device=plans.device)
        result[1:, root] = center + spread * mutation
        # A quarter of the batch continues broad exploration instead of collapse.
        count = max(1, candidates//4)
        result[-count:, root] = 2.*torch.randn((count, *center.shape), generator=generator, device=plans.device)
        result[2, root] = elite[0]  # Re-evaluate the incumbent under new noise.
    result[1].zero_()  # Fresh matched-policy baseline in every search pass.
    return result.clamp(-4., 4.)


class BranchCTI:
    version = 8

    def __init__(self, runtime, gait, *, roots=16, alternatives=12, search_iterations=3,
                 horizon_seconds=6., intervention_seconds=2., block_steps=20):
        if (not 1 <= roots <= 64 or not 2 <= alternatives <= 32 or
                not 1 <= search_iterations <= 5 or runtime.worlds != roots*(alternatives+1)):
            raise ValueError('CTI runtime must contain roots times (alternatives + factual) worlds')
        if getattr(runtime, 'task_profile', None) != GRAPH_PROFILE:
            raise ValueError('Branch CTI requires the temporal reward graph')
        if not 0 < intervention_seconds < horizon_seconds or block_steps < 1:
            raise ValueError('Invalid CTI search horizon')
        self.runtime, self.gait = runtime, gait
        self.roots, self.alternatives, self.search_iterations = roots, alternatives, search_iterations
        self.horizon_steps = round(horizon_seconds/runtime.control_dt)
        self.intervention_steps = round(intervention_seconds/runtime.control_dt)
        self.block_steps = block_steps
        self.rounds = 0

    def _replay_check(self, recorded, now, progress, terminated, truncated, active, tick, drift):
        import warp as wp
        rt, n = self.runtime, self.roots
        expected = wp.to_torch(rt.data.qpos).clone()
        expected[:n] = recorded['qpos']
        errors = {key: value[:n] for key, value in replay_position_errors(rt, {'qpos': expected}).items()}
        mismatch = ((errors['base_m'] > (1e-5 if tick == 0 else .002)) |
                    (errors['fruit_m'] > (1e-5 if tick == 0 else .01)) |
                    (errors['robot_rad'] > (1e-4 if tick == 0 else .01)))
        if tick == 0:
            mismatch |= errors['fruit_orientation_rad'] > .001
        mismatch |= ~torch.stack(list(errors.values())).isfinite().all(0)
        mismatch |= (wp.to_torch(rt.control.targets)[:n]-recorded['targets']).abs().amax(1) > 1e-6
        for key, value in now.items():
            if value.dtype == torch.bool:
                mismatch |= value[:n] != recorded['signals'][key]
        mismatch |= (terminated[:n] != recorded['terminated']) | (truncated[:n] != recorded['truncated'])
        mismatch |= progress.task_reward[:n] != recorded['task_reward']
        checked = active[:n] & recorded['active']
        drift += torch.where(checked, progress.gamma**tick*(progress.shaping_reward[:n]-recorded['shaping_reward']), 0.)
        mismatch |= drift.abs() > .05
        if 'graph_state' in recorded:
            mismatch |= progress.graph_stage[:n] != recorded['graph_state']['graph_stage']
        return checked & mismatch, int(checked.sum()), errors

    def _pass(self, policy, batch, plans, seed, value_policy=None):
        import warp as wp
        rt, n, c = self.runtime, self.roots, self.alternatives+1
        mapping = torch.arange(n, device=rt.device_name).repeat(c)
        app = restore_worlds(rt, batch.snapshot, source_indices=mapping)
        memory, progress = expand_application(app, mapping)
        active = torch.ones(rt.worlds, device=rt.device_name, dtype=torch.bool)
        replay_invalid = torch.zeros(n, device=rt.device_name, dtype=torch.bool)
        numerical = torch.zeros_like(active)
        drift = torch.zeros(n, device=rt.device_name)
        bootstrap = torch.zeros(rt.worlds, device=rt.device_name)
        returns = torch.zeros_like(bootstrap)
        final_grade = progress.graph_score.clone()
        success, failed = torch.zeros_like(active), torch.zeros_like(active)
        checked_steps, transitions, rows = 0, 0, []
        generator = torch.Generator(device=rt.device_name).manual_seed(seed)
        max_errors = {}
        for tick in range(self.horizon_steps):
            if not bool(active.any()):
                break
            obs = rt.observe().detach().clone()
            priv = privileged_observation(rt, obs).clone()
            memory_in = memory.clone()
            mean, logstd, _, memory = policy(priv, obs, memory)
            std = logstd.exp().expand_as(mean)
            bias = plans[:, :, min(tick//self.block_steps, plans.shape[2]-1)].reshape(rt.worlds, 7)
            proposal_mean = mean + std*bias if tick < self.intervention_steps else mean
            # A fresh draw conditional on the fixed proposal plan. Corresponding
            # roots share noise across candidates, reducing comparison variance.
            noise = torch.randn((n, 7), generator=generator, device=rt.device_name)
            recorded = batch.factual[tick] if tick < len(batch.factual) else None
            if recorded is not None:
                if recorded.get('noise') is None:
                    raise ValueError('Branch CTI requires recorded factual policy noise')
            noise = noise.repeat(c, 1)
            raw = proposal_mean + std*noise
            gait = self.gait(obs).detach()
            if recorded is not None:
                raw[:n] = recorded['raw_action']
                gait[:n] = recorded['gait']
            action = raw.tanh()
            if recorded is not None:
                action[:n] = recorded['action']
            behavior_log_prob = torch.distributions.Normal(proposal_mean, std).log_prob(raw).sum(-1)
            if recorded is not None:
                # Original factual density, independent of tolerated replay drift.
                behavior_log_prob[:n] = (-.5*recorded['noise'].square()-std[:n].log()-.5*math.log(2*math.pi)).sum(-1)
            rt.set_gait_actions(gait)
            _, _, done, _ = rt.step(action)
            transitions += rt.worlds
            flags = wp.to_torch(rt._flags).bool()
            numerical |= active & flags
            # Preserve the existing numerical fail-fast guard. No corrupt branch
            # can be trained or silently reset into apparently valid evidence.
            rt.check()
            now = signals(rt)
            before = active.clone()
            reward, terminated, truncated, _ = progress.step(now, done)
            if recorded is not None:
                invalid, count, errors = self._replay_check(recorded, now, progress, terminated,
                    truncated, before, tick, drift)
                replay_invalid |= invalid
                checked_steps += count
                for key, value in errors.items():
                    max_errors[key] = torch.maximum(max_errors.get(key, torch.zeros_like(value)), value)
            rows.append(dict(privileged=priv, r84=obs, memory=memory_in, action=action.clone(),
                raw_action=raw.clone(), behavior_log_prob=behavior_log_prob.clone(),
                reward=reward.clone(), terminated=terminated.clone(), truncated=truncated.clone(),
                mask=before, physical_failure=now['failed'].clone()))
            returns += torch.where(before, progress.gamma**tick*reward, 0.)
            finish = before & (terminated | truncated | (tick+1 == self.horizon_steps))
            if bool(finish.any()):
                final_obs = rt.observe().detach().clone()
                _, _, final_value, _ = (value_policy or policy)(privileged_observation(rt, final_obs), final_obs, memory)
                bootstrap[finish] = torch.where(terminated, 0., final_value)[finish]
                final_grade[finish] = progress.graph_score[finish]
                success[finish], failed[finish] = now['success'][finish], now['failed'][finish]
            active &= ~finish
            source_reset = recorded is not None and recorded.get('source_reset', False)
            if (bool(finish.any()) or source_reset) and bool(active.any()):
                rt.reset(finish.to(torch.uint8))
                progress.reset(finish, signals(rt))
                memory[finish] = 0.
        valid = ~replay_invalid.repeat(c) & ~numerical & torch.isfinite(returns)
        for row in rows:
            row['mask'] &= valid
        bootstrap = torch.where(valid, bootstrap, 0.)
        return dict(rows=rows, bootstrap=bootstrap, returns=returns.reshape(c, n),
                    valid=valid.reshape(c, n), grades=final_grade.reshape(c, n),
                    success=success.reshape(c, n), failed=failed.reshape(c, n),
                    replay_invalid=replay_invalid, replay_steps=checked_steps,
                    transitions=transitions, replay_errors=max_errors)

    def run(self, policy, batch):
        if batch is None or len(batch.source_world_ids) != self.roots:
            raise ValueError('CTI requires an actual PPO decision batch with matching roots')
        if batch.initial_progress.reward_profile != GRAPH_PROFILE:
            raise ValueError('CTI roots must use the same reward graph')
        started = time.perf_counter()
        self.rounds += 1
        device = self.runtime.device_name
        generator = torch.Generator(device=device).manual_seed(1307+self.rounds)
        devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        passes, records = [], []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            frozen = deepcopy(policy).eval()
            frozen.load_state_dict(batch.policy_state)
            plans = initial_plans(self.roots, self.alternatives,
                math.ceil(self.intervention_steps/self.block_steps), generator=generator, device=device)
            for iteration in range(self.search_iterations):
                result = self._pass(frozen, batch, plans, seed=71009+100*self.rounds+iteration, value_policy=policy)
                passes.append(result)
                for candidate in range(self.alternatives+1):
                    for world in range(self.roots):
                        valid = bool(result['valid'][candidate, world])
                        records.append(dict(source='ppo', **batch.sources[world],
                            source_policy_version=batch.policy_version, round=self.rounds,
                            pass_name=f'search-{iteration+1}', candidate=candidate, world=world,
                            skill='factual' if candidate == 0 else 'matched-policy' if candidate == 1 else 'sampled-gaussian-proposal',
                            score=float(result['returns'][candidate, world]),
                            graph_score=float(result['grades'][candidate, world]),
                            success=bool(result['success'][candidate, world]),
                            failed=bool(result['failed'][candidate, world]),
                            eligible=valid, learning_data=valid,
                            rejection_reasons=[] if valid else ['factual_replay_mismatch_or_numerical'],
                            score_delta_vs_factual=float(result['returns'][candidate, world]-result['returns'][0, world])))
                if iteration+1 < self.search_iterations:
                    plans = refine_plans(plans, result['returns'], result['valid'], generator=generator)
        # Keep each search pass as a distinct trajectory block along the world
        # axis. Padding is masked; no transition from another attempt leaks in.
        rows = []
        for tick in range(max(len(result['rows']) for result in passes)):
            parts = []
            for result in passes:
                if tick < len(result['rows']):
                    parts.append(result['rows'][tick])
                else:
                    parts.append({key: torch.zeros_like(value) for key, value in result['rows'][-1].items()})
            rows.append({key: torch.cat([part[key] for part in parts]) for key in parts[0]})
        bootstrap = torch.cat([result['bootstrap'] for result in passes])
        valid_transitions = sum(int(row['mask'].sum()) for row in rows)
        valid_branches = sum(int(result['valid'].sum()) for result in passes)
        improvements = sum(int(((result['returns'][1:] > result['returns'][:1]+.005) & result['valid'][1:]).sum()) for result in passes)
        repairs = sum(int((result['success'][1:] & ~result['success'][:1] & result['valid'][1:]).sum()) for result in passes)
        avoided = sum(int((~result['failed'][1:] & result['failed'][:1] & result['valid'][1:]).sum()) for result in passes)
        metrics = dict(version=8, source='ppo', seconds=time.perf_counter()-started,
            transitions=sum(result['transitions'] for result in passes), candidate_branches=self.alternatives+1,
            roots_searched=self.roots, pilot_events=self.roots, search_iterations=self.search_iterations,
            alternatives_compared=self.roots*self.alternatives*self.search_iterations,
            valid_branches=valid_branches, valid_transitions=valid_transitions,
            improved_branches=improvements, successful_repairs=repairs, failure_avoidance=avoided,
            selected_worlds=0, target_actions=0, source_policy_version=batch.policy_version,
            branch_replay_rejected_worlds=int(torch.stack([result['replay_invalid'] for result in passes]).any(0).sum()),
            branch_replay_checked_steps=sum(result['replay_steps'] for result in passes),
            branch_records=records)
        stages = batch.initial_progress.graph_stage
        for stage in range(7):
            source = stages == stage
            count = int(source.sum()) * self.alternatives * len(passes)
            metrics[f'stage_{stage}/branches'] = count
            if count:
                gained = sum(int(((r['grades'][1:] > r['grades'][:1]+.005) & r['valid'][1:] & source).sum()) for r in passes)
                metrics[f'stage_{stage}/physical_progress_fraction'] = gained/count
        return (rows, bootstrap), metrics
