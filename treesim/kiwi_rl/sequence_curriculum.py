"""Increasing episode objectives, with physical prerequisites and dense feedback."""
import torch
from .harvest_training import EpisodeProgress, REWARD_PROFILE
from .reward_graph import SEQUENCE_PROFILE, STAGE_NAMES

CHECKPOINTS = ('position', 'grip', 'extract', 'carry', 'deposit')
# The unlocked objective is deliberately NOT an actor input. A one-hot of the
# curriculum level switched behaviour off at every promotion: the same weights
# grasped 64/64 under the level-2 code and 0/64 under the unseen level-3 code.
# Episode checkpoint history is enough; the level only changes where episodes end.
SEQUENCE_OBSERVATION_DIM = 16
SEQUENCE_SCHEMA = 'fast-harvest-sequence-r84/v5'  # observation layout unchanged by the seated-grasp rule
# Undiscounted returns made detours free: the critic's value peaked 28 cm from
# the fruit and PPO followed that gradient until the policy parked there. A 4 s
# horizon (50 Hz) makes delay cost about half a milestone; each stage's own
# milestone stays well inside the horizon.
SEQUENCE_GAMMA = .995
LIVE_CREDIT = .5
# Position counts only when the hand has stopped with the fruit between the
# jaws: a 60 ms pass-through taught touch-and-go, which the grip stage then
# had to unlearn.
POSITION_HOLD_S = .5
POSITION_HOLD_SLIP_M_S = .05
# A secure grasp is a seated grasp: fruit centre no further out than this past the
# jaw TCP. Randomised positions taught tip grasps (2.8-3.3 cm out) that were shed
# after detachment; seated grasps (<=1.5 cm) carried. Shaping pulls toward zero.
SEATED_DEPTH_M = .02
TIME_BUDGET = .1
FAILURE_COST = .25
BELOW_RIM_FACTOR = .25    # carry credit kept while the held fruit is below the rim outside the basket
BELOW_RIM_STEP_COST = .002  # per control step (0.1 per second) while held fruit is below the rim outside
CRUISE_HEIGHT_M = .075    # release height of the fruit centre above the rim: ~radius plus clearance
# Release posture over the basket centre (sh0, sh1, el0, el1, wr0, wr1), found
# by collision-checked sampling on the fixed-base scene; the joint-space line
# from the extraction posture to it clears the body, basket and canopy.
CARRY_TARGET_Q = (2.93, -1.86, 2.19, -0.78, -0.79, -0.69)
CARRY_DEVIATION_RAD = 2.  # joint-space distance from the line at which deviation credit halves
RELEASE_TOLERANCE_M = .06 # release may happen from this far above the cruise height ...
BASKET_DEPTH_M = .28      # ... down to the basket floor: a low release inside the basket is the safest deposit
# Live credit per stage; carry gets twice the others so the height floor and
# horizontal progress dominate exploration noise over a long joint-space path.
STAGE_LIVE_CREDIT = (.5, .5, .5, 2., .5)
CARRY_REWARD_VERSION = 'carry-v8: joint-space line (linear potential, loose deviation), stall keeps live credit, carry credit x4 from extraction posture to a fixed release posture, rim-dip cost, carry credit x2'



class Curriculum:
    def __init__(self, level=1, streak=0, max_level=5):
        if level not in range(1, 6) or max_level not in range(1, 6):
            raise ValueError('Sequence objective must be in [1, 5]')
        self.level, self.streak, self.max_level = level, streak, max_level

    def consider(self, evaluations):
        """Every fresh validation trial must pass, twice; count episodes, not frames."""
        passed = len(evaluations) == 2 and all(
            r.get('episodes', 0) >= 32 and r.get('prefix_success', 0) >= .9
            and r.get('physical_failure', 1) <= .05 for r in evaluations)
        self.streak = self.streak + 1 if passed else 0
        promoted = self.level < self.max_level and self.streak >= 2
        if promoted:
            self.level += 1
            self.streak = 0
        return promoted

    def state(self):
        return dict(level=self.level, streak=self.streak, max_level=self.max_level)


class SequenceProgress(EpisodeProgress):
    """Checkpoints record ordered history; current control is checked separately.

    Events pay once, immediately, independently of the unlocked objective.
    Live guidance is a separate bounded balance; failure clears only that balance.
    At the finite deadline, safe partial quality remains explicit partial credit.
    This is curriculum guidance, not policy-invariant shaping or skill retention.
    """
    def __init__(self, initial, *, curriculum_stage=1, rehearse=False, **kwargs):
        kwargs.pop('reward_profile', None)
        if kwargs.pop('gamma', SEQUENCE_GAMMA) != SEQUENCE_GAMMA:
            raise ValueError(f'Sequence rewards use the fixed discount {SEQUENCE_GAMMA}')
        kwargs['gamma'] = SEQUENCE_GAMMA
        super().__init__(initial, reward_profile=REWARD_PROFILE, curriculum_stage=0, **kwargs)
        if curriculum_stage not in range(1, 6):
            raise ValueError('Sequence objective must be in [1, 5]')
        self.reward_profile = SEQUENCE_PROFILE
        self.curriculum_stage = curriculum_stage
        self.curriculum_stage_ids.fill_(curriculum_stage)
        self.rehearse = rehearse
        if rehearse and curriculum_stage > 1:
            # Twenty percent of worlds keep earlier objectives, always from home.
            indices = torch.arange(self.age.numel(), device=self.age.device)
            older = 1 + (indices // 5) % (curriculum_stage - 1)
            self.curriculum_stage_ids = torch.where(indices % 5 == 0, older, self.curriculum_stage_ids)
        self.prefix_success = torch.zeros_like(self.ever_grasp)
        self.timeout = torch.zeros_like(self.ever_grasp)
        self.failure = torch.zeros_like(self.ever_grasp)
        self.sequence_invalid = torch.zeros_like(self.ever_grasp)
        self.carry_dwell = torch.zeros_like(self.closest)
        self.extract_dwell = torch.zeros_like(self.closest)
        self.unheld_time = torch.zeros_like(self.closest)
        self.graph_components = {k: torch.zeros_like(self.closest) for k in STAGE_NAMES}
        self.milestone_reward = {k: torch.zeros_like(self.closest) for k in STAGE_NAMES}
        self.live_components = {k: torch.zeros_like(self.closest) for k in STAGE_NAMES}
        self.carry_start_q = initial['arm_q'].clone()
        self._previous_over_basket = torch.zeros_like(self.ever_grasp)
        self.live_components['position'] = LIVE_CREDIT * self.qualities(initial)[0]
        self.graph_components['position'] = self.live_components['position'].clone()
        self.graph_score = self.live_components['position'].clone()
        self.graph_reference = self.graph_score.clone()
        self.time_reward = torch.zeros_like(self.closest)
        self.failure_reward = torch.zeros_like(self.closest)
        self._below_rim = torch.zeros_like(self.ever_grasp)
        self._previous_below_rim = torch.zeros_like(self.ever_grasp)

    def qualities(self, now):
        gap = now['finger_gap'] + now['jaw_gap']
        # Far-field approach remains dense; physical surfaces guide the last cm.
        position = .6 * now['insertion'] + .4 / (1 + gap / .1)
        # Measured bilateral contact takes precedence over approximate geometry
        # and GJK witness directions, which are ambiguous during penetration.
        position = torch.where(now['bilateral'], 1., position)
        contact = (now['grasp_time'] / .1).clamp(0, 1)
        # Keep insertion and opposing geometry relevant after the position event.
        # These are continuous scores: contact exploration has no boolean gate.
        alignment = now['insertion'].clamp(0, 1) * now['opposition'].clamp(0, 1)
        alignment = torch.where(now['bilateral'], 1., alignment)
        seat = 1 / (1 + now['grasp_depth'].clamp_min(0) / .02)
        grip = .5 * position * (.5 + .5 * seat) + .5 * alignment * seat * (.7 / (1 + gap / .04) + .3 * contact)
        loading = torch.where(now['detached'], (self.extract_dwell / .1).clamp(0, 1),
                              (now['stem_force'] / 8.).clamp(0, 1))
        extract = .5 * grip + .5 * self.graph_grip * loading
        held = self.graph_valid_extract & self.graph_grip
        # Carry follows a straight line in joint space from the posture at
        # extraction to a fixed, collision-checked release posture over the
        # basket centre (the Cartesian straight line needs a posture flip
        # halfway and cannot be traced by this arm). Progress is the projection
        # onto that line; deviation from it costs. Dipping below the rim
        # beside the basket still loses most of the credit.
        inside = now['basket_horizontal'] <= 0.
        below_rim = (now['basket_height'] < 0.) & ~inside
        target = now['arm_q'].new_tensor(CARRY_TARGET_Q)
        line = target - self.carry_start_q
        length2 = line.square().sum(-1).clamp_min(1e-6)
        offset = now['arm_q'] - self.carry_start_q
        # Progress is a linear potential (negative when moving away, so every
        # exploratory move has a gradient); deviation only bites for gross
        # departures, otherwise any first move off the line is punished and
        # freezing becomes the local optimum.
        progress = ((offset * line).sum(-1) / length2).clamp(-.5, 1)
        deviation = (offset - progress[:, None] * line).norm(dim=-1)
        carry = held * torch.where(below_rim, BELOW_RIM_FACTOR, 1.) * (.8 * progress + .2 / (1 + deviation / CARRY_DEVIATION_RAD))
        self._below_rim = held & below_rim
        # Opening creates physical clearance. Release credit is available only
        # over the basket, after a valid extraction and transport.
        over = held & self.over_basket(now)
        clearance = (torch.maximum(now['finger_gap'], now['jaw_gap']) / .02).clamp(0, 1)
        dropping = (-now['basket_height'] / .1).clamp(0, 1)   # descending below the rim inside the basket
        deposit = torch.where(self.graph_valid_release, .5 + .5 * dropping, over * (.25 + .25 * clearance))
        return (position, grip, extract, carry, deposit)

    def over_basket(self, now):
        """Held fruit anywhere inside the basket footprint, from just above the rim down to the floor."""
        return ((now['basket_horizontal'] <= 0.) & (now['basket_height'] <= CRUISE_HEIGHT_M + RELEASE_TOLERANCE_M)
                & (now['basket_height'] >= -BASKET_DEPTH_M))

    def observation(self):
        now = self.previous
        history = torch.stack([self.graph_completed[k] for k in CHECKPOINTS], -1).float()
        clocks = torch.stack((now['grasp_time'] / .1, self.graph_position_dwell / POSITION_HOLD_S,
            self.carry_dwell / .1, self.unheld_time / .12, now['settle_time'] / 2.,
            self.age / self.max_steps, now['finger_gap'] / .1, now['jaw_gap'] / .1,
            now['opposition'], self.graph_valid_release.float(), self.extract_dwell / .1), -1).clamp(-1, 2)
        return torch.cat((history, clocks), -1)

    def reset(self, mask, initial):
        fresh = SequenceProgress(initial, curriculum_stage=self.curriculum_stage,
            stall_steps=self.stall_steps, max_steps=self.max_steps, guidance=self.guidance,
            gamma=self.gamma, control_dt=self.control_dt, rehearse=self.rehearse)
        for name, value in vars(fresh).items():
            current = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                current[mask] = value[mask]
            elif isinstance(value, dict):
                for key, tensor in value.items():
                    current[key][mask] = tensor[mask]

    def step(self, now, physical_done):
        self.age += 1
        self.stale += 1
        previous_completed = {k: v.clone() for k, v in self.graph_completed.items()}
        previous_grip = self.graph_grip.clone()
        previous_position = self.graph_position_dwell >= POSITION_HOLD_S
        previous_detached = self.graph_detached.clone()
        safe = (~now['failed'] & ~(physical_done & ~now['success'])
                & (now['max_load'] <= 15.) & (now['damage'] <= .05))
        positioned = ((now['enclosed'] & (torch.maximum(now['finger_gap'], now['jaw_gap']) <= .04)
                       & (now['opposition'] >= .2)) | now['bilateral']) & safe
        self.graph_position_dwell = torch.where(
            positioned & ~now['detached'] & (now['slip'] < POSITION_HOLD_SLIP_M_S),
            self.graph_position_dwell + self.control_dt, 0.)
        self.graph_grip = (safe & now['holding'] & now['bilateral'] & (now['slip'] < .08)
                           & (now['grasp_depth'] <= SEATED_DEPTH_M))
        completed = self.graph_completed
        completed['position'] |= (self.graph_position_dwell >= POSITION_HOLD_S) | (self.graph_grip & ~now['detached'])
        completed['grip'] |= completed['position'] & self.graph_grip & ~now['detached']
        self.graph_detached |= now['detached']
        new_detach = self.graph_detached & ~previous_detached
        completed['grip'] |= completed['position'] & new_detach & now['held_at_detach'] & self.graph_grip
        # Substep latch verifies grasp at the actual stem break, not merely
        # contact observed in a later 50 Hz sample.
        self.graph_valid_extract |= (new_detach & completed['grip'] & now['held_at_detach'] & self.graph_grip)
        self.graph_invalid_extract |= new_detach & ~self.graph_valid_extract
        held = self.graph_valid_extract & self.graph_grip
        self.extract_dwell = torch.where(held, self.extract_dwell + self.control_dt, 0.)
        extract_before = completed['extract'].clone()
        completed['extract'] |= self.extract_dwell >= .1
        started = completed['extract'] & ~extract_before
        self.carry_start_q = torch.where(started[:, None], now['arm_q'], self.carry_start_q)
        over_basket = self.over_basket(now)
        self.carry_dwell = torch.where(held & over_basket,
                                      self.carry_dwell + self.control_dt, 0.)
        completed['carry'] |= completed['extract'] & (self.carry_dwell >= .1)
        release = (completed['carry'] & previous_grip & ~self.graph_grip & safe
                   & self._previous_over_basket)
        self._previous_over_basket = over_basket
        self.graph_valid_release |= release
        self.graph_valid_release &= safe & ~now['ground_contact'] & ~self.graph_grip
        unheld = self.graph_detached & ~self.graph_grip & ~self.graph_valid_release
        self.unheld_time = torch.where(unheld, self.unheld_time + self.control_dt, 0.)
        self.graph_ground_drop = self.graph_detached & now['ground_contact']
        self.sequence_invalid |= self.graph_invalid_extract | self.graph_ground_drop | (self.unheld_time >= .12)
        # Success is the fruit inside the basket after a valid extraction and
        # release: within the footprint, below the rim, not held. No settle
        # timer: a moving robot never lets a rigid fruit settle.
        inside_basket = ((now['basket_horizontal'] <= 0.) & (now['basket_height'] < 0.)
                         & (now['basket_height'] >= -BASKET_DEPTH_M))
        self.graph_success = (safe & ~self.sequence_invalid & completed['carry'] & ~self.graph_grip
                              & self.graph_valid_release & self.graph_valid_extract & inside_basket)
        completed['deposit'] |= self.graph_success
        completed['complete'] |= self.graph_success
        history = torch.stack([completed[k] for k in CHECKPOINTS], -1)
        count = history.long().sum(-1)
        self.prefix_success = (count >= self.curriculum_stage_ids) & safe & ~self.sequence_invalid
        failure = ~safe | self.sequence_invalid | (physical_done & ~self.graph_success)
        self.failure = failure
        self.timeout = (self.age >= self.max_steps) & ~self.prefix_success & ~failure
        # Sustained inactivity ends the episode. The live credit is KEPT (as at
        # the deadline): forfeiting it made the shaping telescope to exactly
        # zero in every stalled episode, which erased the carry gradient.
        stalled = (self.stale >= self.stall_steps) & ~self.prefix_success & ~failure
        self.timeout |= stalled
        # The deadline ends the task. PPO buffer boundaries still bootstrap.
        terminated = self.prefix_success | failure | self.timeout
        truncated = torch.zeros_like(terminated)
        quality = torch.stack(self.qualities(now), -1).gather(1, count.clamp(max=4)[:, None]).squeeze(1)
        quality = torch.where(self.prefix_success | failure, 0., quality.clamp(0, 1))
        credit = quality.new_tensor(STAGE_LIVE_CREDIT)[count.clamp(max=4)]
        score = count.float() + credit * quality
        improved = score > self.graph_reference + .005
        self.stale[improved] = 0
        self.graph_reference = torch.where(improved | (score < self.graph_reference), score, self.graph_reference)
        self.time_reward = (torch.full_like(score, -TIME_BUDGET / self.max_steps)
                            - BELOW_RIM_STEP_COST * self._below_rim.float())
        self.failure_reward = -FAILURE_COST * failure.float()
        self.graph_score = score
        self.graph_stage = count
        self.graph_enclosed = positioned
        self.graph_slip = now['slip'].clone()
        self.graph_forward += (count > sum(v.long() for k, v in previous_completed.items() if k in CHECKPOINTS)).long()
        self.graph_backward += (previous_grip & ~self.graph_grip & ~self.graph_valid_release).long()
        self.ever_grasp |= completed['grip']
        self.ever_detached |= now['detached']
        self.ever_held_detach |= self.graph_valid_extract
        self.closest = torch.minimum(self.closest, now['distance'])
        for i, name in enumerate(STAGE_NAMES):
            entered = torch.ones_like(self.prefix_success) if i == 0 else completed[STAGE_NAMES[i-1]]
            self.graph_reached[name] |= entered
            self.graph_time[name] += ((count == i) & (self.curriculum_stage_ids > i)) * self.control_dt
            live = credit * quality * (count == i) if i < 5 else torch.zeros_like(score)
            event = (completed[name] & ~previous_completed[name] & ~failure
                     & (self.curriculum_stage_ids > i)).float() if i < 5 else torch.zeros_like(score)
            delta = live - self.live_components[name]
            self.milestone_reward[name] = event
            self.graph_progress_reward[name] = event + delta
            self.graph_gain_reward[name] = delta.clamp_min(0)
            self.graph_loss_reward[name] = (-delta).clamp_min(0)
            self.live_components[name] = live
            self.graph_components[name] = completed[name].float() + live if i < 5 else live
        self.shaping_reward = sum(self.graph_progress_reward.values()) - sum(self.milestone_reward.values())
        self.task_reward = sum(self.milestone_reward.values()) + self.time_reward + self.failure_reward
        self.graph_regressions['position'] += (previous_position & ~positioned & ~now['detached']).long()
        lost_grip = previous_grip & ~self.graph_grip & ~self.graph_valid_release
        self.graph_regressions['grip'] += lost_grip.long()
        self.graph_regressions['carry'] += (lost_grip & previous_detached).long()
        self.graph_regressions['carry'] += (self._below_rim & ~self._previous_below_rim).long()
        self._previous_below_rim = self._below_rim.clone()
        self.graph_regressions['deposit'] += ((self.previous['settle_time'] > 0)
            & (now['settle_time'] == 0) & ~self.graph_success).long()
        self.previous = {k: v.clone() for k, v in now.items()}
        return self.task_reward + self.shaping_reward, terminated, truncated, stalled
