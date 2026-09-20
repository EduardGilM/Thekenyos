"""Scripted deposit after a valid extraction: joint-space carry, lower into the basket, release.

Transport is a planning problem, not a contact problem. The RL policy learns
approach, grasp and extraction; once the fruit is validly extracted and held,
this controller drives the arm targets along two collision-checked joint-space
segments (posture at extraction -> high posture over the basket centre -> low
posture inside the basket) and opens the jaw. Segment feasibility was verified
by forward kinematics on the fixed-base scene; the physics still decides
whether the fruit settles.
"""
import torch

HIGH_Q = (2.93, -1.86, 2.19, -0.78, -0.79, -0.69)   # TCP ~8 cm above the rim, over the centre
LOW_Q = (2.95, -1.65, 2.14, -1.52, -0.39, -0.94)    # TCP ~10 cm below the rim, over the centre
JAW_HOLD = .7
JAW_OPEN = -1.
RELEASE_DWELL_STEPS = 25   # 0.5 s with the jaw open before retracting


class ProprioceptiveHoldDetector:
    """Deployable stand-in for the grasp oracle, from joint state alone.

    Holding: the jaw is commanded (near) closed but the joint has stalled well
    short of fully closed for a while, i.e. it is squeezing something.
    Extracted: while holding, the hand has moved several centimetres from where
    the hold began, so whatever is held came along with it.
    """
    def __init__(self, worlds, device, *, control_dt, jaw_closed_rad=0., stall_margin_rad=.25,
                 command_margin_rad=.35, hold_seconds=.2, travel_m=.05):
        self.control_dt = float(control_dt)
        self.jaw_closed_rad, self.stall_margin_rad = float(jaw_closed_rad), float(stall_margin_rad)
        self.command_margin_rad, self.hold_seconds, self.travel_m = float(command_margin_rad), float(hold_seconds), float(travel_m)
        self.hold_time = torch.zeros(worlds, device=device)
        self.hold_origin = torch.zeros(worlds, 3, device=device)
        self.holding = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.extracted = torch.zeros(worlds, dtype=torch.bool, device=device)

    def reset(self, mask):
        self.hold_time[mask] = 0.
        self.holding[mask] = False
        self.extracted[mask] = False

    def update(self, jaw_angle, jaw_target, hand_position):
        squeezing = ((jaw_target > self.jaw_closed_rad - self.command_margin_rad)
                     & (jaw_angle < self.jaw_closed_rad - self.stall_margin_rad))
        self.hold_time = torch.where(squeezing, self.hold_time + self.control_dt, torch.zeros_like(self.hold_time))
        newly = (self.hold_time >= self.hold_seconds) & ~self.holding
        self.hold_origin = torch.where(newly[:, None], hand_position, self.hold_origin)
        self.holding = self.hold_time >= self.hold_seconds
        travelled = (hand_position - self.hold_origin).norm(dim=-1) > self.travel_m
        self.extracted |= self.holding & travelled
        self.extracted &= self.holding
        return self.holding, self.extracted


class ScriptedDeposit:
    def __init__(self, worlds, device, *, max_delta, settle_steps=15):
        self.max_delta = float(max_delta)
        self.high = torch.tensor(HIGH_Q, device=device)
        self.low = torch.tensor(LOW_Q, device=device)
        self.active = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.s1 = torch.zeros(worlds, device=device)
        self.s2 = torch.zeros(worlds, device=device)
        self.released = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.open_steps = torch.zeros(worlds, dtype=torch.long, device=device)
        self.start_q = torch.zeros(worlds, 6, device=device)
        self.jaw_hold = torch.full((worlds,), JAW_HOLD, device=device)
        self.settle_steps = int(settle_steps)
        self.settle = torch.zeros(worlds, dtype=torch.long, device=device)

    def reset(self, mask):
        self.active[mask] = False
        self.s1[mask] = 0.
        self.s2[mask] = 0.
        self.released[mask] = False
        self.jaw_hold[mask] = JAW_HOLD
        self.settle[mask] = 0
        self.open_steps[mask] = 0

    def act(self, progress, arm_targets, policy_actions, *, engage=None, arm_q=None):
        """Return actions with scripted worlds overridden.

        By default the simulator grasp oracle decides when extraction is complete
        and held. Pass ``engage`` (and the current ``arm_q``) to use another
        trigger, e.g. the proprioceptive detector; the carry then starts from
        the posture at that moment.
        """
        if engage is None:
            held = progress.graph_valid_extract & progress.graph_grip
            engage = progress.graph_completed['extract'] & held
        newly_engaged = engage & ~self.active
        # Continuity at handover: the path starts from the controller's current
        # TARGETS, not the measured angles. Mid-extraction the targets lead the
        # joints (that lead is the pull); snapping them back onto the joints
        # reverses the force while the arm is still moving and sheds the fruit.
        self.start_q = torch.where(newly_engaged[:, None], arm_targets, self.start_q)
        start = self.start_q
        # Never loosen the grip at handover: keep the policy's own jaw command
        # if it was squeezing harder than the default hold.
        self.jaw_hold = torch.where(newly_engaged, policy_actions[:, 6].clamp_min(JAW_HOLD), self.jaw_hold)
        self.active |= engage
        # Hold still briefly so the extraction momentum dies with the grip intact.
        self.settle = torch.where(self.active, self.settle + 1, self.settle)
        moving = self.active & (self.settle > self.settle_steps)
        line1 = self.high - start
        span1 = line1.abs().amax(-1).clamp_min(1e-6)
        seg2 = self.active & (self.s1 >= 1.)
        self.s1 = torch.where(moving, (self.s1 + self.max_delta / span1).clamp(max=1.), self.s1)
        line2 = self.low - self.high
        span2 = line2.abs().amax().clamp_min(1e-6)
        self.s2 = torch.where(seg2, (self.s2 + self.max_delta / span2).clamp(max=1.), self.s2)
        reference = torch.where(seg2[:, None], self.high + self.s2[:, None] * line2, start + self.s1[:, None] * line1)
        at_low = seg2 & (self.s2 >= 1.) & ((arm_targets - self.low).abs().amax(-1) < .02)
        self.released |= at_low
        # After opening, wait for the jaw to clear the fruit, then retract to the
        # high posture so the hand no longer touches the fruit (the settle
        # criterion requires no hand contact).
        self.open_steps = torch.where(self.released, self.open_steps + 1, self.open_steps)
        retract = self.released & (self.open_steps > RELEASE_DWELL_STEPS)
        reference = torch.where(retract[:, None], self.high, reference)
        arm = ((reference - arm_targets) / self.max_delta).clamp(-1., 1.)
        jaw = torch.where(self.released, JAW_OPEN, self.jaw_hold)
        scripted = torch.cat((arm, jaw[:, None]), -1)
        return torch.where(self.active[:, None], scripted, policy_actions)
