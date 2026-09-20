"""Reward/v3: event-based terms with an irreversible per-fruit ledger.

Damage and spill are reported separately and never merged silently.
Each fruit pays deposit/loss/spill at most once; flags are never cleared
by re-entry or phase changes, only by episode reset.

Weights below are initial engineering values (reward/v3), not tuned
results. Callers must log every term separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Initial weights (reward/v3). Tunable with version bump + logging.
W_DEPOSIT = 20.0
W_GRASP_STABLE = 0.5
W_DETACH_HELD = 2.0
# One-shot RL breadcrumbs after detach. Engineering, not a measured harvest
# value. The origin (detach COM) is never paid; eval keeps guidance at 0.
W_CARRY_LINE = 8.0
CARRY_LINE_POINTS = 8
CARRY_LINE_RADIUS_M = 0.10
CARRY_LINE_MAX_POINTS = 8
W_LOSS = -25.0
W_SPILL = -25.0
W_DAMAGE_PER_UNIT = -5.0
W_FALL = -100.0
W_TIME_PER_S = -0.05
W_SMOOTH = -0.01
W_FALSE_FINISH = -2.0
W_OBSTACLE_CONTACT_PER_S = -1.0


def carry_line_waypoints_world_m(origin_xyz, target_xyz, n_points=CARRY_LINE_POINTS):
    """Straight detach→opening samples. Skips the start so detach is not free.

    Points sit at t = 1/n … 1. The last sample is the opening target, not the
    liner floor. Fruit stays a free body.
    """
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    n = int(n_points)
    if not np.isfinite(origin).all() or not np.isfinite(target).all():
        raise ValueError('carry-line ends must be finite')
    if n < 1 or n > CARRY_LINE_MAX_POINTS:
        raise ValueError(f'n_points must be in [1, {CARRY_LINE_MAX_POINTS}]')
    ts = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    return origin.reshape(1, 3) * (1.0 - ts.reshape(-1, 1)) + target.reshape(1, 3) * ts.reshape(-1, 1)


def pay_carry_line_once(fruit_xyz, origin_xyz, target_xyz, paid_mask, *,
                        radius_m=CARRY_LINE_RADIUS_M, bonus=W_CARRY_LINE,
                        n_points=CARRY_LINE_POINTS, armed=True, grasped=True):
    """Pay each unpaid waypoint at most once when the fruit COM first enters it.

    The detach origin is not a waypoint. A dropped fruit (not grasped) does not
    collect crumbs while falling. Returns (reward, new_mask, n_hits).
    """
    fruit = np.asarray(fruit_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(fruit).all():
        raise ValueError('fruit_xyz must be finite')
    radius = float(radius_m)
    pay = float(bonus)
    if not np.isfinite(radius) or not 0.0 < radius <= 0.25:
        raise ValueError('radius_m must be finite in (0, 0.25] m')
    if not np.isfinite(pay) or not 0.0 < pay <= 20.0:
        raise ValueError('bonus must be finite in (0, 20]')
    mask = int(paid_mask)
    if mask < 0:
        raise ValueError('paid_mask must be a non-negative bitfield')
    if not armed or not grasped:
        return 0.0, mask, 0
    points = carry_line_waypoints_world_m(origin_xyz, target_xyz, n_points=n_points)
    reward = 0.0
    hits = 0
    for index, point in enumerate(points):
        flag = 1 << index
        if mask & flag:
            continue
        if float(np.linalg.norm(fruit - point)) <= radius:
            mask |= flag
            reward += pay
            hits += 1
    return float(reward), int(mask), int(hits)


def approach_center_potential(fruit_xyz, tcp_xyz, basket_xyz, length_m=0.60):
    """Hand-heavy mix of fruit-to-target and hand-XY-to-target potentials.

    Easy mode passes the open hover (rim + 28 cm), not the liner floor:
    a 10 cm-over-hole TCP puts the ~0.20 m wrist through the crate, and
    paying fruit 3D to the floor taught the arm to ram the walls. Hand
    stays XY-only so PPO is not paid to dive the wrist after release.
    Engineering shaping, not a measured harvest value.
    """
    fruit = np.asarray(fruit_xyz, dtype=np.float64).reshape(3)
    tcp = np.asarray(tcp_xyz, dtype=np.float64).reshape(3)
    basket = np.asarray(basket_xyz, dtype=np.float64).reshape(3)
    length = float(length_m)
    if not np.isfinite(fruit).all() or not np.isfinite(tcp).all() or not np.isfinite(basket).all():
        raise ValueError('approach-center inputs must be finite')
    if not np.isfinite(length) or not 0.05 <= length <= 2.0:
        raise ValueError('length_m must be finite in [0.05, 2.0] m')
    d_fruit = float(np.linalg.norm(fruit - basket))
    d_hand_xy = float(np.linalg.norm(tcp[:2] - basket[:2]))
    return 0.25 * float(np.exp(-d_fruit / length)) + 0.75 * float(np.exp(-d_hand_xy / length))


@dataclass
class FruitLedger:
    """Irreversible per-fruit event state for one episode.

    fruit_ids: stable integer ids (e.g. body indices) fixed at reset.
    """

    fruit_ids: list
    deposit_paid: np.ndarray = field(init=False)
    lost_once: np.ndarray = field(init=False)
    spilled_once: np.ndarray = field(init=False)
    damage_sum: float = 0.0

    def __post_init__(self):
        n = len(self.fruit_ids)
        self.deposit_paid = np.zeros(n, dtype=bool)
        self.lost_once = np.zeros(n, dtype=bool)
        self.spilled_once = np.zeros(n, dtype=bool)
        self.index_of = {fid: i for i, fid in enumerate(self.fruit_ids)}

    def reset(self):
        self.deposit_paid[:] = False
        self.lost_once[:] = False
        self.spilled_once[:] = False
        self.damage_sum = 0.0

    def _idx(self, fid) -> int:
        try:
            return self.index_of[fid]
        except KeyError:
            raise ValueError(f"unknown fruit id {fid!r}") from None


@dataclass
class RewardState:
    """Accumulators for dense shaping baselines (per skill reference)."""

    phi_prev: float | None = None
    ref_mode: str | None = None


def shaping_potential(distance_m: float, scale_m: float = 0.25) -> float:
    """Potential Phi(d) = exp(-d/scale); finite-input validated."""
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError(f"distance_m={distance_m} must be finite >= 0")
    return float(np.exp(-distance_m / scale_m))


def shaping_step(state: RewardState, distance_m: float, gamma: float,
                 ref_mode: str) -> tuple[float, float]:
    """Potential-based shaping 2*(g*Phi' - Phi); resets baseline on ref change.

    Returns (reward, phi). Never rewards a baseline jump: changing reference
    or mode restarts the baseline with zero reward for that step.
    """
    phi = shaping_potential(distance_m)
    if state.ref_mode != ref_mode or state.phi_prev is None:
        state.ref_mode, state.phi_prev = ref_mode, phi
        return 0.0, phi
    r = 2.0 * (gamma * phi - state.phi_prev)
    state.phi_prev = phi
    return float(r), phi


class RewardEvaluator:
    """Computes separated reward terms from physical evaluator state.

    The evaluator (not the arbiter/actor) observes ground truth: detached,
    held, stored, spilled, damage increments. Actors never see this object.
    """

    def __init__(self, fruit_ids: list):
        self.ledger = FruitLedger(list(fruit_ids))
        self.terms: dict[str, float] = {}

    def reset(self, fruit_ids: list | None = None):
        if fruit_ids is not None:
            self.ledger = FruitLedger(list(fruit_ids))
        else:
            self.ledger.reset()
        self.terms = {}

    def event_deposit(self, fid) -> float:
        """+W_DEPOSIT once per fruit on confirmed stored (evaluator-side)."""
        i = self.ledger._idx(fid)
        if self.ledger.deposit_paid[i] or self.ledger.lost_once[i] or self.ledger.spilled_once[i]:
            return 0.0
        self.ledger.deposit_paid[i] = True
        return W_DEPOSIT

    def event_loss(self, fid) -> float:
        """W_LOSS once per fruit lost before storage (ground/outside)."""
        i = self.ledger._idx(fid)
        if self.ledger.lost_once[i] or self.ledger.spilled_once[i]:
            return 0.0
        self.ledger.lost_once[i] = True
        return W_LOSS

    def event_spill(self, fid) -> float:
        """W_SPILL once per previously-stored (or preloaded) fruit spilled."""
        i = self.ledger._idx(fid)
        if self.ledger.spilled_once[i] or self.ledger.lost_once[i]:
            return 0.0
        self.ledger.spilled_once[i] = True
        return W_SPILL

    def damage_increment(self, delta_damage: float) -> float:
        """Damage term kept separate from spill; proxy, not bruise prediction."""
        if not np.isfinite(delta_damage) or delta_damage < 0:
            raise ValueError(f"delta_damage={delta_damage} must be finite >= 0")
        self.ledger.damage_sum += float(delta_damage)
        return W_DAMAGE_PER_UNIT * float(delta_damage)

    def register_preloaded(self, fid):
        """Preloaded basket fruit counts as STORED without deposit reward."""
        i = self.ledger._idx(fid)
        self.ledger.deposit_paid[i] = True
