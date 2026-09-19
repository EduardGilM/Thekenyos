"""TK-RL-003 v3.0 task curriculum (docs/rl-blueprint.html §5).

Six stages: deposit from pixels, grasp/detach, stationary harvest,
visual approach, multi-fruit mission, then held-out generalisation.
This module owns goals, mix, reset recipes and promotion gates. It does
not implement V3 ResNet-18, 240×320 pixels, or separate N3/M3 GRUs;
those remain the full-spec networks. The batched rigid trainer uses
these recipes on the compact RGB-D actor.

training_ready stays false. A promoted stage is not field harvest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schemas as S

GOALS = S.GOALS
GOAL_ID = {name: i for i, name in enumerate(GOALS)}

RESET_HANGING = 0
RESET_DEPOSIT = 1
RESET_PREGRASP = 2
RESET_APPROACH = 3

RESET_FOR_GOAL = {
    'DEPOSIT_ONLY': RESET_DEPOSIT,
    'DETACH_ONLY': RESET_PREGRASP,
    'HARVEST': RESET_HANGING,
}

# Blueprint §8.2 engineering gates: two consecutive evals, 200 episodes/skill.
PROMOTE_STREAK = 2
POLICY_DT_S = 0.02
# Full stage budgets are 30–900 s. A 2560-world eval cannot store RGBD for that
# long; cap wall-clock so promotion can still observe a deposit (0.5 s settle)
# without OOM. Multi-fruit needs a longer cap than one-fruit stages.
EVAL_CAP_S = 45.0
MULTI_EVAL_CAP_S = 90.0
EVAL_PROFILES = ('default', 'speedrun')
# Speedrun eval caps keep the 0.5 s settle visible; they do not shorten training
# episode timeouts or lower promotion gates. Deposit 8 s is ~16× settle time.
SPEEDRUN_EVAL_CAP_S = {
    'deposit_pixels': 8.0,
    'grasp_detach': 15.0,
    'stationary_harvest': 20.0,
    'visual_approach': 20.0,
    'multi_harvest': 45.0,
    'generalise': 45.0,
}
# Wall-clock / PPO knobs only. Fruit stays free; oracle stays an evaluator.
SPEEDRUN_PRESET = {
    'eval_every': 100,
    'checkpoint_every': 50,
    'entropy_coef': 0.01,
    'video_every': 50,
    'mask_idle_locomotion': True,
    'eval_profile': 'speedrun',
}
# Experimental student-side facilitation. Does not weld fruit, teleport into
# the liner, change promotion gates, or turn the oracle into an action teacher.
# Eval still uses guidance_weight=0 and teacher_mix=0.
EASY_PRESET = {
    'teacher_mix': 0.0,
    'teacher_horizon_updates': 60,
    'shaping_coef': 5.0,
    'open_xy_m': 0.15,
    'hover_clearance_m': 0.28,
    # Carry from the original outside-crate start. Reward fruit 3D and hand
    # XY toward the basket centre; force the jaw open once both are over
    # the hole. Over-opening starts stay available behind this flag.
    # The hold sweep stays at 0.40 m / 0.28 m so a closer student pose
    # cannot poison close-fraction.
    # 0.25 m shaping is flat at 0.7–1.2 m; 0.60 m is an engineering lever,
    # not a measured length.
    'start_over_opening': False,
    'shape_hand_and_fruit': True,
    'release_at_center': True,
    'start_open_radius_m': 0.06,
    'start_inset_x_m': 0.0,
    'start_margin_m': 0.32,
    'start_clearance_m': 0.10,
    'shaping_length_m': 0.60,
    'n_start_poses': 24,
    'start_x_span_m': 0.08,
    'start_y_span_m': 0.04,
    # Kept for the outside-crate restore path. Over-opening samples a disk
    # around CENTER + [inset_x, 0], not this side Y.
    'start_side_y_m': 0.10,
    'start_z_span_m': 0.08,
    # +20 deposit lost to -25 ground, so worlds that almost succeed learn to
    # stay away. 100 was still small next to a window of ground hits.
    # 500 is the enable_easy ceiling: an engineering jackpot, not a
    # measured harvest value.
    'deposit_reward': 500.0,
    'ik_accept_err_m': 0.025,
    'n_hold_levels': 10,
    'hold_close_min': 0.25,
    'hold_close_max': 0.70,
    'far_horizon_updates': 200,
    'default_shaping_coef': 2.0,
    'default_shaping_length_m': 0.25,
}

# Hold sweep stays farther out / higher than training starts so a closer
# student pose cannot knock the free fruit into the front wall.
HOLD_SWEEP_MARGIN_M = 0.40
HOLD_SWEEP_CLEARANCE_M = 0.28


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    goal: str
    budget_s: float
    trains: tuple[str, ...]
    mix: dict[str, float]
    reset: int
    allow_locomotion: bool
    fruit_count: int
    continue_after_success: bool
    gate_success_rate: float
    gate_episodes: int
    guidance_weight: float
    randomize_layout: bool
    description: str

    def __post_init__(self):
        if self.goal not in GOALS:
            raise ValueError(f'unknown goal {self.goal!r}')
        if not np.isfinite([self.budget_s, self.gate_success_rate, self.guidance_weight]).all():
            raise ValueError('stage numeric fields must be finite')
        if self.budget_s <= 0 or not 0 <= self.gate_success_rate <= 1 or self.guidance_weight < 0:
            raise ValueError('invalid budget, gate or guidance')
        total = 0.0
        for name, weight in self.mix.items():
            if name not in GOALS or not np.isfinite(weight) or weight < 0:
                raise ValueError(f'invalid mix entry {name}={weight}')
            total += float(weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'stage {self.name} mix must sum to 1, got {total}')
        if self.mix.get(self.goal, 0.0) <= 0:
            raise ValueError(f'stage {self.name} mix must include its own goal')
        if not 1 <= self.fruit_count <= 40:
            raise ValueError('fruit_count must be in [1, 40]')


STAGES = (
    Stage(
        1, 'deposit_pixels', 'DEPOSIT_ONLY', 30.0, ('V3', 'M3', 'G1'),
        {'DEPOSIT_ONLY': 1.0}, RESET_DEPOSIT, False, 1, False, 0.90, 200, 1.0, False,
        'Fruto ya suelto en la pinza; acercar, soltar y asentar en la cesta.',
    ),
    Stage(
        2, 'grasp_detach', 'DETACH_ONLY', 60.0, ('V3', 'M3', 'G1'),
        {'DETACH_ONLY': 0.75, 'DEPOSIT_ONLY': 0.25}, RESET_PREGRASP, False, 1, False, 0.85, 200, 1.0, False,
        'Kiwi visible, pinza abierta; agarrar, inclinar y separar el pedúnculo.',
    ),
    Stage(
        3, 'stationary_harvest', 'HARVEST', 180.0, ('V3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_HANGING, False, 1, False, 0.80, 200, 1.0, False,
        'Base quieta: de fruto colgante a cesta (ciclo completo).',
    ),
    Stage(
        4, 'visual_approach', 'HARVEST', 240.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_APPROACH, True, 1, False, 0.75, 200, 1.0, False,
        'N3 explora y se coloca; M3 cosecha. Sin waypoint verdadero de kiwi.',
    ),
    Stage(
        5, 'multi_harvest', 'HARVEST', 900.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_APPROACH, True, 5, True, 0.75, 200, 1.0, False,
        'Explorar, almacenar y repetir. Un cuerpo de fruta no es un censo de 40.',
    ),
    Stage(
        6, 'generalise', 'HARVEST', 900.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 1.0}, RESET_APPROACH, True, 5, True, 0.70, 200, 0.5, True,
        'Layouts y apariencia no vistos; shaping reducido. No es despliegue.',
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def stage_named(name: str) -> Stage:
    if name not in STAGE_BY_NAME:
        raise ValueError(f'unknown curriculum stage {name!r}; expected one of {list(STAGE_BY_NAME)}')
    return STAGE_BY_NAME[name]


def next_stage(stage: Stage) -> Stage | None:
    if stage.index >= len(STAGES):
        return None
    nxt = STAGES[stage.index]
    return nxt if nxt.index == stage.index + 1 else None


def first_unsatisfied_stage(stage: Stage, n_fruits: int) -> Stage | None:
    """First remaining stage whose fruit_count exceeds the assembled scene."""
    if not isinstance(n_fruits, int) or isinstance(n_fruits, bool) or n_fruits < 0:
        raise ValueError('n_fruits must be a non-negative integer')
    for item in STAGES:
        if item.index >= stage.index and item.fruit_count > n_fruits:
            return item
    return None


def fruit_block_reason(stage: Stage, n_fruits: int) -> str | None:
    blocked = first_unsatisfied_stage(stage, n_fruits)
    if blocked is None:
        return None
    return (
        f'scene has {n_fruits} fruit bodies; {blocked.name} needs {blocked.fruit_count}. '
        f'Re-export with --fruit-count {blocked.fruit_count}.'
    )


def reset_mode_for_goal(goal: str, stage: Stage) -> int:
    if goal not in GOALS:
        raise ValueError(f'unknown goal {goal!r}')
    if goal == 'HARVEST' and stage.allow_locomotion:
        return RESET_APPROACH
    return RESET_FOR_GOAL[goal]


def sample_world_skills(stage: Stage, worlds: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-world goal/reset/timeout mix. 25% previous skills after stage 01."""
    if not isinstance(worlds, int) or not 1 <= worlds <= 4096:
        raise ValueError('worlds must be an integer in [1, 4096]')
    names = tuple(stage.mix)
    weights = np.array([stage.mix[name] for name in names], dtype=np.float64)
    choices = rng.choice(len(names), size=worlds, p=weights / weights.sum())
    goals = np.array([GOAL_ID[names[i]] for i in choices], dtype=np.int32)
    resets = np.array([reset_mode_for_goal(names[i], stage) for i in choices], dtype=np.int32)
    timeouts = np.array([
        STAGES[0].budget_s if names[i] == 'DEPOSIT_ONLY'
        else STAGES[1].budget_s if names[i] == 'DETACH_ONLY'
        else stage.budget_s
        for i in choices
    ], dtype=np.float32)
    primary = np.array([names[i] == stage.goal for i in choices], dtype=np.bool_)
    continue_after = np.array(
        [np.uint8(stage.continue_after_success and names[i] == 'HARVEST') for i in choices],
        dtype=np.uint8)
    required = np.where(continue_after != 0, stage.fruit_count, 1).astype(np.int32)
    if stage.randomize_layout:
        layout_dx = rng.uniform(-0.25, 0.25, worlds).astype(np.float32)
        layout_dy = rng.uniform(-0.25, 0.25, worlds).astype(np.float32)
    else:
        layout_dx = np.zeros(worlds, dtype=np.float32)
        layout_dy = np.zeros(worlds, dtype=np.float32)
    return {
        'goal_id': goals,
        'reset_mode': resets,
        'timeout_s': timeouts,
        'guidance_weight': np.full(worlds, stage.guidance_weight, dtype=np.float32),
        'allow_locomotion': np.full(worlds, np.uint8(stage.allow_locomotion)),
        'continue_after_success': continue_after,
        'required_harvests': required,
        'fruit_count': np.full(worlds, stage.fruit_count, dtype=np.int32),
        'randomize_layout': np.full(worlds, np.uint8(stage.randomize_layout)),
        'layout_dx_m': layout_dx,
        'layout_dy_m': layout_dy,
        'primary_mask': primary,
    }


def evaluate_skills(stage: Stage, worlds: int) -> dict[str, np.ndarray]:
    """Deterministic evaluation uses only the current stage goal."""
    if not isinstance(worlds, int) or not 1 <= worlds <= 4096:
        raise ValueError('worlds must be an integer in [1, 4096]')
    continue_after = np.full(worlds, np.uint8(stage.continue_after_success))
    required = np.full(worlds, stage.fruit_count if stage.continue_after_success else 1, dtype=np.int32)
    # Held-out deterministic xy jitter from world index; not the training draws.
    if stage.randomize_layout:
        index = np.arange(worlds, dtype=np.float32)
        layout_dx = ((index * 0.6180339887) % 1.0 - 0.5) * 0.5
        layout_dy = ((index * 0.3819660113) % 1.0 - 0.5) * 0.5
    else:
        layout_dx = np.zeros(worlds, dtype=np.float32)
        layout_dy = np.zeros(worlds, dtype=np.float32)
    return {
        'goal_id': np.full(worlds, GOAL_ID[stage.goal], dtype=np.int32),
        'reset_mode': np.full(worlds, reset_mode_for_goal(stage.goal, stage), dtype=np.int32),
        'timeout_s': np.full(worlds, stage.budget_s, dtype=np.float32),
        'guidance_weight': np.zeros(worlds, dtype=np.float32),  # eval with guidance off
        'allow_locomotion': np.full(worlds, np.uint8(stage.allow_locomotion)),
        'continue_after_success': continue_after,
        'required_harvests': required,
        'fruit_count': np.full(worlds, stage.fruit_count, dtype=np.int32),
        'randomize_layout': np.full(worlds, np.uint8(stage.randomize_layout)),
        'layout_dx_m': layout_dx.astype(np.float32),
        'layout_dy_m': layout_dy.astype(np.float32),
        'primary_mask': np.ones(worlds, dtype=np.bool_),
    }


def evaluation_horizon_s(stage: Stage, *, profile: str = 'default') -> float:
    """Eval wall budget: stage timeout, capped so RGBD is not stacked."""
    if profile not in EVAL_PROFILES:
        raise ValueError(f'unknown eval profile {profile!r}; expected one of {EVAL_PROFILES}')
    if profile == 'speedrun':
        try:
            cap = float(SPEEDRUN_EVAL_CAP_S[stage.name])
        except KeyError as exc:
            raise ValueError(f'speedrun eval cap missing for {stage.name}') from exc
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError(f'invalid speedrun eval cap for {stage.name}')
    else:
        cap = MULTI_EVAL_CAP_S if stage.fruit_count > 1 else EVAL_CAP_S
    return float(min(stage.budget_s, cap))


def evaluation_horizon_steps(stage: Stage, control_dt: float = POLICY_DT_S, *,
                             profile: str = 'default') -> int:
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError('control_dt must be finite and positive')
    steps = int(round(evaluation_horizon_s(stage, profile=profile) / float(control_dt)))
    if steps < 1:
        raise ValueError('evaluation horizon must cover at least one control step')
    return steps


def idle_locomotion_mask(action_dim: int, allow_locomotion: bool):
    """Zero N3 base dims when the stage holds the chassis. None = all actions live."""
    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim not in (7, 10):
        raise ValueError('action_dim must be 7 (arm) or 10 (base+arm)')
    if allow_locomotion or action_dim == 7:
        return None
    mask = np.ones(action_dim, dtype=np.float32)
    mask[:3] = 0.0
    return mask


def apply_speedrun_preset(values: dict) -> dict:
    """Overlay speedrun trainer knobs. Does not change gates or weld fruit."""
    if not isinstance(values, dict):
        raise TypeError('values must be a dict')
    out = dict(values)
    out.update(SPEEDRUN_PRESET)
    return out


def apply_easy_preset(values: dict) -> dict:
    """Overlay privileged deposit facilitation. Gates and fruit freedom stay."""
    if not isinstance(values, dict):
        raise TypeError('values must be a dict')
    out = dict(values)
    out.update(EASY_PRESET)
    return out


def easy_start_far_frac(update_index, horizon=None):
    """How far from the crate the easy start may sample. 0=nearest outside."""
    if horizon is None:
        horizon = EASY_PRESET['far_horizon_updates']
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 0:
        raise ValueError('update_index must be a non-negative integer')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    return float(min(1.0, update_index / float(horizon)))


def easy_teacher_mix(update_index, start_mix=None, horizon=None, anneal_after=0):
    """Hold privileged mix, then drop it linearly so eval is unassisted.

    Eval still forces teacher_mix=0. Easy10 annealed from update 0 while
    harvest stayed 0 and ground contact rose as mix fell; keep start_mix
    until ``anneal_after`` (the first update after deposits are seen).
    """
    if start_mix is None:
        start_mix = 1.0
    if horizon is None:
        horizon = EASY_PRESET['teacher_horizon_updates']
    if not isinstance(update_index, int) or isinstance(update_index, bool) or update_index < 0:
        raise ValueError('update_index must be a non-negative integer')
    if not isinstance(anneal_after, int) or isinstance(anneal_after, bool) or anneal_after < 0:
        raise ValueError('anneal_after must be a non-negative integer')
    start_mix = float(start_mix)
    if not np.isfinite(start_mix) or not 0.0 <= start_mix <= 1.0:
        raise ValueError('start_mix must be finite in [0, 1]')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if update_index < anneal_after:
        return float(start_mix)
    return float(start_mix * max(0.0, 1.0 - (update_index - anneal_after) / float(horizon)))


def promotion_ready(success_rates: list[float], stage: Stage, *, episodes_seen: int | None = None) -> bool:
    """Two consecutive evals at or above the stage gate. Not harvest proof."""
    if episodes_seen is not None:
        if not isinstance(episodes_seen, int) or isinstance(episodes_seen, bool) or episodes_seen < 0:
            raise ValueError('episodes_seen must be a non-negative integer')
        if episodes_seen < stage.gate_episodes:
            return False
    if len(success_rates) < PROMOTE_STREAK:
        return False
    window = success_rates[-PROMOTE_STREAK:]
    if any(not np.isfinite(rate) or rate < 0 or rate > 1 for rate in window):
        raise ValueError('success rates must be finite in [0, 1]')
    return all(rate >= stage.gate_success_rate for rate in window)


def summarise_stage(stage: Stage, *, profile: str = 'default') -> dict[str, object]:
    return {
        'index': stage.index,
        'name': stage.name,
        'goal': stage.goal,
        'budget_s': stage.budget_s,
        'mix': dict(stage.mix),
        'allow_locomotion': stage.allow_locomotion,
        'fruit_count': stage.fruit_count,
        'continue_after_success': stage.continue_after_success,
        'randomize_layout': stage.randomize_layout,
        'evaluation_horizon_s': evaluation_horizon_s(stage, profile=profile),
        'eval_profile': profile,
        'gate_success_rate': stage.gate_success_rate,
        'guidance_weight': stage.guidance_weight,
        'trains': list(stage.trains),
        'description': stage.description,
        'training_ready': False,
        'scope': 'task curriculum on the rigid fast runtime; not the full V3/N3/M3 spec',
    }
