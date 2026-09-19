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
        {'DEPOSIT_ONLY': 1.0}, RESET_DEPOSIT, False, 1, False, 0.90, 200, 1.0,
        'Fruto ya suelto en la pinza; acercar, soltar y asentar en la cesta.',
    ),
    Stage(
        2, 'grasp_detach', 'DETACH_ONLY', 60.0, ('V3', 'M3', 'G1'),
        {'DETACH_ONLY': 0.75, 'DEPOSIT_ONLY': 0.25}, RESET_PREGRASP, False, 1, False, 0.85, 200, 1.0,
        'Kiwi visible, pinza abierta; agarrar, inclinar y separar el pedúnculo.',
    ),
    Stage(
        3, 'stationary_harvest', 'HARVEST', 180.0, ('V3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_HANGING, False, 1, False, 0.80, 200, 1.0,
        'Base quieta: de fruto colgante a cesta (ciclo completo).',
    ),
    Stage(
        4, 'visual_approach', 'HARVEST', 240.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_APPROACH, True, 1, False, 0.75, 200, 1.0,
        'N3 explora y se coloca; M3 cosecha. Sin waypoint verdadero de kiwi.',
    ),
    Stage(
        5, 'multi_harvest', 'HARVEST', 900.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 0.75, 'DETACH_ONLY': 0.125, 'DEPOSIT_ONLY': 0.125},
        RESET_APPROACH, True, 5, True, 0.75, 200, 1.0,
        'Explorar, almacenar y repetir. Un cuerpo de fruta no es un censo de 40.',
    ),
    Stage(
        6, 'generalise', 'HARVEST', 900.0, ('V3', 'N3', 'M3', 'G1'),
        {'HARVEST': 1.0}, RESET_APPROACH, True, 5, True, 0.70, 200, 0.5,
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
    return {
        'goal_id': goals,
        'reset_mode': resets,
        'timeout_s': timeouts,
        'guidance_weight': np.full(worlds, stage.guidance_weight, dtype=np.float32),
        'allow_locomotion': np.full(worlds, np.uint8(stage.allow_locomotion)),
        'continue_after_success': np.full(worlds, np.uint8(stage.continue_after_success)),
        'primary_mask': primary,
    }


def evaluate_skills(stage: Stage, worlds: int) -> dict[str, np.ndarray]:
    """Deterministic evaluation uses only the current stage goal."""
    if not isinstance(worlds, int) or not 1 <= worlds <= 4096:
        raise ValueError('worlds must be an integer in [1, 4096]')
    return {
        'goal_id': np.full(worlds, GOAL_ID[stage.goal], dtype=np.int32),
        'reset_mode': np.full(worlds, reset_mode_for_goal(stage.goal, stage), dtype=np.int32),
        'timeout_s': np.full(worlds, stage.budget_s, dtype=np.float32),
        'guidance_weight': np.zeros(worlds, dtype=np.float32),  # eval with guidance off
        'allow_locomotion': np.full(worlds, np.uint8(stage.allow_locomotion)),
        'continue_after_success': np.zeros(worlds, dtype=np.uint8),
        'primary_mask': np.ones(worlds, dtype=np.bool_),
    }


def promotion_ready(success_rates: list[float], stage: Stage) -> bool:
    """Two consecutive evals at or above the stage gate. Not harvest proof."""
    if len(success_rates) < PROMOTE_STREAK:
        return False
    window = success_rates[-PROMOTE_STREAK:]
    if any(not np.isfinite(rate) or rate < 0 or rate > 1 for rate in window):
        raise ValueError('success rates must be finite in [0, 1]')
    return all(rate >= stage.gate_success_rate for rate in window)


def summarise_stage(stage: Stage) -> dict[str, object]:
    return {
        'index': stage.index,
        'name': stage.name,
        'goal': stage.goal,
        'budget_s': stage.budget_s,
        'mix': dict(stage.mix),
        'allow_locomotion': stage.allow_locomotion,
        'fruit_count': stage.fruit_count,
        'continue_after_success': stage.continue_after_success,
        'gate_success_rate': stage.gate_success_rate,
        'guidance_weight': stage.guidance_weight,
        'trains': list(stage.trains),
        'description': stage.description,
        'training_ready': False,
        'scope': 'task curriculum on the rigid fast runtime; not the full V3/N3/M3 spec',
    }
