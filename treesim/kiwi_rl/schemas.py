"""Versioned tensor contracts for the kiwi-harvest RL system (TK-RL-003 v3.0).

All shapes are batch-first with B environments. All floating vectors are
float32 and finite. Units are part of field names or documented scales;
scales below are numerical normalisation choices, not physical limits.

Layouts (0-based, end-exclusive slices):
- R84  (RELIC contract, read-only here): 3+3+3+3+7+12+3+19+19+12 = 84
- P85  (harvest proprioception): 19+19+3+3+3+7+9+6+1+4+1+4+3+2+1 = 85
- Context14: phase_one_hot(6) + goal_one_hot(3) + 5 scalars = 14
- Basket16: 3+6+1+1+1+3+1 = 16
- N3Input: 256+128+85+14+3+2 = 488
- M3Input: 256+85+14+16+8+3 = 382
- CriticInput: 84+85+14+24+256 = 463
- FruitTruth per fruit: 3+4+3+3+1+3+1+1+1+1+1+1 = 23
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

OBS_SCHEMA_VERSION = "obs/v3"
ACTION_SCHEMA_VERSION = "action/v3"
REWARD_SCHEMA_VERSION = "reward/v3"
SENSOR_SCHEMA_VERSION = "sensorbatch/v3"

PHASES = ("EXPLORE", "SETTLE", "MANIPULATE", "VERIFY", "RECOVER", "DONE")
GOALS = ("DEPOSIT_ONLY", "DETACH_ONLY", "HARVEST")
N3_EVENT_NAMES = ("CONTINUE", "ATTEMPT")
M3_EVENT_NAMES = ("CONTINUE", "FINISH", "RECOVER")

# R84 slices: (name, start, end). Semantics from treesim/spot.py + upstream
# ArmLegJointBasePoseCommand (22 = 7 arm + 12 leg + 3 torso ref).
R84_SLICES = (
    ("lin_vel_body_mps", 0, 3),
    ("ang_vel_body_rps", 3, 6),
    ("gravity_body", 6, 9),
    ("command_vx_vy_wz", 9, 12),
    ("arm_target", 12, 19),
    ("leg_subgoal_interlimb", 19, 31),
    ("torso_ref_roll_pitch_height", 31, 34),
    ("q_rel", 34, 53),
    ("qd", 53, 72),
    ("last_action", 72, 84),
)
R84_DIM = 84

# P85 slices: (name, start, end, scale, unit_note).
P85_SLICES = (
    ("q_rel", 0, 19, np.pi, "rad"),
    ("qd", 19, 38, 10.0, "rad/s"),
    ("v_body_mps", 38, 41, 1.0, "m/s"),
    ("omega_body_rps", 41, 44, 3.0, "rad/s"),
    ("gravity_body", 44, 47, 1.0, "unit"),
    ("arm_target", 47, 54, np.pi, "rad"),
    ("tcp_pos_body_m", 54, 57, 1.0, "m"),
    ("tcp_rot6d_body", 57, 63, 1.0, "unit"),
    ("tcp_vel_body", 63, 69, None, "m/s,rad/s"),
    ("gripper_open_01", 69, 70, 1.0, "01"),
    ("force_hand", 70, 73, 50.0, "N"),
    ("force_valid", 73, 74, 1.0, "01"),
    ("height_m", 74, 75, 1.0, "m"),
    ("foot_contact", 75, 79, 1.0, "01"),
    ("base_command_prev", 79, 82, None, "m/s,m/s,rad/s"),
    ("hand_image_age_valid", 82, 84, 1.0, "s,01"),
    ("grip_contact_score", 84, 85, 1.0, "01"),
)
P85_DIM = 85

CONTEXT_DIM = 14
BASKET_DIM = 16
Z_VISUAL_DIM = 256
MAP_ENC_DIM = 128
N3_INPUT_DIM = 488
M3_INPUT_DIM = 382
CRITIC_INPUT_DIM = 463
FRUIT_TRUTH_DIM = 23
FRUIT_POOL_MAX = 128
PRIV_DIM = 24

N3_CONT_DIM = 3
N3_EVENT_DIM = 2
M3_CONT_DIM = 8
M3_EVENT_DIM = 3
G1_DIM = 12

# Per-field normalisation scales (numerical, not safety limits).
P85_SCALES = {
    "tcp_vel_body": np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float32),
    "base_command_prev": np.array([0.4, 0.3, 0.7], dtype=np.float32),
}
NORM_CLIP = 5.0


def schema_hash() -> str:
    """Content hash of the versioned layouts (for checkpoints/metrics)."""
    payload = {
        "obs": OBS_SCHEMA_VERSION,
        "action": ACTION_SCHEMA_VERSION,
        "reward": REWARD_SCHEMA_VERSION,
        "sensor": SENSOR_SCHEMA_VERSION,
        "r84": R84_SLICES,
        "p85": [(n, a, b) for n, a, b, *_ in P85_SLICES],
        "phases": PHASES,
        "goals": GOALS,
        "dims": {
            "p85": P85_DIM, "context": CONTEXT_DIM, "basket": BASKET_DIM,
            "n3": N3_INPUT_DIM, "m3": M3_INPUT_DIM, "critic": CRITIC_INPUT_DIM,
            "fruit_truth": FRUIT_TRUTH_DIM, "priv": PRIV_DIM,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def validate_finite(name: str, arr: np.ndarray, expected_shape: tuple) -> np.ndarray:
    """Check dtype/shape/finiteness; raise ValueError with context on failure."""
    a = np.asarray(arr)
    if a.shape != expected_shape:
        raise ValueError(f"{name}: shape {a.shape} != {expected_shape}")
    if a.dtype != np.float32:
        raise ValueError(f"{name}: dtype {a.dtype} != float32")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return a


def normalise_p85(raw: dict) -> np.ndarray:
    """Build a normalised P85 vector from a dict of raw fields.

    Each value is validated finite and within a documented physical range
    before scaling; out-of-range raises instead of silently clipping
    (clipping is applied only to the final normalised vector and counted
    by the caller).
    """
    out = np.zeros(P85_DIM, dtype=np.float32)

    def put(slc, values, lo, hi, scale):
        v = np.asarray(values, dtype=np.float64)
        name, a, b = slc[0], slc[1], slc[2]
        if v.shape != (b - a,):
            raise ValueError(f"P85.{name}: shape {v.shape} != {(b - a,)}")
        if not np.isfinite(v).all():
            raise ValueError(f"P85.{name}: non-finite input")
        if not ((v >= lo) & (v <= hi)).all():
            raise ValueError(f"P85.{name}: out of range [{lo}, {hi}]")
        out[a:b] = (v / scale).astype(np.float32)

    get = raw.get
    put(P85_SLICES[0], get("q_rel"), -np.pi, np.pi, np.pi)
    put(P85_SLICES[1], get("qd"), -10.0, 10.0, 10.0)
    put(P85_SLICES[2], get("v_body_mps"), -5.0, 5.0, 1.0)
    put(P85_SLICES[3], get("omega_body_rps"), -10.0, 10.0, 3.0)
    g = np.asarray(get("gravity_body"), dtype=np.float64)
    if g.shape != (3,) or not np.isfinite(g).all():
        raise ValueError("P85.gravity_body: must be finite (3,)")
    out[44:47] = (g / max(np.linalg.norm(g), 1e-9)).astype(np.float32)
    put(P85_SLICES[5], get("arm_target"), -np.pi, np.pi, np.pi)
    put(P85_SLICES[6], get("tcp_pos_body_m"), -2.0, 2.0, 1.0)
    r = np.asarray(get("tcp_rot6d_body"), dtype=np.float64)
    if r.shape != (6,) or not np.isfinite(r).all():
        raise ValueError("P85.tcp_rot6d_body: must be finite (6,)")
    out[57:63] = np.clip(r, -1.0, 1.0).astype(np.float32)
    put(P85_SLICES[8], get("tcp_vel_body"), -5.0, 5.0, P85_SCALES["tcp_vel_body"])
    put(P85_SLICES[9], get("gripper_open_01"), 0.0, 1.0, 1.0)
    put(P85_SLICES[10], get("force_hand_n"), -200.0, 200.0, 50.0)
    put(P85_SLICES[11], get("force_valid"), 0.0, 1.0, 1.0)
    put(P85_SLICES[12], get("height_m"), 0.0, 2.0, 1.0)
    put(P85_SLICES[13], get("foot_contact"), 0.0, 1.0, 1.0)
    put(P85_SLICES[14], get("base_command_prev"), -1.0, 1.0,
        P85_SCALES["base_command_prev"])
    put(P85_SLICES[15], get("hand_image_age_valid"), 0.0, 10.0, 1.0)
    if out[83] not in (0.0, 1.0):
        raise ValueError('P85.hand_image_valid must be zero or one')
    put(P85_SLICES[16], get("grip_contact_score"), 0.0, 1.0, 1.0)
    clipped = int((np.abs(out) > NORM_CLIP).sum())
    np.clip(out, -NORM_CLIP, NORM_CLIP, out=out)
    return out, clipped


def build_context14(phase: str, goal: str, phase_elapsed_s: float,
                    timeout_s: float, retry_count: int, retry_max: int,
                    basket_fill_01: float, basket_fill_valid: float,
                    ik_pos_err_m: float) -> np.ndarray:
    """Context14 vector: phase/goal one-hots + 5 normalised scalars."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}")
    for name, v, lo, hi in (
        ("phase_elapsed_s", phase_elapsed_s, 0.0, 3600.0),
        ("timeout_s", timeout_s, 1e-3, 3600.0),
        ("retry_count", retry_count, 0, 100),
        ("basket_fill_01", basket_fill_01, 0.0, 1.0),
        ("basket_fill_valid", basket_fill_valid, 0.0, 1.0),
        ("ik_pos_err_m", ik_pos_err_m, 0.0, 2.0),
    ):
        if not np.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"context {name}={v} outside [{lo}, {hi}]")
    out = np.zeros(CONTEXT_DIM, dtype=np.float32)
    out[PHASES.index(phase)] = 1.0
    out[6 + GOALS.index(goal)] = 1.0
    out[9] = np.float32(phase_elapsed_s / timeout_s)
    out[10] = np.float32(retry_count / max(retry_max, 1))
    out[11] = np.float32(basket_fill_01)
    out[12] = np.float32(basket_fill_valid)
    out[13] = np.float32(ik_pos_err_m / 0.1)
    np.clip(out, -NORM_CLIP, NORM_CLIP, out=out)
    return out


def empty_basket16() -> np.ndarray:
    """Basket16 with goal_valid=0 (no recent view): safe default."""
    out = np.zeros(BASKET_DIM, dtype=np.float32)
    return out
