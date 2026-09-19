"""Training harness config (TK-RL-003 v3.0, §6).

PPO with recurrent N3/M3 + shared V3 encoder, concurrent low-LR G1
adaptation, separate critics. This module holds versioned hyperparams,
checkpoint manifests and rollout-group geometry. The optimisation loop
itself is integration work (torch + rsl-rl adapter); configs here pin
the initial values so runs are reproducible and comparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from . import schemas as S

# Initial PPO values (§6.2). Engineering defaults, not tuned results.
G1_LR = 5e-6
N3_M3_LR = 3e-4
V3_LR = 1e-4
CRITIC_LR = 3e-4
G1_CRITIC_LR = 1e-4

TRAIN_SEEDS = (11, 22, 33)


@dataclass
class PPOConfig:
    actor_lr: float
    critic_lr: float
    gamma: float
    lam: float
    clip: float
    value_coef: float
    entropy_cont: float
    entropy_event: float = 0.005
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 1.0
    kl_limit: float = 0.02
    seq_len: int = 32
    burn_in: int = 16


G1_PPO = PPOConfig(actor_lr=G1_LR, critic_lr=G1_CRITIC_LR, gamma=0.99,
                   lam=0.95, clip=0.1, value_coef=1.0, entropy_cont=0.002,
                   epochs=1, minibatches=4, kl_limit=0.003,
                   seq_len=0, burn_in=0)
HARVEST_PPO = PPOConfig(actor_lr=N3_M3_LR, critic_lr=CRITIC_LR, gamma=0.9996,
                        lam=0.95, clip=0.2, value_coef=1.0,
                        entropy_cont=0.005, epochs=4, minibatches=4)


@dataclass
class CurriculumStage:
    name: str
    goal: str
    budget_s: float
    train: tuple = field(default_factory=tuple)


STAGES = (
    CurriculumStage("deposit_pixels", "DEPOSIT_ONLY", 30.0, ("V3", "M3", "G1")),
    CurriculumStage("grasp_detach", "DETACH_ONLY", 60.0, ("V3", "M3", "G1")),
    CurriculumStage("stationary_harvest", "HARVEST", 180.0, ("V3", "M3", "G1")),
    CurriculumStage("visual_approach", "HARVEST", 240.0, ("V3", "N3", "M3", "G1")),
    CurriculumStage("multi_harvest", "HARVEST", 900.0, ("V3", "N3", "M3", "G1")),
    CurriculumStage("generalise", "HARVEST", 900.0, ("V3", "N3", "M3", "G1")),
)


def checkpoint_manifest(repo_commit: str, relic_commit: str,
                        config: dict, metrics: dict) -> dict:
    """Manifest saved alongside weights: hashes + seeds + schema versions."""
    body = {
        "obs_schema": S.OBS_SCHEMA_VERSION,
        "action_schema": S.ACTION_SCHEMA_VERSION,
        "reward_schema": S.REWARD_SCHEMA_VERSION,
        "schema_hash": S.schema_hash(),
        "repo_commit": repo_commit,
        "relic_commit": relic_commit,
        "train_seeds": list(TRAIN_SEEDS),
        "config": config,
        "metrics": metrics,
    }
    body["manifest_sha"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return body


def validate_config(d: dict) -> dict:
    """Finite/range checks for PPO hyperparams before a run starts."""
    for k in ("actor_lr", "critic_lr", "gamma", "lam", "clip"):
        v = d.get(k)
        if v is None or not float(v) > 0:
            raise ValueError(f"ppo.{k}={v} must be positive")
    if not 0.9 <= d["gamma"] <= 1.0:
        raise ValueError(f"gamma={d['gamma']} outside [0.9, 1.0]")
    if not 0.0 < d["clip"] <= 0.5:
        raise ValueError(f"clip={d['clip']} outside (0, 0.5]")
    return d
