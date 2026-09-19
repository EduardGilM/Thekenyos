"""Kiwi-harvest RL scaffolding (spec TK-RL-003 v3.0).

Numpy-only contracts, rewards ledger, sensor-blind arbiter, model
interfaces and a single-env wrapper over the existing Newton/MuJoCo
runtime. Torch, cameras and RELIC retraining are integration work;
this package defines the versioned interfaces they must satisfy.

Apple path is untouched: everything here is kiwi/pergola/Spot only.
"""

from .schemas import (
    OBS_SCHEMA_VERSION,
    ACTION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    PHASES,
    GOALS,
    N3_EVENT_NAMES,
    M3_EVENT_NAMES,
    schema_hash,
    validate_finite,
)

__all__ = [
    "OBS_SCHEMA_VERSION",
    "ACTION_SCHEMA_VERSION",
    "REWARD_SCHEMA_VERSION",
    "PHASES",
    "GOALS",
    "N3_EVENT_NAMES",
    "M3_EVENT_NAMES",
    "schema_hash",
    "validate_finite",
]
