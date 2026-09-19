"""Sensor-blind phase arbiter (Phase/v3).

Owns the 6-state machine EXPLORE/SETTLE/MANIPULATE/VERIFY/RECOVER/DONE.
Reads only: policy-requested events (CONTINUE/ATTEMPT/FINISH/RECOVER),
measured body speed/yaw rate, sensor frame age/validity, gripper
closure-under-load score, observed basket/map flags and timers.

Never reads detached/held/stored ground truth, fruit census, exact
payload mass or any privileged flag. FINISH declares an attempt done;
only the physical evaluator decides whether reward is due.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import PHASES, N3_EVENT_NAMES, M3_EVENT_NAMES

SETTLE_SPEED_MPS = 0.03
SETTLE_YAW_RPS = 0.05
SETTLE_HOLD_S = 0.3
SETTLE_TIMEOUT_S = 5.0
FRAME_FRESH_S = 0.3
VERIFY_MIN_S = 1.0
VERIFY_TIMEOUT_S = 5.0
MAX_RETRIES = 3


@dataclass
class ArbiterState:
    phase: str = "EXPLORE"
    phase_time_s: float = 0.0
    retries: int = 0
    active_policy: str = "N3"  # N3 in EXPLORE/SETTLE, M3 in MANIPULATE/VERIFY
    base_command_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self):
        self.phase = "EXPLORE"
        self.phase_time_s = 0.0
        self.retries = 0
        self.active_policy = "N3"
        self.base_command_mps = np.zeros(3, dtype=np.float32)


class Arbiter:
    """Rule-based coordinator over learned event heads."""

    def __init__(self):
        self.state = ArbiterState()

    def reset(self):
        self.state.reset()

    def update(self, dt_s: float, n3_event: str | None, m3_event: str | None,
               speed_mps: float, yaw_rps: float, frame_age_s: float,
               frame_valid: bool, verify_time_s: float = 0.0) -> ArbiterState:
        """Advance the machine one harvest tick (25 Hz).

        Events come from the active policy's categorical head; inactive
        policy events must be passed as None. All inputs finite-validated.
        """
        st = self.state
        for name, v in (("dt_s", dt_s), ("speed_mps", speed_mps),
                        ("yaw_rps", yaw_rps), ("frame_age_s", frame_age_s),
                        ("verify_time_s", verify_time_s)):
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name}={v} must be finite >= 0")
        if n3_event is not None and n3_event not in N3_EVENT_NAMES:
            raise ValueError(f"bad n3_event {n3_event!r}")
        if m3_event is not None and m3_event not in M3_EVENT_NAMES:
            raise ValueError(f"bad m3_event {m3_event!r}")
        st.phase_time_s += float(dt_s)
        fresh = bool(frame_valid) and frame_age_s < FRAME_FRESH_S

        if st.phase == "EXPLORE":
            st.active_policy = "N3"
            st.base_command_mps = np.zeros(3, dtype=np.float32)  # policy fills
            if n3_event == "ATTEMPT" and fresh:
                st.phase, st.phase_time_s = "SETTLE", 0.0
        elif st.phase == "SETTLE":
            st.active_policy = "N3"
            st.base_command_mps = np.zeros(3, dtype=np.float32)
            slow = speed_mps < SETTLE_SPEED_MPS and yaw_rps < SETTLE_YAW_RPS
            if slow and st.phase_time_s >= SETTLE_HOLD_S and fresh:
                st.phase, st.phase_time_s = "MANIPULATE", 0.0
            elif st.phase_time_s >= SETTLE_TIMEOUT_S:
                self._to_recover()
        elif st.phase == "MANIPULATE":
            st.active_policy = "M3"
            if m3_event == "FINISH":
                st.phase, st.phase_time_s = "VERIFY", 0.0
            elif m3_event == "RECOVER":
                self._to_recover()
        elif st.phase == "VERIFY":
            st.active_policy = "M3"
            if (m3_event == "FINISH" and verify_time_s >= VERIFY_MIN_S):
                st.phase, st.phase_time_s = "EXPLORE", 0.0
                st.retries = 0
            elif m3_event == "RECOVER" or st.phase_time_s >= VERIFY_TIMEOUT_S:
                self._to_recover()
        elif st.phase == "RECOVER":
            st.active_policy = "N3"
            st.base_command_mps = np.zeros(3, dtype=np.float32)
            # Caller drives retreat; arbiter re-arms exploration on next tick
            # once the recovery motion reports slow + fresh frame.
            if speed_mps < SETTLE_SPEED_MPS and fresh and st.phase_time_s > 1.0:
                st.phase, st.phase_time_s = "EXPLORE", 0.0
        elif st.phase == "DONE":
            st.active_policy = "N3"
            st.base_command_mps = np.zeros(3, dtype=np.float32)
        else:
            raise ValueError(f"bad phase {st.phase!r}")
        if st.phase not in PHASES:
            raise ValueError(f"bad phase {st.phase!r}")
        return st

    def _to_recover(self):
        st = self.state
        st.retries += 1
        st.phase, st.phase_time_s = "RECOVER", 0.0
        st.active_policy = "N3"
        st.base_command_mps = np.zeros(3, dtype=np.float32)

    def finish_mission(self):
        self.state.phase = "DONE"
        self.state.active_policy = "N3"
        self.state.base_command_mps = np.zeros(3, dtype=np.float32)
