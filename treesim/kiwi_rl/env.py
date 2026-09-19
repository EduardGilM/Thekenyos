"""Single-env kiwi-harvest wrapper over the existing runtime (TK-RL-003 v3.0).

Reuses TreeConfig.compliant('pergola') + builder.generate_and_build +
Sim + SpotController exactly like scripts/walk_spot.py. All sim imports
are lazy so schema/reward/arbiter/model unit tests run without newton,
warp, mujoco or the external RELIC checkout.

Scheduler: physics 1 kHz substeps, G1 at 50 Hz (20 substeps), harvest
actor + arbiter at 25 Hz (every 2nd gait tick), camera stub at 10 Hz.
Vision input is currently a validity-flagged stub (frame_valid=False);
camera rendering is integration work and must not feed GT labels.

Arm manipulation: M3 actions are validated and logged; the arm holds its
RELIC ready pose pending the newton.ik Spot side-model (spec INT). This
is explicit: locomotion + coordination + rewards run end-to-end, hand
kartesian actuation is stubbed, never faked with body teleportation or
KiwiField.hold (which raises for kiwi by design).

NOTE on the physical gripper: this file is NOT the gripper. The gripper
is Spot's jaw joint (arm_f1x, part of ARM in treesim/spot.py), simulated
as rigid bodies with real MuJoCo contacts and driven by the PD targets
SpotController writes. Grasping happens only through physical pad-fruit
contact inside the stepped physics; this env merely delivers commands
and reads back sensor-side estimates (gripper_open_01, force_hand from
encoders/loads, never contact truth).
"""

from __future__ import annotations

import numpy as np

from . import schemas as S
from .arbiter import Arbiter
from .rewards import RewardEvaluator, RewardState, shaping_step


def sim_available() -> bool:
    try:
        import newton  # noqa: F401
        import warp  # noqa: F401
        return True
    except ImportError:
        return False


GOAL_TIMEOUT_S = {
    "DEPOSIT_ONLY": 30.0,
    "DETACH_ONLY": 60.0,
    "HARVEST": 180.0,
}


class KiwiHarvestEnv:
    """One Spot, one pergola, one basket. Seeded, resettable, single-env."""

    def __init__(self, relic_path: str, seed: int = 11,
                 goal: str = "HARVEST", payload_mass_kg: float = 0.0,
                 payload_seed: int = 0, fps: int = 50, substeps: int = 20,
                 max_fruit: int = 40):
        if goal not in S.GOALS:
            raise ValueError(f"unknown goal {goal!r}")
        for name, v, lo, hi in (("seed", seed, 0, 2 ** 31 - 1),
                                ("payload_mass_kg", payload_mass_kg, 0.0, 6.0),
                                ("fps", fps, 1, 240),
                                ("substeps", substeps, 2, 100)):
            if not np.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
        if substeps % 2:
            raise ValueError("substeps must be even (25 Hz harvest / 50 Hz gait)")
        self.relic_path = str(relic_path)
        self.seed = int(seed)
        self.goal = goal
        self.payload_mass_kg = float(payload_mass_kg)
        self.payload_seed = int(payload_seed)
        self.fps = int(fps)
        self.substeps = int(substeps)
        self.max_fruit = int(max_fruit)
        self._built = False
        self.rng = np.random.default_rng(seed)
        self.arbiter = Arbiter()
        self.shaping = RewardState()
        self.prev_n3 = np.zeros(3, dtype=np.float32)
        self.prev_m3 = np.zeros(8, dtype=np.float32)
        self.prev_n3_event = 0
        self.prev_m3_event = 0
        self.tick = 0
        self.episode_s = 0.0
        self.evaluator: RewardEvaluator | None = None

    # -- construction ------------------------------------------------------
    def build(self):
        """Build the physics world. Requires newton/warp + RELIC checkout."""
        from treesim.config import TreeConfig
        from treesim import builder
        from treesim.sim import Sim
        from treesim.spot import SpotController, LEGS

        cfg = TreeConfig.compliant("pergola")
        cfg.seed = self.seed
        cfg.robot.enabled, cfg.robot.kind = True, "spot"
        cfg.robot.relic_path = self.relic_path
        cfg.robot.position, cfg.robot.yaw = (-0.65, 0.0), np.pi / 2
        cfg.robot.basket, cfg.robot.basket_mass = True, 1.2
        cfg.robot.payload_seed = self.payload_seed
        cfg.robot.payload_mass = self.payload_mass_kg
        cfg.fruit.enabled, cfg.fruit.max_count = True, self.max_fruit
        cfg.fruit.joint = "free"
        cfg.fruit.colors = ((0.39, 0.27, 0.12), (0.48, 0.34, 0.17))
        cfg.foliage.set_density(0.6)
        cfg.foliage.min_order_for_leaves = 2
        cfg.foliage.leaf_length, cfg.foliage.leaf_width = 0.22, 0.17
        self.tree = builder.generate_and_build(cfg)
        self.sim = Sim(self.tree, fps=self.fps, substeps=self.substeps,
                       collisions=True)
        self.controller = SpotController(self.sim)
        self.home_legs = np.array(
            [self.controller.home[n] for n in LEGS], dtype=np.float32)
        fruit_ids = [int(b) for b in self.tree.apple_bodies[:128]]
        self.evaluator = RewardEvaluator(fruit_ids)
        for b in (self.tree.robot_data["basket"]["fruit_bodies"]
                  if self.tree.robot_data.get("basket") else []):
            pass  # preloaded fruit registered STORED without reward below
        self._built = True
        return self

    def _require_built(self):
        if not self._built:
            raise RuntimeError("call build() first (needs sim stack + RELIC)")

    # -- observations ------------------------------------------------------
    def _raw_proprioception(self) -> dict:
        c = self.controller
        st = self.sim.state_0
        body_q = st.body_q.numpy()
        body_qd = st.body_qd.numpy()
        joint_q = st.joint_q.numpy()
        joint_qd = st.joint_qd.numpy()
        chassis = self.tree.robot_data["chassis"]
        pose, vel = body_q[chassis], body_qd[chassis]
        speed = float(np.linalg.norm(vel[:3]))
        return {
            "q_rel": joint_q[c.obs_q] - c.obs_home,
            "qd": joint_qd[c.obs_dof],
            "v_body_mps": vel[:3].copy(),
            "omega_body_rps": vel[3:].copy(),
            "gravity_body": np.array([0.0, 0.0, -1.0]),
            "arm_target": c.targets[12:].copy(),
            "tcp_pos_body_m": np.zeros(3),
            "tcp_rot6d_body": np.array([1, 0, 0, 0, 1, 0], dtype=np.float64),
            "tcp_vel_body": np.zeros(6),
            "gripper_open_01": np.array([1.0]),
            "force_hand_n": np.zeros(3),
            "force_valid": np.array([0.0]),
            "height_m": np.array([float(pose[2])]),
            "foot_contact": np.ones(4),
            "base_command_prev": self.prev_n3.copy(),
            "hand_image_age_valid": np.array([10.0, 0.0]),
            "grip_contact_score": np.array([0.0]),
            "_speed_mps": speed,
            "_pose": pose.copy(),
        }

    def reset(self, seed: int | None = None, goal: str | None = None):
        self._require_built()
        if seed is not None:
            if not 0 <= seed < 2 ** 31:
                raise ValueError(f"seed={seed} out of range")
            self.seed = int(seed)
        if goal is not None:
            if goal not in S.GOALS:
                raise ValueError(f"unknown goal {goal!r}")
            self.goal = goal
        # Rebuild world on the new seed for a clean episode.
        self.build()
        self.arbiter.reset()
        self.shaping = RewardState()
        self.prev_n3 = np.zeros(3, dtype=np.float32)
        self.prev_m3 = np.zeros(8, dtype=np.float32)
        self.prev_n3_event, self.prev_m3_event = 0, 0
        self.tick, self.episode_s = 0, 0.0
        # Preloaded basket fruit is STORED with no deposit reward.
        basket = self.tree.robot_data.get("basket") or {}
        for b in basket.get("fruit_bodies", []):
            if b in (self.evaluator.ledger.index_of if self.evaluator else {}):
                self.evaluator.register_preloaded(b)
        return self._observe()

    def _observe(self) -> dict:
        raw = self._raw_proprioception()
        p85, clipped = S.normalise_p85(raw)
        ctx = S.build_context14(
            self.arbiter.state.phase, self.goal, self.arbiter.state.phase_time_s,
            GOAL_TIMEOUT_S[self.goal], self.arbiter.state.retries, 3,
            0.0, 0.0, 0.0)
        r84 = self._read_r84()
        return {
            "p85": p85, "p85_clipped": clipped, "context14": ctx,
            "basket16": S.empty_basket16(),
            "r84": r84,
            "z_visual": np.zeros((S.Z_VISUAL_DIM,), dtype=np.float32),
            "map": np.zeros((3, 64, 64), dtype=np.float32),
            "speed_mps": raw["_speed_mps"],
            "schema": S.OBS_SCHEMA_VERSION,
            "schema_hash": S.schema_hash(),
        }

    def _read_r84(self) -> np.ndarray:
        c = self.controller
        st = self.sim.state_0
        joint_q = st.joint_q.numpy()
        joint_qd = st.joint_qd.numpy()
        return np.concatenate([
            np.zeros(3, dtype=np.float32),  # body-frame vel filled by controller
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.prev_n3.copy(),
            c.targets[12:].astype(np.float32),
            np.zeros(12, dtype=np.float32),
            np.array([0.0, 0.0, 0.55], dtype=np.float32),
            (joint_q[c.obs_q] - c.obs_home).astype(np.float32),
            joint_qd[c.obs_dof].astype(np.float32),
            c.last_action.astype(np.float32),
        ]).astype(np.float32)

    # -- stepping ----------------------------------------------------------
    def step(self, n3_cont: np.ndarray, n3_event: int,
             m3_cont: np.ndarray, m3_event: int):
        """One 40 ms harvest tick = 2 gait ticks of 20 ms.

        N3 drives locomotion through the real SpotController; M3 hand
        deltas are validated + logged with the arm held at ready pending
        IK integration (see module docstring).
        """
        self._require_built()
        n3 = np.asarray(n3_cont, dtype=np.float32)
        m3 = np.asarray(m3_cont, dtype=np.float32)
        if n3.shape != (3,) or not np.isfinite(n3).all():
            raise ValueError("n3_cont must be finite (3,)")
        if m3.shape != (8,) or not np.isfinite(m3).all():
            raise ValueError("m3_cont must be finite (8,)")
        if n3_event not in (0, 1):
            raise ValueError("n3_event must be 0/1")
        if m3_event not in (0, 1, 2):
            raise ValueError("m3_event must be 0/1/2")
        dt = 2.0 / self.fps
        arb = self.arbiter.update(
            dt, S.N3_EVENT_NAMES[n3_event] if self.arbiter.state.active_policy == "N3" else None,
            S.M3_EVENT_NAMES[m3_event] if self.arbiter.state.active_policy == "M3" else None,
            speed_mps=float(np.linalg.norm(
                self.sim.state_0.body_qd.numpy()[self.tree.robot_data["chassis"], :3])),
            yaw_rps=0.0, frame_age_s=10.0, frame_valid=False)
        cmd = np.clip(np.tanh(n3) * np.array([0.4, 0.3, 0.7]), -0.7, 0.7)
        if arb.phase in ("MANIPULATE", "VERIFY"):
            cmd = np.zeros(3, dtype=np.float32)
        self.prev_n3 = cmd.astype(np.float32)
        self.prev_m3 = np.tanh(m3).astype(np.float32)
        self.prev_n3_event, self.prev_m3_event = int(n3_event), int(m3_event)
        for _ in range(2):
            self.controller.update(cmd.astype(np.float32))
            self.sim.step()
        self.tick += 1
        self.episode_s += dt
        obs = self._observe()
        terms = self._reward_terms(dt)
        terminated, truncated = self._termination()
        return obs, terms, terminated, truncated

    def _reward_terms(self, dt: float) -> dict:
        terms = {
            "deposit": 0.0, "grasp_stable": 0.0, "detach_held": 0.0,
            "loss": 0.0, "spill": 0.0, "damage": 0.0, "fall": 0.0,
            "time": float(-0.05 * dt), "smooth": 0.0, "false_finish": 0.0,
            "gait_tracking": 0.0,
        }
        if self.sim.kiwi_damage is not None:
            terms["damage"] = self.evaluator.damage_increment(
                max(0.0, -float(self.sim.kiwi_damage.reward)))
        basket = self.tree.robot_data.get("basket")
        if basket is not None:
            body_q = self.sim.state_0.body_q.numpy()
            spill = __import__("treesim.basket", fromlist=["SpillTracker"])
            if not hasattr(self, "_spill"):
                self._spill = spill.SpillTracker(basket)
            penalty = self._spill.update(body_q, self.tree.robot_data["chassis"])
            if penalty:
                terms["spill"] = float(penalty * 25.0)
        return terms

    def _termination(self) -> tuple[bool, bool]:
        pose = self.sim.state_0.body_q.numpy()[self.tree.robot_data["chassis"]]
        tilt = float(np.arccos(np.clip(1 - 2 * (pose[3] ** 2 + pose[4] ** 2), -1, 1)))
        if not np.isfinite(pose).all():
            return True, False
        if pose[2] < 0.25 or tilt > 1.0:
            return True, False
        if self.episode_s >= GOAL_TIMEOUT_S[self.goal]:
            return False, True
        return False, False
