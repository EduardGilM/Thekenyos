"""Measured rollout boundary for :class:`BatchedDeformableRuntime`.

The runtime bridge exposes only camera and joint-controller measurements. It
does not pass fruit state, contacts, or simulator labels to an actor.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


ACTION_DIM = 7  # six arm joints plus jaw
OBS_VERSION = "physical-rollout/v1"


class UnsupportedObservationError(ValueError):
    pass


class RuntimeObservation:
    """Sensor-only observation from the runtime, batch first."""

    unsupported_fields = (
        "fruit_truth", "contacts", "reward",
    )

    def __init__(self, rgbd, r84, timestamp_s):
        self.rgbd = np.asarray(rgbd, dtype=np.float32)
        self.r84 = np.asarray(r84, dtype=np.float32)
        self.timestamp_s = float(timestamp_s)
        self.version = OBS_VERSION


class RuntimeReachReward:
    """Privileged training reward from runtime geometry, separate from actor input."""

    def __init__(self, runtime, fruit_index=0, success_distance_m=.03):
        self.runtime = runtime
        self.fruit_index = int(fruit_index)
        self.success_distance_m = float(success_distance_m)
        self.previous_distance_m = None
        attachments = runtime.manifest.get("attachments", ())
        if not 0 <= self.fruit_index < len(attachments):
            raise ValueError("fruit_index is outside the scene attachments")
        self.flex_name = attachments[self.fruit_index]["flex"]
        self.tcp_site = runtime.model.site("hand_tcp").id
        import mujoco
        self.flex_id = mujoco.mj_name2id(
            runtime.model, mujoco.mjtObj.mjOBJ_FLEX, self.flex_name)
        if self.flex_id < 0:
            raise ValueError(f"fruit flex {self.flex_name!r} is missing from runtime model")

    def reset(self):
        self.previous_distance_m = self.distance()

    def distance(self):
        model, data = self.runtime.model, self.runtime.data
        start = int(model.flex_vertadr[self.flex_id])
        count = int(model.flex_vertnum[self.flex_id])
        tcp = np.asarray(data.site_xpos.numpy())[:, self.tcp_site]
        vertices = np.asarray(data.flexvert_xpos.numpy())[:, start:start + count]
        distances = np.linalg.norm(vertices - tcp[:, None, :], axis=-1).mean(axis=1)
        if distances.shape != (self.runtime.worlds,) or not np.isfinite(distances).all():
            raise RuntimeError("invalid runtime reach measurement")
        return distances.astype(np.float32)

    def step(self):
        distances = self.distance()
        if self.previous_distance_m is None:
            reward = np.zeros_like(distances)
        else:
            reward = self.previous_distance_m - distances
        self.previous_distance_m = distances
        return reward, {"tcp_fruit_distance_m": distances,
                        "reach_success": distances <= self.success_distance_m}


class BatchedPhysicalRollout:
    """Concrete collector around an existing batched runtime."""

    def __init__(self, runtime, reward: RuntimeReachReward | None = None,
                 camera="hand_camera"):
        required = ("control", "advance", "capture", "worlds", "dt")
        if any(not hasattr(runtime, name) for name in required):
            raise TypeError("runtime must be a BatchedDeformableRuntime")
        self.runtime = runtime
        self.reward = reward
        self.camera = str(camera)
        self.last_action = None
        self._limits = self._arm_target_limits()

    def reset(self):
        self.runtime.reset()
        self.last_action = None
        if self.reward is not None and hasattr(self.reward, "reset"):
            self.reward.reset()
        return self.observe()

    def _arm_target_limits(self):
        contract = self.runtime.control.contract
        joints = np.asarray(contract.joints, dtype=np.intp)[12:19]
        ranges = np.asarray(self.runtime.model.jnt_range, dtype=np.float32)[joints]
        limited = np.asarray(self.runtime.model.jnt_limited, dtype=bool)[joints]
        out = np.tile(np.array([[-np.pi, np.pi]], dtype=np.float32), (ACTION_DIM, 1))
        out[limited] = ranges[limited]
        if out.shape != (ACTION_DIM, 2) or not np.isfinite(out).all() or np.any(out[:, 0] > out[:, 1]):
            raise ValueError("invalid arm/jaw joint limits")
        return out

    def observe(self) -> RuntimeObservation:
        frames = self.runtime.capture()
        if self.camera not in frames:
            raise UnsupportedObservationError(f"camera {self.camera!r} is unavailable")
        frame = frames[self.camera]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        rgbd = np.asarray(frame, dtype=np.float32)
        r84 = np.asarray(self.runtime.control.observe().numpy(), dtype=np.float32)
        if rgbd.ndim != 4 or rgbd.shape[0] != self.runtime.worlds or rgbd.shape[1] != 5:
            raise ValueError("runtime camera must produce [worlds, 5, height, width]")
        if r84.shape != (self.runtime.worlds, 84) or not np.isfinite(r84).all():
            raise ValueError("runtime control must produce finite [worlds, 84] R84")
        return RuntimeObservation(rgbd, r84, self.runtime.step_index * self.runtime.dt)

    def apply_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        worlds = self.runtime.worlds
        if action.shape == (ACTION_DIM,):
            action = np.tile(action, (worlds, 1))
        if action.shape != (worlds, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"action must be finite with shape ({worlds}, 7)")
        bounded = np.clip(action, self._limits[:, 0], self._limits[:, 1])
        targets = np.asarray(self.runtime.control.targets.numpy(), dtype=np.float32).copy()
        targets[:, 12:19] = bounded
        self.runtime.control.set_targets(targets)
        self.last_action = bounded.copy()
        return bounded.copy()

    def step(self, action, control_dt_s=.04, gait_action_fn=None):
        applied = self.apply_action(action)
        self.runtime.advance(control_dt_s=control_dt_s, gait_action_fn=gait_action_fn)
        observation = self.observe()
        reward, info = (self.reward.step() if self.reward is not None else (None, {
            "reward_valid": False, "unsupported": "physical outcome observer"}))
        return observation, applied, reward, info

    def collect(self, actions: Iterable[np.ndarray], control_dt_s=.04,
                gait_action_fn=None) -> list[dict]:
        records = []
        for action in actions:
            observation, applied, reward, info = self.step(action, control_dt_s, gait_action_fn)
            records.append({"observation": observation, "action": applied,
                            "reward": reward, "info": info})
        return records
