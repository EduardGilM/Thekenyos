"""Spot gripper RGB-D rig → SensorBatch/v3. JP-ONLY (needs newton + GPU).

Uses the nominal gripper color/ToF frames from the pinned RELIC URDF, converted
from Boston Dynamics optical axes to Newton's camera-to-world convention
(columns right, up, -forward). Body fisheye cameras are not invented here.

Outputs SensorBatch/v3 with validity masks; invalid depth is stored as
0 + valid=false. Never emits GT masks, overlays or fruit IDs.
"""

from __future__ import annotations

import numpy as np

from .spot_cameras import NOMINAL_OPTICAL_FRAMES, optical_to_mujoco

HAND_W, HAND_H = 320, 240
HAND_COLOR_FOVY_DEG = 46.4


def _require_newton():
    try:
        import newton
        import warp as wp
        from newton.sensors import SensorTiledCamera
    except ImportError as e:
        raise ImportError("sensor stack required (newton + GPU)") from e
    return newton, wp, SensorTiledCamera


def _wxyz_to_matrix(quat):
    w, x, y, z = (float(v) for v in quat)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class SpotCameraRig:
    """Tiled-camera rig bound to a built Spot model. Call update() at 10 Hz."""

    def __init__(self, model, chassis: int, wrist: int,
                 hand_fov_deg: float = HAND_COLOR_FOVY_DEG,
                 max_range_m: float = 4.0):
        import math
        newton, wp, SensorTiledCamera = _require_newton()
        self._wp = wp
        self.model = model
        self.chassis = int(chassis)
        self.wrist = int(wrist)
        self.max_range_m = float(max_range_m)
        pose = optical_to_mujoco(**{k: NOMINAL_OPTICAL_FRAMES['hand_color_sensor'][k]
                                    for k in ('rpy', 'xyz')})
        self._hand_offset = np.asarray(pose['position_m'], dtype=float)
        self._hand_local = _wxyz_to_matrix(pose['quaternion_wxyz'])
        self.sensor = SensorTiledCamera(model=model)
        self.sensor.utils.create_default_light(enable_shadows=False)
        self.hand_rays = self.sensor.utils.compute_pinhole_camera_rays(
            HAND_W, HAND_H, math.radians(hand_fov_deg))
        self.hand_depth = self.sensor.utils.create_depth_image_output(
            HAND_W, HAND_H, 1)
        self.hand_color = self.sensor.utils.create_color_image_output(
            HAND_W, HAND_H, 1)
        self._frame_id = 0

    @staticmethod
    def _rot(q, v):
        u, s = q[:3], q[3]
        return v + 2.0 * np.cross(u, np.cross(u, v) + s * v)

    def _hand_pose(self, body_q) -> tuple:
        p, quat = body_q[self.wrist, :3], body_q[self.wrist, 3:]
        rotation = np.column_stack([
            self._rot(quat, np.array([1.0, 0.0, 0.0])),
            self._rot(quat, np.array([0.0, 1.0, 0.0])),
            self._rot(quat, np.array([0.0, 0.0, 1.0])),
        ])
        pos = p + rotation @ self._hand_offset
        return pos, rotation @ self._hand_local

    def update(self, state, sim_time_s: float) -> dict:
        """Render gripper RGB-D; return SensorBatch/v3 arrays."""
        newton, wp, _ = _require_newton()
        from newton.sensors import SensorTiledCamera as _S
        body_q = state.body_q.numpy()
        pos, R = self._hand_pose(body_q)
        Rm = wp.mat33f(R[0, 0], R[0, 1], R[0, 2],
                       R[1, 0], R[1, 1], R[1, 2],
                       R[2, 0], R[2, 1], R[2, 2])
        tf = wp.transformf(wp.vec3f(*map(float, pos)), wp.quat_from_matrix(Rm))
        cam = wp.array([[tf]], dtype=wp.transformf)
        self.model.bvh_refit_shapes(state)
        clear = _S.ClearData(clear_depth=float(self.max_range_m))
        self.sensor.update(state, cam, self.hand_rays,
                           color_image=self.hand_color,
                           depth_image=self.hand_depth, clear_data=clear)
        depth = self.hand_depth.numpy()[0, 0].astype(np.float32)
        color = self.hand_color.numpy()[0, 0].astype(np.uint8)
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < self.max_range_m)
        self._frame_id += 1
        return {
            "hand_rgb": color.reshape(HAND_H, HAND_W, 3),
            "hand_depth_m": np.where(valid, depth, 0.0),
            "hand_depth_valid": valid,
            "hand_pos": pos.astype(np.float32),
            "hand_R": R.astype(np.float32),
            "frame_id": self._frame_id,
            "capture_time_s": float(sim_time_s),
            "camera": "hand_color_sensor",
        }
