"""Spot camera rig → SensorBatch/v3. JP-ONLY (needs newton + GPU).

Mirrors the proven newton.sensors.SensorTiledCamera call pattern from
treesim/robot.py::WristCamera, but for Spot hardware:
- 1 hand camera (gripper RGB + ToF), 240x320 working resolution
- 5 body stereo pairs, diagnostic grayscale + depth at 120x160

Poses: hand cam from the wrist body pose + bracket offset along the
gripper approach axis (same toe-in construction as WristCamera); body
cams from chassis-frame mounts (provisional extrinsics PENDING
calibration — positions documented below, must be closed with the
integrator before claiming a calibrated rig).

Outputs SensorBatch/v3 with validity masks; invalid depth is stored as
0 + valid=false. Never emits GT masks, overlays or fruit IDs.
"""

from __future__ import annotations

import numpy as np

HAND_W, HAND_H = 320, 240
BODY_W, BODY_H = 160, 120
N_BODY = 5

# Provisional body mounts in chassis frame (x fwd, y left, z up) + yaw.
# PENDING calibration against the real rig; used for raycasting only.
BODY_MOUNTS = (
    {"pos": (0.45, 0.0, 0.25), "yaw": 0.0},     # front
    {"pos": (0.30, 0.20, 0.25), "yaw": 0.9},    # front-left
    {"pos": (0.30, -0.20, 0.25), "yaw": -0.9},  # front-right
    {"pos": (-0.30, 0.18, 0.25), "yaw": 2.4},   # rear-left
    {"pos": (-0.30, -0.18, 0.25), "yaw": -2.4},  # rear-right
)
HAND_BRACKET = (0.11, 0.0, 0.03)  # same convention as WristCamera._LOCAL_POS
HAND_FOCUS_AHEAD_M = 0.28


def _require_newton():
    try:
        import newton
        import warp as wp
        from newton.sensors import SensorTiledCamera
    except ImportError as e:
        raise ImportError("sensor stack required (newton + GPU)") from e
    return newton, wp, SensorTiledCamera


class SpotCameraRig:
    """Tiled-camera rig bound to a built Spot model. Call update() at 10 Hz."""

    def __init__(self, model, chassis: int, wrist: int,
                 hand_fov_deg: float = 60.0, body_fov_deg: float = 90.0,
                 max_range_m: float = 4.0):
        import math
        newton, wp, SensorTiledCamera = _require_newton()
        self._wp = wp
        self.model = model
        self.chassis = int(chassis)
        self.wrist = int(wrist)
        self.max_range_m = float(max_range_m)
        self.sensor = SensorTiledCamera(model=model)
        self.sensor.utils.create_default_light(enable_shadows=False)
        self.hand_rays = self.sensor.utils.compute_pinhole_camera_rays(
            HAND_W, HAND_H, math.radians(hand_fov_deg))
        self.body_rays = self.sensor.utils.compute_pinhole_camera_rays(
            BODY_W, BODY_H, math.radians(body_fov_deg))
        self.hand_depth = self.sensor.utils.create_depth_image_output(
            HAND_W, HAND_H, 1)
        self.hand_color = self.sensor.utils.create_color_image_output(
            HAND_W, HAND_H, 1)
        self.body_depth = self.sensor.utils.create_depth_image_output(
            BODY_W, BODY_H, N_BODY)
        self._frame_id = 0

    @staticmethod
    def _rot(q, v):
        u, s = q[:3], q[3]
        return v + 2.0 * np.cross(u, np.cross(u, v) + s * v)

    def _hand_pose(self, body_q) -> tuple:
        p, quat = body_q[self.wrist, :3], body_q[self.wrist, 3:]
        approach = self._rot(quat, np.array([0.0, 0.0, 1.0]))
        pos = p + self._rot(quat, np.array(HAND_BRACKET))
        focus = (p + self._rot(quat, np.array([0.0, 0.0, 0.10]))
                 + approach * HAND_FOCUS_AHEAD_M)
        fwd = focus - pos
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        up_ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(fwd, up_ref))) > 0.98:
            up_ref = self._rot(quat, np.array([0.0, -1.0, 0.0]))
        right = np.cross(fwd, up_ref)
        right /= np.linalg.norm(right) + 1e-9
        return pos, np.column_stack([right, np.cross(right, fwd), -fwd])

    def _body_poses(self, body_q) -> list:
        import math
        p, quat = body_q[self.chassis, :3], body_q[self.chassis, 3:]
        out = []
        for m in BODY_MOUNTS:
            yaw = m["yaw"]
            c, s = math.cos(yaw), math.sin(yaw)
            fwd = self._rot(quat, np.array([c, s, 0.0]))
            up = self._rot(quat, np.array([0.0, 0.0, 1.0]))
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right) + 1e-9
            pos = p + self._rot(quat, np.array(m["pos"]))
            out.append((pos, np.column_stack([right, np.cross(right, fwd),
                                             -fwd])))
        return out

    def update(self, state, sim_time_s: float) -> dict:
        """Render hand RGB-D + body depth; return SensorBatch/v3 arrays."""
        newton, wp, _ = self._require_newton()
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
        body_tfs = []
        for bp, bR in self._body_poses(body_q):
            Rm = wp.mat33f(bR[0, 0], bR[0, 1], bR[0, 2],
                           bR[1, 0], bR[1, 1], bR[1, 2],
                           bR[2, 0], bR[2, 1], bR[2, 2])
            body_tfs.append(wp.transformf(wp.vec3f(*map(float, bp)),
                                          wp.quat_from_matrix(Rm)))
        body_cam = wp.array([body_tfs], dtype=wp.transformf)
        self.sensor.update(state, body_cam, self.body_rays,
                           depth_image=self.body_depth, clear_data=clear)
        body_depth = self.body_depth.numpy()[0].astype(np.float32)
        body_valid = (np.isfinite(body_depth) & (body_depth > 0.05)
                      & (body_depth < self.max_range_m))
        self._frame_id += 1
        return {
            "hand_rgb": color.reshape(HAND_H, HAND_W, 3),
            "hand_depth_m": np.where(valid, depth, 0.0),
            "hand_depth_valid": valid,
            "hand_pos": pos.astype(np.float32),
            "hand_R": R.astype(np.float32),
            "body_depth_m": np.where(body_valid, body_depth, 0.0),
            "body_depth_valid": body_valid,
            "frame_id": self._frame_id,
            "capture_time_s": float(sim_time_s),
        }
