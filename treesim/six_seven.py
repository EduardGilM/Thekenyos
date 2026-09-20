"""Scripted Spot rear-up and 6-7 dance poses.

This is a kinematic gag, not locomotion, not a harvest demonstration, and not
a learned policy. Joint limits come from RELIC's Spot URDF.
"""
from __future__ import annotations

import math

import numpy as np

LEGS = [f'{leg}_{axis}' for axis in ('hx', 'hy', 'kn')
        for leg in ('fl', 'fr', 'hl', 'hr')]
ARM = ['arm_sh0', 'arm_sh1', 'arm_el0', 'arm_el1', 'arm_wr0', 'arm_wr1', 'arm_f1x']

STAND_Z_M = 0.55
# Magnitude of the wheelie. Applied as negative pitch so +X (nose) points up.
REAR_PITCH_RAD = 1.12
HIND_HIP_X_M = -0.29785
DANCE_HZ = 1.2
STAND_S = 1.0
REAR_S = 2.2
DANCE_S = 6.5

# URDF revolute limits from spot_with_arm.urdf.
JOINT_LIMITS = {
    'fl_hx': (-0.785398, 0.785398),
    'fr_hx': (-0.785398, 0.785398),
    'hl_hx': (-0.785398, 0.785398),
    'hr_hx': (-0.785398, 0.785398),
    'fl_hy': (-0.898845, 2.295108),
    'fr_hy': (-0.898845, 2.295108),
    'hl_hy': (-0.898845, 2.295108),
    'hr_hy': (-0.898845, 2.295108),
    'fl_kn': (-2.7929, -0.2471),
    'fr_kn': (-2.7929, -0.2471),
    'hl_kn': (-2.7929, -0.2471),
    'hr_kn': (-2.7929, -0.2471),
    'arm_sh0': (-2.61799, 3.14159),
    'arm_sh1': (-3.14159, 0.523599),
    'arm_el0': (0.0, 3.14159),
    'arm_el1': (-2.7929, 2.7929),
    'arm_wr0': (-1.8326, 1.8326),
    'arm_wr1': (-2.8798, 2.8798),
    'arm_f1x': (-1.5708, 0.0),
}


def clamp_joint(name: str, value: float) -> float:
    lo, hi = JOINT_LIMITS[name]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(np.clip(value, lo, hi))


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lerp_joints(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    t = float(np.clip(t, 0.0, 1.0))
    keys = set(a) | set(b)
    return {name: clamp_joint(name, (1.0 - t) * float(a[name]) + t * float(b[name]))
            for name in keys}


def stand_joints(home: dict[str, float]) -> dict[str, float]:
    return {name: clamp_joint(name, float(home[name])) for name in (*LEGS, *ARM)}


def rear_joints(home: dict[str, float]) -> dict[str, float]:
    """Hind legs planted, front legs out like two hands."""
    pose = stand_joints(home)
    pose.update({
        'hl_hx': 0.20,
        'hr_hx': -0.20,
        'hl_hy': 0.85,
        'hr_hy': 0.85,
        'hl_kn': -1.35,
        'hr_kn': -1.35,
        'fl_hx': 0.55,
        'fr_hx': -0.55,
        'fl_hy': 1.55,
        'fr_hy': 1.55,
        'fl_kn': -1.55,
        'fr_kn': -1.55,
        'arm_sh0': 0.45,
        'arm_sh1': -0.35,
        'arm_el0': 1.15,
        'arm_el1': 0.0,
        'arm_wr0': -0.55,
        'arm_wr1': 0.35,
        'arm_f1x': -0.90,
    })
    return {name: clamp_joint(name, pose[name]) for name in pose}


def dance_joints(home: dict[str, float], phase_rad: float) -> dict[str, float]:
    """6-7 bounce: both front 'hands' on the beat, arm answering SIX vs SEVEN."""
    base = rear_joints(home)
    six = math.sin(phase_rad)
    seven = math.sin(phase_rad + math.pi)
    bounce = 0.40 * six
    pose = dict(base)
    pose['fl_hy'] = base['fl_hy'] + bounce
    pose['fr_hy'] = base['fr_hy'] + bounce
    pose['fl_hx'] = base['fl_hx'] + 0.10 * seven
    pose['fr_hx'] = base['fr_hx'] - 0.10 * six
    pose['fl_kn'] = base['fl_kn'] - 0.22 * six
    pose['fr_kn'] = base['fr_kn'] - 0.22 * six
    pose['arm_sh0'] = 0.70 * six
    pose['arm_sh1'] = -0.25 + 0.28 * seven
    pose['arm_el0'] = 1.05 + 0.45 * six
    pose['arm_wr1'] = 0.70 * seven
    pose['hl_hx'] = base['hl_hx'] + 0.05 * seven
    pose['hr_hx'] = base['hr_hx'] - 0.05 * six
    return {name: clamp_joint(name, pose[name]) for name in pose}


def caption_for_phase(phase_rad: float) -> str:
    cycle = float(phase_rad) % (2.0 * math.pi)
    return 'SIX' if cycle < math.pi else 'SEVEN'


def chassis_xyz_m(pitch: float, bounce_m: float = 0.0) -> tuple[float, float, float]:
    """Place the freejoint so a Y-pitch keeps the hind hips near their stand x."""
    x = HIND_HIP_X_M * (1.0 - math.cos(pitch))
    z = STAND_Z_M + HIND_HIP_X_M * math.sin(pitch) + bounce_m
    return (x, 0.0, z)


def pose_at(time_s: float, home: dict[str, float]) -> dict:
    """Kinematic pose at ``time_s`` seconds from the start of the gag."""
    t = float(time_s)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError('time_s must be finite and >= 0')
    stand = stand_joints(home)
    rear = rear_joints(home)
    dance_t0 = STAND_S + REAR_S
    if t <= STAND_S:
        pitch = 0.0
        joints = stand
        caption = ''
        phase = 0.0
        bounce = 0.0
    elif t <= dance_t0:
        u = smoothstep((t - STAND_S) / REAR_S)
        pitch = -u * REAR_PITCH_RAD
        joints = lerp_joints(stand, rear, u)
        caption = ''
        phase = 0.0
        bounce = 0.0
    else:
        pitch = -REAR_PITCH_RAD
        phase = 2.0 * math.pi * DANCE_HZ * (t - dance_t0)
        joints = dance_joints(home, phase)
        caption = caption_for_phase(phase)
        bounce = 0.04 * abs(math.sin(phase))
    roll = 0.12 * math.sin(phase) if t > dance_t0 else 0.0
    yaw = 0.08 * math.sin(2.0 * phase) if t > dance_t0 else 0.0
    return {
        'time_s': t,
        'xyz_m': chassis_xyz_m(pitch, bounce),
        'rpy_rad': (roll, pitch, yaw),
        'joints': joints,
        'caption': caption,
        'phase_rad': phase,
        'reared': t >= dance_t0,
        'hind_support': abs(pitch) > 0.15,
    }


def duration_s() -> float:
    return STAND_S + REAR_S + DANCE_S


def rpy_to_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ (roll, pitch, yaw) to MuJoCo wxyz."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
