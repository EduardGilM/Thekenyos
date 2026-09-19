"""Wrist-camera actor observations for the sparse pixel SAC lane.

Demo collection and training MUST call :func:`build_observation` on the same
state dict. Reward stays in :func:`treesim.kiwi_rl.rewards.compute_reward_batch`
and never reads these images.
"""

from __future__ import annotations

import numpy as np

from . import schemas as S
from .sensors import HAND_H, HAND_W

CAMERA_CHANNELS = 5
FRAME_STACK = 3
LATENCY_STEPS = 1
DEPTH_SCALE_M = 3.0
LATCH_NAMES = ('fallen', 'damaged', 'spilled', 'lost')
VECTOR_DIM = S.P85_DIM + len(S.PHASES) + len(LATCH_NAMES)


def encode_wrist(rgb, depth_m, valid):
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(valid)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Wrist RGB must be uint8 [H, W, 3]')
    height, width = rgb.shape[:2]
    if depth.shape != (height, width) or mask.shape != (height, width):
        raise ValueError('Wrist depth and validity must match RGB spatial size')
    if not np.isfinite(depth).all():
        raise ValueError('Wrist depth must be finite')
    rgb_n = rgb.astype(np.float32).transpose(2, 0, 1) / 255.
    depth_n = np.clip(np.where(mask, depth, 0.) / DEPTH_SCALE_M, 0., 1.).astype(np.float32)
    valid_n = np.asarray(mask, dtype=np.float32)
    return np.concatenate([rgb_n, depth_n[None], valid_n[None]], axis=0)


def _phase_one_hot(phase):
    if phase not in S.PHASES:
        raise ValueError(f'unknown phase {phase!r}')
    out = np.zeros(len(S.PHASES), dtype=np.float32)
    out[S.PHASES.index(phase)] = 1.
    return out


def _latches(state):
    flags = np.asarray(state['latches'], dtype=np.float32).reshape(-1)
    if flags.shape != (len(LATCH_NAMES),) or not np.isfinite(flags).all():
        raise ValueError('Latches must be four finite flags')
    if np.any((flags != 0.) & (flags != 1.)):
        raise ValueError('Latches must be 0 or 1')
    return flags


def _queue_frames(state, key, channels):
    items = tuple(np.asarray(frame, dtype=np.float32) for frame in state.get(key, ()))
    for frame in items:
        if frame.shape[0] != channels or frame.ndim != 3:
            raise ValueError(f'{key} frames must be [C, H, W] with C={channels}')
    return items


def build_observation(state):
    """Camera stack + proprio + phase + latches. No fruit coordinates.

    ``state`` is the capture-time buffer: current wrist images, delay queue,
    visible history, proprioception, phase and irreversible latches. Demo
    replay must pass this dict unchanged.
    """
    capture = encode_wrist(state['wrist_rgb'], state['wrist_depth_m'], state['wrist_valid'])
    delay = _queue_frames(state, 'delay_queue', CAMERA_CHANNELS)
    if len(delay) > LATENCY_STEPS:
        raise ValueError('Delay queue exceeds the shared action-delay buffer')
    pending = delay + (capture,)
    visible = pending[0] if len(pending) > LATENCY_STEPS else np.zeros_like(capture)
    history = _queue_frames(state, 'visible_history', CAMERA_CHANNELS)
    stacked = (history + (visible,))[-FRAME_STACK:]
    if stacked[0].shape != capture.shape:
        raise ValueError('Frame-stack spatial size must match the current wrist image')
    while len(stacked) < FRAME_STACK:
        stacked = (np.zeros_like(capture),) + stacked
    proprio = S.validate_finite('proprio', np.asarray(state['proprio'], dtype=np.float32), (S.P85_DIM,))
    vector = np.concatenate([proprio, _phase_one_hot(state['phase']), _latches(state)])
    camera = np.concatenate(stacked, axis=0)
    if camera.shape[0] != CAMERA_CHANNELS * FRAME_STACK or not np.isfinite(camera).all():
        raise ValueError('Invalid stacked wrist observation')
    if 'fruit' in state or 'fruit_xyz' in state or 'target_position' in state:
        raise ValueError('Actor observations must not carry simulator fruit coordinates')
    return dict(camera=camera.astype(np.float32, copy=True), vector=vector.astype(np.float32, copy=True))


def next_observation_state(state):
    """Advance delay and history after observing. Used by demo collection and env steps."""
    capture = encode_wrist(state['wrist_rgb'], state['wrist_depth_m'], state['wrist_valid'])
    delay = _queue_frames(state, 'delay_queue', CAMERA_CHANNELS) + (capture,)
    visible = delay[0] if len(delay) > LATENCY_STEPS else np.zeros_like(capture)
    delay = delay[-LATENCY_STEPS:] if LATENCY_STEPS else ()
    history = (_queue_frames(state, 'visible_history', CAMERA_CHANNELS) + (visible,))[-FRAME_STACK:]
    updated = dict(state)
    updated['delay_queue'] = tuple(frame.copy() for frame in delay)
    updated['visible_history'] = tuple(frame.copy() for frame in history)
    return updated


def snapshot_obs_state(state):
    """Deep-copy the fields :func:`build_observation` reads."""
    rgb = np.asarray(state['wrist_rgb']).copy()
    return dict(
        wrist_rgb=rgb,
        wrist_depth_m=np.asarray(state['wrist_depth_m'], dtype=np.float32).copy(),
        wrist_valid=np.asarray(state['wrist_valid']).copy(),
        proprio=np.asarray(state['proprio'], dtype=np.float32).copy(),
        phase=str(state['phase']),
        latches=np.asarray(state['latches'], dtype=np.float32).copy(),
        delay_queue=tuple(np.asarray(frame, dtype=np.float32).copy()
                          for frame in state.get('delay_queue', ())),
        visible_history=tuple(np.asarray(frame, dtype=np.float32).copy()
                              for frame in state.get('visible_history', ())),
    )


def record_demo_transition(state, action, next_state, privileged, next_privileged):
    """Store actor observations plus privileged pairs for exact reward relabel."""
    obs_state = snapshot_obs_state(state)
    next_obs_state = snapshot_obs_state(next_state)
    return dict(
        observation=build_observation(obs_state),
        next_observation=build_observation(next_obs_state),
        obs_state=obs_state,
        next_obs_state=next_obs_state,
        action=np.asarray(action, dtype=np.float32).copy(),
        privileged_state=dict(privileged),
        next_privileged_state=dict(next_privileged),
    )


def empty_wrist_state(height=HAND_H, width=HAND_W, phase='EXPLORE'):
    return dict(
        wrist_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        wrist_depth_m=np.zeros((height, width), dtype=np.float32),
        wrist_valid=np.zeros((height, width), dtype=bool),
        proprio=np.zeros(S.P85_DIM, dtype=np.float32),
        phase=phase,
        latches=np.zeros(len(LATCH_NAMES), dtype=np.float32),
        delay_queue=(),
        visible_history=(),
    )
