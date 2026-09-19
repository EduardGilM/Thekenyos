"""Model interfaces (numpy, no torch dependency).

V3: RGB-D encoder stub mapping [B,5,240,320] -> z [B,256].
N3: 488 -> (mu[B,3], logits[B,2]) + GRU-256 state passthrough.
M3: 382 -> (mu[B,8], logits[B,3]) + GRU-256 state passthrough.
G1: 84 -> mu[B,12]; inference delegates to the RELIC ONNX session when a
    live Sim is attached, otherwise a zero-mean placeholder. The trainable
    numpy copy is explicitly a placeholder pending weight import + parity.

All sampling uses tanh-squashed diagonal Gaussians with Jacobian-corrected
log-probs for N3/M3; G1 keeps the upstream linear output. Log-std is
trainable per action, clipped to [-5, 1].
"""

from __future__ import annotations

import numpy as np

from . import schemas as S

LOGSTD_MIN, LOGSTD_MAX = -5.0, 1.0
GRU_DIM = 256


def _check_batch(name: str, arr: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"{name}: shape {a.shape} != (B, {dim})")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return a


def tanh_squash(u: np.ndarray) -> np.ndarray:
    return np.tanh(u).astype(np.float32)


def gaussian_logprob_tanh(u: np.ndarray, mu: np.ndarray,
                          logstd: np.ndarray) -> np.ndarray:
    """Log-prob of a=tanh(u) under N(mu, sigma), with Jacobian correction."""
    u = np.asarray(u, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    ls = np.clip(np.asarray(logstd, dtype=np.float32), LOGSTD_MIN, LOGSTD_MAX)
    if u.shape != mu.shape:
        raise ValueError(f"u {u.shape} != mu {mu.shape}")
    var = np.exp(2.0 * ls)
    logp = -0.5 * (((u - mu) ** 2) / var + 2.0 * ls + np.log(2.0 * np.pi))
    logp = logp.sum(axis=-1)
    jacob = np.log(np.maximum(1.0 - np.tanh(u) ** 2, 1e-9)).sum(axis=-1)
    return (logp + jacob).astype(np.float32)


def categorical_logprob(logits: np.ndarray, index: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    i = np.asarray(index, dtype=np.int64)
    if z.ndim != 2 or i.shape != (z.shape[0],):
        raise ValueError("logits (B,K), index (B,)")
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return logp[np.arange(z.shape[0]), i].astype(np.float32)


class VisionEncoderV3:
    """Stub encoder: deterministic projection placeholders, correct shapes.

    Real ResNet-18-GN weights are integration work; this stub validates the
    interface (5-channel input -> z[256]) and keeps RNG-seeded behaviour.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.standard_normal((5, 256)).astype(np.float32) * 0.02

    def forward(self, v3_input: np.ndarray) -> np.ndarray:
        x = np.asarray(v3_input, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (5, 240, 320):
            raise ValueError(f"v3_input shape {x.shape} != (B,5,240,320)")
        if not np.isfinite(x).all():
            raise ValueError("v3_input non-finite")
        chan = x.mean(axis=(2, 3))  # (B,5) spatial summary placeholder
        return (chan @ self.proj).astype(np.float32)


class MapEncoder:
    """Stub map encoder: [B,3,64,64] -> [B,128]."""

    def __init__(self, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.proj = self.rng.standard_normal((3, 128)).astype(np.float32) * 0.05

    def forward(self, map_obs: np.ndarray) -> np.ndarray:
        x = np.asarray(map_obs, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (3, 64, 64):
            raise ValueError(f"map shape {x.shape} != (B,3,64,64)")
        chan = x.mean(axis=(2, 3))
        return (chan @ self.proj).astype(np.float32)


class RecurrentPolicyHead:
    """Shared stub for N3/M3: fixed random projection + zero recurrence.

    Shapes and event semantics are exact; weights are random placeholders.
    GRU state passes through unchanged (integration replaces with real GRU).
    """

    def __init__(self, input_dim: int, cont_dim: int, event_dim: int,
                 seed: int = 0, sigma_init: float = 0.3,
                 event_bias: tuple | None = None):
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((input_dim, cont_dim)).astype(np.float32)
                  * 0.01)
        self.v = (rng.standard_normal((input_dim, event_dim)).astype(np.float32)
                  * 0.01)
        self.bias = (np.zeros(event_dim, dtype=np.float32) if event_bias is None
                     else np.asarray(event_bias, dtype=np.float32))
        self.logstd = np.full(cont_dim, np.log(sigma_init), dtype=np.float32)
        self.input_dim, self.cont_dim, self.event_dim = input_dim, cont_dim, event_dim

    def forward(self, x: np.ndarray, h: np.ndarray):
        x = _check_batch("policy_input", x, self.input_dim)
        h = np.asarray(h, dtype=np.float32)
        if h.shape != (1, x.shape[0], GRU_DIM):
            raise ValueError(f"h shape {h.shape} != (1,B,{GRU_DIM})")
        mu = (x @ self.w).astype(np.float32)
        logits = (x @ self.v + self.bias).astype(np.float32)
        return mu, logits, h.copy()


def make_n3(seed: int = 0) -> RecurrentPolicyHead:
    return RecurrentPolicyHead(S.N3_INPUT_DIM, S.N3_CONT_DIM, S.N3_EVENT_DIM,
                               seed=seed, sigma_init=0.3, event_bias=(2.0, 0.0))


def make_m3(seed: int = 0) -> RecurrentPolicyHead:
    return RecurrentPolicyHead(S.M3_INPUT_DIM, S.M3_CONT_DIM, S.M3_EVENT_DIM,
                               seed=seed, sigma_init=0.2,
                               event_bias=(2.0, 0.0, 0.0))


class GaitPolicyG1:
    """G1 interface: 84 -> 12 with upstream linear scaling.

    With a live SpotController attached, `act` runs the RELIC ONNX session
    (pretrained inference). `trainable_delta` exposes a numpy residual that
    starts at zero so adaptation can be validated without touching ONNX.
    """

    def __init__(self, spot_controller=None):
        self.spot = spot_controller
        self.delta = np.zeros(S.G1_DIM, dtype=np.float32)
        self.logstd = np.full(S.G1_DIM, np.log(0.15), dtype=np.float32)

    def reference_action(self, r84: np.ndarray) -> np.ndarray:
        """Pretrained-inference action (ONNX) or zeros without a controller."""
        r = _check_batch("r84", r84, S.R84_DIM)
        if self.spot is None:
            return np.zeros((r.shape[0], S.G1_DIM), dtype=np.float32)
        out = []
        for row in r:
            self.spot.update(row[9:12].astype(np.float32))
            out.append(self.spot.last_action.copy())
        return np.asarray(out, dtype=np.float32)

    def act(self, r84: np.ndarray) -> np.ndarray:
        return (self.reference_action(r84) + self.delta).astype(np.float32)

    def apply_to_targets(self, home_legs: np.ndarray, action: np.ndarray):
        """Upstream scaling q = home + 0.2*u (order LEGS)."""
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (S.G1_DIM,):
            raise ValueError(f"action shape {a.shape} != (12,)")
        return (np.asarray(home_legs, dtype=np.float32) + 0.2 * a).astype(np.float32)


def onnx_pytorch_parity(onnx_actions: np.ndarray, torch_actions: np.ndarray,
                        tol: float = 1e-5) -> dict:
    """Parity gate for G1 weight import: max abs error over samples."""
    a = np.asarray(onnx_actions, dtype=np.float32)
    b = np.asarray(torch_actions, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != S.G1_DIM:
        raise ValueError("parity inputs must share shape (N,12)")
    err = float(np.abs(a - b).max())
    return {"max_abs_err": err, "tol": tol, "pass": bool(err <= tol)}
