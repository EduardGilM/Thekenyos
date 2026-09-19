#!/usr/bin/env python
"""G1 release gate: verify the external RELIC ONNX policy before training.

Checks the pinned-checkout weights *in place* (never vendored into this
repo, per AGENTS.md / RELIC noncommercial terms):
- file exists at <relic>/source/relic/relic/assets/spot/pretrained/policy.onnx
- I/O contract is obs [1,84] float32 -> actions [1,12] float32
- outputs are finite on seeded valid-range R84 samples
- outputs are deterministic across repeated runs (max abs diff == 0)
- records a manifest (sha256, reference vectors) that the future torch
  G1 import must reproduce within tol 1e-5 (see models.onnx_pytorch_parity)

Verified upstream architecture (from the ONNX graph itself):
MLP 84->512->256->128->12, Elu after each hidden layer, linear output.
policy.pt in the same directory is a TorchScript export (compiled code,
no optimizer state), so it cannot be fine-tuned directly; reconstruct
the actor and load weights, then pass this gate.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

PINNED_COMMIT = "27f8033c5064d32f049a17accb71cd1091422878"
TOL = 1e-5


def sample_valid_r84(n: int, seed: int) -> np.ndarray:
    """Seeded R84 samples inside documented valid ranges (float32)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 84), dtype=np.float32)
    x[:, 0:3] = rng.uniform(-2, 2, (n, 3))        # lin vel [m/s]
    x[:, 3:6] = rng.uniform(-3, 3, (n, 3))        # ang vel [rad/s]
    g = rng.standard_normal((n, 3))
    x[:, 6:9] = g / np.linalg.norm(g, axis=1, keepdims=True)
    x[:, 9] = rng.uniform(-0.4, 0.4, n)           # vx
    x[:, 10] = rng.uniform(-0.3, 0.3, n)          # vy
    x[:, 11] = rng.uniform(-0.7, 0.7, n)          # wz
    x[:, 12:19] = rng.uniform(-0.5, 0.5, (n, 7))  # arm targets [rad]
    x[:, 19:31] = 0.0                             # interlimb leg subgoals
    x[:, 31:34] = np.array([0.0, 0.0, 0.55], dtype=np.float32)
    x[:, 34:53] = rng.uniform(-0.5, 0.5, (n, 19))
    x[:, 53:72] = rng.uniform(-5, 5, (n, 19))
    x[:, 72:84] = rng.uniform(-1, 1, (n, 12))
    assert np.isfinite(x).all()
    return x.astype(np.float32)


def run_parity(relic: Path, samples: int, seed: int) -> dict:
    import onnxruntime as ort

    relic = relic.resolve()
    actual_commit = subprocess.check_output(['git', '-C', str(relic), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != PINNED_COMMIT:
        raise ValueError(f'RELIC checkout is not pinned: {actual_commit}')
    dirty = subprocess.check_output(['git', '-C', str(relic), 'status', '--porcelain', '--',
                                     'source/relic/relic/assets/spot'], text=True).strip()
    if dirty:
        raise ValueError('RELIC assets differ from the pinned checkout')
    if samples < 1:
        raise ValueError('Use at least one parity sample')
    onnx_path = (relic / "source/relic/relic/assets/spot/pretrained"
                 / "policy.onnx")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"missing {onnx_path} (external checkout)")
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if (len(ins) != 1 or list(ins[0].shape) != [1, 84]
            or len(outs) != 1 or list(outs[0].shape) != [1, 12]):
        raise ValueError("I/O contract must be [1,84] -> [1,12]")
    xs = sample_valid_r84(samples, seed)
    first = sess.run(None, {ins[0].name: xs[:1]})[0]
    if first.shape != (1, 12) or not np.isfinite(first).all():
        raise ValueError("non-finite or misshapen ONNX output")
    max_abs = float(np.abs(first).max())
    worst_drift, worst_finite = 0.0, True
    for row in xs:
        a = sess.run(None, {ins[0].name: row[None]})[0][0]
        b = sess.run(None, {ins[0].name: row[None]})[0][0]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            worst_finite = False
            break
        max_abs = max(max_abs, float(np.abs(a).max()))
        worst_drift = max(worst_drift, float(np.abs(a - b).max()))
    ok = worst_finite and worst_drift <= TOL
    return {
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha,
        "pinned_commit": PINNED_COMMIT,
        "samples": samples,
        "seed": seed,
        "io_contract": "[1,84]f32 -> [1,12]f32",
        "reference_zero_obs_out": None,  # filled below (1 more run)
        "max_abs_action": max_abs,
        "determinism_max_abs_diff": worst_drift,
        "tol": TOL,
        "pass": bool(ok),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relic", required=True, type=Path)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=None)
    args = p.parse_args()
    if args.samples <= 0:
        p.error("--samples must be positive")
    import onnxruntime as ort
    res = run_parity(args.relic, args.samples, args.seed)
    sess = ort.InferenceSession(res["onnx_path"], providers=["CPUExecutionProvider"])
    z = np.zeros((1, 84), dtype=np.float32)
    res["reference_zero_obs_out"] = sess.run(
        None, {sess.get_inputs()[0].name: z})[0][0].tolist()
    print(json.dumps(res, indent=2))
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(res, indent=2) + "\n")
    if not res["pass"]:
        raise SystemExit("RELIC parity gate FAILED")


if __name__ == "__main__":
    main()
