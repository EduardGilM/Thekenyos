#!/usr/bin/env python
"""Import the RELIC TorchScript actor into a trainable G1 (JP-ONLY).

Loads <relic>/source/relic/relic/assets/spot/pretrained/policy.pt via
torch.jit (never vendored), checks it against the ONNX parity manifest
from scripts/check_relic_parity.py (tol 1e-5), and saves g1_init.pt:
the JIT module plus metadata. Training wraps this module directly with
a trainable log-std head (see models_torch.load_relic_jit_actor).

Fails loudly if outputs mismatch: do not call that fine-tuning.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relic", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path,
                   help="Parity manifest JSON from check_relic_parity.py")
    p.add_argument("--out", required=True, type=Path,
                   help="Output g1_init.pt (outside git, SSD)")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()
    import torch
    from treesim.kiwi_rl.models_torch import load_relic_jit_actor, jit_actor_forward

    pt = (args.relic / "source/relic/relic/assets/spot/pretrained/policy.pt")
    if not pt.is_file():
        raise FileNotFoundError(f"missing {pt} (external checkout)")
    man = json.loads(args.manifest.read_text())
    mod = load_relic_jit_actor(str(pt))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.samples):
        x = rng.standard_normal((1, 84)).astype(np.float32)
        y = jit_actor_forward(mod, x)
        if y.shape != (1, 12) or not np.isfinite(y).all():
            raise ValueError("JIT actor produced invalid output")
    a = jit_actor_forward(mod, np.zeros((1, 84), dtype=np.float32))[0]
    b = jit_actor_forward(mod, np.zeros((1, 84), dtype=np.float32))[0]
    worst = float(np.abs(a - b).max())
    ref = np.asarray(man["reference_zero_obs_out"], dtype=np.float32)
    cross = float(np.abs(a - ref).max())
    ok = worst <= args.tol and cross <= args.tol
    print(json.dumps({
        "jit_determinism_max_abs_diff": worst,
        "jit_vs_onnx_zero_obs_max_abs_diff": cross,
        "tol": args.tol, "pass": bool(ok)}, indent=2))
    if not ok:
        raise SystemExit("G1 import gate FAILED")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Persist the script module itself (reloadable actor) + sidecar manifest.
    torch.jit.save(mod, str(args.out))
    (args.out.parent / (args.out.stem + ".json")).write_text(json.dumps(
        {"source": str(pt), "onnx_sha256": man.get("onnx_sha256"),
         "tol": args.tol,
         "jit_vs_onnx_zero_obs_max_abs_diff": cross}, indent=2) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
