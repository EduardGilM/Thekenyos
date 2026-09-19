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
import hashlib
import io
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
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--rtol', type=float, default=2e-6)
    args = p.parse_args()
    if args.samples < 10000 or not np.isfinite([args.tol, args.rtol]).all() or not 0 < args.tol <= 1e-5 or not 0 <= args.rtol <= 2e-6:
        p.error('Release parity requires >=10000 samples, atol <=1e-5, and rtol <=2e-6')
    metadata_path = args.out.parent / (args.out.stem + '.json')
    if args.out.exists() or metadata_path.exists():
        p.error('Preserve existing exported actor and metadata')
    import torch
    import onnxruntime as ort
    from scripts.check_relic_parity import run_parity, sample_valid_r84
    from treesim.kiwi_rl.models_torch import load_relic_jit_actor, jit_actor_forward, load_trainable_relic_actor
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')

    pt = (args.relic / "source/relic/relic/assets/spot/pretrained/policy.pt")
    if not pt.is_file():
        raise FileNotFoundError(f"missing {pt} (external checkout)")
    man = json.loads(args.manifest.read_text())
    current = run_parity(args.relic, 1, args.seed)
    if not man.get('pass') or not current['pass'] or man.get('onnx_sha256') != current['onnx_sha256']:
        raise ValueError('Invalid or stale ONNX manifest')
    mod = load_relic_jit_actor(str(pt))
    trainable = load_trainable_relic_actor(str(pt)).to(args.device).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(current['onnx_path'], sess_options=options, providers=['CPUExecutionProvider'])
    observations = sample_valid_r84(args.samples, args.seed)
    from treesim.kiwi_rl.models import onnx_pytorch_parity
    worst_cross, worst_trainable, worst_scaled = 0., 0., 0.
    for start in range(0, args.samples, 128):
        x = observations[start:start + 128]
        reference = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in x])
        y = jit_actor_forward(mod, x)
        with torch.no_grad():
            actual = trainable(torch.from_numpy(x).to(args.device)).cpu().numpy()
        if y.shape != reference.shape or actual.shape != reference.shape or not all(np.isfinite(v).all() for v in (y, actual, reference)):
            raise ValueError('Nonfinite or invalid actor output')
        worst_cross = max(worst_cross, float(np.abs(y - reference).max()))
        worst_trainable = max(worst_trainable, float(np.abs(actual - reference).max()))
        for candidate in (y, actual):
            comparison = onnx_pytorch_parity(reference, candidate, args.tol, args.rtol)
            worst_scaled = max(worst_scaled, comparison['max_scaled_err'])
    a = jit_actor_forward(mod, np.zeros((1, 84), dtype=np.float32))[0]
    b = jit_actor_forward(mod, np.zeros((1, 84), dtype=np.float32))[0]
    worst = float(np.abs(a - b).max())
    ref = np.asarray(man["reference_zero_obs_out"], dtype=np.float32)
    cross = float(np.abs(a - ref).max())
    ok = max(worst, cross) <= args.tol and worst_scaled <= 1.
    report = dict(jit_determinism_max_abs_diff=worst, jit_vs_onnx_zero_obs_max_abs_diff=cross,
                  jit_vs_onnx_max_abs_diff=worst_cross, trainable_vs_onnx_max_abs_diff=worst_trainable,
                  max_scaled_error=worst_scaled, weights_identical=True, samples=args.samples,
                  seed=args.seed, device=args.device, tol=args.tol, rtol=args.rtol, passed=bool(ok))
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit("G1 import gate FAILED")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Persist the script module itself (reloadable actor) + sidecar manifest.
    from treesim.kiwi_rl.ppo import _publish_new
    blob = io.BytesIO()
    torch.jit.save(mod, blob)
    payload = blob.getvalue()
    _publish_new(args.out, payload)
    report.update(source=str(pt), onnx_sha256=current['onnx_sha256'],
                  source_jit_sha256=hashlib.sha256(pt.read_bytes()).hexdigest(),
                  exported_sha256=hashlib.sha256(payload).hexdigest(), pinned_commit=current['pinned_commit'])
    _publish_new(metadata_path, (json.dumps(report, indent=2, allow_nan=False) + '\n').encode())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
