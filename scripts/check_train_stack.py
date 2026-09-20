#!/usr/bin/env python
"""Training-stack smoke test (run on jp before any job).

Verifies: torch + CUDA (Blackwell), torchvision, newton/warp/mujoco
versions matching environment.yml, onnxruntime, external RELIC files,
SSD space and GPU visibility. Fails fast with actionable messages.
Run: python scripts/check_train_stack.py --relic <checkout> [--ssd PATH]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PINNED = {"newton": "1.3.0", "warp": "1.14.0", "mujoco": "3.8.1",
          "onnxruntime": "1.30.0"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relic", required=True, type=Path)
    p.add_argument("--ssd", type=Path, default=Path("/mnt/ssd"))
    args = p.parse_args()
    ok = True

    import torch
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA (Blackwell RTX 5090 expected)");
        ok = False
    else:
        cap = torch.cuda.get_device_capability(0)
        print(f"gpu={torch.cuda.get_device_name(0)} capability={cap}")
        if cap[0] < 12:
            print("WARN: expected Blackwell sm_120+; check torch/CUDA build")
    try:
        import torchvision
        print(f"torchvision={torchvision.__version__}")
    except ImportError:
        print("FAIL: torchvision missing");
        ok = False

    import importlib.metadata as md
    for pkg, want in PINNED.items():
        try:
            got = md.version("warp-lang" if pkg == "warp" else pkg)
        except Exception:
            got = "missing"
        flag = "ok" if got == want else "MISMATCH"
        print(f"{pkg}: {got} (pinned {want}) [{flag}]")
        if got != want:
            ok = False
    try:
        import newton, warp, mujoco  # noqa: F401
        print("newton/warp/mujoco imports ok")
    except ImportError as e:
        print(f"FAIL: sim stack import: {e}");
        ok = False

    for rel in ("source/relic/relic/assets/spot/spot_with_arm.urdf",
                "source/relic/relic/assets/spot/constants.py",
                "source/relic/relic/assets/spot/pretrained/policy.onnx",
                "source/relic/relic/assets/spot/pretrained/policy.pt"):
        hit = (args.relic / rel).is_file()
        print(f"{'ok' if hit else 'MISSING'}: {rel}")
        ok &= hit
    if args.ssd.exists():
        free_gb = shutil.disk_usage(args.ssd).free / 1e9
        print(f"ssd {args.ssd}: {free_gb:.1f} GB free")
        if free_gb < 20:
            print("WARN: <20 GB free on SSD");
    else:
        print(f"WARN: ssd path {args.ssd} absent")
    if not ok:
        raise SystemExit("train-stack check FAILED")
    print("train-stack check PASSED")


if __name__ == "__main__":
    main()
