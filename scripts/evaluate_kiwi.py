#!/usr/bin/env python
"""Evaluate a kiwi-harvest bundle: physics checks + rollout metrics.

Reports stored/retained fruit, falls, spills, damage-proxy and vision
dependence separately. Inference labels (scripted/programmed/learned)
must be accurate; a video never replaces these numbers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treesim.kiwi_rl import schemas as S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True,
                   help="Rollout metrics JSON produced by a run")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    data = json.loads(args.metrics.read_text())
    required = ("episodes", "stored_retained", "falls", "spills",
                "damage_proxy_sum")
    missing = [k for k in required if k not in data]
    if missing:
        p.error(f"metrics missing {missing}")
    summary = {
        "schema": S.REWARD_SCHEMA_VERSION,
        "success_rate": (data["stored_retained"] / max(data["episodes"], 1)),
        "falls": data["falls"],
        "spills": data["spills"],
        "damage_proxy_sum": data["damage_proxy_sum"],
        "note": ("damage is an uncalibrated proxy, not bruise prediction; "
                 "spills counted once per fruit"),
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
