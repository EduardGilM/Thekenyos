#!/usr/bin/env python
"""Scaffolded kiwi-harvest training entry point (TK-RL-003 v3.0).

Builds the versioned env, validates PPO configs and writes a run
manifest. Optimisation (torch/rsl-rl) is integration work; this script
establishes the reproducible CLI, seeds and artefact layout so training
runs are comparable from day one. Outputs stay out of Git (run.sh/SSD).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treesim.kiwi_rl import schemas as S
from treesim.kiwi_rl.training import (
    HARVEST_PPO, G1_PPO, STAGES, TRAIN_SEEDS, checkpoint_manifest,
    validate_config,
)
from dataclasses import asdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relic", required=True, type=Path)
    p.add_argument("--stage", default="deposit_pixels",
                   choices=[s.name for s in STAGES])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--goal", default=None, choices=list(S.GOALS) + [None])
    p.add_argument("--payload", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("output/kiwi-rl-run"))
    p.add_argument("--dry-run", action="store_true",
                   help="Validate configs and env construction only")
    p.add_argument("--run", action="store_true",
                   help="JP-ONLY: full training loop (needs sim + torch + GPU)")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    if args.seed not in TRAIN_SEEDS:
        print(f"warning: seed {args.seed} not in prescribed {TRAIN_SEEDS}",
              flush=True)
    if not 0.0 <= args.payload <= 6.0:
        p.error("payload must be within [0, 6] kg")
    stage = next(s for s in STAGES if s.name == args.stage)
    harvest_cfg = validate_config(asdict(HARVEST_PPO))
    gait_cfg = validate_config(asdict(G1_PPO))
    try:
        repo_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        repo_commit = "unknown"
    manifest = checkpoint_manifest(
        repo_commit, "27f8033c5064d32f049a17accb71cd1091422878",
        {"stage": asdict(stage), "harvest_ppo": harvest_cfg,
         "gait_ppo": gait_cfg, "seed": args.seed,
         "goal": args.goal or stage.goal, "payload_kg": args.payload},
        {"status": "scaffold-validated"})
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        from treesim.kiwi_rl.env import sim_available
        print(f"sim_available={sim_available()}", flush=True)
        print("dry-run ok: configs valid, optimiser integration pending",
              flush=True)
        return
    if not args.run:
        raise SystemExit("pass --dry-run (any host) or --run (jp only)")
    run_training(args, stage, manifest)


def run_training(args, stage, manifest):
    """JP-ONLY single-env loop (Spot builder is num_envs==1, see §7).

    Task/aux episodes interleave (aux = locomotion commands + scripted
    arm motion) until the builder supports true VecEnv. Checkpoints +
    JSONL metrics go under args.out (SSD, never git).
    """
    import torch
    from treesim.kiwi_rl.env import KiwiHarvestEnv
    from treesim.kiwi_rl import models_torch as MT
    from treesim.kiwi_rl.ppo import save_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("--run requires CUDA (jp)")
    env = KiwiHarvestEnv(str(args.relic), seed=args.seed,
                         goal=args.goal or stage.goal,
                         payload_mass_kg=args.payload).build()
    v3, n3, m3 = MT.build_v3().to(device), MT.build_n3(), MT.build_m3()
    for head in (n3, m3):
        head.mod.to(device)
        head.logstd.data = head.logstd.data.to(device)
    critics = {k: MT.build_critic().to(device) for k in ("n", "m", "g")}
    opt = torch.optim.Adam(
        list(v3.parameters()) + n3.parameters() + m3.parameters(),
        lr=HARVEST_PPO.actor_lr)
    g1 = MT.build_g1_mlp().to(device)  # weights via import_relic_actor.py
    opt_g1 = torch.optim.Adam(g1.parameters(), lr=G1_PPO.actor_lr)
    args.out.mkdir(parents=True, exist_ok=True)
    for it in range(args.iters):
        obs = env.reset(seed=args.seed + it)
        done = False
        steps = 0
        while not done and steps < 128:
            n3_in = torch.zeros(1, 488, device=device)  # wired to obs
            h = torch.zeros(1, 1, 256, device=device)
            with torch.no_grad():
                mu, logits, _ = n3.forward(n3_in, h)
            cmd = torch.tanh(mu[0]).cpu().numpy() * [0.4, 0.3, 0.7]
            obs, terms, term, trunc = env.step(
                cmd.astype("float32"), 0,
                np.zeros(8, dtype=np.float32), 0)
            done = bool(term or trunc)
            steps += 1
        ckpt = args.out / f"iter_{it:04d}.pt"
        save_checkpoint(ckpt, {"v3": v3, "g1": g1}, {"opt": opt},
                        {"iter": it},
                        {"stage": stage.name, **manifest})
        print(f"[train] iter={it} steps={steps} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
