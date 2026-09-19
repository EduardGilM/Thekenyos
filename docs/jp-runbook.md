# jp runbook: kiwi-harvest RL (TK-RL-003 v3.0)

Training host: SSH alias `jp` (i9-13900K, RTX 5090 32 GB, 64 GB RAM).
Source lives at `/mnt/ssd/experiments/kiwi-pergola/pergola/`; select it with
`ORCHARDBENCH_SOURCE`. Keep caches, runs and videos on the SSD; the root
disk is space constrained. Check GPU/SSD/processes before a big job and
preserve other workloads. First CUDA compile is slow: inspect the log
before restarting.

RELIC stays an external pinned checkout (`assets/relic/` at
`27f8033c5064d32f049a17accb71cd1091422878`); its weights are never
vendored into this repo (noncommercial research terms).

## 0. One-time setup

```bash
conda activate kiwi-pergola
pip install -r requirements-train.txt   # torch>=2.7 cu128 + torchvision
python scripts/check_train_stack.py --relic assets/relic
```

## 1. Gates (must pass before training)

```bash
# RELIC ONNX release gate (contract, finiteness, determinism)
python scripts/check_relic_parity.py --relic assets/relic --samples 10000 \
  --manifest /mnt/ssd/experiments/kiwi-pergola/logs/parity.json
# G1 torch import (TorchScript -> g1_init.pt + cross-check vs manifest)
python scripts/import_relic_actor.py --relic assets/relic \
  --manifest /mnt/ssd/experiments/kiwi-pergola/logs/parity.json \
  --out /mnt/ssd/experiments/kiwi-pergola/logs/g1_init.pt
# Physics regressions (unchanged)
python -m unittest discover -s tests -v
python scripts/check_loose_fruit.py
python scripts/check_kiwi_physics.py
```

## 2. Torch shape tests + dry run

```bash
python -m unittest tests.test_kiwi_rl_torch -v
python scripts/train_kiwi.py --relic assets/relic --dry-run
```

## 3. Curriculum (stages 01→06, seeds 11/22/33)

```bash
python scripts/train_kiwi.py --relic assets/relic --run \
  --stage deposit_pixels --seed 11 \
  --out /mnt/ssd/experiments/kiwi-pergola/logs/s01-s11
```

Promote per §8 gates (two consecutive passing evals, 200 episodes/skill,
500 for gait, Wilson 95%). Evaluate with `scripts/evaluate_kiwi.py`.

## 4. Known pending work

- True 16+4 VecEnv needs Spot `num_envs>1` builder support (currently
  single-env with interleaved aux episodes).
- Angle-conditioned stem break law (Fang curve) before detach stages.
- Body-camera extrinsic calibration (provisional mounts in sensors.py).
