# Local Windows training results (RTX 4070)

Running log of the local training programme on the Windows workstation.
**Read this at the start of every session before touching training.** It
records what was tried, what worked, what did not, and why — so the same
dead ends are not repeated. Append; do not rewrite history.

Branch: `devin/windows-overnight-training`. Generated artifacts live in the
gitignored `output/` tree on the Windows machine (paths below are relative
to the repo root); only code and this document are committed.

## 0. Setup facts (verified)

- Two conda envs, both isolated, nothing system-wide:
  - `kiwi-pergola` — legacy pinned stack (mujoco-warp 3.8.1). Tests + scene export only.
  - `kiwi-flex-gpu` — mujoco/mujoco-warp **3.13.0**, warp 1.15.0, torch 2.10.0+cu128.
    **All training and evaluation runs here.** FastRuntime needs `put_model(batch_sizes=)`
    which 3.8.1 lacks. Installed from a Windows-filtered copy of
    `.devin/training/requirements-gpu.lock` (`output/requirements-gpu-win.lock`; Linux-only
    nvidia-cufile/nccl/triton removed, colorama added).
- RELIC clone at `../relic`, pinned `27f8033c`. ONNX parity gate passed.
- Bundled artifacts are jp-machine-specific (absolute `/mnt/ssd` paths, missing `.json`
  sidecars, stale `model_sha256` after the bundling commit relativized paths). Locally
  repaired, hash-re-attested copies:
  - `output/local-scenes/sequence-near-001` (fixed-base training scene, hash `8215ae8c…`)
  - `output/local-scenes/sequence-holdout-123` (generated, seed 123, hash `5cfd2662…`)
  - `output/local-scenes/demo-multi-003{,-dense}` (free-base, 6 fruits — needed by CTI)
  - `output/gait/g1-cpu.pt` + sidecar — bundled gait weights verified **bit-identical** to a
    fresh import from pinned RELIC (the strict 1e-5 import gate fails by 1.8e-5 on this
    platform — fp noise, not a weight problem).
  - `output/checkpoints/init-seq-local.pt` — ckpt-800 weights re-attested under the local
    scene hash (every transfer path hard-checks `model_sha256`).
  - `output/checkpoints/init-noseq-demo-multi-003.pt` — same weights with the 16
    sequence-history input columns sliced out of `encoder.0.weight` (for non-sequence arms).
- VRAM: FastRuntime physics costs +0.35 / 0.71 / 1.12 GiB at 256 / 512 / 1024 worlds.
  1024 worlds (jp's config) is comfortable on 12 GB.
- Throughput: ~8.9k transitions/s at 1024 worlds × 128 steps (sequence profile).
- Windows gotchas fixed in code: `ppo.py` directory fsync (POSIX-only `os.open(dir)`);
  `time.monotonic()` keeps running through PC suspend, so a suspend eats training budget.
- `evaluate_full_cycle.py` only builds the **sequence** (132-dim) actor; non-sequence
  checkpoints from `train_harvest_fast.py` fail with "architecture mismatch". Use their
  in-training evaluations instead.

## 1. Baseline (bundled `teacher-sequence-015/checkpoint-000800.pt`)

Full-cycle, proprioceptive deposit trigger, 20 s episodes:

| Scene | worlds | complete harvest | funnel pos/grip/extract/carry |
|---|---|---|---|
| sequence-near-001 (train) | 64 | **34.4%** | .77 / .77 / .66 / .34 |
| sequence-holdout-123 | 64 | 35.9% | .67 / .67 / .52 / .36 |
| sequence-holdout-123 | **256** (seed 7) | **28.9%** | .60 / .60 / .51 / .29 |

Bottleneck: extraction → carry handoff (`invalid_extract` + `unheld` ≈ 47% of failures).
It generalizes (train ≈ held-out). **This is the bar; improvements must beat it on held-out.**

## 2. Overnight comparison, 2026-10-02/03 (6 arms, ~10 h, 0 numerical failures)

| Arm | Script / profile | Init | Scene | Result |
|---|---|---|---|---|
| seq-fresh-001 | `train_sequence_fast` | random | fixed-base | grip **98%/95%** (train/held-out), promoted to level 2 @ upd 208; extract untrained (11–20%); 0 complete. Cleanest curve, no overfit, ran out of time. |
| seq-warm-001 | `train_sequence_fast` | ckpt-800 | fixed-base | 40.6% train / **14.1% held-out**. Over-specialized (45/64 `invalid_extract` on held-out). **Worse than baseline.** |
| milestone-001 | `train_harvest_fast --curriculum` | ckpt-800 sliced | free-base | 0 grasps; diverged (closest 11 cm → 59 cm). |
| cti-001 | `train_harvest_fast --cti` (selective v5) | ckpt-800 sliced | free-base | 0 grasps; 7.4 cm closest. CTI selected 3 roots, learned 0 transitions. |
| graph-001 | `train_harvest_fast --reward-graph` | ckpt-800 sliced | free-base | 0 grasps; 6.2 cm; graph score oscillated .04–1.08; 46–62% physical failures near fruit. |
| graph-cti-001 | `--reward-graph --cti` (BranchCTI v8) | ckpt-800 sliced | free-base | 0 grasps; 1.7 cm; 2.3M branch transitions, **0 targets selected**. |

### What did not work, and why (do not repeat without changing the cause)

1. **Warm start with curriculum reset to level 1** (seq-warm-001). `--initialize-from`
   resets `Curriculum` to level 1. A level-3 policy spent 2 h being graded on position-hold,
   never passed the 90%×2 promotion gate (val peaked .84), and traded held-out
   generalization for training-scene fit. Fix: `--start-level 3` (added).
2. **Sliced-encoder warm start for non-sequence arms.** Dropping the 16 history inputs
   makes the policy effectively random (in-run baseline eval: 0 grasps, 100% stall). All four
   `train_harvest_fast` arms started from this and none bootstrapped a grasp in 1.5–2 h.
   Confounds that must be separated before concluding anything about the *algorithms*:
   free-base 6-fruit scene vs fixed-base single fruit; `jaw_cap_Nm=1.0` in that script vs
   0.6 in the sequence profile (graph-001's physical failures near the fruit look like
   over-squeezing).
3. **CTI on a policy that never succeeds.** Both CTI variants ran correctly (replay accepted,
   15+ rounds) but had nothing to branch toward. CTI is a repair tool for a policy that
   *sometimes* succeeds — point it at one that already grasps, or don't run it.
4. **Judging by training-scene success alone.** seq-warm-001 looked like +6 pts; held-out
   showed −22. Always evaluate held-out, and at ≥256 worlds for decisions (64-world rates
   have ±6–8 pt noise).

## 3. Continuation 1 — `seq-l3-001` (ckpt-800 weights, `--start-level 3`, lr 1e-4, 4 h*)

*PC suspend cost ~70 min of real training.* 729 updates; acceptance gate chose ckpt-507.

| Checkpoint | train 64w | held-out 64w | **held-out 256w** | extract→carry (256w) | unheld |
|---|---|---|---|---|---|
| baseline 800 | 34.4% | 35.9% | 28.9% | .51 → .29 | 38 |
| **l3-001 ckpt-507** | **54.7%** | 37.5% | **47.7%** | .62 → .48 | **19** |
| l3-001 ckpt-729 (final) | — | — | 27.0% | .48 → .27 | 30 |

**First genuine improvement: +19 pts held-out (n=256), gain exactly at the identified
bottleneck (unheld failures halved), generalization preserved.** Training drifted after
~update 507 at lr 1e-4 — the final checkpoint is back at baseline. Trust the acceptance
gate's `best_checkpoint`, not the last one. Remaining dominant failure: `invalid_extract`
(96/256 = 38% on held-out).

Best artifact so far: `output/runs/seq-l3-001/checkpoint-000507.pt`.

## 4. Continuation 2 — `seq-l3-002` (from ckpt-507, level 3, **lr 3e-5**, 4 h) — RUNNING

Hypothesis: lower LR consolidates instead of drifting. Decision metric: best checkpoint on
held-out at 256 worlds vs 47.7%. Results to be appended below.

## 5. Open questions / next candidates (in priority order)

1. If l3-002 plateaus: attack `invalid_extract` directly — inspect what "invalid" means in
   `sequence_curriculum` (pull direction? stem force? jaw load?) and whether the held-out
   fruit pose (different anchor) needs wider `--fruit-reach-fraction` / `--fruit-sector-deg`
   randomization during training.
2. Continue `seq-fresh-001` from ckpt-256 at `--start-level 2` — healthiest learning curve,
   independent of jp's weights.
3. Re-test CTI **only** from a checkpoint that already grasps (e.g. l3-001 ckpt-507) and
   only after making it work on the fixed-base scene or a 1-fruit free-base scene.
4. Student (RGB-D) distillation from the best teacher — the deployable artifact; untested.
5. Multi-seed confirmation of any headline number before claiming it in README.

## 6. How to run things

```
# activate
C:\Users\Usuario\anaconda3\envs\kiwi-flex-gpu\python.exe

# launchers (sequential arms + auto full-cycle eval)
output\overnight\run_overnight.bat      # 6-arm comparison
output\overnight\run_continue.bat       # seq-l3-001
output\overnight\run_continue2.bat      # seq-l3-002
output\overnight\evaluate_runs.py       # full-cycle eval of each run's best ckpt, both scenes

# decision-grade held-out eval (256 worlds)
python scripts/evaluate_full_cycle.py --scene output/local-scenes/sequence-holdout-123 \
  --checkpoint <ckpt> --gait-checkpoint output/gait/g1-cpu.pt --output output/evals/<name> \
  --worlds 256 --seconds 20 --deposit-trigger proprioceptive --seed 7

# graceful stop of a running arm
echo. > output\runs\<arm>\pause-request.json
```

Per-run: `training.jsonl` (one JSON/update; eval rows contain `curriculum/validation_success`,
`evaluation/prefix_success`), `report.json` (`best_checkpoint`), `checkpoint-NNNNNN.pt(.json)`.
Videos: `output/videos/` (see §7).

## 7. Videos

See `output/videos/README.md` (generated with `scripts/render_fast_rollout.py --replay`).

### Independent cross-check of §3 (added after video recording)
10 seeds × 8 worlds on held-out (seeds 1–10, different from the 256-world seed 7):
baseline 23/80 = **28.7%**, l3-507 29/80 = **36.2%**. Same direction, smaller gap.
Honest claim: **ckpt-507 beats baseline on held-out by +7 to +19 pts** depending on
sample; baseline ≈ 29% is stable across both samples. Multi-seed 256-world runs are
needed before quoting a single number.

Videos: `output/videos/{base800,l3-507}-{success,failure}/rollout.mp4`
(see `output/videos/README.md`). Visual observations: l3-507's success is fast
(9.5 s vs 20 s); its failure mode on seed 1 was never reaching position — i.e.
variance across fruit poses, not grasp mechanics.

## 8. Full orchard demo (walk + harvest + deposit, 6 kiwis, free-base `demo-multi-003`)

`scripts/demo_orchard_harvest.py` + `render_demo_timelapse.py` (MUJOCO_GL=glfw).

| Policy | Kiwis in basket | sim time | video |
|---|---|---|---|
| jp's recorded run (ckpt-800, Sep 20) | 3 / 6 | 199 s | checkpoints/demo/demo-roll-800-report.json |
| baseline ckpt-800, local | 2 / 6 | 178 s | `output/demo-base800-video/timelapse.mp4` |
| **seq-l3-001 ckpt-507** | **4 / 6** | 266 s | `output/demo-l3-507-video/timelapse.mp4` |

Single episode each — illustrative, not a statistic. ckpt-507 cleared fruits 0, 5, 2, 3;
failed 4 (never engaged, closest 0.99 m — stand geometry) and 1 (positioned+gripped,
then physical failure on 3 attempts). Baseline failed 4 of 6 with mostly physical
failures at the fruit.

### What did not work: demo default `--fruit-damping 0.05` blows up the solver
On the kiwi-flex-gpu stack, the demo's default fruit joint damping (0.05) produces
solver-iteration overflow → NONFINITE within ~90 control steps of the first harvest
attempt, with absurd loads (stem 128 N, hand 1945 N) at zero contact. Checkpoint-,
scene- (bundled and freshly exported), rolling-friction- and leg-freeze-independent.
**`--fruit-damping 0` fixes it completely** (training also used 0). Always pass
`--fruit-damping 0` on this machine. Not root-caused inside MuJoCo-Warp 3.13.0 —
jp's run used the same flag default and did not blow up, so it may be a 3.13 vs jp
stack difference; recorded as an open stack question, not a scene bug.

Also: every mesh `.obj` hash differs from jp's manifest purely because of CRLF
checkout; content is identical after LF normalization (verified on body.obj).
