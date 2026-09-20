# Thekenyos · Kiwi harvesting robotics

![Spot with a chassis-mounted kiwi basket under a procedural pergola](assets/kiwi-pergola.jpg)

<img width="640" height="360" alt="image" src="https://github.com/user-attachments/assets/ad6cda4e-9269-4b4d-af2e-960015df5153" />


A research simulation for a **Spot quadruped with one arm**, harvesting kiwis
under a **1.6 m pergola canopy** and carrying them in a rear basket.

Built on [OrchardBench](https://github.com/humphreymunn/orchardbench),
[Newton](https://github.com/newton-physics/newton), MuJoCo-Warp and native MuJoCo.
The project is simulation-only. Material and damage models are source-backed
but not calibrated against real fruit; see
[the evidence table](docs/kiwi-material-evidence.md) before changing a value.

## Install

Tested on Linux with an NVIDIA RTX 5090. Python 3.12, Newton 1.3.0, Warp 1.14.0
and MuJoCo/MuJoCo-Warp 3.8.1 are pinned. First runs compile CUDA kernels and can
take several minutes. Do not upgrade Warp alone.

```bash
git clone https://github.com/EduardGilM/Thekenyos.git
cd Thekenyos
conda env create -f environment.yml
conda activate kiwi-pergola
python -m unittest discover -s tests -v
```

Native MuJoCo recordings use `MUJOCO_GL=egl` on Linux. `--no-render` avoids a
display dependency.

### External Spot assets

RELIC assets and policy weights are **not redistributed here**. Review the
[RELIC license](https://github.com/rai-opensource/relic) first: its noncommercial
research terms differ from this project's Apache-2.0 code.

```bash
git clone https://github.com/rai-opensource/relic.git ../relic
git -C ../relic checkout 27f8033c5064d32f049a17accb71cd1091422878
```

## Demos and benches

### Spot with a loaded basket

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 \
  --frames 600 --closeup --video output/spot-basket.mp4 \
  --metrics output/spot-basket.json
```

Omit `--payload` to sample 0–6 kg with `--payload-seed`. Add
`--terrain --terrain-kind orchard` for the seeded orchard floor (aisles,
furrows, slope, noise, friction). `--spill-test` applies a roll torque to
provoke spills; each lost fruit is one negative event. The pretrained gait was
not trained on this surface and the arm holds its ready pose.

### Native deformable kiwi bench

```bash
MUJOCO_GL=egl python scripts/kiwi_compression.py --compression .10 \
  --output output/xuxiang-compression --video output/xuxiang-compression.mp4
```

Tetrahedral fruit with 1.57 MPa homogeneous Xuxiang flesh between ideal
parallel pads. Writes `kiwi.xml`, `measurements.csv`, `metrics.json` and an
optional MP4. Elastic recovery does not clear the persistent damage proxy.

### Actual Spot gripper bench

```bash
MUJOCO_GL=egl python scripts/check_spot_gripper.py --relic ../relic \
  --torque .3 --check --output output/spot-gripper --video output/spot-gripper.mp4
```

### Assisted pick and basket deposit

```bash
MUJOCO_GL=egl python scripts/assisted_harvest_cycle.py \
  --relic ../relic --output output/assisted-cycle --video
python -m unittest tests.test_assisted_harvest_cycle -v
```

Scripted, fixed-base native MuJoCo fixture: approach, close, activate a
labelled ideal grip, rotate, pull down until load-triggered detachment, carry,
release over the basket. Success requires the fruit to settle inside the
collision liner. It uses rigid fruit and an artificial grasp, so it validates
workspace and deposit geometry, not grip strength or fruit safety.

### Native stem extraction bench

```bash
MUJOCO_GL=egl python scripts/check_grasp_pull.py --relic ../relic \
  --output output/grasp-pull --torque 3 --grasp-x .175 \
  --target-fsa 60 --pull-after 5.8 --grip-force 15 --roll-speed .4 --video
python -m unittest tests.test_native_stem -v
```

Spot's jaw meshes against an unpinned fruit on a collidable four-segment stalk
(`treesim/native_stem.py`) with an angle-dependent abscission rule. Add
`--rigid --ideal-grip` for the assisted variant. Unassisted contact-only grasps
slip; that is a real result, not a bug to tune away.

## Physics summary

- **Geometry and mass:** sampled Hayward diameters and axial length; mass from
  volume and density, rejected outside the source envelope.
- **Stems:** beam stiffness from `EA/L` and `3EI/L³`, attached at the fruit
  surface, with a Fang2023 mean-force break threshold by fruit–stem angle.
- **Damage:** contact forces feed a Hertz patch estimate of pressure and
  indentation into a persistent score. It is a training proxy, not a bruise
  probability.
- **Contacts:** native MuJoCo contacts. The benches use MPR collision
  (`nativeccd=disable`) as a workaround for intermittent native CCD failures.
- **Orchard floor:** assumed domain-randomization heightfield, not a measured
  survey. Wet-soil friction is an open gap.

Solver contact time constants, jaw gains and the Xuxiang-flesh/Hayward-stem
mix are engineering assumptions.

## Learning pipeline

The current run is a fixed-base privileged PPO teacher trained on a
progressive sequence curriculum, a scripted deposit, a sensor-only student and
an orchard demo that chains them. Earlier reward-graph and CTI runs are archived;
their contract is in [docs/reward-graph.md](docs/reward-graph.md).

### Sequence curriculum teacher

Five ordered objectives: position the kiwi between the jaws, sustain bilateral
loaded contact, detach while grasping, carry above the basket, release and
settle inside it for two seconds. Each first valid checkpoint pays one point;
dense feedback is a live-quality balance capped at 0.25. An objective unlocks
after two consecutive evaluations reach 90% prefix success on 32 varied poses.
A held-out scene measures generalization without controlling promotion.
Jaw–fruit distances come from the MJWarp GJK support maps of the actual
collision meshes (`treesim/kiwi_rl/surface_distance.py`).

```bash
python scripts/export_fixed_base_scene.py --scene /path/to/harvest-near-001 \
  --output /path/to/sequence-near-001
python scripts/export_fixed_base_scene.py --scene /path/to/fast-occlusion-001 \
  --output /path/to/sequence-holdout-001
python scripts/train_sequence_fast.py \
  --scene /path/to/sequence-near-001 --eval-scene /path/to/sequence-holdout-001 \
  --gait-checkpoint /path/to/g1-cpu.pt --output /path/to/teacher-sequence-002 \
  --worlds 1024 --steps 128 --minibatch-worlds 128 --train-seconds 28800
```

Checkpoints are `checkpoint-NNNNNN.pt` and save curriculum, actor, critic,
Adam, counters and RNG. `--resume-from` needs the latest logged checkpoint and,
for online W&B, `--wandb-run-id`. Key metrics: `curriculum/level`,
`evaluation/prefix_success`, `evaluation/completed/{position,grip,extract,carry,deposit}`
and `evaluation/success` (a complete physical harvest).

### Scripted deposit and full cycle

Once the fruit is extracted and held, `treesim/kiwi_rl/scripted_deposit.py`
drives two collision-checked joint-space segments (over the basket, then into
it), opens the jaw, dwells and retracts. The deployable `proprioceptive`
trigger detects a hold from a stalled jaw plus 5 cm of hand travel; `oracle`
uses the simulator and is diagnostic only.

```bash
python scripts/evaluate_full_cycle.py \
  --scene /path/to/sequence-near-001 --checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --gait-checkpoint /path/to/g1-cpu.pt --output /path/to/full-cycle-001 \
  --worlds 64 --seconds 20 --deposit-trigger proprioceptive
```

Reports per-stage completion, physical success and end reasons, and saves one
world's states for the renderer.

### Sensor-only student

`train_sequence_student.py` distills the teacher into a policy that sees only
hand RGB-D (64×48) and R84 proprioception, DAgger-style on top of PPO. It
starts at the extraction objective and stops there, since carry and deposit
are scripted. Jaw torque is capped at 0.6 Nm (about 9 N) because full torque
reaches the 15 N pad damage limit.

```bash
python scripts/train_sequence_student.py \
  --scene /path/to/sequence-near-001 --eval-scene /path/to/sequence-holdout-001 \
  --gait-checkpoint /path/to/g1-cpu.pt --teacher-checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --output /path/to/student-sequence-001 --worlds 512 --steps 128 --train-seconds 28800
```

### Orchard demo and timelapse

Walk to each kiwi with the RELIC gait, harvest with the RL arm policy, detect
the hold proprioceptively, deposit, stow, move on. Every layer is deployable
except the reward oracle, which only reports.

```bash
python scripts/demo_orchard_harvest.py \
  --scene /path/to/orchard-demo-001 --checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --gait-checkpoint /path/to/g1-cpu.pt --output /path/to/demo-001 \
  --freeze-legs-while-harvesting --deposit-trigger proprioceptive
python scripts/render_demo_timelapse.py \
  --scene /path/to/orchard-demo-001 --replay /path/to/demo-001 \
  --output /path/to/demo-001/timelapse.mp4 --seconds 15
```

`--freeze-legs-while-harvesting` matches the fixed-base training stance.
`--r84-template` replays recorded leg and body proprioception while standing.
The renderer replays `states.npz` at 1080p with a chase camera and harvest
counter; no physics is re-run.

### Dashboard and tests

```bash
python3 scripts/training_dashboard.py --run teacher-sequence-002 --cache output/dashboard
python -m unittest tests.test_sequence_curriculum tests.test_scripted_deposit \
  tests.test_fast_task tests.test_fast_ppo tests.test_training_dashboard -v
```

The dashboard serves `http://127.0.0.1:8765`, polls the training host over
SSH and renders the best evaluated checkpoint with the arm-camera inset.

<img width="1512" height="778" alt="image" src="https://github.com/user-attachments/assets/ff4a0d22-99b3-4c01-8c4e-064fd7c4783d" />

## Layout

| Path | Purpose |
|---|---|
| `treesim/pergola.py` | Procedural structure and fruit placement |
| `treesim/orchard_terrain.py` | Seeded kiwi orchard floor |
| `treesim/kiwi_material.py` | Source-backed constants, geometry sampler, damage helper |
| `treesim/harvest_task.py` | Task contract and privileged oracle ([docs](docs/harvest-task.md)) |
| `treesim/kiwi.py` | Stem dynamics and GPU contact damage |
| `treesim/native_stem.py` | Collidable stalk for native benches |
| `treesim/basket.py` | Mounted basket, loose payload, spill events |
| `treesim/spot.py` | URDF import, observation mapping, policy and PD control |
| `treesim/kiwi_rl/` | Fast GPU runtime, curriculum, rewards, scripted deposit, students |
| `scripts/` | Benches, trainers, evaluators and renderers listed above |
| `docs/` | Evidence table, task definition, reward contracts, RL blueprint |

See [AGENTS.md](AGENTS.md) for implementation and handoff rules.

## Provenance and license

Derivative of Humphrey Munn's OrchardBench at upstream commit
`6313313db8b1a7d23fb2cc3afd67cac46f29399a`. The original apple workflows remain
available; see the [upstream documentation](docs/orchardbench-upstream.md) and
[original physics notes](PHYSICS.md). Code is [Apache-2.0](LICENSE). RELIC
assets have separate terms. Do not add external model weights, robot meshes or
research PDFs without checking their licenses.
