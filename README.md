# Thekenyos · Kiwi harvesting robotics

![Spot with a chassis-mounted kiwi basket under a procedural pergola](assets/kiwi-pergola.jpg)

A research simulation for a **Spot quadruped with one arm**, harvesting kiwis
under a **1.6 m pergola canopy** and carrying them in a rear basket.

Built on [OrchardBench](https://github.com/humphreymunn/orchardbench),
[Newton](https://github.com/newton-physics/newton), MuJoCo-Warp and native MuJoCo.
The first milestone is one robot. All fruit starts harvestable. Multi-robot
coordination, maturity perception and automatic unloading come later.

## What works, and what does not

| Component | Current implementation |
|---|---|
| Pergola | Seeded configurable commercial plantation, 4.5–5 m structural grid, continuous rows, compliant canes and hanging fruit |
| Orchard floor | Optional seeded heightfield: grassed pasillos, bare surcos, slope, noise and friction; posts and Spot sit on the sampled surface |
| Spot | External RELIC robot assets and pretrained ONNX gait; scripted velocity route |
| Basket | Rear chassis-mounted yellow panels, vents, black frame, handles and mounting feet; open-top collision liner |
| Basket payload | Separate free, collidable fruit; 0–6 kg; gravity, rotation, packing and spills |
| Canopy fruit | Coupled Hayward size/mass/density envelope; free ellipsoids with stem-site spring forces |
| Stems | Axial/bending stiffness derived from measured stalk dimensions; irreversible load-triggered detachment |
| Damage | Persistent contact/strain **proxy**, with negative increments available as reward terms |
| Deformable fruit | Separate native MuJoCo tetrahedral compression/release bench with Xuxiang flesh stiffness |
| Task evaluator | Outcome-based single-fruit oracle with optional guidance; [task definition](docs/harvest-task.md) |
| RL training | **Not implemented here yet.** Walking uses an existing policy; penalties do not retrain it |

**The GPU orchard fruit is still rigid collision geometry.** The native flex
bench deforms, but is not yet integrated into the GPU orchard or Spot's jaws.
Neither model is a validated predictor of bruising. Layered skin/core,
viscoelastic/plastic constitutive laws, calibrated wet friction and
calibrated angle/torque-dependent abscission remain open work. Research ranges are not
interchangeable across cultivars and test conditions.

## Install

The tested GPU platform is Linux with an NVIDIA RTX 5090. Python 3.12,
Newton 1.3.0, Warp 1.14.0 and MuJoCo/MuJoCo-Warp 3.8.1 are pinned. First runs
compile CUDA kernels and can take several minutes. The full GPU workflow is
not verified on macOS.

```bash
git clone https://github.com/EduardGilM/Thekenyos.git
cd Thekenyos
conda env create -f environment.yml
conda activate kiwi-pergola
python -m unittest discover -s tests -v
```

`ffmpeg` is included for videos. Use a working OpenGL display for Newton's GL
viewer. Native MuJoCo recordings can use `MUJOCO_GL=egl` on Linux. `--no-render`
on the Spot script avoids a display dependency. Do not upgrade Warp alone:
the earlier Warp 1.17 combination failed on this workstation.

### External Spot assets

RELIC assets and policy weights are **not redistributed in this repository**.
Review its [license](https://github.com/rai-opensource/relic) before use: its
noncommercial research terms differ from this project's Apache-2.0 code.

```bash
git clone https://github.com/rai-opensource/relic.git ../relic
git -C ../relic checkout 27f8033c5064d32f049a17accb71cd1091422878
```

The adapter loads `source/relic/relic/assets/spot/spot_with_arm.urdf`,
`constants.py` and `pretrained/policy.onnx`. It does not require Isaac Sim.

## Run the demos

### Procedural pergola

```bash
python scripts/grow_tree.py --preset pergola --foliage --seed 42 \
  --collisions --substeps 40 --viewer gl
```

Plant the same bay on the orchard floor:

```bash
python scripts/grow_tree.py --preset pergola --foliage --terrain --seed 42 \
  --collisions --substeps 40 --viewer gl
```

The scene is built programmatically in Python; it is not a hand-authored pergola
XML. The default pergola is 40 posts along 45 rows at 5 m centres, about
4.3 hectares. Use `--pergola-rows`, `--pergola-columns`, and
`--pergola-spacing` (4.5–5.0 m) to scale the field; render-only foliage is
enabled by default for this preset, while `--foliage-density 0` disables it.
Use `--fruit-count` to cap the independent kiwi bodies (the default is 600 for
the plantation). Change geometry in `treesim/pergola.py`. Add `--terrain` to
plant the grid on a kiwi orchard floor (grassed aisles, bare planting strips,
sampled slope and noise). The native compression bench writes its own generated
MuJoCo XML to its output directory. A software-rendered flyover of the
plantation (no GPU, no Spot gait) is:

```bash
python scripts/record_orchard_mujoco.py --seed 42 --video output/orchard-mujoco.mp4
```

### Spot with a loaded basket

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 \
  --frames 600 --closeup --video output/spot-basket.mp4 \
  --metrics output/spot-basket.json
```

Omit `--payload` to sample 0–6 kg using `--payload-seed`. Add `--terrain` to walk
the same scripted oval on the orchard floor; sampled slope, rut, noise and
friction are written into the metrics JSON. The pretrained gait was not trained
on this surface. The arm holds its ready pose; random arm poses and payload-aware
locomotion retraining are not complete. For a numerical run, replace the video
options with `--no-render`.

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 --terrain \
  --frames 600 --video output/spot-orchard.mp4 --metrics output/spot-orchard.json
```

`--spill-test` applies a deliberate 400 N·m roll torque (`--spill-torque` overrides it) from 2.0 to 2.4 seconds. This is
an adverse-motion test, not a learned behaviour. Each lost fruit receives one
negative spill event. Contact damage is recorded separately and persists after
release. Normal runs stop if Spot falls; spill tests permit that outcome.

### Native deformable kiwi bench

```bash
MUJOCO_GL=egl python scripts/kiwi_compression.py --compression .10 \
  --output output/xuxiang-compression --video output/xuxiang-compression.mp4

MUJOCO_GL=egl python scripts/kiwi_compression.py --compression .03 \
  --output output/gentle-compression --check-timestep
```

Outputs include `kiwi.xml`, `measurements.csv`, `metrics.json` and optional MP4.
The default is **1.57 MPa homogeneous Xuxiang flesh**, not the earlier 30 kPa
soft toy. The default timestep is 10 microseconds. Compare timestep-refined
forces before changing it. The pads are ideal parallel surfaces, not Spot's
actual gripper. Zero gravity isolates compression; geometry recovery does not
clear the persistent damage proxy.

### Actual Spot gripper bench

```bash
MUJOCO_GL=egl python scripts/check_spot_gripper.py --relic ../relic \
  --torque .3 --check --output output/spot-gripper \
  --video output/spot-gripper.mp4
MUJOCO_GL=egl python scripts/check_spot_gripper.py --relic ../relic \
  --torque .3 --timestep .000005 --check --output output/spot-gripper-halfstep
```

This separate native MuJoCo bench uses the external Spot wrist/finger meshes,
URDF hinge limits and finger inertia. It holds the wrist sideways in a test
fixture. One **free, detached** kiwi is initially weight-compensated, squeezed
with a torque-limited jaw, loaded by gravity at 1.5 s, then released at 2.8 s.
There is no fruit weld, grasp assistance, arm trajectory or stem in this test.
The floor catches the released fruit; this deliberate drop is not a harvesting
success. Videos are scripted and shown at 2x slow motion.

`--rigid --timestep .0005` isolates collision geometry. The default fruit is a
native tetrahedral elastic body with the same homogeneous Xuxiang flesh proxy
as the compression bench. `--torque 1` is a stronger-grip comparison. The motor
torque limit is in **N·m**, distinct from measured per-jaw contact load in **N**;
neither is a calibrated safe grasp setting. The assumed friction coefficient
is 0.44; actual Spot pad friction needs measurement.

Outputs include the generated scene, sub-sampled force/position measurements
and metrics. Forces and numerical warnings are checked each physics step;
shape compression removes rigid rotation and is sampled every millisecond.
The 9–12 mm tetrahedral grid is coarse relative to the teeth. Mesh refinement
and pad calibration are still required; the rigid and flex surfaces also differ
in discretization. The strain damage proxy cannot predict local tooth injury
or delayed bruising.
Grip damage is reported separately from the later floor impact. `--check`
uses a 20 mm displacement tolerance during 1.7–2.7 s, no early ground contact,
bilateral contact during the run and contact-free release
to the floor. Test results apply to this one initial pose and fruit geometry.

## Physics and evidence

Read [the evidence table](docs/kiwi-material-evidence.md) before changing a
material value. It lists primary sources, units, cultivar, loading protocol,
known source inconsistencies and missing measurements.

- **Geometry:** sampled equatorial diameters and measured mean axial length;
  mass derives from volume and density and is rejected outside the source
  envelope. The joint distribution is an engineering assumption.
- **Basket fruit:** fixed packing geometry and equal masses conserve requested
  payload exactly. Sub-fruit payloads are synthetic load tests. Contacts with
  the actual basket liner and other fruit remain uncalibrated.
- **Stems:** beam stiffness derives from `EA/L` and `3EI/L³`. The attachment is
  at the fruit surface, so forces act with a moment arm and react on the cane.
  An implicit spring approximation supports small timesteps. A Fang2023 Hayward
  mean-force proxy sets the tensile break threshold by fruit–stem angle. Some
  points are approximate figure readings; interpolation and fracture dynamics
  remain uncalibrated. It does not prescribe a robot picking trajectory.
- **GPU damage:** native MuJoCo contact forces feed a Hertz equivalent-patch
  estimate of pressure and indentation. A persistent score uses a 5% strain
  reference and 0.26 MPa flesh stress reference, with an **assumed** one-second
  accumulation scale. It is a diagnostic and training proxy, not a bruise
  probability. Multiple contact patches and cross-cultivar transfer limit it.
- **Native flex:** actual elastic deformation and compression/release forces;
  homogeneous tissue, numerical damping, strain-based damage proxy. No physical
  peel rupture, plastic set or fitted relaxation law is claimed.
- **Contacts:** the kiwi scene uses native MuJoCo contacts. The earlier Newton
  collision path allowed a fruit to escape a stationary basket. Planar Spot
  lower-leg collision hulls receive 1 mm thickness for native compatibility;
  the original body inertia is retained.
- **Orchard floor:** `--terrain` on a pergola scene samples an assumed domain-
  randomization heightfield (2 m vine rows, grassed pasillos, ~0.64 m bare
  surcos, slope ±4°, noise 0–4 cm, ruts 0–8 cm deep and 20–60 cm wide,
  friction 0.6–1.3). It is an assumed compact layout, not a measured
  orchard-floor survey.
  Apple `--terrain` remains the older value-noise field. Wet soil friction is
  still an explicit calibration gap.

## Validate a change

```bash
python -m unittest discover -s tests -v
python scripts/check_loose_fruit.py
python scripts/check_loose_fruit.py --timestep .0005
python scripts/check_kiwi_physics.py
python scripts/check_detachment_angles.py
python scripts/check_detachment_angles.py --timestep .0005
python scripts/check_harvest_observer.py --relic ../relic
python scripts/walk_spot.py --relic ../relic --basket --payload 6 \
  --frames 300 --no-render --metrics output/walk-check.json
```

The stationary-basket check requires retention and actual settling motion.
The kiwi check requires rest attachment, detachment under a pull, free fall,
ground contact and measured contact load. The native bench fails on numerical
warnings, inverted sampled tetrahedra or poor release/recovery. Tests and videos
are complementary: inspect contact penetration, mounting, packing and spill
trajectories in recordings as well as checking metrics.

### Latest validation on the RTX 5090

Validated on 18 September 2026; these are checks of the implementation, not
real-fruit calibration or evidence that a harvesting policy has been trained.

| Check | Result |
|---|---|
| Unit/integration suite | 15 tests passed in the Linux environment |
| Stationary basket, 60 fruit | Zero spills over 5 s at both 1 ms and 0.5 ms physics steps |
| Loaded walking | 12 s, 2.83 m travelled, 7.6° maximum tilt, zero spills, zero canopy detachments |
| Deliberate 400 N·m roll disturbance | 36 fruit spilled; total spill penalty −36 |
| Angle fixture, 60° / 120° / 180° | 5.99 / 21.39 / 36.59 N at 1 ms; under 0.07 N change at 0.5 ms |
| Spot observer | Read-only measurement; actual ground contact and forced-drop failure detected |
| Stem/drop regression | Attached at rest; detached under a 50 N pull (updated angle model); landed without tunnelling |
| Original apple / pergola smoke runs | 60 / 120 frames completed |
| Native 3% timestep refinement | 10 µs vs 5 µs: force difference 0.069%; strain difference 0.00046 percentage points |
| Native 10% commanded squeeze | 8.72% measured compression, 71.38 N per pad, no solver warnings; damage proxy saturated |

The 10% command differs from tissue strain because the pads also have compliant
contact. The native 3% case records zero strain-based damage proxy; that is not
a guarantee of unbruised real fruit. Outputs are generated under `output/`.

### Spot jaw benchmark, 19 September 2026

| Model / torque | Result in this fixture |
|---|---|
| Rigid / 0.3 and 0.6 N·m | Slipped out before commanded release |
| Rigid / 1.0 N·m | Passed; maximum hold displacement 7.29 mm |
| Deformable / 0.3 N·m | Passed at 10 and 5 µs; hold displacement 15.94 / 15.95 mm; peak jaw loads 5.58 / 7.44 N |
| Deformable / 1.0 N·m | Failed the 20 mm stability tolerance at both 10 and 5 µs; displacement 23.98 mm; no early drop |

The gentler deformable run reached 0.457% rotation-corrected whole-fruit
compression. Halving the timestep changed peak jaw loads by less than 0.06%
and hold displacement by 0.013 mm. The final floor pose differs, so this is
not evidence that post-release rolling trajectories have converged. All runs completed without MuJoCo numerical warnings. These are
one-pose bench results, not an optimal grip or evidence of bruise-free fruit.
The differing rigid/flex outcomes mean the rigid model cannot yet stand in for
the flex benchmark without further contact and mesh-resolution checks.

## Continue the project

1. Fit compression/hold/release and impact tests to one cultivar and harvest
   condition. Add layered, viscoelastic/plastic response without mixing datasets.
2. Integrate the [task contract](docs/harvest-task.md) with arm actuation and
   substep event capture; validate actual Spot jaw contact and torque failure.
3. Train locomotion over 0–6 kg payload and arm configurations; compare against
   the existing policy on matched seeds, spills, tracking and falls.
4. Train reach, grip, detach and deposit, then integrate a full harvesting task.
5. Add station unloading and, later, multi-robot coordination.

PufferLib is not installed or integrated. Select a GPU training implementation
only after the native material and contact benchmarks agree with measurements.

| Path | Purpose |
|---|---|
| `treesim/pergola.py` | Procedural structure and fruit placement |
| `treesim/orchard_terrain.py` | Seeded kiwi orchard floor (aisles, furrows, slope, noise) |
| `treesim/kiwi_material.py` | Source-backed constants, geometry sampler, damage helper |
| `treesim/harvest_task.py` | Task contract, privileged oracle and simulator measurement bridge |
| `treesim/kiwi.py` | Stem dynamics and GPU contact damage diagnostics |
| `treesim/basket.py` | Mounted basket, loose payload, spill events |
| `treesim/spot.py` | URDF import, observation mapping, policy and PD control |
| `treesim/sim.py` | Solver and stepping integration |
| `scripts/walk_spot.py` | Loaded walking and spill recordings |
| `scripts/record_orchard_mujoco.py` | Native MuJoCo orbit of the orchard heightfield |
| `scripts/kiwi_compression.py` | Native MuJoCo material bench |
| `tests/` | Fast geometry, mass, independence and event checks |
| `docs/kiwi-material-evidence.md` | Research sources and calibration gaps |

See [AGENTS.md](AGENTS.md) for implementation and handoff rules.

## Provenance and license

This is a derivative of Humphrey Munn's OrchardBench, based on upstream commit
`6313313db8b1a7d23fb2cc3afd67cac46f29399a`. The original apple workflows remain
available; see the [upstream documentation](docs/orchardbench-upstream.md) and
[original physics notes](PHYSICS.md). Preserve upstream attribution and the
[Apache-2.0 license](LICENSE). RELIC assets have separate terms. Do not copy
external model weights, robot meshes or research PDFs into this repository
without checking their licenses.
