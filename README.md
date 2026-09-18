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
| Pergola | Seeded 3 × 4 m bay, posts, wires, compliant canes and hanging fruit |
| Spot | External RELIC robot assets and pretrained ONNX gait; scripted velocity route |
| Basket | Rear chassis-mounted yellow panels, vents, black frame, handles and mounting feet; open-top collision liner |
| Basket payload | Separate free, collidable fruit; 0–6 kg; gravity, rotation, packing and spills |
| Canopy fruit | Coupled Hayward size/mass/density envelope; free ellipsoids with stem-site spring forces |
| Stems | Axial/bending stiffness derived from measured stalk dimensions; irreversible load-triggered detachment |
| Damage | Persistent contact/strain **proxy**, with negative increments available as reward terms |
| Deformable fruit | Separate native MuJoCo tetrahedral compression/release bench with Xuxiang flesh stiffness |
| RL training | **Not implemented here yet.** Walking uses an existing policy; penalties do not retrain it |

**The GPU orchard fruit is still rigid collision geometry.** The native flex
bench deforms, but is not yet integrated into the GPU orchard or Spot's jaws.
Neither model is a validated predictor of bruising. Layered skin/core,
viscoelastic/plastic constitutive laws, calibrated wet friction and
angle/torque-dependent abscission remain open work. Research ranges are not
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

The scene is built programmatically in Python; it is not a hand-authored pergola
XML. Change geometry in `treesim/pergola.py`. The native compression bench
writes its own generated MuJoCo XML to its output directory.

### Spot with a loaded basket

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 \
  --frames 600 --closeup --video output/spot-basket.mp4 \
  --metrics output/spot-basket.json
```

Omit `--payload` to sample 0–6 kg using `--payload-seed`. The arm holds its ready
pose; random arm poses and payload-aware locomotion retraining are not complete.
For a numerical run, replace the video options with `--no-render`.

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

### Vineyard (table grapes)

An overhead grape arbor: two support rows around a clear 2.4 m aisle, a
12 m long canopy at 2.3 m, dense lobed leaves and hanging dark grape bunches.
The default has 80 tapered clusters, each rendered as individual berries;
`--no-foliage` exposes the support structure. `--rows` requires at least two
support rows; `--canopy-height` (alias `--cordon-height`) accepts 1.8–3 m.
Ground cover is visual only; the collision ground remains flat.
Each cluster is one rigid free body — a **hidden ellipsoid collision proxy**
plus massless visual rachis and berries — tethered to its shoot by a peduncle
beam (`treesim/grape.py`, stiffness from the assumed modulus in
`treesim/grape_material.py`). A load past the sampled detach strength breaks
the peduncle; `GrapeField.cut(i)` severs it directly for scripted harvesting.
`ClusterContact` reports per-cluster peak contact force with a single-berry
rupture threshold applied to whole-cluster force — a proxy, not a measured
cluster limit. Source and assumption notes: `docs/grape-material-evidence.md`.

```bash
python scripts/grow_vineyard.py --viewer null --frames 60 --device cpu
python scripts/grow_vineyard.py --viewer gl --device cpu --robot \
  --row-length 6 --clusters 16 --robot-x -2.5
python scripts/grow_vineyard.py --viewer null --frames 120 --cut-demo 40 \
  --metrics output/vineyard.json --device cpu
python -m unittest tests.test_vineyard -v
```

#### Using the vineyard with a robot

Run these commands **from the repository root**, after installing and activating
`kiwi-pergola` as described in **Install** above. If using an existing local
virtual environment instead, replace `python` with `.venv/bin/python`.
**The vineyard workflow below is intended for CPU use; no NVIDIA GPU or CUDA
driver is required.** Always pass `--device cpu`: the launcher otherwise defaults
to CUDA. Start with the smaller 6 m vineyard and 16 clusters used below rather
than the full 12 m / 80-cluster scene. CPU execution may be slower than real time;
the simulation's 60 Hz clock is not a promise of 60 wall-clock frames per second.
The first RidgebackFranka launch may download the Franka assets and compile CPU
kernels, so it needs network access if those assets are not cached.

**Interactive driving (Ridgeback base + Franka arm):**

```bash
python scripts/grow_vineyard.py --robot --robot-kind ridgeback \
  --device cpu --viewer gl --seed 42 --row-length 6 --clusters 16 --robot-x -2.5
```

This opens a smaller 6 m vineyard with foliage and 16 clusters. The spawn
override puts the robot inside the entrance, ahead of the initial camera.
Without overrides it starts 1.5 m before the row entrance (x = -4.5 m for this
6 m layout), potentially behind the initial camera. The default heading is
along +x, down the aisle. For custom positioning, use `--robot-x` and
`--robot-y` in metres and `--robot-yaw` in radians; keep the base clear of posts.

Focus the scene rather than a UI input field before using these controls:

| Control | Action |
|---|---|
| W / S | Command forward / backward motion |
| A / D | Command left / right turns |
| Release W/S/A/D | Command zero base velocity |
| Arrow keys, Q / E, mouse | Move the viewer camera (not the robot) |
| Space | Pause / resume simulation |
| Close window or Ctrl-C | End the run and finish writing metrics/video |

The arm holds its home pose. **This is a base-driving environment, not an
implemented grape-picking policy:** no arm/gripper keyboard controls, grape
perception, cutting tool, automatic grasping or collection sequence are wired
into this launcher. The default 2.3 m canopy is a layout choice, not a guarantee
that every bunch is reachable. Although `--no-robot-camera` is accepted, this
launcher does not currently create a wrist-camera image panel.

**Headless robot smoke test and simulated stem cut:**

```bash
python scripts/grow_vineyard.py --robot --device cpu --viewer null \
  --seed 42 --row-length 6 --clusters 16 --frames 120 --cut-demo 40 \
  --metrics output/vineyard-robot-check.json
```

With no keyboard input, the Ridgeback receives zero drive commands; this checks
scene initialization and stepping, not navigation. `--cut-demo 40` severs cluster
0's stem at frame 40 regardless of the robot's position. It demonstrates a
released cluster falling, **not a successful robot harvest**. Metrics contain
the detached count and per-cluster contact-force diagnostics, not pick/place
success or validated berry damage. Omit `--cut-demo` to check attachment at rest.

**Record a preview with the robot:**

```bash
python scripts/grow_vineyard.py --robot --device cpu --viewer gl --headless \
  --seed 42 --row-length 6 --clusters 16 --robot-x -2.5 --frames 120 \
  --snapshot output/vineyard-robot.png --video output/vineyard-robot.mp4 \
  --metrics output/vineyard-robot.json
```

Recording requires `ffmpeg`; screenshots require Pillow (both are in the pinned
environment). `--device cpu` selects CPU physics, not the rendering backend.
GL rendering still needs a working OpenGL/display setup (integrated graphics or
a compatible software renderer), even with `--headless`; use `--viewer null`
for display-free physics checks and to avoid rendering/encoding overhead. The snapshot
is taken at frame 30, so request at least 30 frames. `--frames` limits null, USD
and headless GL runs; an interactive GL window runs until closed. The current
encoder writes every rendered frame at 30 fps while physics runs at 60 steps per
simulated second, so the video is not a real-time timing benchmark. Generated
outputs belong in `output/` and should not be committed.

**Layout and performance:** keep the same `--seed` for repeatable geometry.
Use `--clusters 16 --row-length 6` for a smaller CPU scene; `--no-foliage` helps
inspect the structure. `--rows` counts support rows (minimum 2), `--row-spacing`
sets their separation (minimum 1.5 m), and `--canopy-height` sets the overhead
height (1.8–3 m). Run `python scripts/grow_vineyard.py --help` for all options.
The robot and clusters collide with posts and the ground; thin vineyard wood
and leaves are non-colliding, and bunch collisions use a hidden ellipsoid rather
than individual berries. Do not treat this as a full canopy-contact benchmark.

**Spot status:** `--robot --robot-kind spot --relic-path /path/to/relic` selects
the external Spot assets described in **External Spot assets** above. This path
is experimental and has not been validated in the vineyard. The launcher creates
the Spot controller but does not call its gait-policy `update(command)` method;
the Ridgeback WASD controls do not apply to Spot. Do not use this option as a
working Spot walking/harvesting example without integrating that control loop.

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
  An implicit spring approximation supports small timesteps. Pooled detachment
  forces are sampled; the threshold is conditioned to support static weight.
  This is not a measured angle-conditioned fracture law.
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

## Validate a change

```bash
python -m unittest discover -s tests -v
python scripts/check_loose_fruit.py
python scripts/check_loose_fruit.py --timestep .0005
python scripts/check_kiwi_physics.py
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
| Unit/integration suite | 8 tests passed in the Linux environment |
| Stationary basket, 60 fruit | Zero spills over 5 s at both 1 ms and 0.5 ms physics steps |
| Loaded walking | 12 s, 2.83 m travelled, 7.6° maximum tilt, zero spills, zero canopy detachments |
| Deliberate 400 N·m roll disturbance | 36 fruit spilled; total spill penalty −36 |
| Stem/drop regression | Attached at rest; detached under a 20 N pull; landed without tunnelling |
| Original apple / pergola smoke runs | 60 / 120 frames completed |
| Native 3% timestep refinement | 10 µs vs 5 µs: force difference 0.069%; strain difference 0.00046 percentage points |
| Native 10% commanded squeeze | 8.72% measured compression, 71.38 N per pad, no solver warnings; damage proxy saturated |

The 10% command differs from tissue strain because the pads also have compliant
contact. The native 3% case records zero strain-based damage proxy; that is not
a guarantee of unbruised real fruit. Outputs are generated under `output/`.

## Continue the project

1. Fit compression/hold/release and impact tests to one cultivar and harvest
   condition. Add layered, viscoelastic/plastic response without mixing datasets.
2. Validate the actual Spot jaw contact and a stem angle/torque break law.
3. Train locomotion over 0–6 kg payload and arm configurations; compare against
   the existing policy on matched seeds, spills, tracking and falls.
4. Train reach, grip, detach and deposit, then integrate a full harvesting task.
5. Add station unloading and, later, multi-robot coordination.

PufferLib is not installed or integrated. Select a GPU training implementation
only after the native material and contact benchmarks agree with measurements.

| Path | Purpose |
|---|---|
| `treesim/pergola.py` | Procedural structure and fruit placement |
| `treesim/kiwi_material.py` | Source-backed constants, geometry sampler, damage helper |
| `treesim/kiwi.py` | Stem dynamics and GPU contact damage diagnostics |
| `treesim/basket.py` | Mounted basket, loose payload, spill events |
| `treesim/spot.py` | URDF import, observation mapping, policy and PD control |
| `treesim/sim.py` | Solver and stepping integration |
| `scripts/walk_spot.py` | Loaded walking and spill recordings |
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
