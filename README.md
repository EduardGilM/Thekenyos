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

### Uneven kiwi terrain for future RL episodes

`--terrain` uses a **physical collision heightfield**, not just a visual mesh.
For the kiwi pergola, three octaves of **smooth value noise** (quintic
interpolation, decreasing octave amplitudes) generate continuous rolling ground.
There are no post-centred masks, flat pads or artificial mounds. Posts remain
embedded below the surface, rather than reshaping the ground around them; the
canopy stays at its world-space height. Kiwi terrain gets a **fresh random seed
each launch**, even with a fixed canopy `--seed`. Flat ground remains the default,
and the apple terrain retains its flat trunk area and original seed behaviour.

```bash
python scripts/grow_tree.py --preset pergola --foliage --seed 42 \
  --terrain --terrain-amplitude .24 \
  --terrain-wavelength 1.8 --terrain-extent 6 \
  --device cpu --substeps 40 --viewer gl --headless --frames 360 \
  --camera-orbit 20 --snapshot output/kiwi-uneven-terrain.png \
  --video output/kiwi-uneven-terrain.mp4 --metrics output/kiwi-uneven-terrain.json
```

Use `.venv/bin/python` instead of `python` with the local virtual environment.
Remove `--headless` for an interactive window; remove recording options and use
`--viewer null` for display-free physics. CPU rendering still needs working
OpenGL. The MP4 records every second physics frame at 30 fps, matching the
60 Hz simulation clock. `--snapshot` saves the last frame. The orbit is a
scripted camera motion, not robot motion or learned behaviour.

Terrain controls (engineering assumptions, **not measured orchard soil**):

- `--terrain-amplitude`: nonnegative height range in metres above a 4 mm offset;
  default 0.05 m. The 0.24 m preview deliberately makes relief easier to see;
  start with 0.03–0.05 m for robot experiments. Traversability is not validated.
- `--terrain-wavelength`: positive dominant bump spacing in metres; default 1.8.
  Three noise scales share a finite-resolution grid, so very small wavelengths
  cannot create arbitrarily fine geometry.
- `--terrain-extent`: ground half-extent in metres; 6 gives a 12 × 12 m patch.
  It must cover the kiwi posts plus a 0.5 m margin. Outside it, ground is flat;
  constrain training episodes to the patch (the edge may have a step).
- `--terrain-seed`: specify a nonnegative integer (for example,
  `--terrain-seed 7`) to reproduce exactly the same ground. **Omit it for new
  random kiwi ground on every launch**, independently of the canopy and fruit.
  The chosen seed is printed at startup and saved in metrics. Rebuild at episode
  reset with a new seed or amplitude for curriculum/domain randomization.
  Batched worlds currently share one terrain; independent per-world terrain
  and RL training are not implemented.

Programmatic use: set `cfg.physics.terrain = True`, `terrain_seed`,
`terrain_amplitude`, `terrain_wavelength`, and `terrain_extent` before calling
`builder.generate_and_build(cfg)`. For deterministic programmatic builds,
`terrain_seed=None` still inherits `cfg.seed`; sample a new `terrain_seed` at
each episode reset for independent ground. The demo launcher handles that
sampling automatically. `tree.terrain_height(x, y)` provides a
bilinear height estimate for spawn/planning; actual contact follows the
heightfield triangles. Robot spawning and policy observations are not adapted
here—avoid spawning feet inside bumps when integrating a legged RL task.
Metrics save the scene and terrain seeds and terrain dimensions.
Kiwi terrain uses a 2 ms, critically damped MuJoCo contact reference with higher
contact priority than fruit (`ke=250000`, `kd=1000` in Newton's numerical
mapping). These are rigid-ground solver settings, not measured soil or fruit
stiffness. The old 20 ms blended response let a sustained 20 N pull drive a
small kiwi through the heightfield. Use the demonstrated 40 substeps at 60 Hz;
larger timesteps and GPU execution still need separate validation.

```bash
python -m unittest discover -s tests -v
python scripts/check_kiwi_physics.py --device cpu --terrain
```

The terrain demo, drop regression and unit tests still use MuJoCo-Warp on CPU.

The terrain check verifies attachment at rest, physical pull detachment, falling,
settling on the elevated ground and measured contact force. The suite also
checks high-speed forced impacts against the collision heightfield.
These checks do not establish rough-terrain Spot locomotion or harvesting success.

#### Three reproducible terrain examples

Use terrain seeds **101**, **202**, and **303** to compare three different
surfaces. All three use canopy/fruit seed **42**, the same camera path, a
12 × 12 m patch, 0.24 m height range, 1.8 m dominant wavelength, and 40 kiwis.
Only the terrain seed changes. Each video covers 3 simulated seconds
(180 physics frames, encoded as 90 frames at 30 fps); the PNG is the final view.

Run from the repository root with the pinned environment installed:

```bash
for terrain_seed in 101 202 303; do
  .venv/bin/python scripts/grow_tree.py \
    --preset pergola --foliage --seed 42 \
    --terrain --terrain-seed "$terrain_seed" \
    --terrain-amplitude .24 --terrain-wavelength 1.8 --terrain-extent 6 \
    --device cpu --substeps 40 --viewer gl --headless --frames 180 \
    --camera-orbit 20 --progress-every 60 \
    --snapshot "output/kiwi-terrain-${terrain_seed}.png" \
    --video "output/kiwi-terrain-${terrain_seed}.mp4" \
    --metrics "output/kiwi-terrain-${terrain_seed}.json" || break
done
```

Use `python` instead of `.venv/bin/python` if the conda environment is active.
The loop is sequential to avoid competing render jobs on a small CPU machine.
It overwrites the corresponding generated files when rerun.

| Terrain seed | Snapshot | Video | Metrics |
|---|---|---|---|
| 101 | [PNG](output/kiwi-terrain-101.png) | [MP4](output/kiwi-terrain-101.mp4) | [JSON](output/kiwi-terrain-101.json) |
| 202 | [PNG](output/kiwi-terrain-202.png) | [MP4](output/kiwi-terrain-202.mp4) | [JSON](output/kiwi-terrain-202.json) |
| 303 | [PNG](output/kiwi-terrain-303.png) | [MP4](output/kiwi-terrain-303.mp4) | [JSON](output/kiwi-terrain-303.json) |

These are local generated artifacts, intentionally excluded from Git. The
links work after generating them; a fresh clone or GitHub's README view will
not contain the files. This is a scripted camera preview, not robot navigation.

For interactive viewing of one example:

```bash
.venv/bin/python scripts/grow_tree.py --preset pergola --foliage --seed 42 \
  --terrain --terrain-seed 202 --terrain-amplitude .24 \
  --terrain-wavelength 1.8 --terrain-extent 6 \
  --device cpu --substeps 40 --viewer gl
```

For display-free simulation, use `--viewer null --frames 60` instead of GL,
and omit `--snapshot`, `--video`, `--headless`, and `--camera-orbit`.
To obtain a different random terrain each launch, omit `--terrain-seed`.

#### How generation works (agent handoff)

The implementation is `treesim/builder.py::_add_terrain`; the public scene
entry point is `builder.generate_and_build(cfg)`. For the kiwi path:

1. The terrain seed initializes a local NumPy random generator, separate from
   canopy and fruit sampling. Random lattice values are normally distributed.
2. Three periodic noise layers use target wavelengths `w`, `w/2`, `w/4` and
   weights `1.0`, `0.35`, `0.12`. Integer lattice dimensions make the exact
   wavelength approximate. Quintic interpolation, `6t^5 - 15t^4 + 10t^3`,
   smooths transitions between lattice values.
3. The combined field is normalized to `[0, 1]`, then mapped to world heights
   `0.004 + terrain_amplitude * noise` in metres. Amplitude is the full height
   range, not a standard deviation and not a plus/minus offset.
4. Grid spacing targets 0.08 m, with 48–384 cells per axis; extreme extents or
   tiny wavelengths are resolution-limited. The 12 m examples have 150 cells
   per axis (151 × 151 height samples, including the repeated boundary).
5. The same Newton heightfield supplies rendering and collision geometry.
   Posts are fixed below the surface; there is **no terrain deformation around
   posts**. This models rigid ground, not deformable soil.

Preserve the seed **and** terrain settings, scene geometry, environment count,
code revision and pinned dependency versions to reproduce an experiment.
Metrics record `summary.scene_seed` and `summary.terrain` (seed, amplitude,
wavelength, half-extent, noise algorithm and octave count). Record the code
revision separately with `git rev-parse HEAD`, and note any uncommitted edits.

#### Programmatic episode generation for other agents

This lightweight example builds three fresh CPU episodes with eight kiwis and
no foliage or rendering. It samples a reproducible **sequence** of terrain
seeds from a master seed, suitable for matched-seed comparisons. It does not
implement an RL environment, policy, observation space, action space or reward.

```python
import numpy as np

from treesim import builder
from treesim.config import TreeConfig
from treesim.metrics import Metrics
from treesim.sim import Sim

terrain_rng = np.random.default_rng(2026)
for episode in range(3):
    cfg = TreeConfig.compliant("pergola")
    cfg.device = "cpu"
    cfg.seed = 42
    cfg.physics.terrain = True
    cfg.physics.terrain_seed = int(terrain_rng.integers(0, 2**31))
    cfg.physics.terrain_amplitude = 0.05
    cfg.physics.terrain_wavelength = 1.8
    cfg.physics.terrain_extent = 6.0
    cfg.fruit.enabled = True
    cfg.fruit.max_count = 8
    cfg.fruit.joint = "free"
    cfg.fruit.colors = ((0.39, 0.27, 0.12), (0.48, 0.34, 0.17))

    tree = builder.generate_and_build(cfg)
    sim = Sim(tree, fps=60, substeps=40, collisions=True)
    metrics = Metrics(f"output/terrain-episode-{episode:03d}.json")
    for frame in range(10):
        sim.step()
        metrics.frame()
    metrics.save(sim)
    print(episode, cfg.physics.terrain_seed, tree.terrain_height(0.0, 0.0))
```

Agent integration rules:

- **CLI versus Python:** the kiwi launcher chooses a random terrain seed when
  omitted. In direct Python builds, `terrain_seed=None` inherits `cfg.seed`;
  it does not sample a new seed. Set an explicit sampled seed per episode.
- **Reset:** rebuild the model and `Sim` for a new terrain, as above. Changing
  `cfg.physics.terrain_seed` after construction does not replace the existing
  heightfield. Recreate controllers, state and any CUDA graph tied to that model;
  this is not an in-place vectorized reset implementation.
- **Spawn and clearance:** use `tree.terrain_height(x, y)` for an approximate
  bilinear height query; collision uses triangles. Check all foot/wheel contact
  positions, robot orientation, canopy clearance and post clearance before
  starting an episode. Existing robot spawns do not automatically follow terrain.
- **Batching:** worlds currently share one periodic heightfield; changing
  `num_envs` changes the display tiling and may change the generated field.
  Use separate single-environment builds for different terrain seeds today.
- **Curriculum:** start around 0.03–0.05 m amplitude, then increase deliberately.
  The 0.24 m renders exaggerate relief for inspection; they are not validated
  traversability targets. Keep evaluation seeds fixed and separate from training
  seeds, and stay away from the finite patch boundary.
- **Validation:** run the suite and terrain pull/drop check above after changes.
  Inspect renders as well as metrics. No detachments at rest is only a stability
  check; it is not harvesting success. GPU stepping, terrain-aware Spot control
  and end-to-end RL training still require their own validation.
- **Artifacts:** save unique output names and seeds per episode. Keep generated
  images, videos, metrics and external robot assets out of Git. Do not change
  the pinned physics dependencies to make a run pass.

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
