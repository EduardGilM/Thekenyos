# Thekenyos · Kiwi harvesting robotics

![Spot with a chassis-mounted kiwi basket under a procedural pergola](assets/kiwi-pergola.jpg)

A research simulation for a **Spot quadruped with one arm**, harvesting kiwis
under a **1.6 m pergola canopy** and carrying them in a rear basket.

Built on [OrchardBench](https://github.com/humphreymunn/orchardbench),
[Newton](https://github.com/newton-physics/newton), MuJoCo-Warp and native MuJoCo.

The current graph teacher uses [continuous progress and control-loss graph v5](docs/reward-graph.md).
Use `--reward-graph --role teacher --continue-from /path/to/checkpoint.pt
--learning-rate .0001` to preserve actor, critic and factual Adam state under the
new objective. Add `--cti --cti-time-fraction .2 --cti-worlds 32` only for the
matched CTI arm. Both arms share the same source checkpoint and time budget.
There is no explicit curriculum, stage practice or checkpoint rollback. Archived
v1–v4 profiles retain their original scoring. `--resume-from` is only for the same
reward profile and run; `--initialize-from` resets critic/optimizers instead.

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

### Spot with a loaded basket

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 \
  --frames 600 --closeup --video output/spot-basket.mp4 \
  --metrics output/spot-basket.json
```

Omit `--payload` to sample 0–6 kg using `--payload-seed`. Add
`--terrain --terrain-kind orchard` to walk the same scripted oval on the
retained orchard floor; sampled slope, rut, noise and
friction are written into the metrics JSON. The pretrained gait was not trained
on this surface. The arm holds its ready pose; random arm poses and payload-aware
locomotion retraining are not complete. For a numerical run, replace the video
options with `--no-render`.

```bash
python scripts/walk_spot.py --relic ../relic --basket --payload 6 --terrain --terrain-kind orchard \
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

### Fixed-base harvesting environment

See the [task and runnable Gymnasium example](docs/harvest-task.md#run-the-integration-environment).
The chassis is fixed; the agent controls six arm joints and the jaw. Physics
and failure checks run at every substep. This uses rigid orchard fruit and is
a control/reward integration milestone, not validated material transfer from
the deformable bench.

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
- **Orchard floor:** `--terrain --terrain-kind orchard` samples an assumed domain-
  randomization heightfield (2 m vine rows, grassed pasillos, ~0.64 m bare
  surcos, slope ±4°, noise 0–4 cm, ruts 0–8 cm deep and 20–60 cm wide,
  friction 0.6–1.3). It is an assumed compact layout, not a measured
  orchard-floor survey.
  Apple `--terrain` remains the older value-noise field. Wet soil friction is
  still an explicit calibration gap.


## Continue the project

See the [RL implementation specification](docs/rl-blueprint.html) (Spanish) for
the proposed end-to-end RGB-D policies, concurrent low-learning-rate RELIC
adaptation, tensor contracts, task curriculum and acceptance criteria. Open the
HTML locally in a browser; it works offline and includes a print/PDF layout.
`treesim/kiwi_rl/` scaffolds its contracts (schemas, reward ledger, arbiter,
model interfaces, single-env wrapper) with `scripts/train_kiwi.py --dry-run`;
optimisation, cameras and RELIC retraining remain integration work.
`scripts/check_relic_parity.py --relic ../relic` gates the pinned ONNX policy
(contract [1,84]->[1,12], finiteness, determinism) against an external checkout.
Torch actors/critics/PPO (`treesim/kiwi_rl/models_torch.py`, `ppo.py`), the Spot
camera rig (`sensors.py`), arm IK (`arm_ik.py`) and the jp runbook
(`docs/jp-runbook.md`) are ready for the training host; CPU-only machines run
the numpy contracts and `--dry-run` only.

The agreed sequence is: finish native physics checks; define the shared
observation/action interface and recorder; check full-cycle reachability and
payload; establish a conventional harvesting baseline; train a compact local RL
policy with realistic sensing; then compare imitation or diffusion if the
results justify them. Native and GPU collision equivalence must pass before
GPU training. A failed physics gate must not be bypassed by another pilot.

The native compression bench now derives mass from its tetrahedral volume and
the selected Xuxiang flesh density, 1,030 kg/m³. At mesh count 7 this is about
109.5 g, replacing the inconsistent 90 g assumption. Matched rigid/flex gripper
tests share that mass; their surface geometry still differs slightly. These are
homogeneous flesh benchmarks, not calibrated whole Hayward fruit. Earlier 90 g
gripper results above are historical and require revalidation.

```bash
python scripts/check_compression_convergence.py --workers 4 --video
python scripts/check_gripper_transfer.py --relic ../relic \
  --output output/contact-refined --workers 4
python scripts/check_detachment_angles.py --device cpu \
  --output output/detachment-native.json
python scripts/check_detachment_angles.py --device cpu --timestep .0005 \
  --output output/detachment-native-halfstep.json
```

Compression convergence compares counts 9/11 at 20/10 µs. Engineering tolerances
are 2% force change for timestep refinement, 10% for mesh refinement, and 0.5
percentage points of compression change. A failed child process remains a
failed case. Gripper resume checks source fingerprints before reusing results.
New rigid-fruit training requires `transfer_accepted=true`; archived checkpoint
evaluation remains available. This agreement is necessary, not a complete
physical-calibration or deployment approval.

The ideal-pad compression bench uses a 0.2 ms numerical contact time. The compression
pad-gap command accounts for the flex's 0.3 mm collision radius on each side,
so requested tissue strain is not confused with the inflated contact envelope.
The gripper bench now defaults to count 9 (387 vertices), with matched rigid
and flex mass of about 111.1 g. Retain failed coarse-mesh and old-contact
comparisons; changing density or contact settings invalidates earlier results.

Latest simulation-only screen (2026-09-19):

- Compression: all four count-9/11, 20/10 µs cases passed. Held force was
  14.15/14.30 N per pad; mesh difference 1.04%, timestep difference below
  0.000001%. Actual compression was 2.99% for the 3% command. No inverted
  tetrahedra or numerical warnings. Elastic recovery is not a bruise test.
- Native detachment: 60°, 120° and 180° fixtures passed at 1 and 0.5 ms.
  This verifies the implemented angle law, not its real-world calibration.
- Gripper: the 13-case screen failed rigid/flex agreement. The centred flex
  case moved 24 mm at 20 µs and dropped at 10 µs. The +4 mm flex case retained
  fruit at both timesteps, but peak forces differed substantially. One −4 mm
  flex process exited with SIGSEGV; separate default and alternate-collision
  repeats completed without a crash. The intermittent crash remains unresolved.
- New training is blocked. Next: isolate the native contact failure and initial
  force transients, then repeat gripper timestep checks. Do not tune material
  constants merely to make a grasp succeed.

Raw reports on JP are `output/compression-refined/summary.json`,
`output/contact-refined/summary.json`, `output/detachment-native.json` and
`output/detachment-native-halfstep.json`. Reports include executed source hashes
where available. Generated artifacts stay outside Git.

### Full Spot assisted pick and basket deposit

`scripts/assisted_harvest_cycle.py` runs a scripted, fixed-base native MuJoCo
fixture with the existing Spot arm, chassis-mounted basket and seeded pergola.
It starts at a clear pregrasp pose, approaches the kiwi, closes both jaws,
activates the explicitly labelled ideal grip, rotates, commands a vertical
pull, carries the fruit around the side of the basket, opens the jaw and
removes the assist. The released fruit falls under gravity and must settle
inside the collision liner. Arm motion uses torque-limited joint actuators;
only the initial fixture pose is set directly. IK uses the RELIC URDF link
geometry and joint position limits; actuator effort limits also come from RELIC.
The URDF lists 100 rad/s for every arm joint, so it does not establish usable
hardware speed limits. Manufacturer speed/acceleration limits and complete
self-collision coverage remain unverified.

```bash
MUJOCO_GL=egl python scripts/assisted_harvest_cycle.py \
  --relic /path/to/relic --output output/assisted-cycle --video
MUJOCO_GL=egl python scripts/assisted_harvest_cycle.py \
  --relic /path/to/relic --output output/assisted-cycle-half \
  --timestep 0.00001
python -m unittest tests.test_assisted_harvest_cycle -v
```

The script writes `scene.xml`, `workspace.json`, `result.json`, a final-state
snapshot and, with `--video`, `cycle.mp4` with a close-up inset. It exits nonzero
on failure. Success requires load-triggered detachment during the downward
pull, assist removal, no ground hit, bounded fruit/stalk/arm-obstacle overlap,
and at least 0.5 s of low linear and angular velocity fully inside the basket
with liner contact at the end of the observation period.

This fixture uses rigid fruit and a secure artificial grasp. It does **not**
validate fruit safety, contact-only grasp strength, balance, hardware workspace,
RL or GPU parity. The basket liner uses a 0.5 ms contact time constant with
high impedance to limit numerical penetration. Six-dimensional contacts enable
its existing sliding, torsional and rolling friction coefficients
(`0.7`, `0.005 m`, `0.0001 m`). These are engineering assumptions, not measured
kiwi–liner friction or cushioning.
The fixed stalk root excludes collision only with its two joined canopy canes;
hand/stalk and fruit/stalk collisions remain active. Canopy joints and the
chassis are fixed for this workspace diagnostic. The default RL environment
and its training gate are unchanged.

Validated on JP (2026-09-19): the full 20 µs rerun passed, detaching at
27.024 N with 0.437 mm maximum fruit contact overlap, no ground hit and no
recorded arm/basket or arm/canopy overlap. The 10 µs trajectory detached at
27.023 N with 0.428 mm maximum overlap. Its saved final state passed a 1 s
settling continuation after correcting the containment evaluator to accept
wall contact within 1 µm. The earlier strict-boundary failure report remains
archived; it was not a failure to deposit the fruit. Three containment
regression tests pass, and separate outside-basket falls trigger ground-contact
failure at both timesteps. Reports are in `output/assisted-cycle-verified`,
`output/assisted-cycle-half/settlement-recheck.json` and
`output/assisted-cycle-ground-negative.json` on JP. Peak jaw force was about
62 N in this rigid assisted fixture; no fruit-safety claim follows from it.

### Native stem extraction bench

The native stem-attached diagnostic is `scripts/check_grasp_pull.py`.
It uses Spot's jaw collision meshes, an unpinned fruit and a collidable stalk.
`treesim/native_stem.py` builds four massive capsule segments with bending,
torsional and axial spring joints. Mean He2024 length, diameter, density and
modulus determine segment mass and EA/EI/GJ stiffness. Joint damping and contact
friction remain engineering assumptions. This is a reduced beam model, not a
calibrated stalk fracture or viscoelastic model. The 0.4 mm collision clearance
at the tip avoids initial overlap at the fruit's attachment node.

A separate MuJoCo point connection attaches the stalk to the fruit surface.
The connection's tensile reaction and the **local stalk direction at the
junction** drive the angle-dependent abscission rule. Breaking that connection
leaves the stalk's bodies and collisions active. By default there is no fruit-to-hand
attachment. The explicitly requested `--rigid --ideal-grip` diagnostic adds
a hand/fruit weld after bilateral jaw contact at closure. It matches the
current pose before activation, keeps stalk collisions and load-triggered
detachment active, and labels the video as assisted. This isolates extraction
assuming a secure grasp; it does not validate real grip strength, tissue
damage or an unassisted harvesting policy. The fixed world fixture receives the branch-end reaction.
This native prototype has **not yet replaced the orchard/Newton force-only
stem representation**; the old orchard path does not provide stem collisions.
Do not claim orchard or GPU parity from these native checks.

```bash
MUJOCO_GL=egl python scripts/check_grasp_pull.py --relic ../relic \
  --output output/grasp-pull --torque 3 --grasp-x .175 \
  --target-fsa 60 --pull-after 5.8 --grip-force 15 --roll-speed .4 --video
# Add --rigid for the diagnostic surrogate; repeat with --timestep .00001.
# For a numerical smoke check, omit --pull-after and add --duration 1.

# Generate a rigid scene with the command above plus --rigid, then:
MUJOCO_GL=egl python scripts/check_stem_contacts.py \
  --scene output/grasp-pull/scene.xml --output output/stem-contacts --video
# Repeat at --timestep .00001 and with a generated elastic-fruit scene.
```

The ideal controller closes the jaws and rotates around the observed attachment.
With `--pull-after 5.8`, it stops rotating at 5.8 s and pulls straight down in
world Z at 9 mm/s, keeping wrist XY and orientation fixed. The measured angle
is recorded but does not block the pull. Without `--pull-after`, the earlier
angle-gated diagnostic waits within 3° of its target for 100 ms before pulling.
Wrist rotation is not fruit–stem angle.
A rigid diagnostic with the transition at 4.8 s verified 27 mm downward travel,
zero wrist XY/orientation change, and no contact-limit abort over 10 s. The
fruit remained attached: correct pull direction is not extraction success.
The elastic run with the 5.8 s transition also completed 10 s and verified
27 mm vertical travel with fixed wrist XY/orientation. It remained attached
and slipped out of the jaws (71 mm net fruit-centre motion relative to the
wrist); peak stem load was 27.82 N and maximum contact overlap 0.814 mm.
Reports and motion traces are in `output/extraction-vertical-pull` on JP.

For the explicitly assisted extraction fixture, add `--rigid --ideal-grip`
and use `--pull-after 4`. The 20 µs run activated the attachment after bilateral
contact at 2 s, detached at 4.034 s (29.54 N, 137.52°) and retained the fruit.
The 10 µs repeat also passed, detaching at 29.537 N versus 29.545 N at 20 µs.
Both retained the fruit with less than 4 µm relative drift. These used rigid
fruit and an artificial grip: 346–362 N transient jaw reactions were recorded,
so this is not evidence of safe fruit handling. An initial
misaligned-site activation was rejected; the fixed implementation verifies
coincident site frames before enabling the attachment. Accepted-run artifacts
are under `output/extraction-ideal-grip-aligned`, with the timestep repeat in
`output/extraction-ideal-grip-half`.
The contact-force setpoint is neither a hard force bound nor a measured safe
fruit limit. This privileged controller is separate from the outcome-only RL
evaluator; no angle target or prescribed motion is added to the RL reward.

The direct contact regression first moves the actual hand into the stalk,
stops after 0.5 mm additional travel, and withdraws it. The second case starts
with the fruit detached and lets it strike the stalk. Gravity is disabled in
these collision-isolation checks; they are not harvesting demonstrations.
Both require nonzero contact force, stalk motion, sub-millimetre penetration
and no numerical warning. The elastic check also rejects collapsed elements.
The initial unrestricted prescribed-hand sweep became numerically unstable;
it is archived under `output/physical-stem-contacts` and is not a passing case.

The final palm/stalk and fruit/stalk checks passed with both rigid and elastic
fruit at 20 and 10 µs. Elastic peak loads were 1.980/1.985 N for the hand and
0.26024/0.26015 N for the fruit; peak-force changes were below 0.4% across
all four matched cases. Maximum stalk-contact overlap was 0.366 mm. The elastic
minimum tetrahedral volume ratio stayed above 0.9978. A separate small-load
beam check matched continuum tip deflection within 3.2%; this checks the
reduction, not biological calibration. Run it with
`python -m unittest tests.test_native_stem -v`.
Raw contact reports are `output/stem-verified-{rigid,flex}-{20,10}/metrics.json`
on JP. `--hand-surface jaw` provides an additional, less occluded contact view.

The earlier force-only-stem rigid demonstration detached at 62.27° and 6.02 N,
then retained the fruit. **That result is superseded:** adding stalk collisions
changed the outcome. An early stalk prototype released around 114° and dropped
the fruit. The refined stalk released at 110.24° and retained it, but the
prescribed wrist drove into the stalk (10.4 mm peak overlap). That run is
rejected. The fixture now stops immediately if hand/stalk penetration or
attachment error exceeds 1 mm. The picking motion still needs collision-aware
control; numerical completion alone is not acceptance.
The force-only elastic runs slipped, and an unregulated 3 Nm run collapsed a
tetrahedron. Do not use these as physically verified demonstrations for RL.
Preserve the failed cases while improving the physical model and controller.

These benches use MPR collision (`nativeccd=disable`) as a scoped workaround
for intermittent native CCD failures; their root cause is not established.
The free-fruit bench uses a 4 ms numerical contact response, the grasp/pull
fixture 2 ms, and the ideal-pad compression bench 0.2 ms. These values are
solver settings, not measured tissue compliance. The jaw controller uses
kp=20, kv=0.2 and a torque limit; these are not identified Spot hardware gains.
Hold drift is measured after closure, with settling reported separately.
The pull check measures retention relative to the wrist and rejects excessive
hand/stalk penetration or attachment error. `numerically_completed` and
harvest `passed` are separate results; neither establishes absence of bruising.
The mixed Xuxiang tissue / Hayward abscission model remains uncalibrated.

Continue the agreed sequence above after the numerical contact gate passes.
Real-world calibration remains pending while the project is simulation-only.
PufferLib is not installed or integrated. Select a GPU implementation only after
native contact is stable and matched CPU/GPU checks pass; retain the explicit
limits of the uncalibrated tissue and damage models.

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

### Live reward/CTI run dashboard

For the fresh reward-graph and CTI v7 run, run locally:

```bash
python3 scripts/training_dashboard.py --run teacher-graph-cti-001 --cache output/graph-dashboard
```

Open `http://127.0.0.1:8765`. This standard-library server polls JP every five
seconds and shows physical outcomes, reward components, throughput, CTI comparisons,
and a cached rollout of the best evaluated checkpoint with the arm-camera inset.
Changed best checkpoints render no more often than every three minutes. Video is
recorded, not a live camera feed. Render errors retain the previous video. The
server binds only to localhost; SSH access to `jp` is required. Use `--no-render`
for metrics only. Run selection is explicit; the server reads the active PID and the renderer source from run metadata.
Run `python3 -B -m unittest tests.test_training_dashboard -v` for the focused checks.

### Temporal reward graph and all-branch CTI

The current hackathon teacher uses `--reward-graph --cti`. See
[the reward contract](docs/reward-graph.md) for the six stages, physical gates,
partial ground-drop credit and two-second basket settling. The graph shapes
training reward; it does not choose actions or become a sensor-student input.

CTI v7 replicates 32 actual PPO roots into 416 GPU worlds, runs three refinement
passes, and learns from all replay-valid branch transitions using a separate
V-trace actor/critic optimizer. Worse outcomes and physical failures are included.
The search samples all seven controls, keeps an unbiased comparison branch,
and preserves broad mutations. It does not prescribe jaw closure or harvesting
motions. A mean factual KL limit and rollback constrain auxiliary updates.

```bash
python scripts/train_harvest_fast.py --role teacher \
  --scene "$FAST_SCENE" --eval-scene "$EVAL_SCENE" \
  --gait-checkpoint "$GAIT_CHECKPOINT" --output training/runs/teacher-graph-cti-001 \
  --reward-graph --worlds 4096 --steps 128 --minibatch-worlds 256 \
  --gae-lambda .99 --entropy-coef .001 --cti --cti-worlds 32 \
  --cti-alternatives 12 --cti-search-iterations 3 --cti-time-fraction .2 \
  --train-seconds 3600 --wandb-mode online
```

Omitting initialization/resume flags starts fresh harvest actor and critic weights
and both optimizers; the fixed gait checkpoint remains reused. Preserve old runs.
The CTI time fraction is an amortized cap; an individual search can overshoot it.

Focused checks: `tests.test_reward_graph`, `tests.test_cti_learning`,
`tests.test_branch_cti`, `tests.test_counterfactual_replay`,
`tests.test_harvest_training`, `tests.test_fast_task`, and
`tests.test_training_dashboard`. Set `CTI_GPU_TEST=1`, `FAST_SCENE`, and
`GAIT_CHECKPOINT` for the real PPO-to-CTI GPU test. The full-size integration
check processed 249,600 branch transitions in 4.88 s including learning, used
40 accepted optimizer steps, and rejected zero replay roots. That demonstrates
working data flow and throughput, not improved harvesting.

### Continuous graph v2 run

`teacher-graph-cti-002` starts fresh harvesting weights and both optimizers,
with the fixed gait reused. The corrected graph removes the approach plateau,
settles the physical reset, recognizes accumulated progress and recovery, and
records stage-level progress/reward and actual control behavior. CTI v8 uses
six-second eligible PPO roots and bounded-action entropy. The previous run is
preserved at checkpoint 338. See docs/reward-graph.md for the versioned contract.

Use the graph command above with `--stall-seconds 8 --max-episode-seconds 30`
and a new output directory. Validate with `scripts/check_graph_training.py`
using FAST_SCENE and GAIT_CHECKPOINT, plus the documented graph and CTI tests.
The dashboard now renders latest and best evaluations separately.

### Progressive sequence curriculum (current run)

`train_sequence_fast.py` starts a fresh privileged PPO policy with CTI disabled.
Each episode starts from the ordinary settled pose, with small seeded arm-pose
variation. The robot base is fixed at its authored home pose; legs hold their
settled motor targets. Arm and fruit dynamics remain physical.
There are no demonstrations, scripted harvesting actions, or stage-state resets.

The objectives are: position the kiwi between the jaws; sustain bilateral loaded
contact; detach while grasping; carry above the basket while holding; release and
settle inside the basket for two seconds. A rollout must achieve every preceding
checkpoint in order. Earlier events are remembered; grasp must remain valid during
extraction and transport, and can end during a valid release. A drop elsewhere
cannot become a successful harvest. Two fresh validation trials from the training distribution must each reach 90%
prefix success in two consecutive evaluations before the next objective unlocks.
Each trial uses 32 varied initial poses. A separate held-out scene measures
generalization and does not control curriculum promotion.

The old aperture-width closure score is replaced with distances between the
actual convex jaw collision meshes and the fruit, computed by the pinned MJWarp
GJK support maps. Sustained contact, slip, load, and the physical detachment event
still decide validity. Sequence v2 pays one point immediately on each first valid
unlocked checkpoint. These payments stay unchanged when the next objective unlocks.
Dense feedback is the change in a separate live-quality balance capped at 0.25;
grasp guidance retains insertion and opposing-jaw alignment. Failure clears that
live balance and costs 0.25. The total time cost is at most 0.1 per episode.
Stationary poses and repeated checkpoints pay nothing. Gamma is 1, and the finite
30-second deadline is terminal without critic bootstrap. Safe quality remaining
at the deadline is explicit partial credit, bounded by 0.25. Success clears live
credit. This is curriculum guidance, not policy-invariant shaping or a guarantee
that the policy cannot forget. Inactivity remains diagnostic.

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

Each 131,072-transition rollout receives up to 24 actual Adam steps across three
shuffled recurrent minibatch epochs, subject to a 0.02 KL guard. Factual replay is
verified before the first mutation. Observations include the current objective,
checkpoint history and timers. Checkpoints save the curriculum, actor, critic,
Adam, counters and RNG. `--resume-from` requires the latest logged checkpoint and,
for online W&B, its `--wandb-run-id`; it keeps the original total time budget.
After promotion, 20% of training worlds rehearse earlier objectives, each from
the normal initial pose. Evaluation always tests the current objective alone.
`current_objective/*` separates training outcomes from shorter rehearsal episodes.
`stage_event/*` counts paid achievements per batch; `stage_live/*` shows current
guidance. `stage_reward/*` includes event payments and live-quality changes.
`stage_loss/*` measures negative guidance changes, not optimizer loss.

W&B and the dashboard show `curriculum/level`, `evaluation/prefix_success`,
`evaluation/completed/{position,grip,extract,carry,deposit}`, conditional completion,
per-stage progress rewards, actual optimizer steps and joint motion. Prefix success
is distinct from `evaluation/success`, which means a complete physical harvest.
The renderer reads the saved objective and policy schema. Archived graph profiles
retain their existing scorer and observation schema. The archived sequence-v1
run uses its frozen `sequence-001` source release; v2 checkpoints use a new schema
and cannot resume a v1 optimizer.

### Scripted deposit and full-cycle evaluation

Transport is treated as a planning problem, not a contact problem. The RL
policy learns approach, seated grasp and extraction. Once the fruit is validly
extracted and held, `treesim/kiwi_rl/scripted_deposit.py` drives the arm targets
along two collision-checked joint-space segments (extraction posture, high
posture over the basket centre, low posture inside the basket), opens the jaw,
dwells 0.5 s and retracts. Segment feasibility was checked by forward kinematics
on the fixed-base scene; the physics still decides whether the fruit settles.
The handover keeps a harder squeeze than the policy's own jaw command.

Two triggers can start the deposit. `oracle` uses the simulator grasp oracle and
is a diagnostic. `proprioceptive` uses joint state only: the jaw is commanded
near closed but has stalled well short of closed for 0.2 s, and the hand has
then travelled at least 5 cm from where the hold began. Only the proprioceptive
trigger is deployable.

```bash
python scripts/evaluate_full_cycle.py \
  --scene /path/to/sequence-near-001 --checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --gait-checkpoint /path/to/g1-cpu.pt --output /path/to/full-cycle-001 \
  --worlds 64 --seconds 20 --deposit-trigger proprioceptive
```

The report lists per-stage completion, physical success, episode end reasons,
agreement steps between the two triggers and the numerical flag count. One
world's states are saved in the renderer's replay format. `--role` defaults to
the checkpoint's role, so the same command evaluates a student checkpoint.
Checkpoints are numbered `checkpoint-NNNNNN.pt`; the training log records
which one scored best.

Jaw distances in the sequence scorer come from
`treesim/kiwi_rl/surface_distance.py`, which reuses the pinned MJWarp GJK support
maps to measure nonpenetrating distances between the actual convex jaw
collision meshes and the fruit. A regression test compares them with native
MuJoCo at a recorded false optimum of the old aperture-width proxy.

### Sensor-only student

`train_sequence_student.py` distills the privileged sequence teacher into a
policy that never receives simulator state. It observes hand RGB-D at 64×48 by
default plus the R84 proprioception vector. Training is DAgger-style on top of
the same PPO objective: the student acts, the teacher labels its actions, and
`--teacher-distill-weight` scales the imitation term. The student starts at the
extraction objective (`--start-level 3`) because the teacher already masters
the prefix, and `--max-level 3` is the ceiling since carry and deposit are
scripted. The jaw torque command is capped at 0.6 Nm: full 1 Nm torque loads
the pads to the 15 N damage limit, so a fully closed command on the fruit would
always fail, while 0.6 Nm squeezes at about 9 N.

```bash
python scripts/train_sequence_student.py \
  --scene /path/to/sequence-near-001 --eval-scene /path/to/sequence-holdout-001 \
  --gait-checkpoint /path/to/g1-cpu.pt --teacher-checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --output /path/to/student-sequence-001 --worlds 512 --steps 128 --train-seconds 28800
```

The student schema is `fast-harvest-sequence-student-rgbd-r84/v1`. Resume with
`--resume-from` and, for online W&B, `--wandb-run-id`.

### Orchard demo and timelapse

`demo_orchard_harvest.py` runs the whole cycle in the orchard scene: walk to a
stand point beside each kiwi with the RELIC gait under a simple route
controller, harvest with the RL arm policy, detect the hold proprioceptively,
deposit with the scripted carry, stow and move on. Every layer is deployable
except the reward oracle, which only reports. Fruit order defaults to
nearest-first greedy; `--order` overrides it. `--freeze-legs-while-harvesting`
holds the current stance targets from stand through stow, matching the
fixed-base training. `--r84-template` replays leg and body proprioception
recorded on the fixed-base training scene while the gait stands, keeping arm
entries live, which closes the gap between the training and demo observation
distributions. `--yaw-sign` and `--vy-sign` correct the gait command
conventions if the robot turns or strafes the wrong way.

```bash
python scripts/demo_orchard_harvest.py \
  --scene /path/to/orchard-demo-001 --checkpoint /path/to/teacher-sequence-002/checkpoint-000120.pt \
  --gait-checkpoint /path/to/g1-cpu.pt --output /path/to/demo-001 \
  --freeze-legs-while-harvesting --deposit-trigger proprioceptive
python scripts/render_demo_timelapse.py \
  --scene /path/to/orchard-demo-001 --replay /path/to/demo-001 \
  --output /path/to/demo-001/timelapse.mp4 --seconds 15
```

The demo writes `report.json` with the visit events and harvest count and
`states.npz` with one world's joint states. The timelapse renderer replays those
states in native MuJoCo at 1920×1080 with a chase camera and a harvest-count
HUD, compressed to a fixed output length. No physics is re-run, and a
visual-only scene variant can be substituted for rendering.

Validate these additions with:

```bash
python -m unittest tests.test_sequence_curriculum tests.test_scripted_deposit -v
```

The sequence tests cover ordered checkpoints, one-time event payments,
rehearsal, terminal deadlines, carry progress along the joint line and the
native-versus-GPU surface-distance regression. The deposit tests check that the
controller engages only after a held extraction, follows both segments before
releasing, and keeps the harder squeeze at handover.
