# Working on Thekenyos

## Objective and current scope

Build a reproducible kiwi-harvesting simulation: one Spot with one arm, a 1.6 m
pergola and a rear basket. The 0–6 kg fruit range is a simulation stress-test
range, not a supported hardware load: Spot's 14 kg combined payload includes
the 8 kg arm, basket, mounts and other payloads. All fruit starts harvestable. Preserve
OrchardBench's existing apple workflows. Multi-robot coordination and automatic
unloading are later work.

Current programme is simulation-only; no physical Spot or fruit calibration
data is available. Follow the agreed sequence: native physics checks, shared
observation/action contract and recording, full-cycle workspace/payload check,
conventional full-cycle baseline, compact local RL with realistic sensing, then
compare imitation/diffusion only when evidence warrants it. Do not resume the
rigid-fruit pilot while its rigid/flex contact agreement gate fails. Real-fruit
calibration remains separate from numerical and literature consistency.

Read README.md, then the files affected by the task. For material changes read
docs/kiwi-material-evidence.md. Check git status before editing; preserve work
already present. State a short plan before substantial implementation.

## Correctness rules

- Separate measured values, fitted values, engineering assumptions and unknowns.
  Keep source, units, cultivar and test conditions with material parameters.
- Do not describe rigid contacts as deformable tissue, an elastic recovery as
  absence of bruising, a damage proxy as a validated predictor, or inference as
  training. Do not invent a safe gripper force from a tissue stress value.
- Preserve separate free fruit bodies, mass, inertia, rotation and collisions.
  Do not make basket fruit visual ballast or glue it to the robot to hide spills.
- Basket geometry must attach to the chassis and contribute the intended mass
  and inertia. The liner is a simplified collision surface, not the visual vents.
- The optional native `--ideal-grip` weld is an explicitly assisted extraction
  diagnostic, not a validated grasp or an RL demonstration. Preserve the
  unassisted default, video label and physical stem release rule.
- A force-only stem or rendered line does not establish physical stem contact.
  Native stalk changes require hand/stalk and fruit/stalk contact checks at two
  timesteps, including after detachment. Keep native and orchard/GPU support
  separate: the collidable stalk exists in the native bench and standalone full-Spot
  assisted fixture, not the default orchard/GPU environment.
- Full-cycle assisted changes require `scripts/assisted_harvest_cycle.py` at
  20 and 10 microseconds, numerical result checks and video inspection. Preserve
  load-triggered detachment, assist removal before gravity deposition and the
  final settled-in-basket gate. A successful assisted cycle is not an RL or
  contact-only grasp demonstration.
- A stem must transmit load at its attachment site. Apply equal/opposite forces
  and moment arms. Detachment must respond to physical contact as well as pulls.
- Damage and spill state must be irreversible within an episode. Penalize a
  newly spilled fruit once. Report damage rewards separately from spill rewards.
- Unknown wet/liner friction, creep weights, plastic response and detachment
  torque must remain explicit calibration gaps. Do not sample unrelated source
  extremes as if they form one measured distribution.

## Harvesting objective

Read docs/harvest-task.md before adding RL controls. The oracle evaluates
physical outcomes; it must not enforce a paper angle or a grasp sequence.
Keep optional guidance separate from the persistent task objective and evaluate
with guidance disabled. An evaluator is not an action teacher. Any future
imitation must use physically verified demonstrations and permit divergence.
The generic measurement bridge samples contacts when called. The Gymnasium
adapter uses one physics substep per call and evaluates failures immediately.
Preserve this behavior when batching or adding CUDA graphs. RL interface changes
require `scripts/check_harvest_env.py` with the external RELIC assets.

## Implementation

Use the smallest change that meets the task. Reuse the existing runtime and
libraries. Keep the original apple path isolated from kiwi-specific mechanics.
Prefer seeded generators and machine-readable metrics. Include units in names
and output fields. Validate finite inputs and physical ranges.

The tested stack is environment.yml: Python 3.12, Newton 1.3.0, Warp 1.14.0,
MuJoCo/MuJoCo-Warp 3.8.1. Do not independently upgrade these. RELIC stays in an
external pinned checkout; do not vendor its assets or weights.

Newton uses xyzw quaternions. Body spatial velocities/forces store linear
components first, angular components second. Body wrenches reference the COM.
CUDA graphs use persistent device buffers; avoid replacing arrays inside a
captured loop. Validate buffer swaps if changing substep counts.

## Validation and delivery

Run the affected checks in README.md. Material changes require native
compression/release results and timestep sensitivity. Contact changes require
stationary retention, falls/ground contact and adversarial spill tests. Actual
jaw geometry changes require `scripts/check_spot_gripper.py` and timestep
comparison; keep failed torque cases visible. Policy changes require matched-seed
tracking/fall/spill comparisons. A video does not
replace numerical checks; metrics do not replace visual inspection.

For the reach/grasp pilot, retain matched-seed untrained/trained evaluations
with guidance off. Keep failed contact cases and native crashes in the report;
model disagreement blocks new rigid-fruit training, but archived checkpoint
evaluation remains available. CPU contact
adapter changes require `check_harvest_env.py --device cpu` at both timesteps.
Whole-hand contacts require `check_hand_contacts.py`: a force on one jaw does
not prove collision coverage of the palm, opposite jaw or teeth. Preserve the
independent geometric intersection check and the archived failed-pilot replay.
The CPU native-contact midphase bypass is a pinned-stack workaround; do not
remove it without passing coverage and dynamic checks. GPU equivalence and
calibrated deformable fruit remain gates before further harvesting training.

Show the user a video when a useful visual milestone is ready. Label scripted
motions, pretrained inference and learned behaviour accurately. Report what
passed, what failed and what remains untested. Do not claim task completion
while a required physical regression still fails.

Keep generated videos, measurements, caches and environments out of Git unless
explicitly requested for publication. Update README commands and state when
behaviour changes. Preserve LICENSE and upstream attribution. Before a push,
inspect staged changes and exclude credentials, external restricted assets and
large generated files. Never force-push or overwrite unrelated remote work.

## Remote workstation

The current training host is SSH alias `jp` (i9-13900K, RTX 5090 32 GB, 64 GB
RAM). Existing runs use `/mnt/ssd/experiments/kiwi-pergola`, with `pergola/` for
this source, `conda/` for the environment, `assets/relic/`, `logs/` and caches.
`run.sh` selects the environment; set `ORCHARDBENCH_SOURCE` to the source path.
These are deployment details, not requirements for other contributors.

Check GPU use, free SSD space and running processes before a substantial job.
Preserve other workloads. Keep caches and outputs on the SSD; the root disk is
space constrained. Do not kill unrelated processes. First CUDA compilation can
be slow; inspect the log before restarting a job. Desktop DISPLAY/XAUTHORITY
values are session-specific, so verify them rather than hardcoding new scripts.

## Isolated deformable training development

The approved training work uses `training/envs/flex-gpu` under the existing SSD
experiment root, separately from `conda/` and `pergola/`. Its candidate stack is
MuJoCo/MuJoCo-Warp 3.13.0, Warp 1.15.0 and CUDA Torch 2.10.0. Do not upgrade the
legacy environment to these versions. Dependency inputs are under
`.devin/training/`; backend acceptance and the full training launcher are not yet complete.

Run learning tests with `python -B -m unittest discover -s tests -p 'test_kiwi_rl_*.py' -v`.
GPU monitor and mesh-frame tests require the isolated JP environment. A feature
screen from `scripts/check_deformable_backend.py` is not a training-readiness report.

MJWarp 3.13 flex contact filtering can reject offset jaw meshes because its
mesh-convex bounding-sphere test adds the original mesh offset again. The
`normalize_collision_meshes` helper normalizes authored mesh coordinates while
checking world-space collision surfaces and mass/inertia invariance. Preserve
its geometry-equivalence and actual GPU-contact regressions; do not substitute
simpler gripper colliders or zero model fields without validation.

In mixed rigid/flex GPU contacts, valid geom IDs take precedence over stale flex
IDs, matching the solver. Use `contact_flex_ids` or the device observer rather
than identifying fruit contact from `contact.flex` alone. Numerical overflow,
nonfinite states and element inversion must remain latched across substeps.

The isolated full-scene exporter has explicit options for floating-base and
multi-fruit scene generation. Legacy fixed-base defaults remain unchanged.
Preserve the initial-body-frame comparison and asset hashes. New assembled
scenes use the supported native/GPU midphase path; do not change the legacy CPU
contact-adapter bypass as part of this separate path.

RELIC R84 values in `spot.py` are raw velocities and absolute arm targets:
do not add Isaac-style scaling or subtract arm-home targets without evidence.
MuJoCo spatial velocity is angular-first and COM-referenced differently from
Newton. The native/device control tests compare the body COM velocity against
an independent Jacobian and verify that optimizer changes reach motor commands.

MJWarp `get_depth` produces clipped display-normalized values, not metric
measurements. Use the raw planar-depth buffer and explicit range validity.
Render-buffer camera indices refer to the active-camera list, not global model
camera IDs. Preserve the non-square, inactive-camera, metric-plane and immutable
frame tests. GPU scene/vision bring-up does not clear physical training gates.

## Hackathon precision (user direction)

Prioritize a working training demonstration over material calibration or fine
mesh resolution. The current isolated profile uses coarse deformable fruit,
static canopy supports, a one-segment collidable stalk, and a 2 mm sampled
hand/fruit overlap screen. Keep the
strict contact result visible, and keep failures for nonfinite state, overflow,
inversion, retention and release. Do not reopen finer calibration as a blocker
for this approved profile. Verify learned progress with the teacher disabled;
one-fruit reaching is not a complete harvest or generalization result.

## Fast hackathon training (approved approximation)

The user explicitly approved a separate rigid-fruit bulk-training profile to
maximize learning per GPU hour. Use `export_fast_scene.py`, `FastRuntime`, and
`train_fast.py`: 200 Hz physics, 50 Hz policy/gait, and 25 Hz RGBD cameras by
default. 500 Hz physics is also supported. Use CUDA FP32 pretrained gait in
this profile. The existing detailed flex and legacy CPU profiles stay separate.
Do not reopen tissue calibration or the older rigid/flex agreement gate as a
blocker for this approved approximate profile. Do keep numerical failures,
overflow, real collision loads, gravity, stem release, and world-local resets.
The 8 N stem release and 15 N force-based damage limit are uncalibrated engineering
assumptions, not biological measurements. Do not claim successful harvesting or
policy improvement from throughput or lower optimizer loss. Evaluate the actor
without privileged inputs and report physical outcomes separately.

Use `train_harvest_fast.py` for sustained harvesting experiments. Its collector
preserves physics and GRU memory across optimizer buffers; `train_fast.py` remains
a short reaching benchmark. Keep best-so-far stall detection separate from the
optimizer horizon and bootstrap hard timeouts from the final pre-reset state.
Grasp metrics require sustained loaded contact on both actual finger/jaw bodies.
Run `tests.test_harvest_training`, `tests.test_fast_ppo`, and `tests.test_fast_task`
in the JP GPU environment with FAST_SCENE and GAIT_CHECKPOINT set after changes.

The fast profile uses a 1 N·m jaw cap and a 15 N limit per jaw/non-pad contact
group; legacy controller defaults remain separate. Teacher training uses the
explicit `--role teacher` privileged actor/critic. Student distillation requires
the exact teacher checkpoint and successful unguided evaluation report. Keep
student observations RGB-D/R84 only and teacher parameters frozen during
distillation. Run `tests.test_fast_teacher` for changes to this path.
CTI requires replay acceptance and meaningful physical outcome comparisons, not
a successful full harvest first. Early action-branch experiments may develop
executor competence; keep their data separate from factual PPO. Process-local
snapshots alone are not evidence of CTI policy training or benefit.

## Continuous visual leaf roof

The reproducible local capture and generation details are in README.md under
"Continuous leaf roof (render-only)". Use `--canopy-spacing .08` for area-wide
infill; increasing `--leaves` alone only thickens the existing sparse cane lines.
The recipe uses a 3x3 post grid, scene seed 42, terrain seed 202 and 40 kiwis.
`treesim/foliage.py::place_canopy_leaves` generates the seeded placements;
`treesim/builder.py` attaches shared visual meshes to supported cane bodies.
Keep this optional layer massless and non-colliding, with no extra bodies/DOFs;
it is neither physical shoot growth nor PBR postprocessing. Default spacing zero
preserves the original foliage, and apple placement must remain unaffected.
The layer is capped at 100,000 leaves; use cropped plots rather than enabling it
blindly over the commercial field. Run `python -B -m unittest tests.test_pergola -v`
after changing it, and inspect a newly generated image as well.
