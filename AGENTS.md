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

## Terrain navigation checks

`treesim.spot_navigation` is separate from harvesting: frozen RELIC gait plus
high-level velocity actions. Test with:

```bash
SPOT_NAV_RELIC=../relic SPOT_NAV_DEVICE=cuda:0 python -m unittest discover -s tests -p test_spot_navigation.py -v
```

Use the registered Gymnasium spec for its GPU nondeterminism declaration;
keep explicit seeded-reset and graph-versus-eager numerical comparisons.
For this pinned stack, the navigation launcher must set `MUJOCO_GL=egl` before
importing MuJoCo. Native solver conversion omits visual-only robot meshes;
the navigation renderer mirrors original URDF visuals and the exact physical
heightfield into a separate, never-stepped model. Preserve the render/physics
pose and terrain equivalence checks rather than changing collisions for looks.

## Explicit assisted reach/grab task

The user-approved `treesim.assisted_kiwi_env` is an isolated easy-grasp RL task,
not a relaxation of contact-only harvesting validation. It uses native CPU
MuJoCo, frozen RELIC walking, ideal target observations and artificial stem/grip
welds; never present it as physical grasp, detachment or damage validation.
Run `ASSISTED_KIWI_RELIC=../relic python -B -m unittest tests.test_assisted_kiwi_env -v`.
Preserve hanging-fruit, no-teleport attachment-frame, seeded-reset, walking,
fall, and two-timestep assisted-retention checks. Dynamic hand sites must use
`mjSAMEFRAME_NONE`; otherwise MuJoCo skips changed site transforms. Anchor stem
constraints at each fruit rather than using a distant world-origin weld frame.
The launcher `scripts/train_assisted_kiwi.py` writes periodic text/PNG evaluations
and checkpoints to a new directory, and checks actual weight changes. EGL must
be selected before importing MuJoCo. On dayone the isolated learning environment
is `/home/ubuntu/assisted-kiwi-venv`; the source worktree is
`/home/ubuntu/Thekenyos-assisted-training`. Do not change the legacy environment.
Longer runs use `/home/ubuntu/Thekenyos-assisted-training-v2`, preserving the
original source and checkpoints. The launcher has subprocess workers and
weight-only warm starts: trainer changes may be accepted, but environment/Spot
hashes, physical task settings and observation/action spaces must still match.
Exact resume also checks trainer hashes and learning configuration. Keep per-stage
best checkpoints, compare initial/best policies on fresh held-out seeds, and
retain failed evaluations. Test the harness via
`python -B -m unittest tests.test_assisted_kiwi_env.TrainingHarnessTest -v`.
The body-workspace curriculum lives in the trainer's `WorkspaceApproach` wrapper,
not the physical environment: `--workspace-weight 1` enables training-only
shaping and combined workspace/grab promotion. Evaluations always set this weight
to zero; `--eval-all-stages` compares one monitoring-selected checkpoint on all
three distances. Preserve the action/trajectory-equivalence test, one-time bonus,
original anchor goal and frozen RELIC hash check. The workspace is a fixture-specific
engineering heuristic, not calibrated robot reachability. Updated source snapshots
use `/home/ubuntu/Thekenyos-assisted-training-v3`, preserving both earlier versions.

## Assisted basket collection

`treesim.basket_kiwi_env` adds the existing 1.2 kg rear basket without changing
`assisted_kiwi_env` or its archived 78-value checkpoints. The basket task has
99 observations and explicit approach/carry/settle phases. Deposits require an
inactive grip, full-ellipsoid containment with 0.2 mm numerical tolerance,
support contact connected to the basket, and 0.5 s below 0.05 m/s relative speed
and 1 rad/s relative spin. Previously deposited fruit remains free and can spill;
final success requires all requested fruit settled, not just historical counts.
Keep the drop/spill, action/reset, two-timestep deposit, and body-frame tests:
`ASSISTED_KIWI_RELIC=../relic python -B -m unittest tests.test_basket_kiwi_env -v`.

MuJoCo `mj_objectVelocity(..., mjOBJ_BODY, ..., local=1)` uses the principal
inertia frame, which changes when the basket is added. Query world velocity and
rotate by the chassis body rotation before passing it to RELIC. Do not alter
inertia or gait weights to compensate for a sensor-frame error. Basket contacts
retain the existing friction values and use six-dimensional contact friction;
Cartesian arm IK includes joint-limit and basket-clearance posture correction.

`scripts/check_basket_kiwi.py --relic ../relic --output NEW_DIR --picks 6 --video`
is a scripted physical check, not a trained harvesting policy. Repeat at
`--physics-hz 2000`, inspect videos and keep failures. `train_assisted_kiwi.py
--basket` uses the new task; `--start-phase release/carry` are reset-only curriculum
fixtures with a fruit already held, not full harvesting successes. Evaluation
turns off the new task's shaping. Warm starts between basket lessons may change
start phase and requested count, but all environment/basket hashes and physical
settings must match. Never load a 78-input reach checkpoint as a 99-input basket
policy. Basket source snapshots use `/home/ubuntu/Thekenyos-assisted-training-v4`.

## Camera-policy assisted task

`treesim.visual_kiwi_env` isolates camera observations and ten-action wrist-aim
control from archived ideal-state tasks. There is no chassis head camera. Hand
RGB/ToF offsets and optical direction reference the commented sensor frames in
the external RELIC URDF. Named MuJoCo cameras `ee_cam`/`ee_depth` use the Boston
Dynamics gripper vertical FOV (46.4° RGB, 44° depth). Keep self-occlusion, actual
rendered depth, validity masks, latency, frame history, and independent seeded
sensor randomness. Actor RGB is 3 channels per history frame (wrist only). Archived
6-channel head+hand visual checkpoints cannot load or encoder-warm-start.
Never use diagnostic target overlays, segmentation, fruit coordinates or IDs in
actor inputs. The actor may see phase / has_grasped / has_placed; the critic
reads privileged 99-D basket state. Body twist/attitude are still noisy estimator
surrogates, not calibrated sensors.

Run `MUJOCO_GL=egl ASSISTED_KIWI_RELIC=../relic python -B -m unittest
 tests.test_visual_kiwi_env -v` (one line). It checks metric rendered depth,
no hidden-state observation/camera tracking, frame delay, camera blackout versus
physics equivalence, bounded wrist commands and free deposition at both rates.
Use `train_assisted_kiwi.py --vision --visual-lesson grab|collect`; grab is an
assisted-retention lesson, not collection. Stationary evaluations keep cameras
on. Keep the original grasp/basket tests and harvesting-interface regressions.
Dayone visual experiments used `/home/ubuntu/Thekenyos-visual-training-v1`; the
first post-training snapshot is `/home/ubuntu/Thekenyos-visual-training-v2`.
The latter additionally allows an explicit visual weight-only warm start to
change the episode horizon for longer collection lessons; exact resume and
physical success criteria remain strict. Preserve the original experiment
snapshot and its source hashes.

Camera geometry follow-up uses `/home/ubuntu/Thekenyos-visual-training-v4`;
`v3` contains development pilots. The scene's old 6.644 cm rendering near plane
clipped geometry at the hand TCP. Keep the robot-camera-only 5 mm near-plane
fix and matching metric-depth conversion, restore shared visual settings after
rendering, and retain the independent 0.15 m ToF validity cutoff. Preserve off-axis
projection and close-surface occlusion regressions. Never confuse camera axial
depth, Euclidean range, visible surface position and gripper-relative fruit center.

`visual_servo.py` and `check_visual_approach.py` are an explicit scripted baseline,
not learned PPO actions. The sensor packet contains only images and calibrated
robot geometry/odometry; no target IDs or simulator fruit coordinates. It retains
capture-time camera transforms for latency compensation. RGB-derived brown masks
are not simulator segmentation. There is no head-camera size prior. The remaining
hand RGB/ToF color heuristic is not validated real-kiwi perception. Stop translation on missing depth and do not revert to
monocular range after hand-depth acquisition. Keep all failed/contactful attempts,
matched body-position comparisons and camera ablations; a brief assisted hold
still does not validate contact-only grasping, basket collection or damage safety.

## Stationary close-fruit camera lesson

`treesim.stationary_kiwi_env` is an arm-only camera grab lesson, not a walking
policy, physical grasp test or basket collector. Stage 0 (S0v) places fruit in
the wrist depth FOV at 20–40 cm and scores pregrasp. Stages 1–3 place fruit along
the starting TCP axis after the usual robot spawn, then offset in the chassis YZ
plane by more than the 12 cm assist radius, so the base pose stays the same while
distance increases and a constant body-forward reach cannot capture. Zero the body and
wrist-rotation commands; keep the frozen standing gait. Actor observations stay
on the visual sensor contract: no fruit coordinates, segmentation or target IDs.
The critic may read privileged basket state. Categorical XYZ plus jaw actions must
not be mixed with archived 10-action visual checkpoints.

Run `MUJOCO_GL=egl ASSISTED_KIWI_RELIC=../relic python -B -m unittest
 tests.test_stationary_kiwi_env -v` (one line). Preserve identical-pose/farther-fruit,
off-axis placement, hand-depth visibility, passive-closure failure, open-loop
miss and scripted capture at both rates. `train_assisted_kiwi.py --stationary`
trains that lesson with privileged distance/capture shaping off and view shaping
on. Evaluations use `stationary_success` with cameras on and view shaping off.
S0v promotes on one ≥80% in-FOV pregrasp pass; later levels need two consecutive
≥75% arm-only assisted holds across 24/32/44 cm (body travel <0.10 m, arm motion
>0.015 m). Walking is a separate `--vision --visual-lesson grab`
run with ten continuous actions and 30 s episodes; do not load the 4-action actor into it.
`--encoder-warm-start` may copy only the visual CNN. After that copy, walking
action means start at zero and motion log-std is capped at 0.15 so random arm
commands do not immediately fail on the rear basket. Do not jump from that
standing encoder to three-fruit collect: queue walking grab, one-fruit collect,
then three-fruit collect with `scripts/queue_visual_harvest.py`, waiting for any
existing trainer PID and never signalling other processes. Full-weight warm
starts require held-out success. `scripts/replay_assisted_kiwi.py`
writes deterministic sensor-inset videos for
review at 10 fps (one frame per 10 Hz control step); it is not training. Collect+pick places requested fruit at
increasing chassis-forward distances (`COLLECT_FORWARD_M`, about 0.75/1.15/1.60 m
for three picks) and keeps the basket deposit rules. Do not treat a grab checkpoint
as collection. Dayone source is
`/home/ubuntu/Thekenyos-stationary-training-v1`; do not overwrite visual-training
snapshots or `assisted-kiwi-venv`. The off-axis standing lesson is a new
run; do not overwrite `stationary-kiwi-ppo-01`.

## Estimated-target assisted task

`treesim.estimated_kiwi_env` wraps `assisted_kiwi_env` or `basket_kiwi_env` so the
78-D / 7-action policy sees camera-estimated fruit XYZ instead of simulator fruit
coordinates. Keep delayed hand RGB/ToF packets, brown-mask estimates, odometry hold and
a far dummy on never-seen fruit; never restore privileged fruit pose when detection
fails. A held fruit uses gripper TCP, not the simulator kiwi. `estimate_error_m` is
diagnostic info only. Workspace shaping may stay on the privileged target during
grab training; evaluations set that weight to zero and still require camera
blackouts. `--estimate --basket` keeps that 78-D actor and scores free rear-basket
deposit; do not expand it to a 99-input basket checkpoint. This is not a visual
CNN policy and not contact-only grasping.

Run `MUJOCO_GL=egl ASSISTED_KIWI_RELIC=../relic python -B -m unittest
 tests.test_estimated_kiwi_env -v` (one line).
`train_assisted_kiwi.py --estimate` trains the grab wrap. `--estimate --basket
--picks 1` trains one-fruit collection. Weight-only warm start from a matching
78-input assisted or estimated-grab checkpoint may add estimator/basket source
files and change the episode horizon; assisted/spot hashes, capture/hold/physics
and the 7-action space must still match. Do not load a 10-action visual checkpoint
or a 99-input basket policy. Queue behind existing harvest jobs with
`scripts/queue_estimated_target.py` and never signal them. Hand-only camera
source is `/home/ubuntu/Thekenyos-estimated-training-v2`; preserve
`Thekenyos-estimated-training-v1`, stationary, and visual-training snapshots.
