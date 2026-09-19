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
  separate: the current collidable stalk exists only in the native bench.
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
