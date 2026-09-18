# Single-fruit harvesting task

The RL policy chooses how to collect the target. The oracle is a privileged
**evaluator**, not a controller or an action demonstration. It never moves the
robot, attaches fruit to a jaw, or commands a picking angle.

## Objective and outcomes

Start with one attached, harvestable target and a basket on Spot. Success means
the target is detached, released from the jaws, fully inside the basket, in
contact with its liner or payload, and moving below 0.05 m/s relative to the
basket for 0.5 s. Any strategy that meets this condition can succeed, including
one that never establishes the evaluator's bilateral grasp event.

Ground contact, excess damage proxy, jaw overload, forbidden target–robot
contact, collateral fruit loss, or timeout ends the episode. Damage history and
losses cannot be refunded. These rules evaluate outcomes, not a required sequence.

`treesim/harvest_task.py` implements the contract and evaluator. Defaults are
engineering curriculum choices: 30 s horizon, 15 N maximum summed contact load
per jaw, damage-proxy limit 0.05, and 0.2 N contact detection threshold. These are
not validated real-fruit safety limits. No 60-degree pose earns reward.

## Actions, observations and initial curriculum

The action specification is seven normalized joint velocity commands: six arm
joints and one jaw, initially limited to 0.5 rad/s each. The future environment
adapter must enforce asset joint/effort limits and integrate absolute actuator
work at physics substeps. It must also latch transient collisions and force
peaks between policy steps; frame-end samples alone can miss harmful contacts.

Begin with a fixed chassis, legs in the standing pose and a reachable target.
This reduces exploration difficulty; it is a proposed curriculum, not a claim
that the current code implements a fixed-base training environment. Later use
the standing controller and then permit locomotion.

The actor should receive joint state, target/TCP and basket relative poses,
contact information and previous action. Exact stem threshold, damage history
and attachment state belong to the evaluator or privileged critic; do not
silently expose them as deployable sensors. `SimHarvestObserver` reads simulator
state, actual native contact forces and full-ellipsoid containment. Its caller
selects the actual pad bodies and TCP offset. It is a measurement bridge, not a
complete Gym environment or training loop.

## Guidance that can be removed

`TaskDefinition(guidance_weight=1.)` enables bounded reach-potential shaping and
one-time stable-grasp / retained-detachment bonuses. Set this weight between
episodes; anneal to **zero** as unguided evaluation success improves. At zero,
only collection, failure, damage, loss, elapsed time and actuator-work terms
remain. Keep safety and physical outcome criteria active throughout training.

The reach shaping uses `gamma_per_second ** dt`; the trainer must use the same
discount convention. Grasp/detachment bonuses are explicit temporary biases,
not guarantees that the optimal policy is unchanged. Compare seeded training
with and without them. Report evaluation success and damage with guidance off,
not just training reward.

An evaluator cannot initialize action weights. If a physical controller later
produces successful, low-damage demonstrations with this actual gripper, they
can initialize a policy or provide a temporary imitation loss. Anneal that loss
as well; do not penalize departure from the demonstrated trajectory. Start with
easy target positions before adding this extra machinery.

## Detachment physics and its limits

The Fang2023 Hayward curve changes the physical tensile break threshold with
fruit–stem angle. A straight hanging fruit is 180 degrees. There is no angle
term in the task objective. Gripper contact, stability, tissue damage, reach,
time and energy can make the best robot motion differ from the paper fixture.

The mean curve is approximate, not a complete fracture model. See the
[evidence table](kiwi-material-evidence.md). Evaluate against withheld contact,
geometry and strength settings and eventually measured fruit trials. Otherwise
RL may exploit simulator errors, even with no imitation or guidance reward.

## Checks

```bash
python -m unittest discover -s tests -v
python scripts/check_detachment_angles.py
python scripts/check_detachment_angles.py --timestep .0005 \
  --output output/detachment-halfstep.json
python scripts/check_detachment_angles.py --video output/detachment-angles.mp4
python scripts/check_kiwi_physics.py
```

The angle fixture holds orientation and ramps axial load; it validates the
implemented break rule, not robot picking performance or real-fruit accuracy.
The pull stops on the physics substep of detachment. Gravity acts after
release; a speed check detects accidental continued pulling. Recordings show
4x slow motion with stems, measured loads and attachment state.
The evaluator tests use explicit observation sequences to exercise success,
alternative strategies, irreversible penalties and repeated-event protection.
