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
joints and one jaw, initially limited to 0.5 rad/s each. The environment enforces joint-target and effort limits and estimates absolute
actuator work at physics substeps. It evaluates the oracle each physics step,
so transient collision and load failures end the episode immediately.

Begin with a fixed chassis, legs in the standing pose and a reachable target.
The Gymnasium integration environment implements this fixture. The chassis is
fixed and leg joints are held by PD control; the legs are not welded. Later use
the standing controller and then permit locomotion.

The actor should receive joint state, target/TCP and basket relative poses,
contact information and previous action. Exact stem threshold, damage history
and attachment state belong to the evaluator or privileged critic; do not
silently expose them as deployable sensors. `SimHarvestObserver` reads simulator
state, actual native contact forces and full-ellipsoid containment. Its caller
selects the actual pad bodies and TCP offset. It is a measurement bridge. `SpotHarvestEnv` supplies the Gymnasium interface;
a policy optimizer is not implemented.

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


## Run the integration environment

Install `python -m pip install -e '.[rl]'` in the pinned simulation environment.
The optional extra adds Gymnasium 1.3.0; walking does not need it.

```python
import gymnasium as gym
import treesim.harvest_env  # Registers the environment.

env = gym.make('Thekenyos/SpotHarvest-v0', relic='../relic')
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

This is a **rigid-fruit integration surrogate**, not a calibrated substitute for
the native deformable bench. Do not start production-scale training or claim
material transfer from this implementation. It deliberately reuses the existing
fruit, stem, basket and robot physics instead of silently changing their models.

Each 50 Hz action updates bounded arm/jaw position targets at at most 0.5 rad/s.
PD effort caps remain active, with a provisional 0.3 N·m jaw cap. No locomotion
policy runs. Reset creates one fruit under a 1.6 m canopy in a small 2×2 fixture;
the full plantation default is unchanged. Initial fruit size, canopy geometry
and lateral position vary with the seed. The rear basket starts empty.

Observations are a dictionary of joint position/velocity/targets, fruit relative
position, quaternion, velocity and radii, TCP orientation, basket relative
position, jaw loads, previous action and remaining task time. Poses are ideal
simulator measurements for the initial curriculum, not validated perception.
Stem state, force threshold and damage appear only in diagnostic `info`.

The oracle runs at 1000 Hz (or 2000 with `physics_hz=2000`), including absolute
work integration and contact checks. Arm contact with non-target objects above
2 N summed load is a provisional failure threshold. Rewards aggregate substep
terms with the task discount; configure a future learner with
`gamma=task.gamma_per_second ** .02`. A task timeout is a finite-horizon failure
(`terminated=True`), not an external rollout truncation. Calls after termination
require reset. Safety criteria remain active when `guidance_weight=0`.

The registered environment declares GPU nondeterminism: exact resets are
checked, but step comparisons use numerical tolerances because parallel force
reductions can differ in the last bits. Reset rebuilds the solver to avoid stale
contact, damage or attachment state. Per-substep host observation copies and
model rebuilds make this a correctness-first, single-instance implementation;
batched throughput and fast reset remain future work.

```bash
python scripts/check_harvest_env.py --relic ../relic
python scripts/check_harvest_env.py --relic ../relic --physics-hz 2000 \
  --output output/harvest-env-halfstep.json
python scripts/check_harvest_env.py --relic ../relic \
  --video output/harvest-env-actions.mp4
```

The video contains scripted shoulder/wrist actions, not a trained pick. The
checks cover the Gymnasium API, seeded reset, action validation, fixed chassis,
kinematics, a physically forced loss, a one-substep overload, timeout and a
ground-drop and basket-settling fixtures. These fixtures initialize an already
detached fruit at known positions; it is not evidence of a successful robot transfer.
