# Continuous progress and control-loss graph v5

Current profile: `graph-harvest/v5`. Continuous progress reward is restored:
`gamma * current_component - previous_component`, gamma .999, for all graph
components. Recovery can earn positive feedback again; discounted closed loops
remain nonpositive. No episode-peak reward restriction or blanket 2x backward
motion cost applies. Stalls retain partial physical progress; real drops and
invalid extraction still remove dependent credit.

Additional one-step control-loss costs: .25 for losing an enclosure sustained
for .1 seconds, .75 for losing secure grip, 1.5 for losing controlled carry,
and .5 for losing basket settling. Qualified basket release is exempt from
lost-grip/carry penalties. These are physical events, not learned-policy mastery
estimates. No explicit curriculum, stage practice or evaluation rollback.
Physical success still requires controlled extraction and two-second settling.

`--continue-from` starts a new recorded run from the exact actor, critic and
factual Adam state. It applies the requested learning rate (1e-4 for this
experiment), resets physical episodes, and records source provenance. It does
not reset the critic. An optional CTI arm uses a fresh independent auxiliary
Adam optimizer. PPO and CTI use identical reward and termination rules.

The matched comparison continues checkpoint 95 from teacher-graph-ppo-004.
Both arms have 30 additional minutes, the same evaluation schedule, scenes,
seed, policy, factual optimizer and learning rate. Run sequentially on the same
GPU; CTI uses its previous 20% wall-time budget. This is a one-seed experiment,
not definitive evidence of general benefit. Report both wall time and transitions.
Use `tests.test_control_graph`, archived reward checks, and `tests.test_branch_cti`
with `CTI_GPU_TEST=1 GRAPH_TEST_PROFILE=graph-harvest/v5` before launch.

## Archived weighted regression graph v4

The current profile is `graph-harvest/v4`, a PPO-only experiment with a random
harvesting actor and critic. The fixed gait is reused. CTI, stage practice,
explicit curriculum stages and evaluation-triggered checkpoint rollback are off.
All seven actions remain available at every step.

Physical grades and success prerequisites remain those of v3 below. An unheld
extraction never earns extraction, carry or success credit. It receives a one-time
penalty of 2, rather than immediately resetting the episode. Ground drops,
physical failures, inactivity and episode limits still end episodes. Valid
completion still requires controlled extraction and two seconds of basket settling.

For each additive graph component, reward is `weight * (new_peak_gain - 2*loss)`.
The episode peak starts at the initial component value. Only a new high earns
positive credit; recovering previously lost progress does not repay it. Loss is
the decrease from the previous physical component value, not a missing future
ability. Weights are position 1, grip 1.5, extraction/carry/deposit/completion 2.
This makes regression more expensive than equal progress and prevents profitable
repeated acquire/release cycles. Peaks reset per physical episode. These are
engineering weights, not calibrated guarantees of cross-update skill retention.

Unlike v3 potential shaping, earned partial progress is not erased by an artificial
terminal debit. This deliberately changes the objective to value partial competence.
Time costs .001/control step, stall .5, physical failure/drop 5, and valid success
pays 20. A fruit lost in reality still loses its dependent grade and incurs weighted
regression costs. Numerical corruption remains fail-fast. Timeouts still bootstrap.

Evaluation covers the same two scenes and seeded trials. It selects a best recorded
checkpoint for display only; it never changes active policy or optimizer state.
`acceptance/rollback` is always zero. Stage reward and physical completion charts
remain separate. `stage_gain/*` and `stage_loss/*` expose the weighted positive
and negative terms independently in W&B and the dashboard. Check `tests.test_soft_graph` plus archived graph tests, GPU
collector/optimizer checks and a full-size smoke run before launching.

## Archived prerequisite graph v3

The archived profile is `graph-harvest/v3`. This contract supersedes the v1/v2
partial-drop rewards below; archived checkpoints retain their original profile.
It is an engineering training objective, not a physical calibration claim.

Position, grip, extraction, carry, deposit and completion have physical
prerequisites. Position uses continuous collision-region distance and completes after 0.1 s
of safe enclosure. Grip requires
safe bilateral contact, enclosure, low slip and 0.1 s dwell. Extraction qualifies
only when secure grip holds on both observed sides of the detachment interval.
An unheld detachment terminates as a task failure; it does not count as extraction.
Loss of grip removes dependent carry credit. Recovery of a valid extraction is
possible, but recovering a prerequisite pays no repeatable event bonus.

A release is valid only following controlled extraction, from a secure grip at
most 4 cm from the computed release region above the basket. It retains transit
credit while falling into the basket. Physical basket containment/contact and two
continuous settling seconds are still required. Release in transit, a ground
drop, and physical success reached without the prerequisite chain earn no success.

Valid state grades are: continuous position 0–2; enclosure/jaw fit 2–3;
secure attached grip/extraction load 4–5; held extraction/carry 6–8;
qualified release 8; basket settling 10–11; completion 12. The gaps reserve credit
for completed prerequisites. A backward stage transition additionally costs
0.5 per lost stage. This is bounded and does not prescribe motor actions.
Shaping remains `gamma*next_grade-previous_grade`, gamma .999, but every true
terminal has zero next potential. Truncations retain final-value bootstrap.
Thus a failed or deliberately shortened episode cannot retain partial shaping
credit. Time costs .001/step, task failure costs 5, stall costs .5, and valid
completion pays 20. Numeric corruption remains fail-fast, not training data.

## Practice and acceptance

At most one quarter of reset slots use an in-memory bank of up to eight
physically reached states per boundary: near enclosure, enclosed, secure grip,
held extraction, and release-ready. Physics/controller buffers and prerequisite
history use the existing world snapshot contract. New practice episodes reset
policy memory and episode timers; they do not splice stale hidden states into
current policies. No assisted motion or invented grasp pose is used. The bank is
rebuilt after process restart. Full-start episode metrics exclude practice starts.

Every candidate evaluation covers the training scene and distinct held-out
scene, each with one deterministic trial and two seeded stochastic trials.
These are robustness checks over two scenes, not broad orchard generalization.
The accepted policy may not lose more than 5 percentage points in any established
stage completion rate, increase physical failure or combined task-failure rates by more
than 5 points, or worsen closest approach by more than 2 cm, in either scene.
The combined failure gate permits replacing an invalid extraction with a valid
extraction followed by a drop while learning retention; it does not promote the
drop itself. These are explicit engineering tolerances, not statistical confidence bounds.
Protected rates use the best accepted historical values, preventing a sequence
of individually small regressions from erasing a skill. Among candidates that pass, rank physical completion, deposition, carry,
controlled extraction, grip and positioning before failure rates and distance.
Promotion additionally requires at least a 5-point outcome-rate gain or a 1 cm
approach improvement; insignificant score changes do not promote candidates.
A candidate with a material regression restores the accepted actor, critic and
both optimizers, resets live episodes, and clears stale CTI roots. A candidate
without improvement continues training but is not promoted. `accepted.json`
records the selected checkpoint; rejected candidate files remain inspectable.

New reward profiles warm-start actor weights but reset the critic and optimizer
states. Checkpoint metadata names the source checkpoint. A process resume under
the same reward profile restores both optimizers and the accepted checkpoint.

## Diagnostics and checks

W&B and the local dashboard report full-episode stage entries, completion rates,
completion conditional on entry, time per stage, and regressions by prerequisite.
Practice outcomes have a separate namespace. Candidate evaluation, accepted
skills, promotions and rollbacks are separate; the latest video may be a rejected
candidate. Additive stage shaping includes terminal debits; completion reward is
reported separately. PPO and CTI share graph state, rewards and termination.

Run `tests.test_prerequisite_graph`, the archived reward tests, and the GPU
practice/replay test with `GRAPH_GPU_TEST=1`, `GRAPH_CHECKPOINT`, `FAST_SCENE`
and `GAIT_CHECKPOINT`. Also run the existing CTI learning/replay checks,
`scripts/check_graph_training.py`, and a full-size launch smoke check.

## Archived continuous harvesting graph v2


The current profile is `graph-harvest/v2` with CTI v8. It supersedes the v1
approach radius and stall rules below; v1 checkpoints retain their old scoring.

- Position grade is `2 / (1 + jaw_region_error_m / .15)`. The error measures
  distance of the fruit core to the valid jaw opening, including aperture
  deficiency. There is no 0.5 m plateau or hard distance cutoff.
- Safe enclosure adds bounded jaw-fit and sustained-contact credit. Secure
  grip, safe extraction, retained carry and two-second settling remain physical
  outcomes, never prescribed actions. Carry distance also uses smooth decay.
- Before reward history or policy memory starts, each runtime settles for four
  seconds under fixed gait and zero harvesting increments. Reset restores the
  settled pose, velocity, controller targets and gait history in each world.
  Damage, detachment or robot failure during settling blocks training.
- A 0.005 score improvement accumulates relative to the last progress reference.
  Regression lowers the reference without resetting the timer, allowing real
  recovery to earn time. The run uses an eight-second inactivity limit and a
  thirty-second hard episode limit. Repeated motion earns no repeatable bonus.
- CTI collects only roots with at least six seconds remaining before both
  inactivity and hard limits. It preserves timers and factual replay; it does
  not silently give alternatives different termination rules.
- PPO and CTI both regularize entropy after tanh. CTI weights this term by
  clipped importance ratios. All valid branches retain V-trace actor/critic
  learning, factual KL rollback and numerical checks.

W&B records stage_reward/* and stage_progress/* for position, grip, extract,
carry, deposit and completion. The additive progress bands sum to shaping;
completion reward is separate. Episode metrics record stage reach, duration,
forward/backward transitions and termination reasons. Joint-named metrics
record bounded command mean/spread/saturation, measured speed and target error.
CTI reports physical grade improvement over the factual branch by source stage.
The local dashboard shows both the latest and best evaluated videos.

Validation: `scripts/check_graph_training.py` checks settled drift, physical
command sensitivity and isolated world resets. The GPU branch test checks
settled PPO roots, factual replay, rejected corrupt evidence, actor and critic
learning. These checks establish a usable learning path, not harvest competence.

## Archived v1 contract


This profile implements the user's six-stage task. It is a training objective
and evaluator, not a scripted action controller. The privileged teacher still
chooses all seven controls from physical observations and recurrent memory.
A future sensor-only student must infer task state from sensors and memory.

## Required end state

A new run starts with random policy weights and fresh PPO and CTI optimizers.
The previous run is preserved. The fixed pretrained gait is reused; the harvesting actor and critic start from zero. One full success means the detached fruit was
released, remained contained and in contact with the basket, and stayed below
the physical settling-speed threshold for two continuous seconds.

The graph is reversible where physics is reversible: moving away, leaving the
jaw space, opening a loose grasp and slipping reduce progress. Detachment is
irreversible within an episode. An intact ground drop is a partial terminal
outcome, never a harvest and never automatically fruit damage.

| State | Requirement and progress | Score band |
| --- | --- | --- |
| Approach | Distance credit saturates at exactly 0.5 m and falls outside it | 0–1 |
| Enclose | Fruit central core enters the actual jaw support window | 1–2 |
| Secure grasp | Enclosure, opposing loaded contact, safe load, low slip and dwell | 2–3 |
| Extract | Maintain secure grasp while increasing safe extraction load | 3–3.5 |
| Detached, uncontrolled | Intact detachment is retained, including a ground drop | 3.6 |
| Carry | Detached and securely held; approach a valid position above the basket | 4–5 |
| Deposit | Release into the basket and maintain physical settling conditions | 5–6 |
| Complete | Two seconds of continuous settling; physical success | 6 plus success reward |

Approach starts saturated in the current near-fruit scene, whose initial
hand-to-fruit distance is already less than 0.5 m. Inside that radius, closer
TCP distance alone cannot earn additional approach reward. Insertion progress
must use jaw geometry. Empty jaw closure must not earn grasp credit, and
smaller jaw separation is not rewarded below a safe fruit-sized gap.

## Reward accounting

The graph contribution is `gamma * current_score - previous_score`, with
`gamma = .999`. Time costs .001 per control step. Full harvest earns 20;
physical damage or robot failure costs 5; stalling costs .5. Numerical failure
remains a hard validity failure. Ordinary stalls, timeouts and intact ground
drops preserve their partial terminal score; they do not erase all preceding
progress. This intentionally optimizes partial progress as well as completion.
It is not a claim of policy-invariant reward shaping.

History is used for hysteresis, contact dwell, settling and irreversibility.
It must be reset only for the appropriate episode and fully restored in
counterfactual replay. Moving out and back must not create a positive discounted
reward cycle. Stage changes must not award repeatable event bonuses.

## CTI integration

PPO and CTI use the same graph contribution, task rewards and physical outcome
classification. CTI compares recorded PPO decision states with their frozen
collection policy and sampled alternatives. It must not substitute the older
reach/grasp potential, add an unrelated leaf bonus or reject every unheld
fruit. An intact dropped fruit may beat an attached baseline; retained control
and basket settling rank higher.

All seven controls remain freely sampled. CTI v7 replicates each actual PPO root
into a factual world plus 12 alternatives, processed together on the GPU. Three
search passes refine coherent Gaussian proposal biases using the four best
returns per root while preserving broad mutations. Every step draws fresh
Gaussian action noise. Alternatives share fresh noise with an unbiased policy
resample; the separate factual lane replays recorded actions for validation.
Refinement passes draw new noise so conditioning on earlier search outcomes does
not invalidate the logged Gaussian proposal densities.
The actual conditional proposal density is logged; no stage prescribes an action.

Every valid branch transition trains a separate actor/critic V-trace update,
including failed, worse and unchanged branches. There is no winning-branch or
harvest-success gate. Replay mismatches reject all descendants of that root;
numerical corruption remains fail-fast. Physical damage is valid negative
experience. Search uses exactly the graph's discounted reward, without a second
leaf bonus; the learner bootstraps nonterminal branch endings from the current
critic and uses clipped importance ratios to correct exploration bias.

This uses the [IMPALA V-trace equations](https://proceedings.mlr.press/v80/espeholt18a.html).
The recurrent approximation evaluates stored, detached collection-policy memories
instead of rebuilding the entire current-policy history. Independent CTI Adam
state is checkpointed. A factual Gaussian KL limit of .01 constrains changes;
backtracking restores both policy and optimizer before retrying a smaller step.
Only accepted optimizer work counts as learned transitions in the dashboard.
The value and policy updates remain separate from factual PPO buffers.

Enclosure uses a quarter-radius central core and 8 mm tolerance against transformed
collision mesh support bounds. This is an explicit coarse geometric criterion,
validated against a recorded bilateral grasp; it is not full mesh containment.

## Acceptance checks

- Exact approach threshold and regressions; no closer-distance bonus inside it.
- Geometry-aware enclosure, including rotation and empty closed-jaw cases.
- Secure contact, slip/opening regressions and overload rejection.
- Intact ground drop ranks above attached grasp, below secure held detachment.
- Carry destination is above the basket; arbitrary release is not deposition.
- Continuous two-second settling and timer reset when conditions cease.
- No positive discounted cycles; world-local reset and graph-state replay.
- PPO and CTI use identical scores and terminal outcome rules.
- Invalid replay is rejected; generic action exploration remains unconstrained.
- Training, evaluation and rendering use the same physical task profile.

Legacy profiles retain their previous ground-contact and settling behavior.
The graph uses the existing approximate rigid-fruit physics and engineering
load limits; it does not establish calibrated biological damage or hardware
performance. Physical task outcomes, not graph score alone, determine harvest
success and readiness for sensor-only distillation.
