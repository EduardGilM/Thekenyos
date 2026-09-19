# Continuous harvesting graph v2

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
