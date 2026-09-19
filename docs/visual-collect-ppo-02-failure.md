# visual-collect-ppo-02: why three-fruit camera collect did not learn

Stopped 19 Sep 2026 after 65,536 env steps (~48 min wall on dayone). This is a
learning-curriculum failure on the **assisted** camera task, not a physical
grasp, contact-only harvest, or damage-safety result. The 12 cm weld and
artificial stem/grip constraints were still in force.

Remote output: `/home/ubuntu/visual-collect-ppo-02`. Best zip (TCP gap only):
`policy-00057344.zip`. Source snapshot: `/home/ubuntu/Thekenyos-stationary-training-v1`.

## What was trained

PPO on `VisualKiwiEnv` with `--visual-lesson collect --start-phase pick --picks 3`.
Fruit hung at about 0.75 / 1.15 / 1.60 m chassis-forward. Actions were the 10-D
Gaussian walking/arm/wrist/jaw set. The standing 4-action grabber
(`stationary-kiwi-ppo-01/policy-00032768.zip`) contributed **encoder weights
only**. Body, wrist and deposit heads started random. Evaluations used
`guidance_weight=0`.

Success required three assisted picks, free release, full-ellipsoid basket
containment and 0.5 s settle. That never happened.

## Measured outcome

| Checkpoint | Eval successes | Failures (of 4) | Mean best TCP–fruit gap (m) | Mean net body travel (m) |
| --- | --- | --- | --- | --- |
| initial | 0 | 2 collision, 2 timeout | 0.383 | 0.534 |
| 8,192 | 0 | 1 collision, 3 timeout | 0.362 | 0.532 |
| 16,384 | 0 | 1 collision, 3 timeout | 0.383 | 0.237 |
| 24,576 | 0 | 2 collision, 2 timeout | 0.371 | 0.461 |
| 32,768 | 0 | 3 collision, 1 timeout | 0.383 | 0.359 |
| 40,960 | 0 | 3 collision, 1 timeout | 0.383 | 0.165 |
| 49,152 | 0 | 4 collision | 0.383 | 0.204 |
| 57,344 (best) | 0 | 1 collision, 3 timeout | 0.298 | 0.656 |

Train rollouts: **0 / 604** full collections (`exploratory_successes`). Fruit
contact time was **0.0 s** on every logged eval episode. Closest single look was
seed 10000 at 57,344 steps: **0.198 m**, still short of the **0.12 m** assist
radius. Mean train episode length peaked near 19 s, then fell to **10.8 s** at
65,536 steps as collisions returned. Held-out videos were not written: the job
was killed during the final eval.

These are recorded eval JSON / `progress.jsonl` / `progress.csv` values, not
inferences.

## Why it stalled

Ranked by how directly the logs support them.

1. **The lesson skipped the skills this policy did not have.** The standing
   encoder was trained with body and wrist commands zeroed and fruit 24–44 cm
   from the TCP. Collection needs walking ~0.75 m, aiming the wrist, then
   deposit. Copying the CNN does not copy those action heads. The 4-D
   categorical grabber cannot be loaded as this 10-D Gaussian policy.

2. **Almost no episode reached the fruit, so the encoder never saw its training
   distribution.** Until 57,344 steps the mean best TCP gap sat at the spawn
   value (~0.38 m). Capture, deposit and success bonuses never fired. With
   default `guidance_weight=1` during rollouts, the only dense term that can
   fire without a capture is distance progress, and that is drowned by
   `time=-0.015` per 0.1 s plus `-10` on `arm_basket_collision`.

3. **Basket collision is a hard stop that truncates exploration.** Mid-run
   evals went to 4/4 collisions in 3–7 s (49,152 steps) with mean path 0.49 m.
   The robot is then scoring the “do not hit the rear basket” failure, not
   “walk to the hanging fruit.” That is consistent with body travel shrinking
   from 0.53 m to 0.17 m between the 8k and 41k evals.

4. **65k steps is the wrong budget for this sparse objective.** Four workers,
   ~10–20 s mean episodes, 604 train episodes, zero captures. Standing grab
   learned at 24–32 cm because random close-and-hold plus the weld already
   worked. That sample-efficiency does not transfer to walk–aim–pick–carry–
   deposit ×3.

5. **The 57k “best” zip is not a collection policy.** The trainer ranks by
   success, then retained count, then TCP gap. Gap improved (0.30 m mean,
   0.20 m best seed) while successes stayed 0. That is a slightly closer miss,
   not a pick.

## What this does not show

- That the standing CNN is useless. Fruit never entered the close ToF/RGB
  regime it was trained on, except one 0.20 m look.
- That camera noise, latency or FOV blocked learning. Those were not isolated;
  the policy did not get a capture to ablate against.
- That contact-only grasping, basket physics or fruit damage failed. Those
  were not exercised.

## What to run instead

Keep the standing encoder. Do not jump it to three-fruit collect. The queued
order in `scripts/queue_visual_harvest.py` is: walking **grab** (same 10-D
actions, fruit still close), then **one** hanging fruit collect, then three
fruit, with a full-weight warm start only after held-out success. If a stage
has zero held-out successes, keep `--encoder-warm-start` rather than copying a
failed walking head.
