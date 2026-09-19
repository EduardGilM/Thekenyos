# Wrist-camera assisted RL on Vast (Sep 2026)

These notes are from the wrist-camera standing-grab work on Vast
`ssh -p 13280 root@79.160.189.79`. They are a training log, not a physical
grasp, contact-only harvest, or damage-safety result. The 12 cm assist weld
and artificial stem/grip constraints were in force. The instance overlay disk
is **not** a volume: copy run directories off before destroy/recycle.

Pinned stack on that host: Python 3.12 in `/venv/kiwi-pergola`, MuJoCo 3.8.1,
Warp 1.14.0, Newton 1.3.0, torch 2.10.0+cu126, SB3 2.9.0. RELIC
`27f8033c` at `/workspace/relic`. `MUJOCO_GL=egl`. Do not independently
upgrade those packages.

## Actor contract (this branch)

- Wrist only. Named MuJoCo cameras `ee_cam` / `ee_depth` on
  `spot_with_arm_arm_link_wr1`. No chassis head camera.
- Boston Dynamics gripper **vertical** FOV: RGB 46.4°, depth 44°.
  https://dev.bostondynamics.com/docs/concepts/arm/arm_specification
- Offsets from commented RELIC URDF frames (`hand_color_sensor`,
  `hand_depth_sensor`).
- Actor: 4-frame RGB-D, 22-D proprioception, phase, `has_grasped` /
  `has_placed`, ages. No fruit XYZ, IDs or segmentation.
- Critic: privileged 99-D basket state plus proprioception.
- ToF valid on the optical axis in **[0.15, 3) m**.
- Evaluations keep cameras on. Stationary training uses `view_weight=1` and
  `guidance_weight=0`; evaluation sets both to 0.

## Standing lesson (`--stationary`)

Arm-only 4-action MultiCategorical PPO. Body and wrist-rotation commands are
zero. Stage 0 (S0v) places fruit in the wrist depth FOV at 20–40 cm and
scores pregrasp. Stages 1–3 are off-axis 24/32/44 cm assisted holds. Do not
load a 10-action walking checkpoint into this contract, or the reverse.

```bash
MUJOCO_GL=egl python scripts/train_assisted_kiwi.py --relic /workspace/relic \
  --stationary --output /workspace/NEW_DIR --steps 32768 --num-envs 4 \
  --rollout-steps 128 --batch-size 128 --learning-rate .0001 --entropy .002 \
  --eval-every 4096 --eval-episodes 8 --heldout-seed 86000 \
  --policy-device cuda --record-video
```

Do not overwrite `stationary-kiwi-ppo-01` (archived on-axis) or
`Thekenyos-stationary-training-v1`.

## Basket deposit is a later lesson

Standing grab cannot walk or score rear-basket settle. Collection is
`--vision --visual-lesson collect` after a **walking** 10-action grab. Queue
with `scripts/queue_visual_harvest.py`. Three-fruit collect must not start
unless one-fruit held-out actually succeeded. Jumping from a standing encoder
to three hanging fruit failed; see
[visual-collect-ppo-02-failure.md](visual-collect-ppo-02-failure.md).

## Vast runs from this chat

### `stationary-kiwi-ppo-offaxis-01` (finished, old cameras)

Head+hand 6-channel CNN, off-axis fruit, cameras-on eval after blackout was
removed. Recommended `policy-00028672.zip`: 3/8 greedy, 2/8 held-out, stayed
on stage 0. **Cannot load** into the wrist-only observation space.

### `stationary-kiwi-ppo-s0v-01` (finished, wrist S0v)

PID 15774, RTX 4090, 32,768 steps, 20.6 min wall, ~28 samples/s. New
asymmetric wrist policy. Local copies of the held-out clips:
`output/stationary-kiwi-ppo-s0v-01/` (gitignored).

| Check | Result |
| --- | --- |
| Untrained greedy (seeds 10000–10007) | 7/8 S0v pregrasps |
| Greedy evals at 4k–32k steps | 0/8 timeouts every time |
| Recommended checkpoint | `initial-policy.zip` |
| Held-out (same initial policy) | 6/8 |
| Stochastic train episodes | 555/984 (~56%) |
| Final stage | 0 (not promoted) |

The untrained policy already pregrasped because fruit starts in the wrist FOV.
The later greedy policy froze (arm motion ~0.8 cm on several evals) while
random exploration kept succeeding. Treat `initial-policy.zip` as a random
close-fruit baseline, not a trained look-and-reach skill.

Held-out videos (untrained greedy, assisted pregrasp):

- `heldout-best-stage-0-seed-86001.mp4` / `86002` / `86003`: pregrasp
- `heldout-best-stage-0.mp4` (seed 86000): timeout

### Walking-grab queue (cancelled)

`visual-harvest-queue-01` started `visual-walk-grab-03` with
`--encoder-warm-start` from `stationary-kiwi-ppo-s0v-01/initial-policy.zip`.
Untrained walking eval 0/8. User stopped all trainers before collect. Partial
`/workspace/visual-walk-grab-03` is not a finished run.

## Checks

```bash
MUJOCO_GL=egl ASSISTED_KIWI_RELIC=../relic python -B -m unittest tests.test_visual_kiwi_env -v
MUJOCO_GL=egl ASSISTED_KIWI_RELIC=../relic python -B -m unittest tests.test_stationary_kiwi_env -v
```
