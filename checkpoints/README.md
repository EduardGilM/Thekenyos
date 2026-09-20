# Checkpoints and demo assets

| Path | What it is |
|---|---|
| `teacher-sequence-015/checkpoint-000800.pt` | Privileged arm teacher (PPO + GRU), sequence curriculum, randomized kiwi position. Approach, seated grasp and extraction. Used in the orchard demo. |
| `teacher-sequence-015/run.json` | Training configuration of that run. |
| `gait-relic-parity-002/g1-cpu.pt` | Pretrained RELIC locomotion (gait) policy, reused as-is for walking and station keeping. |
| `demo/r84_template.npy` | Leg/body proprioception template recorded on the fixed-base training scene; replayed while the robot stands so the arm policy sees in-distribution inputs. |
| `demo/demo-roll-800-report.json` | Event log of the demo run behind the timelapse video (3 of 6 kiwis in the basket, rolling friction on). |
| `scenes/sequence-near-001` | Fixed-base training scene (one kiwi, pergola). |
| `scenes/demo-multi-003` | Orchard demo scene with 6 kiwis. |
| `scenes/demo-multi-003-dense` | Same scene with dense visual foliage, used for rendering only. |

Reproduce the demo (paths relative to the repo root, GPU with MuJoCo Warp required):

```bash
python scripts/demo_orchard_harvest.py \
  --scene checkpoints/scenes/demo-multi-003 \
  --checkpoint checkpoints/teacher-sequence-015/checkpoint-000800.pt \
  --gait-checkpoint checkpoints/gait-relic-parity-002/g1-cpu.pt \
  --r84-template checkpoints/demo/r84_template.npy \
  --output output/demo --stand-distance .47 --deposit-trigger proprioceptive \
  --fruit-roll-friction .02 --detach-requires-hold
python scripts/render_demo_timelapse.py --scene checkpoints/scenes/demo-multi-003-dense \
  --replay output/demo --output output/demo-video --distance 2.8 --elevation -10 --azimuth-offset 120
```
