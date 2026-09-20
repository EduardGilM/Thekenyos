#!/usr/bin/env python
"""Deterministic camera-policy replay with sensor-inset video. Not training."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_assisted_kiwi import evaluate, emit


def main():
    parser = argparse.ArgumentParser(description='Replay an assisted/visual checkpoint for local review')
    parser.add_argument('--relic', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(84000, 84004)))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--video-limit', type=int, default=4)
    parser.add_argument('--video-fps', type=int, default=10)
    args = parser.parse_args()
    saved = json.loads(args.checkpoint.with_suffix('.json').read_text())
    if min(args.seeds) < 0 or args.video_limit < 1 or args.video_fps not in range(5, 13):
        parser.error('Seeds must be nonnegative; video limit at least 1; video fps 5-12')
    args.output.mkdir(parents=True, exist_ok=False)
    from treesim.basket_kiwi_env import BasketTask
    from treesim.visual_kiwi_env import VisualKiwiEnv, VisionConfig
    from stable_baselines3 import PPO
    task = BasketTask(**saved['task'])
    vision_config = dict(saved['vision']) if saved.get('vision') else None
    if vision_config and 'cameras' in vision_config:
        vision_config['cameras'] = tuple(vision_config['cameras'])
    vision = VisionConfig(**vision_config) if vision_config else None
    if vision is None:
        parser.error('Replay is for camera checkpoints')
    env_class = VisualKiwiEnv
    if saved.get('stationary'):
        from treesim.stationary_kiwi_env import StationaryKiwiEnv
        env_class = StationaryKiwiEnv
    search_spawn = bool((saved.get('training_config') or {}).get('search_rewards'))
    env = env_class(args.relic, task=replace(task), vision=vision, guidance_weight=0.,
                    view_weight=0., search_rewards=False, search_spawn=search_spawn,
                    render_mode='rgb_array')
    env.set_stage(saved.get('stage', 0))
    model = PPO.load(args.checkpoint, device=args.device)
    label = f'inference-stage-{env.stage}'
    report = evaluate(model, env, args.seeds, args.output, label, True, record_video=True,
                      video_limit=args.video_limit, video_fps=args.video_fps)
    summary = dict(event='replay', checkpoint=str(args.checkpoint.resolve()),
                   stationary=bool(saved.get('stationary')),
                   **{k: report[k] for k in report if k not in ('records',)})
    emit(args.output/'progress.jsonl', {k: summary[k] for k in summary})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    env.close()


if __name__ == '__main__':
    main()
