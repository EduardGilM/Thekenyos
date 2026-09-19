"""Watch a train_fast run: live HTML charts and periodic CPU progress videos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from treesim.kiwi_rl.training_monitor import (
    LiveDashboard, checkpoint_paths, due_checkpoints, due_latest_checkpoint,
    record_progress_video, serve_monitor, spawn_progress_video,
)


def record_due_videos(dashboard: LiveDashboard, *, every: int, steps: int, camera_every: int | None):
    spawned = []
    if every <= 0:
        return spawned
    for checkpoint in due_checkpoints(dashboard.run_dir, every):
        update = int(checkpoint.stem.split('-')[1])
        output = dashboard.video_path(update)
        if output.exists() or output.with_name(output.stem + '.partial.mp4').exists():
            continue
        info = checkpoint_paths(checkpoint)
        interval = info['camera_every'] if camera_every is None else camera_every
        print(json.dumps(dict(event='record_progress_video', update=update,
                              checkpoint=str(checkpoint), output=str(output))), flush=True)
        process = spawn_progress_video(checkpoint, output, steps=steps, camera_every=interval)
        if process is None:
            break
        spawned.append(dict(update=update, pid=process.pid, output=str(output)))
        break
    if spawned:
        return spawned
    latest = due_latest_checkpoint(dashboard.run_dir, every)
    if latest is None:
        return spawned
    info = checkpoint_paths(latest)
    update = int(info['completed_updates'])
    output = dashboard.video_path(update)
    if output.exists() or output.with_name(output.stem + '.partial.mp4').exists():
        return spawned
    interval = info['camera_every'] if camera_every is None else camera_every
    print(json.dumps(dict(event='record_progress_video', update=update,
                          checkpoint=str(latest), output=str(output))), flush=True)
    process = spawn_progress_video(latest, output, steps=steps, camera_every=interval)
    if process is None:
        return spawned
    spawned.append(dict(update=update, pid=process.pid, output=str(output)))
    return spawned


def watch(args):
    dashboard = LiveDashboard(args.run, hub=args.hub)
    payload = dashboard.refresh()
    print(json.dumps(dict(event='monitor_ready', run=str(dashboard.run_dir),
                          rows=payload['rows'], hub=str(args.hub) if args.hub else None)), flush=True)
    server = serve_monitor(Path(args.hub).resolve() if args.hub else dashboard.monitor_dir, args.http_port) if args.http_port else None
    if server is not None:
        print(json.dumps(dict(event='monitor_http', url=f'http://127.0.0.1:{args.http_port}/index.html')), flush=True)
    try:
        while True:
            dashboard.refresh()
            if args.video_every:
                record_due_videos(dashboard, every=args.video_every, steps=args.video_steps,
                                  camera_every=args.camera_every)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        if server is not None:
            server.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='train_fast output directory with training.jsonl')
    parser.add_argument('--hub', type=Path, help='Extra dashboard copy, e.g. /workspace/training/monitor')
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    parser.add_argument('--video-every', type=int, default=10)
    parser.add_argument('--video-steps', type=int, default=256)
    parser.add_argument('--camera-every', type=int)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--http-port', type=int)
    parser.add_argument('--record-checkpoint', type=Path)
    parser.add_argument('--video-output', type=Path)
    args = parser.parse_args()
    if args.record_checkpoint is not None:
        if args.video_output is None:
            parser.error('--video-output is required with --record-checkpoint')
        result = record_progress_video(args.record_checkpoint, args.video_output,
                                       steps=args.video_steps, camera_every=args.camera_every)
        print(json.dumps(result), flush=True)
        return
    if args.run is None:
        parser.error('--run is required unless --record-checkpoint is set')
    if not 0 <= args.video_every <= 10000 or not 8 <= args.video_steps <= 512:
        parser.error('Invalid video-every or video-steps')
    if not (0.5 <= args.poll_seconds <= 60):
        parser.error('poll-seconds must be in [0.5, 60]')
    watch(args)


if __name__ == '__main__':
    main()
