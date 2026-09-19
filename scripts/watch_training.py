"""Watch a train_fast run: live HTML charts and periodic CPU progress videos."""

from __future__ import annotations

import argparse
import json
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from treesim.kiwi_rl.training_monitor import (
    LiveDashboard, checkpoint_paths, due_checkpoints, record_progress_video,
)


def serve_monitor(directory: Path, port: int) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def record_due_videos(dashboard: LiveDashboard, *, every: int, steps: int, camera_every: int | None):
    recorded = []
    if every <= 0:
        return recorded
    for checkpoint in due_checkpoints(dashboard.run_dir, every):
        update = int(checkpoint.stem.split('-')[1])
        output = dashboard.video_path(update)
        if output.exists():
            continue
        info = checkpoint_paths(checkpoint)
        interval = info['camera_every'] if camera_every is None else camera_every
        print(json.dumps(dict(event='record_progress_video', update=update,
                              checkpoint=str(checkpoint), output=str(output))), flush=True)
        result = record_progress_video(checkpoint, output, steps=steps, camera_every=interval)
        if result.get('skipped'):
            break
        recorded.append(result)
        dashboard.refresh()
    return recorded


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
    parser.add_argument('--video-steps', type=int, default=64)
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
    if not 0 <= args.video_every <= 10000 or not 8 <= args.video_steps <= 256:
        parser.error('Invalid video-every or video-steps')
    if not (0.5 <= args.poll_seconds <= 60):
        parser.error('poll-seconds must be in [0.5, 60]')
    watch(args)


if __name__ == '__main__':
    main()
