"""Serve a read-only view of the live CTI teacher run."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

DEFAULT_RUN = 'teacher-reward-cti-001'
RUN = DEFAULT_RUN
REMOTE_ROOT = '/mnt/ssd/experiments/kiwi-pergola'
REMOTE_RUN = f'{REMOTE_ROOT}/training/runs/{RUN}'
WANDB_URL = 'https://wandb.ai/juampab/Thekenyos/runs/jh72ewg6'
POLL_SECONDS = 5
RENDER_INTERVAL = 180
REMOTE_TIMEOUT = 45
RENDER_TIMEOUT = 1800
CHECKPOINT_RE = re.compile(r'checkpoint-(\d{6})\.pt\Z')

_REMOTE_PROBE_TEMPLATE = r'''import json, os, pathlib, time
from collections import deque
root = pathlib.Path(__REMOTE_RUN__)
run = __RUN__
def read(name):
    try: return json.loads((root/name).read_text())
    except FileNotFoundError: return None
    except Exception: raise
try:
    rows=[]
    with (root/'training.jsonl').open() as f: lines=f.readlines()
    for index,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith(('\n','\r')): continue
            raise
    log_mtime=(root/'training.jsonl').stat().st_mtime
except FileNotFoundError:
    rows=[]; log_mtime=None
active=read('active-process.json') or {}
active=active if isinstance(active,dict) else {}
pid=active.get('pid',1545354)
alive=False
try:
    if isinstance(pid,int) and not isinstance(pid,bool) and pid > 0:
        os.kill(pid, 0)
        cmd=pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode(errors='replace')
        alive='train_harvest_fast.py' in cmd and run in cmd
except (OSError, FileNotFoundError): pass
def tail_jsonl(name, limit):
    try:
        with (root/name).open('rb') as f:
            lines=deque(f,maxlen=limit)
        records=[]
        for index,line in enumerate(lines):
            try: records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines)-1 and not line.endswith((b'\n',b'\r')): continue
                raise
        return records
    except FileNotFoundError: return []
events=tail_jsonl('phase-events.jsonl',200)
branches=tail_jsonl('cti-branches.jsonl',3)
phase=active.get('phase')
if not phase and events: phase=events[-1].get('phase')
if not phase and rows: phase='cti-v'+str(rows[-1]['cti/version']) if rows[-1].get('cti/version') in (2,3,4,5,6,7) else 'ppo'
status=active.get('status') if active.get('status') in ('paused','running') else None
print(json.dumps({'rows':rows,'report':read('report.json'),'failure':read('failure.json'),
                  'log_mtime':log_mtime,'process_alive':alive,'active_process':active,
                  'active_status':status,'phase':phase,'phase_events':events,'branch_diagnostics':branches}))
'''


def _remote_probe(run):
    remote_run = f'{REMOTE_ROOT}/training/runs/{run}'
    return (_REMOTE_PROBE_TEMPLATE.replace('__REMOTE_RUN__', repr(remote_run))
            .replace('__RUN__', repr(run)))


REMOTE_PROBE = _remote_probe(RUN)


def _configure_run(run, wandb_url=None):
    global RUN, REMOTE_RUN, WANDB_URL, REMOTE_PROBE
    RUN = run
    REMOTE_RUN = f'{REMOTE_ROOT}/training/runs/{RUN}'
    WANDB_URL = wandb_url if wandb_url is not None else (
        'https://wandb.ai/juampab/Thekenyos/runs/jh72ewg6' if RUN == DEFAULT_RUN else None)
    REMOTE_PROBE = _remote_probe(RUN)


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _row_score(row):
    success = _number(row.get('evaluation/success'))
    failure = _number(row.get('evaluation/physical_failure'))
    grasp = _number(row.get('evaluation/grasp'))
    distance = _number(row.get('evaluation/closest_distance_m'))
    # Missing metrics stay missing; do not let fabricated zeros rank a checkpoint.
    if any(value is None for value in (success, failure, grasp, distance)):
        return None
    graph=_number(row.get('evaluation/graph_score'))
    if graph is not None:
        return success, -failure, graph, _number(row.get('evaluation/held_detach')) or 0., grasp, -distance
    return success, -failure, grasp, -distance


def select_checkpoint(rows, report):
    """Return best evaluated checkpoint, including baseline step zero."""
    best_row = None
    best_score = None
    for row in rows:
        if _row_score(row) is None:
            continue
        step = row.get('step', row.get('update'))
        if not isinstance(step, int) or step < 0:
            continue
        score = _row_score(row)
        if best_score is None or score > best_score:
            best_row, best_score = row, score
    checkpoint = f'checkpoint-{int(best_row.get("step", best_row.get("update"))):06d}.pt' if best_row else None
    if isinstance(report, dict):
        reported = report.get('best_checkpoint') or report.get('teacher_checkpoint')
        if isinstance(reported, str):
            match = CHECKPOINT_RE.search(Path(reported).name)
            if match:
                checkpoint = Path(reported).name
    return checkpoint, best_row


def _status(snapshot):
    if snapshot.get('failure'):
        return 'failed'
    if snapshot.get('report'):
        return 'completed'
    if snapshot.get('process_alive'):
        return 'running'
    return 'stale'


def _phase(snapshot):
    """Choose the explicit phase, then event history, then CTI-v2 row marker."""
    phase = snapshot.get('phase')
    if isinstance(phase, str) and phase:
        return phase
    events = snapshot.get('phase_events')
    if isinstance(events, list):
        for event in reversed(events):
            if isinstance(event, dict) and isinstance(event.get('phase'), str):
                return event['phase']
    rows = snapshot.get('rows')
    if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
        if rows[-1].get('cti/version') in (2,3,4,5,6,7):
            return 'cti-v'+str(rows[-1]['cti/version'])
    return 'ppo'


def _decode_jsonl_tail(lines):
    """Decode complete newline-delimited JSON records from a deque tail."""
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith((b'\n', b'\r')):
                continue
            raise
    return records


class Dashboard:
    def __init__(self, cache: Path, no_render=False, run=RUN, wandb_url=None):
        self.cache = cache.expanduser().resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.no_render = no_render
        self.run = run
        self.remote_run = f'{REMOTE_ROOT}/training/runs/{run}'
        self.wandb_url = wandb_url if wandb_url is not None else (WANDB_URL if run == DEFAULT_RUN else None)
        self.lock = threading.Lock()
        self.state = dict(run=self.run, wandb_url=self.wandb_url, fetched_at=None, error=None,
                          status='stale', rows=[], video=None, render={'status': 'idle', 'error': None,
                          'checkpoint': None}, log_mtime=None, process_alive=False,
                          budget_seconds=3600, best_checkpoint=None, best_metrics=None,
                          last_video_checkpoint=None, elapsed_seconds=None,
                          phase='ppo', phase_events=[], branch_diagnostics=[], active_process=None,
                          active_status=None)
        self.last_render_checkpoint = None
        self.last_render_at = 0.0
        self.render_busy = False

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def _remote_snapshot(self):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'jp', 'python3', '-'],
                                input=_remote_probe(self.run), text=True, capture_output=True,
                                timeout=REMOTE_TIMEOUT, check=True)
        return json.loads(result.stdout)

    def poll(self):
        try:
            remote = self._remote_snapshot()
            rows = remote.get('rows') if isinstance(remote.get('rows'), list) else []
            report = remote.get('report')
            failure = remote.get('failure')
            checkpoint, best_row = select_checkpoint(rows, report)
            report_wandb_url = report.get('wandb_url') if isinstance(report, dict) else None
            wandb_url = self.wandb_url or (report_wandb_url if isinstance(report_wandb_url, str) else None)
            with self.lock:
                self.state.update(rows=rows, wandb_url=wandb_url, log_mtime=remote.get('log_mtime'),
                                  process_alive=bool(remote.get('process_alive')),
                                  active_process=remote.get('active_process'),
                                  active_status=remote.get('active_status'),
                                  phase=_phase(remote), phase_events=remote.get('phase_events') or [],
                                  branch_diagnostics=remote.get('branch_diagnostics') or [],
                                  fetched_at=time.time(), error=None,
                                  status=_status(dict(report=report, failure=failure,
                                                      process_alive=remote.get('process_alive'))),
                                  best_checkpoint=checkpoint, best_metrics=best_row,
                                  elapsed_seconds=(report.get('elapsed_seconds') if isinstance(report, dict)
                                                   else (rows[-1].get('elapsed_seconds') if rows else None)))
                if failure:
                    self.state['error'] = failure.get('error') if isinstance(failure, dict) else str(failure)
                if remote.get('active_status') == 'paused' and not failure and not report:
                    self.state['status'] = 'paused'
            if checkpoint and not self.no_render and checkpoint != self.last_render_checkpoint and \
                    time.monotonic() - self.last_render_at >= RENDER_INTERVAL:
                with self.lock:
                    if not self.render_busy:
                        self.render_busy = True
                        threading.Thread(target=self.render, args=(checkpoint,),
                                         name='dashboard-render', daemon=True).start()
        except Exception as exc:
            with self.lock:
                self.state.update(fetched_at=time.time(), error=f'{type(exc).__name__}: {exc}', status='stale')

    def render(self, checkpoint):
        match = CHECKPOINT_RE.fullmatch(checkpoint)
        if not match:
            return
        number = match.group(1)
        remote_base = f'{self.remote_run}/dashboard-videos/{Path(checkpoint).stem}'
        try:
            with self.lock:
                self.state['render'] = {'status': 'rendering', 'error': None, 'checkpoint': checkpoint}
            # The renderer refuses an existing output directory. Reuse a complete
            # artifact; otherwise render into a unique retry directory.
            probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'jp', 'python3', '-'],
                input=("import json,pathlib\np=pathlib.Path(" + repr(remote_base) + ")\n"
                       "m=json.loads(pathlib.Path(" + repr(f'{self.remote_run}/{checkpoint}.json') + ").read_text())\n"
                       "print(json.dumps({'complete':(p/'rollout.mp4').is_file() and (p/'report.json').is_file(), 'config':m.get('config',{})}))\n"),
                text=True, capture_output=True, timeout=REMOTE_TIMEOUT, check=True)
            artifact = json.loads(probe.stdout)
            complete = artifact.get('complete', False)
            render_config = artifact.get('config', {})
            render_source = render_config.get('source_dir', f'{REMOTE_ROOT}/training/releases/reward-cti-001')
            source_dir = remote_base
            if not complete:
                exists = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'jp', 'test', '-e', remote_base],
                                         capture_output=True, timeout=REMOTE_TIMEOUT).returncode == 0
                if exists:
                    source_dir = remote_base + f'-retry-{int(time.time())}'
                output = source_dir
                argv = [f'{REMOTE_ROOT}/training/envs/flex-gpu/bin/python',
                        f'{render_source}/scripts/render_fast_rollout.py',
                        '--role', 'teacher', '--scene', render_config.get('scene', f'{REMOTE_ROOT}/training/scenes/harvest-near-001'),
                        '--checkpoint', f'{self.remote_run}/{checkpoint}', '--gait-checkpoint',
                        render_config.get('gait_checkpoint', f'{REMOTE_ROOT}/training/runs/relic-parity-002/g1-cpu.pt'),
                        '--steps', '1500', '--output', output]
                script = ('import os,subprocess\n'
                          f'os.environ.update(WARP_CACHE_PATH={REMOTE_ROOT + "/training/cache/warp"!r}, '
                          f'TMPDIR={REMOTE_ROOT + "/tmp"!r}, MUJOCO_GL="egl")\n'
                          f'subprocess.run({argv!r},check=True)\n')
                subprocess.run(['ssh', '-o', 'BatchMode=yes', 'jp', 'python3', '-'],
                               input=script, capture_output=True, text=True,
                               timeout=RENDER_TIMEOUT, check=True)
            staging = Path(tempfile.mkdtemp(prefix=f'.{Path(checkpoint).stem}-', dir=self.cache))
            try:
                subprocess.run(['scp', '-q', '-o', 'BatchMode=yes',
                                f'jp:{source_dir}/rollout.mp4', f'jp:{source_dir}/report.json',
                                f'jp:{source_dir}/preview.png', str(staging) + '/'],
                               capture_output=True, timeout=REMOTE_TIMEOUT, check=True)
                report_path = staging / 'report.json'
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if not (staging / 'rollout.mp4').is_file() or report.get('checkpoint') and \
                        Path(report['checkpoint']).name != checkpoint:
                    raise RuntimeError('Rendered artifact does not match selected checkpoint')
                final_dir = self.cache / Path(checkpoint).stem
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                os.replace(staging, final_dir)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            created = time.time()
            video = dict(url=f'/media/{Path(checkpoint).stem}/rollout.mp4',
                         preview_url=f'/media/{Path(checkpoint).stem}/preview.png',
                         checkpoint=checkpoint, created_at=created, report=report)
            with self.lock:
                self.state['video'] = video
                self.state['last_video_checkpoint'] = checkpoint
                self.state['render'] = {'status': 'complete', 'error': None, 'checkpoint': checkpoint}
            self.last_render_checkpoint = checkpoint
            self.last_render_at = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.state['render'] = {'status': 'failed',
                                        'error': f'{type(exc).__name__}: {exc}',
                                        'checkpoint': checkpoint}
        finally:
            self.last_render_at = time.monotonic()
            with self.lock:
                self.render_busy = False

    def worker(self):
        while True:
            self.poll()
            time.sleep(POLL_SECONDS)


def handler_for(dashboard, html_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/state':
                body = json.dumps(dashboard.snapshot(), allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body)
            elif path == '/' or path == '/index.html':
                try: body = html_path.read_bytes()
                except OSError: body = b'Dashboard HTML file is not available yet.\n'
                self.send_response(200 if html_path.is_file() else 503)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body)
            elif path.startswith('/media/'):
                relative = unquote(path[len('/media/'):])
                candidate = (dashboard.cache / relative).resolve()
                try: candidate.relative_to(dashboard.cache)
                except ValueError:
                    self.send_error(404); return
                if candidate.name not in {'rollout.mp4', 'preview.png'} or not candidate.is_file():
                    self.send_error(404); return
                content_type = 'video/mp4' if candidate.suffix == '.mp4' else 'image/png'
                size = candidate.stat().st_size
                start, end, status = 0, size - 1, 200
                requested = self.headers.get('Range')
                if requested and requested.startswith('bytes=') and ',' not in requested:
                    left, _, right = requested[6:].partition('-')
                    try:
                        if left:
                            start = int(left); end = min(int(right), size - 1) if right else size - 1
                        else:
                            length = int(right); start = max(0, size - length)
                        if start < 0 or start > end or start >= size: raise ValueError
                        status = 206
                    except (ValueError, TypeError):
                        self.send_response(416); self.send_header('Content-Range', f'bytes */{size}')
                        self.end_headers(); return
                self.send_response(status); self.send_header('Content-Type', content_type)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end - start + 1)); self.send_header('Cache-Control', 'no-cache')
                if status == 206: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                with candidate.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
            else:
                self.send_error(404)

        def log_message(self, fmt, *args):
            return
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--cache', type=Path,
                        default=Path(tempfile.gettempdir()) / 'thekenyos-live-dashboard')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--run', default=RUN, help='remote training run directory')
    parser.add_argument('--wandb-url', help='W&B run URL shown in dashboard metadata')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    _configure_run(args.run, args.wandb_url)
    dashboard = Dashboard(args.cache, args.no_render, RUN, WANDB_URL)
    threading.Thread(target=dashboard.worker, name='dashboard-poller', daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                 handler_for(dashboard, Path(__file__).with_name('training_dashboard.html')))
    print(f'http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
