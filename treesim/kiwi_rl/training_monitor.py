"""Local training dashboard and CPU progress videos.

Charts come from ``training.jsonl`` so a sidecar can update them while a GPU
job holds the 4090. Progress clips are a CPU-native MuJoCo preview of the
latest checkpoint: third-person viewer plus the gripper RGB the policy sees.
They are not harvest demonstrations and they do not use the training GPU
runtime.
"""

from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = 'training-monitor/v1'
VIDEO_SCHEMA = 'progress-video/v1'
PRIORITY_CHARTS = (
    'loss', 'kl', 'entropy', 'logstd_mean', 'grad_norm', 'reward_mean', 'reward_std',
    'distance_mean_closest_m', 'distance_final_m', 'distance_closest_m',
    'evaluation/mean_closest_distance_m', 'evaluation/final_distance_m',
    'evaluation/closest_distance_m', 'evaluation/harvest_successes',
    'evaluation/success_rate', 'evaluation/detach_rate', 'evaluation/grasp_rate',
    'harvest_successes', 'grasp_events', 'detach_events',
    'evaluation/terminal_transitions', 'terminal_transitions',
    'curriculum_index', 'guidance_weight',
    'training_transitions_per_second', 'rollout_transitions_per_second',
    'torch_peak_allocated_gb', 'rollout_seconds', 'update_seconds',
)
SKIP_CHARTS = {'step', 'update', 'minibatches', 'optimized_transitions', 'transitions'}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load metric rows, ignoring a truncated final line while a writer appends."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get('step'), int) and not isinstance(row['step'], bool):
            rows.append(row)
    return rows


def is_chartable(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, (bytes, str, dict, list)):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def series_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        step = int(row['step'])
        for key, value in row.items():
            if key in SKIP_CHARTS or not is_chartable(value):
                continue
            bucket = series.setdefault(key, {'steps': [], 'values': []})
            bucket['steps'].append(step)
            bucket['values'].append(float(value))
    return series


def due_checkpoints(run_dir: str | Path, every: int) -> list[Path]:
    if not isinstance(every, int) or every < 0:
        raise ValueError('video-every must be a non-negative integer')
    if every == 0:
        return []
    run_dir = Path(run_dir)
    due = []
    for path in run_dir.glob('checkpoint-*.pt'):
        try:
            update = int(path.stem.split('-')[1])
        except (IndexError, ValueError):
            continue
        if update % every == 0:
            due.append((update, path))
    return [path for _, path in sorted(due)]


def compose_progress_frame(scene_rgb, gripper_rgb, *, overlay_px: int = 128, inset: int = 8):
    """Place a nearest-neighbour gripper view on a third-person frame."""
    import numpy as np
    frame = np.asarray(scene_rgb, dtype=np.uint8).copy()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('scene frame must be an HxWx3 uint8 image')
    gripper = np.asarray(gripper_rgb, dtype=np.uint8)
    if gripper.ndim != 3 or gripper.shape[2] != 3:
        raise ValueError('gripper frame must be an HxWx3 uint8 image')
    if overlay_px < 16 or inset < 1 or frame.shape[0] < overlay_px + 2 * inset or frame.shape[1] < overlay_px + 2 * inset:
        raise ValueError('overlay does not fit on the scene frame')
    gy, gx = gripper.shape[:2]
    ys = (np.arange(overlay_px) * gy / overlay_px).astype(int)
    xs = (np.arange(overlay_px) * gx / overlay_px).astype(int)
    small = gripper[ys][:, xs]
    top, left = inset, frame.shape[1] - inset - overlay_px
    frame[top:top + overlay_px, left:left + overlay_px] = small
    frame[top - 1:top + overlay_px + 1, left - 1] = (250, 210, 50)
    frame[top - 1:top + overlay_px + 1, left + overlay_px] = (250, 210, 50)
    frame[top - 1, left - 1:left + overlay_px + 1] = (250, 210, 50)
    frame[top + overlay_px, left - 1:left + overlay_px + 1] = (250, 210, 50)
    return frame


def svg_chart(steps: Sequence[float], values: Sequence[float], title: str,
              *, width: int = 720, height: int = 220) -> str:
    title = html.escape(title)
    if not steps:
        return (f'<svg viewBox="0 0 {width} {height}" class="chart">'
                f'<text x="20" y="28" fill="#9aa">{title}: sin datos</text></svg>')
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 64, width - 12, 32, height - 28
    span = max(steps[-1] - steps[0], 1)

    def x_of(step: float) -> float:
        return left + (float(step) - steps[0]) / span * (right - left)

    def y_of(value: float) -> float:
        return bottom - (float(value) - lo) / (hi - lo) * (bottom - top)

    points = ' '.join(f'{x_of(step):.1f},{y_of(value):.1f}' for step, value in zip(steps, values))
    y_ticks = [(lo, y_of(lo)), ((lo + hi) / 2, y_of((lo + hi) / 2)), (hi, y_of(hi))]
    ticks = ''.join(
        f'<text x="8" y="{y + 4:.1f}" fill="#8a93a6" font-size="11">{html.escape(f"{value:.4g}")}</text>'
        for value, y in y_ticks)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{title}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#171a21" rx="10"/>'
        f'<text x="16" y="22" fill="#f2f5ff" font-size="13">{title}</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#2c3344"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#2c3344"/>'
        f'{ticks}'
        f'<text x="{left}" y="{height - 8}" fill="#8a93a6" font-size="11">{html.escape(str(steps[0]))}</text>'
        f'<text x="{right - 24}" y="{height - 8}" fill="#8a93a6" font-size="11">{html.escape(str(steps[-1]))}</text>'
        f'<polyline fill="none" stroke="#7dd3a0" stroke-width="2" points="{points}"/>'
        f'<circle cx="{x_of(steps[-1]):.1f}" cy="{y_of(values[-1]):.1f}" r="3.5" fill="#f4d35e"/>'
        f'</svg>'
    )


def list_progress_videos(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    videos = []
    if not directory.exists():
        return videos
    for path in sorted(directory.glob('progress-*.mp4')):
        try:
            step = int(path.stem.split('-')[1])
        except (IndexError, ValueError):
            continue
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        videos.append(dict(step=step, file=path.name, url=f'videos/{path.name}',
                           label=meta.get('label', 'CPU progress preview'),
                           mean_tcp_fruit_distance_m=meta.get('mean_tcp_fruit_distance_m'),
                           min_tcp_fruit_distance_m=meta.get('min_tcp_fruit_distance_m')))
    return videos


def _latest_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(rows[-1]) if rows else {}


def latest_progress_video(videos: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    return dict(videos[-1]) if videos else None


def _video_figure(video: Mapping[str, Any]) -> str:
    distance = video.get('min_tcp_fruit_distance_m')
    extra = f" · min TCP-fruta {float(distance):.3f} m" if is_chartable(distance) else ''
    return (
        f'<figure><figcaption>update {html.escape(str(video.get("step")))}'
        f'{html.escape(extra)} · {html.escape(str(video.get("label", "")))}</figcaption>'
        f'<video controls preload="metadata" src="{html.escape(str(video.get("url")))}"></video></figure>'
    )


_DASHBOARD_SCRIPT = r'''
<script>
const CARD_KEYS = ["step", "curriculum_index", "loss", "entropy", "reward_mean",
  "evaluation/success_rate", "evaluation/mean_closest_distance_m",
  "evaluation/harvest_successes", "training_transitions_per_second",
  "torch_peak_allocated_gb"];
const PRIORITY = ["loss", "kl", "entropy", "logstd_mean", "grad_norm", "reward_mean", "reward_std",
  "distance_mean_closest_m", "distance_final_m", "distance_closest_m",
  "evaluation/mean_closest_distance_m", "evaluation/final_distance_m",
  "evaluation/closest_distance_m", "evaluation/harvest_successes",
  "evaluation/success_rate", "evaluation/detach_rate", "evaluation/grasp_rate",
  "harvest_successes", "grasp_events", "detach_events",
  "evaluation/terminal_transitions", "terminal_transitions",
  "curriculum_index", "guidance_weight",
  "training_transitions_per_second", "rollout_transitions_per_second",
  "torch_peak_allocated_gb", "rollout_seconds", "update_seconds"];
const SKIP = new Set(["step", "update", "minibatches", "optimized_transitions", "transitions"]);

function esc(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}
function chartable(value) {
  return typeof value === "number" && Number.isFinite(value);
}
function fmt(value) {
  return Number(value).toPrecision(5).replace(/\.?0+$/, "").replace(/\.$/, "");
}
function svgChart(steps, values, title) {
  const width = 720, height = 220;
  if (!steps.length) {
    return `<svg viewBox="0 0 ${width} ${height}" class="chart"><text x="20" y="28" fill="#9aa">${esc(title)}: sin datos</text></svg>`;
  }
  let lo = Math.min(...values), hi = Math.max(...values);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = 0.08 * (hi - lo);
  lo -= pad; hi += pad;
  const left = 64, right = width - 12, top = 32, bottom = height - 28;
  const span = Math.max(steps[steps.length - 1] - steps[0], 1);
  const xOf = step => left + (step - steps[0]) / span * (right - left);
  const yOf = value => bottom - (value - lo) / (hi - lo) * (bottom - top);
  const points = steps.map((step, i) => `${xOf(step).toFixed(1)},${yOf(values[i]).toFixed(1)}`).join(" ");
  const ticks = [lo, (lo + hi) / 2, hi].map(value =>
    `<text x="8" y="${yOf(value) + 4}" fill="#8a93a6" font-size="11">${esc(fmt(value))}</text>`).join("");
  const lastX = xOf(steps[steps.length - 1]), lastY = yOf(values[values.length - 1]);
  return `<svg viewBox="0 0 ${width} ${height}" class="chart" role="img" aria-label="${esc(title)}">
    <rect x="0" y="0" width="${width}" height="${height}" fill="#171a21" rx="10"/>
    <text x="16" y="22" fill="#f2f5ff" font-size="13">${esc(title)}</text>
    <line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#2c3344"/>
    <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#2c3344"/>
    ${ticks}
    <text x="${left}" y="${height - 8}" fill="#8a93a6" font-size="11">${esc(steps[0])}</text>
    <text x="${right - 24}" y="${height - 8}" fill="#8a93a6" font-size="11">${esc(steps[steps.length - 1])}</text>
    <polyline fill="none" stroke="#7dd3a0" stroke-width="2" points="${points}"/>
    <circle cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="3.5" fill="#f4d35e"/>
  </svg>`;
}
function renderCards(latest) {
  const cards = CARD_KEYS.filter(key => chartable(latest[key])).map(key =>
    `<div class="card"><div class="k">${esc(key)}</div><div class="v">${esc(fmt(latest[key]))}</div></div>`);
  return cards.join("") || '<div class="card">Esperando training.jsonl</div>';
}
function renderCharts(series) {
  const names = PRIORITY.filter(name => series[name])
    .concat(Object.keys(series).filter(name => !PRIORITY.includes(name) && !SKIP.has(name)));
  if (!names.length) return '<p class="empty">Sin métricas numéricas todavía.</p>';
  return names.map(name => svgChart(series[name].steps, series[name].values, name)).join("\n");
}
function renderVideo(video) {
  const extra = chartable(video.min_tcp_fruit_distance_m)
    ? ` · min TCP-fruta ${Number(video.min_tcp_fruit_distance_m).toFixed(3)} m` : "";
  return `<figure><figcaption>update ${esc(video.step)}${esc(extra)} · ${esc(video.label || "")}</figcaption>
    <video controls preload="metadata" src="${esc(video.url)}"></video></figure>`;
}
let currentVideo = null;
async function tick() {
  const payload = await fetch("metrics.json?t=" + Date.now(), {cache: "no-store"}).then(r => r.json());
  document.getElementById("sub").textContent =
    `${payload.run} · ${payload.rows} filas · actualizado ${payload.generated_at}`;
  document.getElementById("cards").innerHTML = renderCards(payload.latest || {});
  document.getElementById("charts").innerHTML = renderCharts(payload.series || {});
  const videos = payload.videos || [];
  const video = videos.length ? videos[videos.length - 1] : null;
  const slot = document.getElementById("clip");
  if (!video) {
    currentVideo = null;
    slot.innerHTML = '<p class="empty">Aún no hay vídeos. El sidecar los graba cada --video-every updates.</p>';
    return;
  }
  if (video.url !== currentVideo) {
    currentVideo = video.url;
    slot.innerHTML = renderVideo(video);
  }
}
tick();
setInterval(() => { tick().catch(() => {}); }, 2000);
</script>
'''


def render_dashboard_html(payload: Mapping[str, Any]) -> str:
    latest = payload.get('latest') or {}
    series = payload.get('series') or {}
    videos = list(payload.get('videos') or [])
    cards = []
    for key in ('step', 'curriculum_index', 'loss', 'entropy', 'reward_mean',
                'evaluation/success_rate', 'evaluation/mean_closest_distance_m',
                'evaluation/harvest_successes', 'training_transitions_per_second',
                'torch_peak_allocated_gb'):
        if key in latest and is_chartable(latest[key]):
            cards.append(
                f'<div class="card"><div class="k">{html.escape(key)}</div>'
                f'<div class="v">{html.escape(f"{float(latest[key]):.5g}")}</div></div>')
    ordered = [name for name in PRIORITY_CHARTS if name in series]
    ordered.extend(name for name in series if name not in ordered)
    charts = '\n'.join(svg_chart(series[name]['steps'], series[name]['values'], name) for name in ordered)
    last = latest_progress_video(videos)
    clip = _video_figure(last) if last is not None else (
        '<p class="empty">Aún no hay vídeos. El sidecar los graba cada --video-every updates.</p>')
    generated = html.escape(str(payload.get('generated_at', '')))
    run = html.escape(str(payload.get('run', '')))
    rows = int(payload.get('rows', 0))
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <title>Thekenyos · monitor de entrenamiento</title>
  <style>
    body {{ margin: 0; background: #0f1115; color: #e8edf7; font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }}
    header, main {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 22px; margin-bottom: 6px; }}
    .sub, .empty {{ color: #9aa3b5; }}
    .warn {{ background: #3a2a12; color: #f4d35e; padding: 10px 12px; border-radius: 8px; margin: 12px 0 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .card {{ background: #171a21; padding: 12px; border-radius: 10px; }}
    .k {{ color: #8a93a6; font-size: 11px; }}
    .v {{ font-size: 20px; margin-top: 4px; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 18px; }}
    .chart {{ width: 100%; height: auto; }}
    figure {{ margin: 0 0 16px; }}
    video {{ width: 100%; background: #000; border-radius: 8px; }}
    figcaption {{ color: #9aa3b5; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Monitor de entrenamiento</h1>
    <div class="sub" id="sub">{run} · {rows} filas · actualizado {generated}</div>
    <div class="warn">Curriculum TK-RL-003 sobre fruta rígida: depositar → agarrar/desprender → cosecha estacionaria → aproximación → varios → generalizar. <code>training_ready</code> sigue en false. Un depósito simulado no es cosecha de campo. El recuadro amarillo es la RGB del gripper RELIC. Las gráficas se actualizan sin recargar la página.</div>
    <div class="cards" id="cards">{''.join(cards) or '<div class="card">Esperando training.jsonl</div>'}</div>
  </header>
  <main>
    <h2>Gráficas</h2>
    <div class="charts" id="charts">{charts or '<p class="empty">Sin métricas numéricas todavía.</p>'}</div>
    <h2>Último vídeo de progreso</h2>
    <div id="clip">{clip}</div>
  </main>
{_DASHBOARD_SCRIPT}
</body>
</html>
"""


def dashboard_payload(rows: Sequence[Mapping[str, Any]], *, run: str | Path,
                      videos: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    series = series_from_rows(rows)
    return dict(schema=SCHEMA, training_ready=False,
                run=str(run), rows=len(rows), latest=_latest_row(rows),
                series=series, videos=list(videos or ()),
                generated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                caveat='Curriculum metrics and CPU previews, not field harvest')


def write_dashboard(rows: Sequence[Mapping[str, Any]], monitor_dir: str | Path,
                    *, run: str | Path, videos: Iterable[Mapping[str, Any]] | None = None,
                    hub: str | Path | None = None) -> dict[str, Any]:
    monitor_dir = Path(monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    (monitor_dir / 'videos').mkdir(parents=True, exist_ok=True)
    payload = dashboard_payload(rows, run=run, videos=videos)
    (monitor_dir / 'metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (monitor_dir / 'index.html').write_text(render_dashboard_html(payload), encoding='utf-8')
    if hub is not None:
        publish_hub(monitor_dir, hub)
    return payload


def publish_hub(monitor_dir: str | Path, hub: str | Path) -> None:
    monitor_dir, hub = Path(monitor_dir).resolve(), Path(hub).resolve()
    if hub == monitor_dir:
        return
    hub.mkdir(parents=True, exist_ok=True)
    for name in ('index.html', 'metrics.json'):
        source = monitor_dir / name
        if source.exists():
            shutil.copy2(source, hub / name)
    videos, target = monitor_dir / 'videos', hub / 'videos'
    if target.is_symlink() or target.is_file():
        target.unlink()
    if videos.exists() and not target.exists():
        target.symlink_to(videos)


class LiveDashboard:
    """Rewrite ``monitor/index.html`` from the run's jsonl and video folder."""

    def __init__(self, run_dir: str | Path, hub: str | Path | None = None):
        self.run_dir = Path(run_dir)
        self.monitor_dir = self.run_dir / 'monitor'
        self.jsonl = self.run_dir / 'training.jsonl'
        self.hub = Path(hub) if hub is not None else None
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        (self.monitor_dir / 'videos').mkdir(parents=True, exist_ok=True)

    def video_path(self, update: int) -> Path:
        if not isinstance(update, int) or isinstance(update, bool) or update < 0:
            raise ValueError('update must be a non-negative integer')
        return self.monitor_dir / 'videos' / f'progress-{update:04d}.mp4'

    def refresh(self) -> dict[str, Any]:
        return write_dashboard(read_jsonl(self.jsonl), self.monitor_dir, run=self.run_dir,
                               videos=list_progress_videos(self.monitor_dir / 'videos'),
                               hub=self.hub)


def add_monitor_args(parser) -> None:
    parser.add_argument('--video-every', type=int, default=10,
                        help='Record a CPU progress clip every N updates; 0 disables')
    parser.add_argument('--video-steps', type=int, default=256,
                        help='Policy steps in each CPU progress clip (about 10 s at 25 fps)')
    parser.add_argument('--monitor-hub', type=Path,
                        help='Optional extra copy of index.html for Jupyter /files')


def _select_mujoco_gl(env: dict[str, str]) -> dict[str, str]:
    if env.get('MUJOCO_GL'):
        return env
    for library in (
        '/usr/lib/x86_64-linux-gnu/libOSMesa.so.8',
        '/usr/lib64/libOSMesa.so.8',
        '/usr/lib/libOSMesa.so.8',
    ):
        if Path(library).exists():
            env['MUJOCO_GL'] = 'osmesa'
            return env
    env['MUJOCO_GL'] = 'egl'
    return env


def checkpoint_paths(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    meta = json.loads(Path(str(checkpoint) + '.json').read_text(encoding='utf-8'))
    config = dict(meta.get('config') or {})
    scene, gait = config.get('scene'), config.get('gait_checkpoint')
    if not scene or not gait:
        raise ValueError('Checkpoint sidecar is missing scene or gait_checkpoint')
    return dict(checkpoint=checkpoint, scene=Path(scene), gait_checkpoint=Path(gait),
                camera=meta.get('camera', 'hand_color_sensor'),
                camera_every=int(config.get('camera_every') or 2),
                completed_updates=int(meta.get('completed_updates', 0)),
                config=config, meta=meta)


def _tcp_fruit_ids(model, manifest):
    robot = manifest['robot']
    fruit = manifest.get('fruits') or manifest.get('fruit') or []
    if not fruit:
        raise ValueError('Fast scene must contain fruit for a progress video')
    fruit_body = int(model.body(fruit[0]['body']).id)
    names = [model.site(i).name for i in range(model.nsite)]
    tcp_name = robot.get('tcp_site', 'hand_tcp')
    if tcp_name not in names:
        tcp_body = robot.get('tcp_body', robot.get('prefix', '') + 'arm_link_fngr')
        tcp_name = next((model.site(i).name for i in range(model.nsite)
                         if model.site(i).bodyid == model.body(tcp_body).id), None)
    if tcp_name is None:
        raise ValueError('Fast scene robot must provide a TCP site')
    return int(model.site(tcp_name).id), fruit_body


def _arm_limits(model, controller):
    import numpy as np
    lower = np.full(7, -np.inf, dtype=np.float32)
    upper = np.full(7, np.inf, dtype=np.float32)
    joints = np.asarray(controller.joints[12:], dtype=int)
    limited = np.asarray(model.jnt_limited, dtype=bool)[joints]
    ranges = np.asarray(model.jnt_range, dtype=np.float32)[joints]
    lower[limited] = ranges[limited, 0]
    upper[limited] = ranges[limited, 1]
    return lower, upper


def _capture_rgbd(renderer, data, camera, *, minimum_m=.05, maximum_m=4.):
    import numpy as np
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    rgb_u8 = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    valid = np.isfinite(depth) & (depth >= minimum_m) & (depth <= maximum_m)
    depth_n = np.where(valid, np.clip(depth, 0., maximum_m) / maximum_m, 0.).astype(np.float32)
    rgb = rgb_u8.astype(np.float32) / 255.
    rgbd = np.concatenate((np.transpose(rgb, (2, 0, 1)), depth_n[None], valid.astype(np.float32)[None]), axis=0)
    return rgbd[None].astype(np.float32), rgb_u8


def record_progress_video(checkpoint: str | Path, output: str | Path, *,
                          steps: int = 256, camera_every: int | None = None,
                          control_dt: float = .02, width: int = 640, height: int = 360,
                          resolution: int = 64, fps: int = 25) -> dict[str, Any]:
    """Deterministic CPU rollout of one checkpoint. Not a harvest demo."""
    if not isinstance(steps, int) or not 8 <= steps <= 512:
        raise ValueError('video steps must be an integer in [8, 512]')
    if width % 2 or height % 2:
        raise ValueError('ffmpeg yuv420p needs even width and height')
    info = checkpoint_paths(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve existing progress video: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / '.recording.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return dict(schema=VIDEO_SCHEMA, skipped=True, reason='recorder busy', output=str(output))
    os.write(lock_fd, str(os.getpid()).encode())
    try:
        return _record_progress_video_locked(
            info, output, steps=steps, camera_every=camera_every,
            control_dt=control_dt, width=width, height=height,
            resolution=resolution, fps=fps)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _record_progress_video_locked(info, output, *, steps, camera_every, control_dt,
                                  width, height, resolution, fps):
    partial = output.with_name(output.stem + '.partial.mp4')
    camera_every = info['camera_every'] if camera_every is None else int(camera_every)
    if not 1 <= camera_every <= 5:
        raise ValueError('camera_every must be in [1, 5]')

    os.environ['MUJOCO_GL'] = _select_mujoco_gl(dict(os.environ)).get('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    import mujoco
    from PIL import Image, ImageDraw
    from treesim.kiwi_rl.fast_scene import load_fast_scene
    from treesim.kiwi_rl.control import NativeSpotControl, load_gait_artifact
    from treesim.kiwi_rl.spot_cameras import require_mujoco_gripper_cameras
    from treesim.kiwi_rl.ppo import load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from train_fast import build_policy

    model, data, manifest = load_fast_scene(info['scene'])
    require_mujoco_gripper_cameras(model, manifest['robot'])
    camera = info['camera']
    gait = load_gait_artifact(info['gait_checkpoint'], precision_profile='cuda-fp32').to('cpu').eval()
    controller = NativeSpotControl(model, manifest['robot'], gait)
    policy = build_policy().to('cpu').eval()
    load_checkpoint(info['checkpoint'], {'student': policy}, expected_meta={'camera': camera})
    tcp_site, fruit_body = _tcp_fruit_ids(model, manifest)
    chassis = controller.chassis
    lower, upper = _arm_limits(model, controller)
    max_delta = 2.5 * control_dt
    substeps = round(control_dt / float(model.opt.timestep))
    if substeps < 1:
        raise ValueError('control_dt must cover at least one physics step')

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg must be installed to record progress videos')
    encoder = subprocess.Popen([
        ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-f', 'mp4', str(partial),
    ], stdin=subprocess.PIPE)
    scene_renderer = mujoco.Renderer(model, height=height, width=width)
    policy_renderer = mujoco.Renderer(model, height=resolution, width=resolution)
    viewer = mujoco.MjvCamera()
    memory = torch.zeros(1, 64)
    rgbd = None
    distances = []
    try:
        for index in range(steps):
            mujoco.mj_camlight(model, data)
            if rgbd is None or index % camera_every == 0:
                rgbd, grip_u8 = _capture_rgbd(policy_renderer, data, camera)
            else:
                policy_renderer.disable_depth_rendering()
                policy_renderer.update_scene(data, camera=camera)
                grip_u8 = policy_renderer.render().copy()
            r84 = controller.observe(data, np.zeros(3, dtype=np.float32))[None]
            with torch.no_grad():
                mean, _, _, memory = policy(torch.as_tensor(rgbd), torch.as_tensor(r84), memory)
                action = mean.tanh().numpy()[0]
            arm = action[-7:]
            controller.update_gait(data, np.zeros(3, dtype=np.float32))
            controller.targets[12:] = np.clip(
                controller.targets[12:] + np.clip(arm, -1., 1.) * max_delta, lower, upper)
            for _ in range(substeps):
                controller.apply(data)
                mujoco.mj_step(model, data)
                if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                    raise RuntimeError('Nonfinite native state while recording a progress video')
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            distance = float(np.linalg.norm(data.site_xpos[tcp_site] - data.xipos[fruit_body]))
            distances.append(distance)
            viewer.lookat[:] = data.xpos[chassis] + [0., 0., .35]
            viewer.distance, viewer.azimuth, viewer.elevation = 3., 135., -20.
            scene_renderer.update_scene(data, camera=viewer)
            frame = compose_progress_frame(scene_renderer.render(), grip_u8)
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            update = info['completed_updates']
            draw.text((8, 8), f'CPU preview  update {update}  step {index + 1}/{steps}', fill=(255, 255, 255))
            draw.text((8, 22), f'TCP-fruit {distance:.3f} m  gripper RGB overlay  curriculum preview not harvest proof', fill=(244, 211, 94))
            try:
                encoder.stdin.write(np.asarray(image).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError('ffmpeg exited while writing the progress video') from exc
    finally:
        scene_renderer.close()
        policy_renderer.close()
        encoder.stdin.close()
        status = encoder.wait()
    if status:
        partial.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg failed to encode the progress video')
    os.replace(partial, output)
    result = dict(schema=VIDEO_SCHEMA, training_ready=False,
                  label='CPU native curriculum preview; not a harvest demonstration',
                  checkpoint=str(info['checkpoint']), scene=str(info['scene']),
                  gait_checkpoint=str(info['gait_checkpoint']), camera=camera,
                  update=info['completed_updates'], steps=steps, fps=fps,
                  control_dt_s=control_dt, backend=f'cpu-native-mujoco-{os.environ.get("MUJOCO_GL", "egl")}',
                  mean_tcp_fruit_distance_m=float(np.mean(distances)),
                  min_tcp_fruit_distance_m=float(np.min(distances)),
                  final_tcp_fruit_distance_m=float(distances[-1]),
                  output=str(output))
    output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def spawn_progress_video(checkpoint: str | Path, output: str | Path, *,
                         steps: int = 256, camera_every: int = 2,
                         python: str | None = None) -> subprocess.Popen | None:
    """Start a CPU recording subprocess that leaves the training GPU alone."""
    output = Path(output)
    if output.exists() or output.with_name(output.stem + '.partial.mp4').exists() or (output.parent / '.recording.lock').exists():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'watch_training.py'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env = _select_mujoco_gl(env)
    log_path = output.with_suffix('.record.log')
    command = [python or sys.executable, '-B', str(script),
               '--record-checkpoint', str(checkpoint), '--video-output', str(output),
               '--video-steps', str(steps), '--camera-every', str(camera_every)]
    with log_path.open('w', encoding='utf-8') as log:
        return subprocess.Popen(command, env=env, start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)
