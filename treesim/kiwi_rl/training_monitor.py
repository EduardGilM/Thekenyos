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
    'loss', 'kl', 'entropy', 'entropy_per_dim', 'entropy_gaussian', 'logstd_mean', 'grad_norm',
    'reward_mean', 'success_window_return_mean', 'deposit_return_sum', 'fail_return_sum',
    'harvest_jackpot_sum', 'reward_window_mean', 'reward_transition_mean', 'reward_std',
    'harvest_successes', 'evaluation/success_rate', 'evaluation/harvest_fraction',
    'basket_distance_mean_m', 'basket_distance_closest_m', 'basket_xy_mean_m',
    'evaluation/mean_closest_basket_distance_m', 'evaluation/final_basket_distance_m',
    'ground_contact_worlds', 'fallen_worlds', 'failed_worlds', 'hand_load_max_N',
    'distance_mean_closest_m', 'distance_final_m', 'distance_closest_m',
    'evaluation/mean_closest_distance_m', 'evaluation/final_distance_m',
    'evaluation/closest_distance_m', 'evaluation/harvest_successes',
    'evaluation/detach_rate', 'evaluation/grasp_rate',
    'grasp_events', 'detach_events', 'harvested_mean',
    'recovered_worlds', 'overflow_worlds', 'evaluation/recovered_worlds', 'evaluation/overflow_worlds',
    'evaluation/terminal_transitions', 'terminal_transitions',
    'curriculum_index', 'guidance_weight', 'teacher_mix', 'shaping_coef',
    'easy_far_frac', 'easy_start_index_mean', 'easy_start_index_max', 'easy_hold_close_mean',
    'teacher_anneal_after',
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
        if update == 0:
            continue
        if update == 1 or update % every == 0:
            due.append((update, path))
    return [path for _, path in sorted(due)]


def due_latest_checkpoint(run_dir: str | Path, every: int) -> Path | None:
    """Use latest.pt at the same cadence as numbered checkpoints.

    Speedrun numbered files only exist every 50 updates; latest.pt is
    overwritten every update so the CPU sidecar can still preview the student.
    """
    if not isinstance(every, int) or every < 0:
        raise ValueError('video-every must be a non-negative integer')
    if every == 0:
        return None
    latest = Path(run_dir) / 'latest.pt'
    sidecar = Path(str(latest) + '.json')
    if not latest.exists() or not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        update = int(meta.get('completed_updates', 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if update < 1:
        return None
    if update != 1 and update % every != 0:
        return None
    return latest


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
    basket = video.get('min_basket_distance_m')
    extra = f" · min TCP-fruta {float(distance):.3f} m" if is_chartable(distance) else ''
    if is_chartable(basket):
        extra += f' · min cesta {float(basket):.3f} m'
    if video.get('easy'):
        extra += ' · easy-carry'
    return (
        f'<figure><figcaption>update {html.escape(str(video.get("step")))}'
        f'{html.escape(extra)} · {html.escape(str(video.get("label", "")))}</figcaption>'
        f'<video controls preload="metadata" src="{html.escape(str(video.get("url")))}"></video></figure>'
    )


_DASHBOARD_SCRIPT = r'''
<script>
const CARD_KEYS = ["step", "curriculum_index", "loss", "entropy", "entropy_per_dim", "reward_mean",
  "success_window_return_mean", "deposit_return_sum", "fail_return_sum", "harvest_jackpot_sum", "harvest_successes", "evaluation/success_rate", "evaluation/harvest_fraction",
  "basket_distance_mean_m", "basket_xy_mean_m", "easy_far_frac", "teacher_mix",
  "evaluation/mean_closest_basket_distance_m",
  "ground_contact_worlds", "nonfinite_worlds", "evaluation/mean_closest_distance_m",
  "evaluation/harvest_successes", "training_transitions_per_second",
  "torch_peak_allocated_gb"];
const PRIORITY = ["loss", "kl", "entropy", "entropy_per_dim", "entropy_gaussian", "logstd_mean", "grad_norm", "reward_mean", "success_window_return_mean", "deposit_return_sum", "fail_return_sum", "harvest_jackpot_sum", "reward_window_mean", "reward_transition_mean", "reward_std",
  "harvest_successes", "evaluation/success_rate", "evaluation/harvest_fraction",
  "basket_distance_mean_m", "basket_distance_closest_m", "basket_xy_mean_m",
  "easy_far_frac", "easy_start_index_mean", "easy_start_index_max", "teacher_mix",
  "easy_hold_close_mean", "hand_load_mean_N", "hand_load_max_N",
  "evaluation/mean_closest_basket_distance_m", "evaluation/final_basket_distance_m",
  "ground_contact_worlds", "fallen_worlds", "failed_worlds", "hand_load_max_N",
  "distance_mean_closest_m", "distance_final_m", "distance_closest_m",
  "evaluation/mean_closest_distance_m", "evaluation/final_distance_m",
  "evaluation/closest_distance_m", "evaluation/harvest_successes",
  "evaluation/detach_rate", "evaluation/grasp_rate",
  "grasp_events", "detach_events", "harvested_mean",
  "evaluation/terminal_transitions", "terminal_transitions",
  "curriculum_index", "guidance_weight", "teacher_mix", "shaping_coef",
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
function stageLabel(payload) {
  if (payload.curriculum_label) return payload.curriculum_label;
  const latest = payload.latest || {};
  if (latest.curriculum_stage && chartable(latest.curriculum_index)) {
    return String(latest.curriculum_index).padStart(3, "0") + " · " + latest.curriculum_stage;
  }
  return latest.curriculum_stage || "esperando etapa";
}
function renderCards(payload) {
  const latest = payload.latest || {};
  const label = stageLabel(payload);
  const stage = label && label !== "esperando etapa"
    ? `<div class="card"><div class="k">curriculum_stage</div><div class="v">${esc(label)}</div></div>`
    : "";
  const cards = CARD_KEYS.filter(key => chartable(latest[key])).map(key =>
    `<div class="card"><div class="k">${esc(key)}</div><div class="v">${esc(fmt(latest[key]))}</div></div>`);
  return (stage + cards.join("")) || '<div class="card">Esperando training.jsonl</div>';
}
function renderCharts(series) {
  const names = PRIORITY.filter(name => series[name])
    .concat(Object.keys(series).filter(name => !PRIORITY.includes(name) && !SKIP.has(name)));
  if (!names.length) return '<p class="empty">Sin métricas numéricas todavía.</p>';
  return names.map(name => svgChart(series[name].steps, series[name].values, name)).join("\n");
}
function queryToken() {
  try {
    const token = new URLSearchParams(location.search).get("token") || "";
    return token && token !== "None" ? token : "";
  } catch (err) {
    return "";
  }
}
function withAuth(path) {
  const token = queryToken();
  if (!token) return path;
  return path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}
function renderVideo(video) {
  let extra = chartable(video.min_tcp_fruit_distance_m)
    ? ` · min TCP-fruta ${Number(video.min_tcp_fruit_distance_m).toFixed(3)} m` : "";
  if (chartable(video.min_basket_distance_m)) {
    extra += ` · min cesta ${Number(video.min_basket_distance_m).toFixed(3)} m`;
  }
  if (video.easy) extra += " · easy-carry";
  return `<figure><figcaption>update ${esc(video.step)}${esc(extra)} · ${esc(video.label || "")}</figcaption>
    <video controls preload="metadata" src="${esc(withAuth(video.url))}"></video></figure>`;
}
let currentVideo = null;
function showFetchHint(message) {
  if (document.getElementById("auth-hint")) return;
  const sub = document.getElementById("sub");
  if (!sub) return;
  const hint = document.createElement("div");
  hint.id = "auth-hint";
  hint.className = "warn";
  hint.textContent = message;
  sub.after(hint);
}
async function tick() {
  const response = await fetch(withAuth("metrics.json?t=" + Date.now()), {
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!response.ok) {
    showFetchHint("No se pudieron leer las métricas (HTTP " + response.status +
      "). Si Caddy pide token, abre /index.html?token=… o Jupyter /files/workspace/training/monitor-live/index.html.");
    throw new Error("metrics http " + response.status);
  }
  const payload = await response.json();
  const hint = document.getElementById("auth-hint");
  if (hint) hint.remove();
  const stage = stageLabel(payload);
  document.getElementById("sub").textContent =
    `${payload.run} · etapa ${stage} · ${payload.rows} filas · actualizado ${payload.generated_at}`;
  document.getElementById("cards").innerHTML = renderCards(payload);
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
    stage = html.escape(str(payload.get('curriculum_label') or latest.get('curriculum_stage') or 'esperando etapa'))
    if stage and stage != 'esperando etapa':
        cards.append(
            f'<div class="card"><div class="k">curriculum_stage</div>'
            f'<div class="v">{stage}</div></div>')
    for key in ('step', 'curriculum_index', 'loss', 'entropy', 'entropy_per_dim', 'reward_mean',
                'success_window_return_mean', 'deposit_return_sum', 'fail_return_sum', 'harvest_jackpot_sum',
                'harvest_successes', 'evaluation/success_rate', 'evaluation/harvest_fraction',
                'basket_distance_mean_m', 'evaluation/mean_closest_basket_distance_m',
                'ground_contact_worlds', 'evaluation/mean_closest_distance_m',
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
    <div class="sub" id="sub">{run} · etapa {stage} · {rows} filas · actualizado {generated}</div>
    <div class="warn">Curriculum TK-RL-003 sobre fruta rígida: depositar → agarrar/desprender → cosecha estacionaria → aproximación → varios → generalizar. <code>training_ready</code> sigue en false. Un depósito simulado no es cosecha de campo. El recuadro amarillo es la RGB del gripper RELIC. Las gráficas se actualizan sin recargar la página. La entropía es diferencial (nats) de una tanh-Gaussiana; con logstd negativo puede ser &lt; 0 y no es un fallo numérico. Un overflow (flag 2) o acción no finita (flag 4) reinicia ese mundo; qpos/qvel no finito aborta el trabajo.</div>
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


def _run_config(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = run_dir / 'config.json'
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    sidecars = sorted(run_dir.glob('checkpoint-*.pt.json'))
    if not sidecars:
        return {}
    meta = json.loads(sidecars[-1].read_text(encoding='utf-8'))
    config = dict(meta.get('config') or {})
    if meta.get('curriculum_stage') and 'curriculum' not in config:
        config['stage'] = meta['curriculum_stage']
    return config


def curriculum_status(run_dir: str | Path, latest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stage label for the dashboard. Index 1 is 001 · deposit_pixels."""
    latest = dict(latest or {})
    name = latest.get('curriculum_stage')
    index = latest.get('curriculum_index')
    if not name:
        config = _run_config(run_dir)
        curriculum = config.get('curriculum') if isinstance(config.get('curriculum'), dict) else {}
        name = curriculum.get('name') or config.get('stage')
        index = curriculum.get('index') if index is None else index
    if not name:
        return dict(curriculum_stage=None, curriculum_index=None, curriculum_label='esperando etapa')
    try:
        from treesim.kiwi_rl.curriculum import stage_named
        stage = stage_named(str(name))
        name = stage.name
        if index is None:
            index = stage.index
    except ValueError:
        pass
    if isinstance(index, bool) or not isinstance(index, int):
        index = None
    label = f'{index:03d} · {name}' if index is not None else str(name)
    return dict(curriculum_stage=str(name), curriculum_index=index, curriculum_label=label)


def dashboard_payload(rows: Sequence[Mapping[str, Any]], *, run: str | Path,
                      videos: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    series = series_from_rows(rows)
    latest = _latest_row(rows)
    status = curriculum_status(run, latest)
    if status['curriculum_stage'] and 'curriculum_stage' not in latest:
        latest = dict(latest, curriculum_stage=status['curriculum_stage'],
                      curriculum_index=status['curriculum_index'])
    return dict(schema=SCHEMA, training_ready=False,
                run=str(run), rows=len(rows), latest=latest,
                series=series, videos=list(videos or ()),
                generated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                caveat='Curriculum metrics and CPU previews, not field harvest',
                **status)


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
    if not videos.exists():
        return
    desired = videos.resolve()
    if target.is_symlink():
        try:
            current = target.resolve()
        except OSError:
            current = None
        if current == desired:
            return
        target.unlink()
    elif target.is_file():
        target.unlink()
    elif target.is_dir():
        try:
            next(target.iterdir())
        except StopIteration:
            target.rmdir()
        else:
            return
    target.symlink_to(desired)


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


def serve_monitor(directory: str | Path, port: int):
    """Serve the hub on 0.0.0.0 so Vast Caddy can reverse-proxy it."""
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import threading

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f'monitor directory does not exist: {directory}')
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError('http-port must be an integer in [0, 65535]')

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):
            return

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


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


def curriculum_preview_from_checkpoint(info: Mapping[str, Any]) -> dict[str, Any]:
    """CPU clip reset/locomotion flags from checkpoint metadata. Not a harvest demo."""
    from treesim.kiwi_rl.curriculum import reset_mode_for_goal, stage_named
    meta = dict(info.get('meta') or {})
    config = dict(info.get('config') or {})
    name = meta.get('curriculum_stage') or config.get('stage')
    easy = bool(config.get('easy', False))
    far_frac = 0.0
    hold_frac = 0.75
    if easy:
        raw = config.get('easy_far_frac', 0.0)
        far_frac = float(raw)
        if not math.isfinite(far_frac) or not 0.0 <= far_frac <= 1.0:
            raise ValueError('easy_far_frac must be finite in [0, 1]')
        raw_hold = config.get('hold_close_frac', 0.75)
        if raw_hold is None:
            raw_hold = 0.75
        hold_frac = float(raw_hold)
        if not math.isfinite(hold_frac) or not 0.0 <= hold_frac <= 1.0:
            raise ValueError('hold_close_frac must be finite in [0, 1]')
    if not name:
        return dict(stage=None, reset_mode=0, allow_locomotion=False, easy=easy,
                    easy_far_frac=far_frac, hold_close_frac=hold_frac)
    stage = stage_named(str(name))
    return dict(stage=stage.name, reset_mode=int(reset_mode_for_goal(stage.goal, stage)),
                allow_locomotion=bool(stage.allow_locomotion), goal=stage.goal, easy=easy,
                easy_far_frac=far_frac, hold_close_frac=hold_frac)


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


def apply_native_easy_hover(model, data, controller, tcp_site: int, *,
                            clearance_m: float | None = None) -> float:
    """Move the native arm to the privileged basket-hover TCP. Fruit is not written.

    This is the teacher drop pose above the opening, not the episode start.
    """
    import mujoco
    import numpy as np
    from treesim.kiwi_rl.curriculum import EASY_PRESET
    from treesim.kiwi_rl.reach_teacher import hover_tcp_world_m, solve_tcp_hover
    if clearance_m is None:
        clearance_m = float(EASY_PRESET['hover_clearance_m'])
    mujoco.mj_kinematics(model, data)
    qids = np.asarray(controller.qids[12:18], dtype=int)
    dofs = np.asarray(controller.dofs[12:18], dtype=int)
    joints = np.asarray(controller.joints[12:18], dtype=int)
    ranges = np.tile(np.array([-np.pi, np.pi], dtype=np.float64), (6, 1))
    limited = np.asarray(model.jnt_limited[joints], dtype=bool)
    ranges[limited] = np.asarray(model.jnt_range[joints], dtype=np.float64)[limited]
    q_init = np.asarray(data.qpos[qids], dtype=np.float64)
    target = hover_tcp_world_m(
        data.xpos[controller.chassis], data.xmat[controller.chassis], clearance_m)
    arm_q, err = solve_tcp_hover(
        model, data.qpos, int(tcp_site), target, qids, dofs, q_init, ranges)
    if not np.isfinite(arm_q).all() or not np.isfinite(err):
        return float('inf')
    data.qpos[qids] = arm_q
    controller.targets[12:18] = arm_q
    mujoco.mj_forward(model, data)
    return float(err)


def apply_native_easy_start(model, data, controller, tcp_site: int, *,
                            far_frac: float = 0.0) -> float:
    """Move the native arm to the easy start pose. Fruit is not written."""
    import mujoco
    import numpy as np
    from treesim.kiwi_rl.curriculum import EASY_PRESET
    from treesim.kiwi_rl.reach_teacher import (
        easy_over_opening_local_m, random_easy_start_local_m, solve_tcp_hover,
        tcp_outside_basket, tcp_over_opening_above_rim,
    )
    frac = float(far_frac)
    if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError('far_frac must be finite in [0, 1]')
    mujoco.mj_kinematics(model, data)
    qids = np.asarray(controller.qids[12:18], dtype=int)
    dofs = np.asarray(controller.dofs[12:18], dtype=int)
    joints = np.asarray(controller.joints[12:18], dtype=int)
    ranges = np.tile(np.array([-np.pi, np.pi], dtype=np.float64), (6, 1))
    limited = np.asarray(model.jnt_limited[joints], dtype=bool)
    ranges[limited] = np.asarray(model.jnt_range[joints], dtype=np.float64)[limited]
    chassis_p = np.asarray(data.xpos[controller.chassis], dtype=np.float64)
    chassis_R = np.asarray(data.xmat[controller.chassis], dtype=np.float64).reshape(3, 3)
    home_tcp = np.asarray(data.site_xpos[int(tcp_site)], dtype=np.float64)
    home_local = chassis_R.T @ (home_tcp - chassis_p)
    rng = np.random.default_rng(7 + int(round(frac * 10_000)))
    if EASY_PRESET.get('start_over_opening'):
        local = easy_over_opening_local_m(rng)
        if not tcp_over_opening_above_rim(local):
            raise ValueError('easy start TCP is not over the opening above the rim')
    else:
        local = random_easy_start_local_m(
            rng, home_local,
            margin_m=EASY_PRESET['start_margin_m'],
            clearance_m=EASY_PRESET['start_clearance_m'])
        if not tcp_outside_basket(local, margin_m=0.04, above_rim_m=0.0):
            raise ValueError('easy start TCP still intersects the crate volume')
    target = chassis_p + chassis_R @ local
    q_init = np.asarray(data.qpos[qids], dtype=np.float64)
    arm_q, err = solve_tcp_hover(
        model, data.qpos, int(tcp_site), target, qids, dofs, q_init, ranges)
    if not np.isfinite(arm_q).all() or not np.isfinite(err):
        return float('inf')
    data.qpos[qids] = arm_q
    controller.targets[12:18] = arm_q
    mujoco.mj_forward(model, data)
    return float(err)


def apply_native_skill_reset(model, data, manifest, controller, *, reset_mode: int,
                             approach_offset_m: float = 1.0, easy: bool = False,
                             far_frac: float = 0.0, hold_close_frac: float | None = None) -> None:
    """Match GPU deposit/approach resets on CPU native MuJoCo. Fruit stays a free body."""
    import mujoco
    import numpy as np
    if reset_mode not in (0, 1, 2, 3):
        raise ValueError('reset_mode must be 0, 1, 2 or 3')
    if not np.isfinite(approach_offset_m) or approach_offset_m < 0:
        raise ValueError('approach_offset_m must be finite and >= 0')
    mujoco.mj_forward(model, data)
    tcp_site, fruit_body = _tcp_fruit_ids(model, manifest)
    fruit = (manifest.get('fruits') or manifest.get('fruit') or [None])[0]
    if not fruit:
        raise ValueError('Fast scene must contain fruit for a skill reset')
    joint = int(model.body_jntadr[fruit_body])
    if joint < 0:
        raise ValueError('Fruit body must keep an independent free joint')
    qposadr = int(model.jnt_qposadr[joint])
    dofadr = int(model.jnt_dofadr[joint])
    if easy and reset_mode == 1:
        apply_native_easy_start(model, data, controller, tcp_site, far_frac=far_frac)
    if reset_mode == 1:
        from treesim.kiwi_rl.reach_teacher import grasp_pocket_world_m, jaw_hold_q, jaw_open_closed_q
        opened, closed = jaw_open_closed_q(model, int(controller.qids[18]), data)
        frac = 0.75 if hold_close_frac is None else float(hold_close_frac)
        if not easy:
            frac = 1.0
        if not np.isfinite(frac) or not 0.0 <= frac <= 1.0:
            raise ValueError('hold_close_frac must be finite in [0, 1]')
        hold = jaw_hold_q(frac, opened, closed)
        data.qpos[int(controller.qids[18])] = hold
        controller.targets[18] = hold
        mujoco.mj_forward(model, data)
        if easy:
            pocket = grasp_pocket_world_m(model, data, tcp_site)
            data.qpos[qposadr:qposadr + 3] = pocket
        else:
            tcp = np.asarray(data.site_xpos[tcp_site], dtype=np.float64)
            data.qpos[qposadr:qposadr + 3] = tcp
        data.qpos[qposadr + 3:qposadr + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[dofadr:dofadr + 6] = 0.0
        equality = fruit.get('equality')
        if equality:
            data.eq_active[int(model.equality(equality).id)] = 0
    if reset_mode == 2:
        from treesim.kiwi_rl.reach_teacher import jaw_open_closed_q
        opened, _closed = jaw_open_closed_q(model, int(controller.qids[18]), data)
        data.qpos[int(controller.qids[18])] = opened
        controller.targets[18] = opened
    if reset_mode == 3:
        chassis_joint = int(model.body_jntadr[controller.chassis])
        data.qpos[int(model.jnt_qposadr[chassis_joint])] -= approach_offset_m
    mujoco.mj_forward(model, data)


N3_SCALE_MPS = (0.4, 0.3, 0.7)
N3_SLEW_MPS = (0.02, 0.02, 0.04)


def _n3_command(previous, raw_base):
    import numpy as np
    previous = np.asarray(previous, dtype=np.float32)
    desired = np.clip(np.asarray(raw_base, dtype=np.float32), -1.0, 1.0) * np.asarray(N3_SCALE_MPS, dtype=np.float32)
    slew = np.asarray(N3_SLEW_MPS, dtype=np.float32)
    return previous + np.clip(desired - previous, -slew, slew)


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
    from treesim.basket import CENTER, SIZE
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from train_fast import build_policy

    model, data, manifest = load_fast_scene(info['scene'])
    require_mujoco_gripper_cameras(model, manifest['robot'])
    camera = info['camera']
    gait = load_gait_artifact(info['gait_checkpoint'], precision_profile='cuda-fp32').to('cpu').eval()
    controller = NativeSpotControl(model, manifest['robot'], gait)
    policy = build_policy().to('cpu').eval()
    load_checkpoint(info['checkpoint'], {'student': policy}, expected_meta={'camera': camera})
    preview = curriculum_preview_from_checkpoint(info)
    apply_native_skill_reset(model, data, manifest, controller, reset_mode=preview['reset_mode'],
                             easy=bool(preview.get('easy')),
                             far_frac=float(preview.get('easy_far_frac') or 0.0),
                             hold_close_frac=preview.get('hold_close_frac'))
    tcp_site, fruit_body = _tcp_fruit_ids(model, manifest)
    chassis = controller.chassis
    basket_local = np.asarray(CENTER, dtype=np.float64)
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
    basket_distances = []
    command = np.zeros(3, dtype=np.float32)
    easy_hold_q = None
    stage_label = preview['stage'] or 'hanging'
    easy_tag = (
        f'easy-carry far_frac={float(preview.get("easy_far_frac") or 0.0):.2f} '
        if preview.get('easy') else '')
    try:
        for index in range(steps):
            mujoco.mj_camlight(model, data)
            if rgbd is None or index % camera_every == 0:
                rgbd, grip_u8 = _capture_rgbd(policy_renderer, data, camera)
            else:
                policy_renderer.disable_depth_rendering()
                policy_renderer.update_scene(data, camera=camera)
                grip_u8 = policy_renderer.render().copy()
            r84 = controller.observe(data, command)[None]
            with torch.no_grad():
                mean, _, _, memory = policy(torch.as_tensor(rgbd), torch.as_tensor(r84), memory)
                action = mean.tanh().numpy()[0]
            arm = action[-7:]
            if preview.get('easy'):
                from treesim.kiwi_rl.curriculum import EASY_PRESET
                from treesim.kiwi_rl.reach_teacher import (
                    adapt_scripted_hold_q, fruit_in_release_zone, jaw_hold_q,
                    jaw_open_closed_q, scripted_jaw_target,
                )
                opened, closed = jaw_open_closed_q(model, int(controller.qids[18]), data)
                frac = preview.get('hold_close_frac')
                if easy_hold_q is None:
                    easy_hold_q = jaw_hold_q(0.6 if frac is None else float(frac), opened, closed)
                hold = float(easy_hold_q)
                fruit_xyz = np.asarray(data.xpos[fruit_body], dtype=np.float64)
                basket_xyz = (
                    np.asarray(data.xpos[chassis], dtype=np.float64)
                    + np.asarray(data.xmat[chassis], dtype=np.float64).reshape(3, 3) @ basket_local)
                rim_z = float(SIZE[2])
                tcp_xyz = np.asarray(data.site_xpos[tcp_site], dtype=np.float64)
                release_center = bool(EASY_PRESET.get('release_at_center'))
                release_opening = bool(EASY_PRESET.get('release_over_opening'))
                inset = float(EASY_PRESET.get('release_opening_inset_m', 0.04))
                max_above = EASY_PRESET.get('release_max_above_rim_m')
                open_xy = float(EASY_PRESET['open_xy_m'])
                rotation = np.asarray(data.xmat[chassis], dtype=np.float64).reshape(3, 3)
                desired = scripted_jaw_target(
                    fruit_xyz, basket_xyz, hold, opened, open_xy_m=open_xy, rim_z_m=rim_z,
                    tcp_xy=tcp_xyz, release_at_center=release_center,
                    rotation=rotation, release_over_opening=release_opening,
                    inset_m=inset, max_above_rim_m=max_above)
                slip = float(np.linalg.norm(tcp_xyz - data.xpos[fruit_body]))
                over = fruit_in_release_zone(
                    fruit_xyz, basket_xyz, open_xy_m=open_xy, rim_z_m=rim_z,
                    tcp_xyz=tcp_xyz, release_at_center=release_center,
                    rotation=rotation, release_over_opening=release_opening,
                    inset_m=inset, max_above_rim_m=max_above)
                hold = adapt_scripted_hold_q(
                    hold, opened, closed, slip_m=slip, over_basket=over)
                easy_hold_q = hold
                desired = scripted_jaw_target(
                    fruit_xyz, basket_xyz, hold, opened, open_xy_m=open_xy, rim_z_m=rim_z,
                    tcp_xy=tcp_xyz, release_at_center=release_center,
                    rotation=rotation, release_over_opening=release_opening,
                    inset_m=inset, max_above_rim_m=max_above)
                arm = np.asarray(arm, dtype=np.float64).copy()
                arm[6] = np.clip((desired - float(controller.targets[18])) / max_delta, -1.0, 1.0)
            controller.update_gait(data, command)
            if preview['allow_locomotion'] and action.shape[0] >= 10:
                command = _n3_command(command, action[:3])
            else:
                command = np.zeros(3, dtype=np.float32)
            controller.targets[12:] = np.clip(
                controller.targets[12:] + np.clip(arm, -1., 1.) * max_delta, lower, upper)
            for _ in range(substeps):
                if preview.get('easy'):
                    data.qpos[int(controller.qids[18])] = desired
                    data.qvel[int(controller.dofs[18])] = 0.0
                    controller.targets[18] = desired
                controller.apply(data)
                mujoco.mj_step(model, data)
                if any(w.number for w in data.warning) or not np.isfinite(data.qpos).all():
                    raise RuntimeError('Nonfinite native state while recording a progress video')
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            distance = float(np.linalg.norm(data.site_xpos[tcp_site] - data.xipos[fruit_body]))
            distances.append(distance)
            basket_world = np.asarray(data.xpos[chassis], dtype=np.float64) + (
                np.asarray(data.xmat[chassis], dtype=np.float64).reshape(3, 3) @ basket_local)
            basket_distance = float(np.linalg.norm(
                np.asarray(data.xpos[fruit_body], dtype=np.float64) - basket_world))
            basket_distances.append(basket_distance)
            viewer.lookat[:] = data.xpos[chassis] + [0., 0., .35]
            viewer.distance, viewer.azimuth, viewer.elevation = 3., 135., -20.
            scene_renderer.update_scene(data, camera=viewer)
            frame = compose_progress_frame(scene_renderer.render(), grip_u8)
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            update = info['completed_updates']
            draw.text((8, 8), f'CPU preview  {stage_label}  update {update}  step {index + 1}/{steps}', fill=(255, 255, 255))
            draw.text((8, 22), (
                f'TCP-fruit {distance:.3f} m  basket {basket_distance:.3f} m  {easy_tag}'
                'gripper RGB overlay  curriculum preview not harvest proof'), fill=(244, 211, 94))
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
                  curriculum_stage=preview['stage'], reset_mode=preview['reset_mode'],
                  allow_locomotion=preview['allow_locomotion'],
                  easy=bool(preview.get('easy')),
                  easy_far_frac=float(preview.get('easy_far_frac') or 0.0),
                  update=info['completed_updates'], steps=steps, fps=fps,
                  control_dt_s=control_dt, backend=f'cpu-native-mujoco-{os.environ.get("MUJOCO_GL", "egl")}',
                  mean_tcp_fruit_distance_m=float(np.mean(distances)),
                  min_tcp_fruit_distance_m=float(np.min(distances)),
                  final_tcp_fruit_distance_m=float(distances[-1]),
                  mean_basket_distance_m=float(np.mean(basket_distances)),
                  min_basket_distance_m=float(np.min(basket_distances)),
                  final_basket_distance_m=float(basket_distances[-1]),
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
