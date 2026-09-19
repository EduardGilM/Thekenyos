#!/usr/bin/env python
"""Record a 30 fps MP4 on the NVIDIA Newton GL viewer.

python scripts/record_scene.py --video output/plantation-gpu.mp4 --orbit \
    --preset pergola --terrain --foliage --seed 42 --frames 600
Remaining arguments are passed to grow_tree.py. Physics is unchanged.
This is a scripted camera, not Spot gait. Exits if no NVIDIA GPU is present.
"""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treesim.gl_backend import nvidia_gpu_present


def _orbit_eye(sim, demo):
    """Look at the skeleton. A 200 m block needs a steep flyover, not a ground-level orbit."""
    lo, hi = sim.tree.skeleton.bounds()
    center = (lo + hi) / 2
    span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 2.0)
    angle = math.radians(45 + 12 * sim.sim_time)
    if span > 40.0:
        dist = 0.95 * (0.5 * span)
        elev = math.radians(58.0)
        eye = center + np.array([
            dist * math.cos(elev) * math.cos(angle),
            dist * math.cos(elev) * math.sin(angle),
            dist * math.sin(elev),
        ])
    else:
        radius = 1.5 * span
        eye = center + np.array([
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.5 * sim.tree.skeleton.height(),
        ])
    direction = center - eye
    sim.viewer.set_camera(
        pos=demo.wp.vec3(*map(float, eye)),
        yaw=math.degrees(math.atan2(direction[1], direction[0])),
        pitch=math.degrees(math.atan2(direction[2], np.linalg.norm(direction[:2]))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--orbit", action="store_true")
    args, scene_args = parser.parse_known_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg must be installed to record MP4")
    if not nvidia_gpu_present():
        parser.error(
            "No NVIDIA device is visible (/dev/nvidia* or nvidia-smi). "
            "Newton GL will not run. On a 4 GB GTX 1650 use the MuJoCo EGL "
            "flyover for the full 45x40 block; Newton foliage needs a cropped "
            "grid, e.g. --pergola-rows 5 --pergola-columns 4."
        )
    args.video.parent.mkdir(parents=True, exist_ok=True)

    import grow_tree as demo

    original_render = demo.Sim.render
    encoder = None
    frame = 0

    def record_render(sim):
        nonlocal encoder, frame
        if args.orbit:
            _orbit_eye(sim, demo)
        original_render(sim)
        frame += 1
        if frame % 2:  # Simulator runs at 60 Hz; retain every second frame.
            return
        pixels = sim.viewer.get_frame().numpy()
        if encoder is None:
            h, w = pixels.shape[:2]
            label = (
                "scale=1280:-2,"
                "drawtext=text='NEWTON GL GPU  scripted orbit  not Spot gait':"
                "x=28:y=28:fontsize=20:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1"
            )
            encoder = subprocess.Popen([
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                "-framerate", "30", "-i", "pipe:0", "-an",
                "-vf", label, "-c:v", "libx264", "-preset", "fast",
                "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.video),
            ], stdin=subprocess.PIPE)
        encoder.stdin.write(pixels.tobytes())

    demo.Sim.render = record_render
    sys.argv = [sys.argv[0], *scene_args, "--viewer", "gl", "--headless"]
    try:
        demo.main()
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the recording")
    if encoder is None:
        raise RuntimeError("No video frames were recorded; use --frames 2 or more")
    print(f"[video] {args.video} ({frame // 2} frames at 30 fps)")


if __name__ == "__main__":
    main()
