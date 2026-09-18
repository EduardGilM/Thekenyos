#!/usr/bin/env python
"""Record a 30 fps MP4 at simulation speed; requires ffmpeg on PATH.

python scripts/record_scene.py --video output/pergola.mp4 --orbit \
    --preset pergola --foliage --seed 42 --frames 600
Remaining arguments are passed to grow_tree.py. Physics is unchanged.
"""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--orbit", action="store_true")
    args, scene_args = parser.parse_known_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg must be installed to record MP4")
    args.video.parent.mkdir(parents=True, exist_ok=True)

    import grow_tree as demo

    original_render = demo.Sim.render
    encoder = None
    frame = 0

    def record_render(sim):
        nonlocal encoder, frame
        if args.orbit:
            lo, hi = sim.tree.skeleton.bounds()
            center = (lo + hi) / 2
            radius = 1.5 * max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 2.0)
            angle = math.radians(45 + 12 * sim.sim_time)
            eye = center + np.array([radius * math.cos(angle), radius * math.sin(angle),
                                     0.5 * sim.tree.skeleton.height()])
            direction = center - eye
            sim.viewer.set_camera(
                pos=demo.wp.vec3(*map(float, eye)),
                yaw=math.degrees(math.atan2(direction[1], direction[0])),
                pitch=math.degrees(math.atan2(direction[2], np.linalg.norm(direction[:2]))),
            )
        original_render(sim)
        frame += 1
        if frame % 2:  # Simulator runs at 60 Hz; retain every second frame.
            return
        pixels = sim.viewer.get_frame().numpy()
        if encoder is None:
            h, w = pixels.shape[:2]
            encoder = subprocess.Popen([
                ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                "-framerate", "30", "-i", "pipe:0", "-an",
                "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "fast",
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
