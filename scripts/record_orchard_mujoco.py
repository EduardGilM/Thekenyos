#!/usr/bin/env python
"""Native MuJoCo preview of the kiwi orchard floor plus planted pergola.

This is a scripted camera orbit of the same seeded heightfield used by
``--terrain``, compiled as a MuJoCo hfield. It is not Newton GL, not Spot's
URDF, and not a learned policy.

    MUJOCO_GL=osmesa python scripts/record_orchard_mujoco.py \
        --seed 42 --video output/orchard-mujoco.mp4
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treesim.config import FruitParams
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import sample_orchard_floor
from treesim.pergola import generate, place_fruit


def _rgba(rgb, a=1.0) -> str:
    return " ".join(f"{float(c):.3f}" for c in (*rgb, a))


def mjcf(floor, skeleton, fruit) -> str:
    heights = np.asarray(floor.heights_m, dtype=np.float64)
    min_z = float(heights.min())
    max_z = float(heights.max())
    elevation = max(max_z - min_z, 1e-4)
    nrow, ncol = heights.shape
    half = float(floor.half_extent_m)
    mu = float(floor.friction)
    geoms = []
    wood = ((0.45, 0.28, 0.12), (0.35, 0.22, 0.10), (0.28, 0.42, 0.14))
    for seg in skeleton:
        rgb = wood[min(seg.order, 2)]
        geoms.append(
            f'    <geom type="capsule" fromto="'
            f'{seg.start[0]:.5f} {seg.start[1]:.5f} {seg.start[2]:.5f} '
            f'{seg.end[0]:.5f} {seg.end[1]:.5f} {seg.end[2]:.5f}" '
            f'size="{seg.mean_radius:.5f}" rgba="{_rgba(rgb)}" '
            f'contype="0" conaffinity="0"/>'
        )
    for f in fruit:
        center = f.attach - np.array([0.0, 0.0, STEM_LENGTH + float(f.radii[2])])
        rx, ry, rz = (float(v) for v in f.radii)
        geoms.append(
            f'    <geom type="ellipsoid" pos="{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}" '
            f'size="{rx:.5f} {ry:.5f} {rz:.5f}" rgba="{_rgba(f.color)}" '
            f'contype="0" conaffinity="0"/>'
        )
    look_z = 0.5 * (min_z + float(floor.canopy_z(0.0, 0.0)))
    return f'''<mujoco model="kiwi_orchard_floor">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="125" elevation="-22" fovy="48"/>
    <headlight ambient=".28 .30 .24" diffuse=".62 .62 .55" specular=".12 .12 .10"/>
    <rgba haze=".70 .80 .88 1"/>
    <map fogstart="14" fogend="48" znear=".12" zfar="80"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".48 .66 .84" rgb2=".90 .93 .95"
             width="256" height="256"/>
    <hfield name="orchard_ground" nrow="{nrow}" ncol="{ncol}"
            size="{half} {half} {elevation:.5f} 0.08"/>
  </asset>
  <worldbody>
    <light pos="6 -10 14" dir="-0.25 0.45 -1" directional="true"
           diffuse=".85 .80 .68" specular=".25 .25 .22"/>
    <light pos="-8 6 9" diffuse=".18 .22 .16"/>
    <camera name="orbit" pos="8 -11 6" xyaxes="0.81 0.59 0 -0.18 0.25 0.95"
            fovy="48"/>
    <geom name="ground" type="hfield" hfield="orchard_ground"
          pos="0 0 {min_z:.5f}" rgba="0.36 0.42 0.22 1"
          friction="{mu:.3f} 0.01 0.001"/>
{chr(10).join(geoms)}
    <geom type="sphere" pos="0 0 {look_z:.4f}" size="0.001" rgba="0 0 0 0"
          contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
'''


def apply_hfield(model, floor) -> None:
    heights = np.asarray(floor.heights_m, dtype=np.float64)
    min_z = float(heights.min())
    span = max(float(heights.max()) - min_z, 1e-4)
    model.hfield_data[:] = ((heights - min_z) / span).astype(np.float64).ravel()


def camera_pose(frame, n_frames, look_high, look_low):
    t = frame / max(n_frames - 1, 1)
    s = 0.5 - 0.5 * math.cos(math.pi * t)
    azimuth = 30.0 + 300.0 * t
    elevation = -50.0 + 32.0 * s
    distance = 17.0 - 8.8 * s
    look = (1.0 - s) * look_high + s * look_low
    return look, distance, azimuth, elevation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--video", type=Path, default=Path("output/orchard-mujoco.mp4"))
    p.add_argument("--xml", type=Path, help="Optional MJCF dump (not required)")
    args = p.parse_args()
    if args.frames < 2:
        p.error("--frames must be at least 2")

    # MuJoCo 3.8.1 fails if OSMesa is selected before the package import.
    os.environ.pop("MUJOCO_GL", None)
    import mujoco
    os.environ["MUJOCO_GL"] = "osmesa"

    floor = sample_orchard_floor(args.seed, canopy_height_m=1.6)
    skeleton = generate(height=1.6, seed=args.seed, ground_z=floor.ground_z,
                        canopy_z=floor.canopy_z)
    fruit = place_fruit(skeleton, FruitParams(
        max_count=40, joint="free",
        colors=((0.39, 0.27, 0.12), (0.48, 0.34, 0.17)),
    ), seed=args.seed)
    xml = mjcf(floor, skeleton, fruit)
    if args.xml:
        args.xml.parent.mkdir(parents=True, exist_ok=True)
        args.xml.write_text(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    apply_hfield(model, floor)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    look_high = np.array([0.0, 0.0, float(floor.canopy_z(0.0, 0.0)) + 0.2])
    look_low = np.array([0.0, 0.4, float(floor.ground_z(0.0, 0.0)) + 0.55])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(model, height=720, width=1280)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    args.video.parent.mkdir(parents=True, exist_ok=True)
    m = floor.metrics()
    label = (f"drawtext=text='NATIVE MUJOCO hfield  seed {args.seed}':"
             f"x=28:y=28:fontsize=22:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1,"
             f"drawtext=text='slope {m['slope_deg']:.1f} deg   ruts {100*m['rut_depth_m']:.0f} cm x "
             f"{100*m['rut_width_m']:.0f} cm   noise {100*m['ground_noise_m']:.1f} cm   "
             f"mu={m['friction']:.2f}':x=28:y=60:fontsize=18:fontcolor=white:"
             f"shadowcolor=black:shadowx=1:shadowy=1,"
             f"drawtext=text='scripted orbit  -  pergola planted on sampled floor  -  not Spot gait':"
             f"x=28:y=92:fontsize=16:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1")
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", "30", "-i", "-", "-an", "-vf", label,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.video),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(args.frames):
            lookat, distance, azimuth, elevation = camera_pose(frame, args.frames, look_high, look_low)
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera)
            encoder.stdin.write(renderer.render().tobytes())
            if frame % 30 == 0:
                print(f"[orchard-mujoco] frame {frame}/{args.frames}", flush=True)
    finally:
        renderer.close()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    print(f"[orchard-mujoco] {args.video} ({args.frames} frames at 30 fps)")
    print(m)


if __name__ == "__main__":
    main()
