#!/usr/bin/env python
"""Native MuJoCo preview of the commercial kiwi plantation.

This is a scripted flyover of the seeded pergola grid plus orchard floor,
compiled as a MuJoCo hfield. It is not Newton GL, not Spot's URDF, and
not a learned policy. Render-only foliage from the GPU path is omitted.

On an NVIDIA host the default is MuJoCo EGL. Pass ``--require-gpu`` to
refuse OSMesa. The Newton GL recording is ``scripts/record_scene.py``.

    python scripts/record_orchard_mujoco.py --seed 42 --require-gpu \
        --video output/orchard-mujoco.mp4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treesim.config import FruitParams
from treesim.gl_backend import bind_mujoco_gl
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
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
    return f'''<mujoco model="kiwi_plantation">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720" azimuth="125" elevation="-22" fovy="48"/>
    <headlight ambient=".22 .22 .21" diffuse=".50 .50 .48" specular=".08 .08 .07"/>
    <rgba haze=".72 .78 .84 1"/>
    <map fogstart="80" fogend="320" znear=".20" zfar="420"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".46 .64 .82" rgb2=".88 .91 .94"
             width="256" height="256"/>
    <texture type="2d" name="orchard" file="orchard_ground.png"/>
    <material name="orchard" texture="orchard" texrepeat="1 1" texuniform="false"
              reflectance="0.01" rgba="1 1 1 1"/>
    <hfield name="orchard_ground" nrow="{nrow}" ncol="{ncol}"
            size="{half} {half} {elevation:.5f} 0.08"/>
  </asset>
  <worldbody>
    <light pos="40 -80 90" dir="-0.18 0.35 -1" directional="true"
           diffuse=".62 .60 .52" specular=".10 .10 .08"/>
    <light pos="-50 40 50" diffuse=".14 .15 .12"/>
    <camera name="orbit" pos="80 -110 60" xyaxes="0.81 0.59 0 -0.18 0.25 0.95"
            fovy="48"/>
    <geom name="ground" type="hfield" hfield="orchard_ground" material="orchard"
          pos="0 0 {min_z:.5f}" rgba="1 1 1 1"
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


def camera_pose(frame, n_frames, floor, half_span_m: float):
    """High flyover of the full plantation; stay steep enough to fill the frame."""
    t = frame / max(n_frames - 1, 1)
    s = 0.5 - 0.5 * math.cos(math.pi * t)
    look = np.array([0.0, -0.08 * half_span_m + 0.16 * half_span_m * s,
                     float(floor.canopy_z(0.0, 0.0))])
    azimuth = 35.0 + 90.0 * t
    elevation = -58.0
    distance = 0.95 * half_span_m + 0.08 * half_span_m * (1.0 - s)
    return look, distance, azimuth, elevation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--video", type=Path, default=Path("output/orchard-mujoco.mp4"))
    p.add_argument("--xml", type=Path, help="Optional MJCF dump (not required)")
    p.add_argument("--pergola-rows", type=int, default=45)
    p.add_argument("--pergola-columns", type=int, default=40)
    p.add_argument("--pergola-spacing", type=float, default=5.0)
    p.add_argument("--fruit-count", type=int, default=600)
    p.add_argument("--gl", choices=("auto", "egl", "osmesa"), default="auto",
                   help="MuJoCo GL backend (auto = EGL on NVIDIA, else OSMesa)")
    p.add_argument("--require-gpu", action="store_true",
                   help="refuse OSMesa / CPU fallback; exit if no NVIDIA GPU")
    args = p.parse_args()
    if args.frames < 2:
        p.error("--frames must be at least 2")
    if args.pergola_rows < 2 or args.pergola_columns < 2:
        p.error("pergola rows and columns must be at least 2")
    if not 4.5 <= args.pergola_spacing <= 5.0:
        p.error("pergola spacing must stay inside [4.5, 5.0] m")
    if args.fruit_count < 0:
        p.error("--fruit-count must be nonnegative")

    try:
        gl_backend = bind_mujoco_gl(args.gl, require_gpu=args.require_gpu)
    except RuntimeError as exc:
        p.error(str(exc))
    import mujoco

    spacing = float(args.pergola_spacing)
    cover = floor_kwargs_for_plantation(
        args.pergola_rows, args.pergola_columns, spacing)
    half = float(cover["half_extent_m"])
    floor = sample_orchard_floor(
        args.seed, canopy_height_m=1.6,
        slope_deg=0.0, noise_m=0.01, rut_depth_m=0.05, rut_width_m=0.40,
        friction=1.0, slope_azimuth_deg=0.0, **cover,
    )
    skeleton = generate(
        height=1.6, seed=args.seed,
        rows=args.pergola_rows, columns=args.pergola_columns, spacing=spacing,
        ground_z=floor.ground_z, canopy_z=floor.canopy_z,
    )
    fruit = place_fruit(skeleton, FruitParams(
        max_count=args.fruit_count, joint="free",
        colors=((0.39, 0.27, 0.12), (0.48, 0.34, 0.17)),
    ), seed=args.seed)
    print(
        f"[orchard-mujoco] posts {args.pergola_rows}x{args.pergola_columns} "
        f"at {spacing:.1f} m; segments {len(skeleton)}; fruit {len(fruit)}; "
        f"floor {2*half:.0f}x{2*half:.0f} m; GL {gl_backend}",
        flush=True,
    )
    xml = mjcf(floor, skeleton, fruit)
    if args.xml:
        args.xml.parent.mkdir(parents=True, exist_ok=True)
        args.xml.write_text(xml)

    model = mujoco.MjModel.from_xml_string(
        xml, assets={"orchard_ground.png": floor.texture_png_bytes()})
    apply_hfield(model, floor)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(model, height=720, width=1280, max_geom=30000)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    args.video.parent.mkdir(parents=True, exist_ok=True)
    ha = (args.pergola_rows - 1) * spacing * (args.pergola_columns - 1) * spacing / 10000.0
    label = (
        f"drawtext=text='NATIVE MUJOCO plantation  seed {args.seed}':"
        f"x=28:y=28:fontsize=22:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1,"
        f"drawtext=text='{args.pergola_rows} x {args.pergola_columns} posts at "
        f"{spacing:.1f} m   ~{ha:.1f} ha   fruit {len(fruit)}':"
        f"x=28:y=60:fontsize=18:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1,"
        f"drawtext=text='scripted flyover  -  Ines plantation on orchard floor  -  "
        f"GL {gl_backend}  -  not Spot gait  -  no foliage here':"
        f"x=28:y=92:fontsize=16:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1"
    )
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x720", "-r", "30", "-i", "-", "-an", "-vf", label,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.video),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(args.frames):
            lookat, distance, azimuth, elevation = camera_pose(
                frame, args.frames, floor, half)
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
    print(floor.metrics())


if __name__ == "__main__":
    main()
