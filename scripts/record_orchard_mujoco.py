#!/usr/bin/env python
"""Native MuJoCo preview of the commercial kiwi plantation.

This is a scripted approach into a row of the seeded pergola grid plus orchard
floor, compiled as a MuJoCo hfield. The heightfield sits on a visual earth
bulk so the hillside is not a floating card. It is not Newton GL, not Spot's
URDF, and not a learned policy.

    python scripts/record_orchard_mujoco.py --seed 42 --require-gpu \
        --hillside --canopy-spacing .15 --pergola-rows 9 --pergola-columns 7 \
        --fruit-count 180 --video output/orchard-mujoco.mp4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from treesim.config import FoliageParams, FruitParams
from treesim.gl_backend import bind_mujoco_gl
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit


def _rgba(rgb, a=1.0) -> str:
    return " ".join(f"{float(c):.3f}" for c in (*rgb, a))


def _leaf_mjcf(skeleton, fp: FoliageParams, seed: int) -> tuple[list[str], list[str]]:
    from treesim.foliage import (
        LEAF_SIZE_CLASSES, leaf_blade_arrays, leaf_blade_style,
        place_canopy_leaves, place_leaves,
    )
    placements = place_leaves(skeleton, fp, seed=seed)
    if fp.canopy_spacing_m:
        placements.extend(place_canopy_leaves(skeleton, fp, seed=seed))
    assets = []
    style = leaf_blade_style(fp.leaf_shape)
    for i, scale in enumerate(LEAF_SIZE_CLASSES):
        verts, faces = leaf_blade_arrays(
            fp.leaf_length * scale, fp.leaf_width * scale, **style)
        vertex = " ".join(f"{float(v):.5f}" for v in verts.reshape(-1))
        face = " ".join(str(int(i)) for i in faces.reshape(-1))
        assets.append(f'    <mesh name="kiwi_leaf_{i}" vertex="{vertex}" face="{face}"/>')
    rgba = _rgba(fp.leaf_color)
    geoms = []
    nominal = max(float(fp.leaf_length), 1e-9)
    for p in placements:
        cls = int(np.argmin([abs(p.length / nominal - s) for s in LEAF_SIZE_CLASSES]))
        x, y, z = (float(v) for v in p.attach)
        qx, qy, qz, qw = (float(v) for v in p.frame)
        geoms.append(
            f'    <geom type="mesh" mesh="kiwi_leaf_{cls}" '
            f'pos="{x:.4f} {y:.4f} {z:.4f}" quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}" '
            f'rgba="{rgba}" contype="0" conaffinity="0" group="2"/>'
        )
    return assets, geoms


def mjcf(floor, skeleton, fruit, leaf_assets=(), leaf_geoms=(),
         width=1920, height=1080) -> str:
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
    geoms.extend(leaf_geoms)
    look_z = 0.5 * (min_z + float(floor.canopy_z(0.0, 0.0)))
    leaf_xml = "\n".join(leaf_assets)
    # Visual earth bulk so the heightfield is a hillside cut, not a floating card.
    bulk = max(6.0, 0.35 * elevation)
    skirt = 0.45
    earth = "0.27 0.17 0.08 1"
    soil_geoms = [
        f'    <geom name="earth_mass" type="box" size="{half + 0.8:.3f} {half + 0.8:.3f} {bulk:.3f}" '
        f'pos="0 0 {min_z - bulk:.4f}" rgba="{earth}" contype="0" conaffinity="0"/>',
        f'    <geom name="earth_x_pos" type="box" size="{skirt:.3f} {half:.3f} {(elevation + bulk) * 0.5:.3f}" '
        f'pos="{half:.4f} 0 {min_z - bulk + 0.5 * (elevation + bulk):.4f}" rgba="{earth}" '
        f'contype="0" conaffinity="0"/>',
        f'    <geom name="earth_x_neg" type="box" size="{skirt:.3f} {half:.3f} {(elevation + bulk) * 0.5:.3f}" '
        f'pos="{-half:.4f} 0 {min_z - bulk + 0.5 * (elevation + bulk):.4f}" rgba="{earth}" '
        f'contype="0" conaffinity="0"/>',
        f'    <geom name="earth_y_pos" type="box" size="{half:.3f} {skirt:.3f} {(elevation + bulk) * 0.5:.3f}" '
        f'pos="0 {half:.4f} {min_z - bulk + 0.5 * (elevation + bulk):.4f}" rgba="{earth}" '
        f'contype="0" conaffinity="0"/>',
        f'    <geom name="earth_y_neg" type="box" size="{half:.3f} {skirt:.3f} {(elevation + bulk) * 0.5:.3f}" '
        f'pos="0 {-half:.4f} {min_z - bulk + 0.5 * (elevation + bulk):.4f}" rgba="{earth}" '
        f'contype="0" conaffinity="0"/>',
    ]
    return f'''<mujoco model="kiwi_plantation">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{int(width)}" offheight="{int(height)}" azimuth="125" elevation="-22" fovy="46"/>
    <headlight ambient=".18 .19 .16" diffuse=".42 .44 .38" specular=".12 .12 .10"/>
    <rgba haze=".70 .78 .86 1"/>
    <map fogstart="35" fogend="180" znear=".05" zfar="420"/>
    <quality shadowsize="2048" offsamples="4"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".42 .62 .86" rgb2=".90 .93 .96"
             width="512" height="512"/>
    <texture type="2d" name="orchard" file="orchard_ground.png"/>
    <material name="orchard" texture="orchard" texrepeat="1 1" texuniform="false"
              reflectance="0.02" rgba="1 1 1 1"/>
    <hfield name="orchard_ground" nrow="{nrow}" ncol="{ncol}"
            size="{half} {half} {elevation:.5f} {bulk:.3f}"/>
{leaf_xml}
  </asset>
  <worldbody>
    <light pos="30 -70 80" dir="-0.15 0.42 -1" directional="true"
           diffuse=".70 .66 .52" specular=".18 .16 .12"/>
    <light pos="-40 30 55" diffuse=".16 .18 .14"/>
    <camera name="orbit" pos="80 -110 60" xyaxes="0.81 0.59 0 -0.18 0.25 0.95"
            fovy="46"/>
    <geom name="ground" type="hfield" hfield="orchard_ground" material="orchard"
          pos="0 0 {min_z:.5f}" rgba="1 1 1 1"
          friction="{mu:.3f} 0.01 0.001"/>
{chr(10).join(soil_geoms)}
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


def free_camera_eye(lookat, distance, azimuth, elevation) -> np.ndarray:
    """World-space eye of a MuJoCo free camera (matches ``mjv_updateCamera``)."""
    az = math.radians(float(azimuth))
    el = math.radians(float(elevation))
    d = float(distance)
    lookat = np.asarray(lookat, dtype=float)
    return lookat + d * np.array([
        math.cos(el) * math.sin(az),
        -math.cos(el) * math.cos(az),
        math.sin(el),
    ])


def mjv_from_eye_target(eye, target):
    """Convert an explicit eye/target pair into MuJoCo free-camera sphericals."""
    lookat = np.asarray(target, dtype=float)
    delta = np.asarray(eye, dtype=float) - lookat
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance < 1e-6:
        raise ValueError("camera eye and target must be distinct finite points")
    azimuth = math.degrees(math.atan2(delta[0], -delta[1]))
    elevation = math.degrees(math.asin(float(np.clip(delta[2] / distance, -1.0, 1.0))))
    return lookat, distance, azimuth, elevation


def _aisle_eye_z(floor, x, y) -> float:
    """Keep the eye in the working volume: above the aisle, below the leaf roof."""
    ground = float(floor.ground_z(x, y))
    canopy = float(floor.canopy_z(x, y))
    z = min(ground + 1.22, canopy - 0.38)
    z = max(z, ground + 0.95)
    return min(z, canopy - 0.28)


def camera_pose(frame, n_frames, floor, half_span_m: float, spacing: float = 5.0):
    """Enter from the south margin along a grass aisle, always under the canopy."""
    t = frame / max(n_frames - 1, 1)
    s = t * t * (3.0 - 2.0 * t)
    x = 0.0
    y0 = -float(half_span_m) + 2.2
    y1 = -0.35 * float(spacing)
    y = (1.0 - s) * y0 + s * y1
    look_ahead = 5.8 - 1.8 * s
    x_look = 0.32 * float(spacing)
    eye = np.array([x, y, _aisle_eye_z(floor, x, y)])
    target = np.array([x_look, y + look_ahead,
                       _aisle_eye_z(floor, x_look, y + look_ahead) - 0.06])
    return mjv_from_eye_target(eye, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--video", type=Path, default=Path("output/orchard-mujoco.mp4"))
    p.add_argument("--xml", type=Path, help="Optional MJCF dump (not required)")
    p.add_argument("--pergola-rows", type=int, default=9)
    p.add_argument("--pergola-columns", type=int, default=7)
    p.add_argument("--pergola-spacing", type=float, default=5.0)
    p.add_argument("--fruit-count", type=int, default=180)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--slope-deg", type=float, default=3.5,
                   help="mild residual orchard-floor tilt [deg]; assumed, not a survey")
    p.add_argument("--slope-azimuth-deg", type=float, default=38.0)
    p.add_argument("--landform-m", type=float, default=2.4,
                   help="rolling value-noise landform amplitude [m]; 0 disables")
    p.add_argument("--landform-wavelength-m", type=float, default=18.0,
                   help="dominant landform wavelength [m]")
    p.add_argument("--canopy-spacing", type=float, default=0.15,
                   help="render-only leaf-roof spacing [m]; 0 disables infill")
    p.add_argument("--hillside", action="store_true", default=True)
    p.add_argument("--flat", action="store_true",
                   help="disable the default hillside tilt")
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
    if args.width < 64 or args.height < 64:
        p.error("resolution must be at least 64x64")
    if not math.isfinite(args.canopy_spacing) or args.canopy_spacing < 0 or 0 < args.canopy_spacing < .03:
        p.error("--canopy-spacing must be zero or finite and at least 0.03 m")
    if args.flat:
        args.slope_deg = 0.0
        args.landform_m = 0.0
    if not math.isfinite(args.landform_m) or args.landform_m < 0.0:
        p.error("--landform-m must be finite and >= 0")

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
        slope_deg=float(args.slope_deg), noise_m=0.04, rut_depth_m=0.05, rut_width_m=0.40,
        friction=1.0, slope_azimuth_deg=float(args.slope_azimuth_deg),
        landform_m=float(args.landform_m),
        landform_wavelength_m=float(args.landform_wavelength_m), **cover,
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
    fp = FoliageParams(
        enabled=True, leaves_per_terminal=8, min_order_for_leaves=2,
        leaf_length=0.22, leaf_width=0.17, leaf_shape="cordate",
        leaf_color=(0.14, 0.36, 0.10),
        canopy_spacing_m=float(args.canopy_spacing),
    )
    leaf_assets, leaf_geoms = _leaf_mjcf(skeleton, fp, args.seed)
    print(
        f"[orchard-mujoco] posts {args.pergola_rows}x{args.pergola_columns} "
        f"at {spacing:.1f} m; segments {len(skeleton)}; fruit {len(fruit)}; "
        f"leaves {len(leaf_geoms)}; floor {2*half:.0f}x{2*half:.0f} m; "
        f"slope {args.slope_deg:.1f} deg; landform {args.landform_m:.1f} m; "
        f"GL {gl_backend}",
        flush=True,
    )
    xml = mjcf(floor, skeleton, fruit, leaf_assets, leaf_geoms,
               width=args.width, height=args.height)
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
    max_geom = max(30000, int(model.ngeom) + 2048)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width, max_geom=max_geom)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
    args.video.parent.mkdir(parents=True, exist_ok=True)
    ha = (args.pergola_rows - 1) * spacing * (args.pergola_columns - 1) * spacing / 10000.0
    label = (
        f"drawtext=text='NATIVE MUJOCO hillside kiwi block  seed {args.seed}':"
        f"x=28:y=28:fontsize=22:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1,"
        f"drawtext=text='{args.pergola_rows} x {args.pergola_columns} posts at "
        f"{spacing:.1f} m   ~{ha:.2f} ha   fruit {len(fruit)}   leaves {len(leaf_geoms)}':"
        f"x=28:y=60:fontsize=18:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1,"
        f"drawtext=text='scripted aisle entry  -  rolling landform {args.landform_m:.1f} m  "
        f"residual {args.slope_deg:.1f} deg  -  render-only foliage  -  GL {gl_backend}  -  not Spot gait':"
        f"x=28:y=92:fontsize=16:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1"
    )
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", "30", "-i", "-", "-an", "-vf", label,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.video),
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(args.frames):
            lookat, distance, azimuth, elevation = camera_pose(
                frame, args.frames, floor, half, spacing)
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
