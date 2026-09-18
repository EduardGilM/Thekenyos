#!/usr/bin/env python
"""Grow and simulate a VSP table-grape vineyard in NVIDIA Newton.

Rigid clusters of visual berries hang from compliant shoots by a peduncle
beam tether (treesim/grape.py); a hard pull or ``--cut-demo`` severs it.

Examples
--------
    # Interactive OpenGL window
    python scripts/grow_vineyard.py --viewer gl

    # Headless smoke test
    python scripts/grow_vineyard.py --viewer null --frames 60 --device cpu

    # Scripted harvest: sever cluster 0 at frame 40 and dump metrics
    python scripts/grow_vineyard.py --viewer null --frames 120 --cut-demo 40 \
        --metrics output/vineyard.json --device cpu

    # Record to USD for offline rendering
    python scripts/grow_vineyard.py --viewer usd --output output/vineyard.usda \
        --frames 300 --foliage

Interactive controls (OpenGL viewer): orbit = left-drag, pan = middle/right-drag,
zoom = scroll, apply force = grab a body and drag, space = pause.
With --robot: W/S = drive, A/D = turn (the camera keeps arrows/Q/E/mouse).
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prefer_nvidia_gl() -> None:
    """On Linux hybrid-graphics machines, render the OpenGL viewer on the NVIDIA
    GPU rather than the integrated one; otherwise the GL window falls back to slow
    CPU copies every frame. Only acts when an NVIDIA GPU is present and the user
    hasn't set these already. Must run before any GL context is created."""
    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("__NV_PRIME_RENDER_OFFLOAD") or \
       os.environ.get("__GLX_VENDOR_LIBRARY_NAME"):
        return
    import glob
    has_nvidia = bool(glob.glob("/dev/nvidia[0-9]*")) or \
        os.path.isdir("/proc/driver/nvidia/gpus")
    if not has_nvidia:
        import shutil, subprocess
        if shutil.which("nvidia-smi"):
            try:
                has_nvidia = subprocess.run(
                    ["nvidia-smi", "-L"], capture_output=True, timeout=5
                ).returncode == 0
            except Exception:
                has_nvidia = False
    if has_nvidia:
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        print("[grow_vineyard] NVIDIA GPU detected; preferring it for OpenGL "
              "rendering (PRIME offload). First run is slower while Warp compiles "
              "its CUDA kernels; later runs are cached.", file=sys.stderr)


_prefer_nvidia_gl()

import warp as wp
import newton

from treesim.config import TreeConfig, preset, StiffnessModel
from treesim import builder
from treesim.sim import Sim


def make_config(args) -> TreeConfig:
    cfg = TreeConfig(lsystem=preset("vineyard"))
    cfg.lsystem.target_height = args.cordon_height
    cfg.lsystem.vy_rows = args.rows
    cfg.lsystem.vy_row_length = args.row_length
    cfg.lsystem.vy_row_spacing = args.row_spacing
    cfg.lsystem.vy_vine_spacing = args.vine_spacing
    cfg.seed = args.seed
    cfg.device = args.device or "cuda"

    cfg.deformable = True
    cfg.physics.model = StiffnessModel.BEAM
    # Shoots are thin (4-6 mm) and carry up to 2 kg clusters: at the apple
    # damping ratio they ring hard enough that the peduncle's damping force
    # alone exceeds the detach threshold.  0.3 keeps the whip sub-critical.
    cfg.physics.damping_ratio = 0.3

    cfg.fruit.enabled = True
    cfg.fruit.max_count = args.clusters
    cfg.fruit.joint = "free"
    # red / black / green table-grape colour classes (initial scene values)
    cfg.fruit.colors = ((.16, .12, .23), (.19, .15, .27), (.22, .17, .30))

    cfg.foliage.enabled = args.foliage
    cfg.foliage.min_order_for_leaves = 2
    cfg.foliage.leaf_length = 0.26
    cfg.foliage.leaf_width = 0.25
    cfg.foliage.leaves_per_terminal = 14
    cfg.foliage.leaf_color = (.22, .40, .055)
    cfg.render.wood_tint = (.55, .63, .55)

    cfg.robot.enabled = args.robot
    cfg.robot.kind = args.robot_kind
    cfg.robot.camera = args.robot and not args.no_robot_camera
    # Spawn in the alley, just past the -x row end, facing +x down the alley.
    alley_y = (1 - (args.rows - 1) / 2) * args.row_spacing - args.row_spacing / 2
    rx = -(args.row_length / 2 + 1.5) if args.robot_x is None else args.robot_x
    ry = alley_y if args.robot_y is None else args.robot_y
    cfg.robot.position = (rx, ry)
    cfg.robot.yaw = 0.0 if args.robot_yaw is None else args.robot_yaw
    if args.relic_path:
        cfg.robot.relic_path = args.relic_path
    return cfg


def _unpin_host_allocs_for_cpu() -> None:
    """On CPU-only machines there is no CUDA driver, so the GL viewer's
    pinned host allocations (wp.empty/wp.array(..., pinned=True)) fail.
    Pinning is a DMA optimisation only, so make Warp's pinned CPU allocator
    fall back to regular host memory there."""
    if wp.get_device().is_cuda:
        return
    from warp._src.context import CpuPinnedAllocator, runtime

    def allocate(self, size_in_bytes):
        ptr = runtime.core.wp_alloc_host(size_in_bytes, None)
        if not ptr:
            raise RuntimeError(
                f"Failed to allocate {size_in_bytes} bytes on device '{self.device}'")
        return ptr

    def deallocate(self, ptr, size_in_bytes):
        runtime.core.wp_free_host(ptr)

    CpuPinnedAllocator.allocate = allocate
    CpuPinnedAllocator.deallocate = deallocate


def make_viewer(args):
    import newton.viewer as V
    if args.viewer == "gl":
        _unpin_host_allocs_for_cpu()
        viewer = V.ViewerGL(headless=args.headless, paused=args.paused)
        viewer.renderer.ambient_sky = (.95, 1., .90)
        viewer.renderer.ambient_ground = (.75, .80, .62)
        viewer.renderer.sky_lower = (.62, .70, .65)
        viewer.renderer.sky_upper = (.38, .64, .86)
        viewer.renderer.spotlight_enabled = False
        return viewer
    if args.viewer == "usd":
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        return V.ViewerUSD(output_path=args.output, num_frames=args.frames)
    if args.viewer == "null":
        return V.ViewerNull(num_frames=args.frames)
    raise ValueError(args.viewer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("vineyard")
    g.add_argument("--rows", type=int, default=2, help="support rows (at least two)")
    g.add_argument("--row-length", type=float, default=12.0,
                   help="row extent along x [m]")
    g.add_argument("--row-spacing", type=float, default=2.4,
                   help="alley width between rows [m]")
    g.add_argument("--vine-spacing", type=float, default=1.5,
                   help="vine spacing along the row [m]")
    g.add_argument("--cordon-height", "--canopy-height", type=float, default=2.3,
                   help="overhead canopy height above ground [m] (1.8..3.0)")
    g.add_argument("--clusters", type=int, default=80,
                   help="max grape clusters (each is a free body)")
    g.add_argument("--seed", type=int, default=-1, help="-1 = random each run")
    g.add_argument("--foliage", action=argparse.BooleanOptionalAction, default=True,
                   help="render-only leaf cards on the shoots (massless, no physics)")
    g.add_argument("--cut-demo", type=int, default=-1, metavar="N",
                   help="sever cluster 0's peduncle after frame N (scripted harvest)")

    ro = p.add_argument_group("robot")
    ro.add_argument("--robot", action="store_true",
                    help="add a mobile manipulator (RidgebackFranka; --robot-kind spot "
                         "for the RELIC Spot, needs --relic-path)")
    ro.add_argument("--robot-kind", default="ridgeback",
                    choices=["ridgeback", "spot"])
    ro.add_argument("--relic-path", default="",
                    help="external RELIC checkout (spot only)")
    ro.add_argument("--no-robot-camera", action="store_true")
    ro.add_argument("--robot-x", type=float, default=None,
                    help="spawn x [m] (default: 1.5 m past the -x row end)")
    ro.add_argument("--robot-y", type=float, default=None,
                    help="spawn y [m] (default: alley centre between rows, or "
                         "-1.2 m beside a single row)")
    ro.add_argument("--robot-yaw", type=float, default=None,
                    help="spawn heading [rad] (default: 0 = facing +x down the alley)")

    r = p.add_argument_group("render/sim")
    r.add_argument("--viewer", default="gl", choices=["gl", "usd", "null"],
                   help="gl = interactive window; null = NO rendering (physics-only)")
    r.add_argument("--headless", action="store_true")
    r.add_argument("--paused", action="store_true")
    r.add_argument("--frames", type=int, default=300,
                   help="frames for usd/null viewers (and headless gl cap)")
    r.add_argument("--substeps", type=int, default=3)
    r.add_argument("--device", default=None)
    r.add_argument("--output", default="output/vineyard.usda",
                   help="USD output path (usd viewer)")
    r.add_argument("--metrics", default=None, metavar="FILE",
                   help="optional JSON: detached count + ClusterContact metrics")
    r.add_argument("--snapshot", default=None, metavar="FILE.png",
                   help="gl viewer only: save one rendered frame at frame 30")
    r.add_argument("--video", default=None, metavar="FILE.mp4",
                   help="gl viewer only: record every other frame via ffmpeg")
    r.add_argument("--max-bodies", type=int, default=60000)
    return p.parse_args()


def main():
    import math
    import random
    args = parse_args()
    if args.seed < 0:
        args.seed = random.randrange(2**31)
    if args.device:
        wp.set_device(args.device)

    cfg = make_config(args)
    print(f"[grow_vineyard] building vineyard (rows={args.rows}, "
          f"row_length={args.row_length}, cordon={args.cordon_height}, "
          f"clusters={args.clusters}, seed={args.seed}) ...")
    tm = builder.generate_and_build(cfg, max_bodies=args.max_bodies)
    print(f"[grow_vineyard] bodies={tm.model.body_count} "
          f"joints={tm.model.joint_count} dof={tm.model.joint_dof_count} "
          f"clusters={len(tm.apple_bodies)}")

    sim = Sim(tm, solver="mujoco", fps=60, substeps=args.substeps,
              collisions=True)
    print(f"[grow_vineyard] solver=mujoco substeps={args.substeps} "
          f"collisions={sim.collisions}")

    viewer = make_viewer(args)
    sim.set_viewer(viewer)

    # frame the row
    try:
        L = args.row_length
        alley_y = (1 - (args.rows - 1) / 2) * args.row_spacing - args.row_spacing / 2
        pos = wp.vec3(-L / 2 - .8, alley_y + .12, 1.35)
        look = wp.vec3(L / 2, alley_y, 1.65)
        d = look - pos
        d = d / (wp.length(d) + 1e-9)
        viewer.set_camera(pos=pos,
                          pitch=float(math.degrees(math.asin(d[2]))),
                          yaw=float(math.degrees(math.atan2(d[1], d[0]))))
    except Exception:
        pass

    robot_driver = None
    if tm.robot_data is not None:
        if cfg.robot.kind == "spot":
            from treesim import spot as _spot
            _spot.SpotController(sim)          # registers itself on the sim
            print("[robot] RELIC Spot policy controller on.")
        else:
            from treesim import robot as _robot
            robot_driver = _robot.RobotDriver(sim, tm, cfg.robot)
            if _robot.take_over_wasd(viewer):
                print("[robot] W/S = drive, A/D = turn.  Camera: arrows / Q / E / mouse.")

    print("[grow_vineyard] running. Close the window (or Ctrl-C) to stop.")
    total_frames = (args.frames if (args.viewer in ("usd", "null")
                                    or args.headless) else 0)
    frame = 0
    cut_done = False
    snap_done = False
    encoder = None
    try:
        while viewer.is_running() and not (total_frames and frame >= total_frames):
            if viewer.should_step():
                sim.step()
                frame += 1
                if args.cut_demo >= 0 and not cut_done and frame >= args.cut_demo:
                    if sim.apples is not None and sim.apples.cut(0):
                        print(f"[cut-demo] frame {frame}: severed cluster 0 peduncle")
                    cut_done = True
                if robot_driver is not None:
                    robot_driver.update(viewer)
            sim.render()
            # frame grab (gl viewer, incl. headless): snapshot once, video always
            if args.viewer == "gl" and ((args.snapshot and not snap_done
                                         and frame >= 30) or args.video):
                pixels = viewer.get_frame().numpy()
                if args.snapshot and not snap_done and frame >= 30:
                    from PIL import Image
                    out = os.path.abspath(args.snapshot)
                    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                    Image.fromarray(pixels).save(out)
                    print(f"[snapshot] saved {out} (frame {frame})")
                    snap_done = True
                if args.video:
                    if encoder is None:
                        import subprocess, shutil
                        if shutil.which("ffmpeg") is None:
                            raise RuntimeError("--video needs ffmpeg")
                        h, w = pixels.shape[:2]
                        out = os.path.abspath(args.video)
                        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                        encoder = subprocess.Popen(
                            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                             "-framerate", "30", "-i", "pipe:0", "-an",
                             "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)
                    encoder.stdin.write(pixels.tobytes())
    except KeyboardInterrupt:
        pass
    viewer.close()
    if encoder is not None:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the recording")
        print(f"[video] saved {args.video} ({frame} frames)")

    detached = int(sim.apples.broken_count) if sim.apples is not None else 0
    print(f"[grow_vineyard] clusters detached: {detached}")
    if sim.grape_contact is not None:
        m = sim.grape_contact.metrics()
        peak = m["peak_force_N"]
        print(f"[grow_vineyard] cluster contact peak force [N]: "
              f"max={max(peak) if peak else 0.0:.2f} "
              f"rupture_risk={sum(m['rupture_risk'])}/{len(peak)}")
        if args.metrics:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics)) or ".",
                        exist_ok=True)
            with open(args.metrics, "w") as fh:
                json.dump(dict(detached=detached, cluster_contact=m), fh, indent=2)
            print(f"[metrics] saved to {args.metrics}")
    elif args.metrics:
        import json
        with open(args.metrics, "w") as fh:
            json.dump(dict(detached=detached), fh, indent=2)

    print(f"[grow_vineyard] done ({frame} frames).")


if __name__ == "__main__":
    main()
