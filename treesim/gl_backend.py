"""NVIDIA / MuJoCo GL selection for recordings.

Newton GL and MuJoCo EGL both need a real GPU on Linux. This module does not
import Warp or Newton, so the native MuJoCo preview can use it on CPU-only
hosts. Detecting a GPU is not the same as having created a GL context.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess


def nvidia_gpu_present() -> bool:
    """True when this Linux host exposes an NVIDIA device."""
    if glob.glob("/dev/nvidia[0-9]*") or os.path.isdir("/proc/driver/nvidia/gpus"):
        return True
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        return subprocess.run(
            [smi, "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def bind_mujoco_gl(mode: str = "auto", require_gpu: bool = False) -> str:
    """Import MuJoCo, then bind ``MUJOCO_GL``.

    MuJoCo 3.8.1 crashes if OSMesa is selected before the package import.
    ``auto`` uses EGL on an NVIDIA host and OSMesa otherwise. ``require_gpu``
    refuses OSMesa so a CPU recording cannot be labelled as a GPU render.
    """
    mode = str(mode).lower()
    if mode not in {"auto", "egl", "osmesa"}:
        raise ValueError("mode must be auto, egl or osmesa")
    has_gpu = nvidia_gpu_present()
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "No NVIDIA device is visible (/dev/nvidia* or nvidia-smi). "
            "MuJoCo EGL needs that GPU. Do not fall back to OSMesa. "
            "On the GTX 1650 machine: python scripts/record_orchard_mujoco.py "
            "--seed 42 --require-gpu --video output/orchard-mujoco.mp4"
        )
    if mode == "egl" and not has_gpu:
        raise RuntimeError(
            "MUJOCO_GL=egl was requested but no NVIDIA device is visible"
        )
    if mode == "osmesa":
        backend = "osmesa"
    elif mode == "egl" or has_gpu:
        backend = "egl"
    else:
        backend = "osmesa"
    # MuJoCo 3.8.1 picks the platform library at import. OSMesa must not be
    # selected before import (it can abort). EGL must be selected before import
    # or Renderer falls through to GLFW and needs a DISPLAY.
    if backend == "egl":
        os.environ["MUJOCO_GL"] = "egl"
        import mujoco  # noqa: F401
    else:
        os.environ.pop("MUJOCO_GL", None)
        import mujoco  # noqa: F401
        os.environ["MUJOCO_GL"] = backend
    return backend
