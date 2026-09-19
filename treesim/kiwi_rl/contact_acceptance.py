"""Acceptance screen for deformable backend probe reports.

This is a diagnostic screen only. It does not establish training readiness or
harvesting success.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any


_PROFILE = ("timestep_s", "integrator", "solver", "iterations", "tolerance",
            "contact_time_s")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _profile(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("numerical_profile", manifest.get("profile"))
    if not isinstance(value, dict):
        raise ValueError("manifest lacks numerical_profile")
    return value


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def read_contact_gate(path: str | Path, manifest: dict[str, Any],
                      max_penetration_m: float = .002) -> dict[str, Any]:
    """Evaluate raw grip evidence against a scene manifest.

    ``passed`` from the probe is retained as ``strict_pass``. The returned
    ``accepted`` value applies the explicit hackathon penetration allowance;
    all other raw evidence gates remain mandatory.
    """
    if not isinstance(manifest, dict) or not _finite(max_penetration_m) or max_penetration_m <= 0:
        raise ValueError("valid manifest and positive penetration limit required")
    report = json.loads(Path(path).read_text())
    if not isinstance(report, dict):
        raise ValueError("contact report must be an object")
    if report.get("report_kind") != "deformable-backend-screen/v2":
        raise ValueError("unsupported contact report kind")
    if report.get("case") != "grip" or report.get("backend") != "gpu":
        raise ValueError("contact gate requires the GPU grip screen")
    if report.get("finite_all_worlds") is not True:
        raise ValueError("contact report contains a non-finite world")
    for section in ("gpu_numerical", "contacts"):
        flags = report.get(section, {}).get("flags", ()) if isinstance(report.get(section, {}), dict) else ()
        if not isinstance(flags, list) or not flags or any(flags):
            raise ValueError(f"{section} flags report a failure")
    for key in ("source_sha256", "scene_sha256"):
        if not isinstance(report.get(key), str) or not _HEX64.fullmatch(report[key]):
            raise ValueError(f"invalid {key}")
    profile = _profile(manifest)
    for key in _PROFILE:
        if key not in report or key not in profile or not _same(report[key], profile[key]):
            raise ValueError(f"contact profile mismatch: {key}")
    material = manifest.get("material", {})
    if "mesh_count" in material and report.get("mesh_count") != material["mesh_count"]:
        raise ValueError("contact material mesh-count mismatch")
    for normalization in (manifest.get("mesh_normalization"), report.get("mesh_normalization")):
        if not isinstance(normalization, dict):
            raise ValueError("Geometry-preserving mesh normalization is required")
        error = normalization.get("max_surface_error_m")
        if not _finite(error) or error > 1e-6:
            raise ValueError("Mesh normalization changed collision geometry")
    for key in ("source_sha256", "scene_sha256"):
        expected = manifest.get(key) or manifest.get("hashes", {}).get(key)
        if expected is not None and expected != report[key]:
            raise ValueError(f"contact source mismatch: {key}")
    required = ("mesh_flex_contacts", "minimum_volume_ratio", "hold_samples",
                "sampled_bilateral_fraction", "max_hold_motion_m",
                "post_release_jaw_load_N", "max_hand_penetration_m")
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError("contact report missing raw evidence: " + ", ".join(missing))
    worlds = int(report.get("worlds", len(report["sampled_bilateral_fraction"])))
    fractions = report["sampled_bilateral_fraction"]
    hold_motion = report["max_hold_motion_m"]
    release = report["post_release_jaw_load_N"]
    arrays = (fractions, hold_motion, release)
    if worlds < 1 or any(not isinstance(x, list) or len(x) != worlds for x in arrays):
        raise ValueError("per-world contact evidence has inconsistent shape")
    ground_steps = report["contacts"].get("first_ground_step", [])
    volumes = report["gpu_numerical"].get("minimum_volume_ratio", [])
    if len(ground_steps) != worlds or len(volumes) != worlds:
        raise ValueError("Missing per-world ground or volume evidence")
    ground_times = [row[0] * report["timestep_s"] for row in ground_steps if len(row) == 1]
    raw_ok = bool(
        report["mesh_flex_contacts"] > 0 and _finite(report["minimum_volume_ratio"]) and
        report["minimum_volume_ratio"] > .1 and report["hold_samples"] > 0 and
        all(_finite(x) and x >= .95 for x in fractions) and
        all(_finite(x) and x < .02 for x in hold_motion) and
        all(_finite(x) and x < .01 for x in release) and
        len(ground_times) == worlds and all(_finite(x) and 2.8 <= x <= report.get("simulated_seconds", 0.) for x in ground_times) and
        all(_finite(x) and x > .1 for x in volumes) and
        _finite(report["max_hand_penetration_m"]) and
        report["max_hand_penetration_m"] <= max_penetration_m
    )
    per_world_penetration = report.get("max_hand_penetration_by_world_m")
    if per_world_penetration is not None:
        if (not isinstance(per_world_penetration, list) or len(per_world_penetration) != worlds or
                not all(_finite(x) and x <= max_penetration_m for x in per_world_penetration)):
            raw_ok = False
    strict_penetration = report["max_hand_penetration_m"] <= .001
    sampled = int(report.get("penetration_worlds_sampled", 1))
    limited = sampled < worlds and "max_hand_penetration_by_world_m" not in report
    payload_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "accepted": raw_ok,
        "strict_pass": bool(report.get("passed") is True),
        "strict_penetration_pass": strict_penetration,
        "approximation": "hackathon_max_penetration_2mm",
        "max_penetration_m": float(max_penetration_m),
        "penetration_sampling_limited": limited,
        "penetration_worlds_sampled": sampled,
        "report_sha256": payload_hash,
        "observed_max_penetration_m": float(report["max_hand_penetration_m"]),
        "observed_hold_fraction_min": float(min(fractions)),
        "observed_hold_drift_max_m": float(max(hold_motion)),
        "observed_post_release_load_max_N": float(max(release)),
        "observed_ground_times_s": ground_times,
        "report_path": str(Path(path).resolve()),
        "scope": "GPU deformable grip contact screen; not full readiness or solved harvesting",
        "source_sha256": report["source_sha256"],
        "scene_sha256": report["scene_sha256"],
    }
