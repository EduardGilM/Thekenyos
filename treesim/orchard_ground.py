"""Seeded kiwi-orchard heightfield under the pergola.

The cross-section is the requested pasillo / surco / pasillo pattern along x,
with furrow spacing taken from the existing 3 m pergola bay. Episode slope,
noise, ruts and sliding friction are domain-randomised. This is rigid contact
geometry, not soil physics; wet/liner friction stays an uncalibrated gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pergola import BAY_X_M, POST_X0_M

# Specified orchard-floor profile [m]. Engineering scene values, not a survey.
PEAK_HEIGHT_M = 0.08
AISLE_HEIGHT_M = 0.04
FURROW_HEIGHT_M = 0.00
# Half-extent matching Eduard's MuJoCo sketch size="15 15 ...".
EXTENT_M = 15.0
# MuJoCo hfield base thickness; used only as a catch-plane offset, not soil.
BASE_THICKNESS_M = 0.05
# Resolve 20 cm ruts; coarser than this smears the furrow shoulders.
CELL_M = 0.05
# Assumed dual wheel-track offset inside each aisle (Spot hip scale).
RUT_TRACK_HALF_M = 0.18
# Existing Spot floating-base stand height on flat ground.
SPOT_STAND_Z_M = 0.65
# Sketch sliding / torsional / rolling. Sliding is sampled; the other two stay
# the sketch constants and are numerical contact parameters, not soil.
FRICTION_TORSIONAL = 0.01
FRICTION_ROLLING = 0.001
NOISE_WAVELENGTH_M = 1.8


def _require_finite_range(lo, hi, name):
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError(f"{name} must be a finite range with lo <= hi")
    return lo, hi


def _uniform(rng, lo, hi, name):
    lo, hi = _require_finite_range(lo, hi, name)
    return float(rng.uniform(lo, hi))


def row_pitch_m():
    """Pergola bay width along x: the only row pitch in the current farm model."""
    return float(BAY_X_M)


def aisle_x_m(index=0):
    """World x of an aisle centre. index 0 is the aisle left of the bay midline."""
    return float(POST_X0_M + (0.25 + 0.5 * int(index)) * BAY_X_M)


def nearest_aisle_x_m(x, pitch=None, peak_x=None):
    """Snap x onto the nearest pasillo, never a surco."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("x must be finite")
    pitch = float(row_pitch_m() if pitch is None else pitch)
    peak_x = float(POST_X0_M if peak_x is None else peak_x)
    if pitch <= 0 or not np.isfinite(pitch) or not np.isfinite(peak_x):
        raise ValueError("row pitch and peak x must be finite and pitch > 0")
    u = (x - peak_x) / pitch
    base = np.floor(u)
    candidates = peak_x + (base + np.array([0.25, 0.75, 1.25])) * pitch
    return float(candidates[np.argmin(np.abs(candidates - x))])


def profile_height_m(x, pitch=None, peak_x=None):
    """Cosine cross-section: peak 0.08, aisle 0.04, furrow 0.00, repeating."""
    pitch = float(row_pitch_m() if pitch is None else pitch)
    peak_x = float(POST_X0_M if peak_x is None else peak_x)
    u = (np.asarray(x, dtype=np.float64) - peak_x) / pitch
    mid = 0.5 * (PEAK_HEIGHT_M + FURROW_HEIGHT_M)
    amp = 0.5 * (PEAK_HEIGHT_M - FURROW_HEIGHT_M)
    z = mid + amp * np.cos(2.0 * np.pi * u)
    return z if z.ndim else float(z)


def _value_noise(xs, ys, rng, amplitude_m, wavelength_m=NOISE_WAVELENGTH_M):
    if amplitude_m <= 0:
        return np.zeros((ys.size, xs.size), dtype=np.float64)
    px = float(xs[-1] - xs[0])
    py = float(ys[-1] - ys[0])
    tile = np.zeros((ys.size, xs.size), dtype=np.float64)
    for octave, w in enumerate([1.0, 0.5, 0.25]):
        wl = wavelength_m / (octave + 1)
        cx, cy = max(int(round(px / wl)), 2), max(int(round(py / wl)), 2)
        g = rng.standard_normal((cy, cx))
        u = (xs - xs[0]) / max(px, 1e-9) * cx
        v = (ys - ys[0]) / max(py, 1e-9) * cy
        j0 = np.floor(u).astype(int) % cx
        i0 = np.floor(v).astype(int) % cy
        j1, i1 = (j0 + 1) % cx, (i0 + 1) % cy
        fu, fv = u - np.floor(u), (v - np.floor(v))[:, None]
        top = g[np.ix_(i0, j0)] * (1 - fu) + g[np.ix_(i0, j1)] * fu
        bot = g[np.ix_(i1, j0)] * (1 - fu) + g[np.ix_(i1, j1)] * fu
        tile += w * (top + (bot - top) * fv)
    tile -= tile.mean()
    scale = np.max(np.abs(tile))
    if scale < 1e-12:
        return np.zeros_like(tile)
    return tile * (amplitude_m / scale)


def _rut_delta_m(xx, pitch, peak_x, depth_m, width_m):
    if depth_m <= 0 or width_m <= 0:
        return np.zeros_like(xx)
    half = 0.5 * width_m
    n0 = int(np.floor((xx.min() - peak_x) / pitch)) - 1
    n1 = int(np.ceil((xx.max() - peak_x) / pitch)) + 1
    delta = np.zeros_like(xx)
    for n in range(n0, n1 + 1):
        for frac in (0.25, 0.75):
            aisle = peak_x + (n + frac) * pitch
            for track in (-RUT_TRACK_HALF_M, RUT_TRACK_HALF_M):
                t = np.clip(1.0 - np.abs(xx - (aisle + track)) / half, 0.0, 1.0)
                delta -= depth_m * (t * t * (3.0 - 2.0 * t))
    return delta


@dataclass
class OrchardGround:
    slope_deg: float
    ground_noise_m: float
    rut_depth_m: float
    rut_width_m: float
    friction: float
    row_pitch_m: float
    extent_m: float
    seed: int
    xs_m: np.ndarray
    ys_m: np.ndarray
    heights_m: np.ndarray

    def height_m(self, x, y):
        x, y = float(x), float(y)
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("sample coordinates must be finite")
        dx = float(self.xs_m[1] - self.xs_m[0])
        dy = float(self.ys_m[1] - self.ys_m[0])
        u = (x - float(self.xs_m[0])) / dx
        v = (y - float(self.ys_m[0])) / dy
        if u < 0 or v < 0 or u > self.xs_m.size - 1 or v > self.ys_m.size - 1:
            return 0.0
        j0 = min(int(u), self.xs_m.size - 2)
        i0 = min(int(v), self.ys_m.size - 2)
        j1, i1 = j0 + 1, i0 + 1
        fu, fv = u - j0, v - i0
        g = self.heights_m
        return float((g[i0, j0] * (1 - fu) + g[i0, j1] * fu) * (1 - fv)
                     + (g[i1, j0] * (1 - fu) + g[i1, j1] * fu) * fv)

    def nearest_aisle_x_m(self, x):
        return nearest_aisle_x_m(x, self.row_pitch_m)

    def spawn_height_m(self, x, y, stand_z_m=SPOT_STAND_Z_M):
        return float(stand_z_m) + self.height_m(x, y)

    def metrics(self):
        return dict(
            kind="orchard_hfield",
            provenance="assumed_orchard_floor_profile_plus_domain_randomisation",
            slope_deg=self.slope_deg,
            ground_noise_m=self.ground_noise_m,
            rut_depth_m=self.rut_depth_m,
            rut_width_m=self.rut_width_m,
            friction=self.friction,
            friction_torsional=FRICTION_TORSIONAL,
            friction_rolling=FRICTION_ROLLING,
            row_pitch_m=self.row_pitch_m,
            peak_height_m=PEAK_HEIGHT_M,
            aisle_height_m=AISLE_HEIGHT_M,
            furrow_height_m=FURROW_HEIGHT_M,
            extent_m=self.extent_m,
            cell_m=CELL_M,
            seed=self.seed,
            height_min_m=float(self.heights_m.min()),
            height_max_m=float(self.heights_m.max()),
            note="hfield sliding friction is not soil physics; wet/liner friction remains uncalibrated",
        )


def generate(physics, seed: int) -> OrchardGround:
    """Sample one episode and bake a 15 m × 15 m heightfield in metres."""
    rng = np.random.default_rng((int(seed) * 2654435761) & 0x7FFFFFFF)
    slope_deg = _uniform(rng, *physics.orchard_slope_deg, "orchard_slope_deg")
    noise_m = _uniform(rng, *physics.orchard_ground_noise_m, "orchard_ground_noise_m")
    rut_depth_m = _uniform(rng, *physics.orchard_rut_depth_m, "orchard_rut_depth_m")
    rut_width_m = _uniform(rng, *physics.orchard_rut_width_m, "orchard_rut_width_m")
    friction = _uniform(rng, *physics.orchard_friction, "orchard_friction")
    if rut_width_m < 0.05 or friction <= 0:
        raise ValueError("rut width must be >= 5 cm and friction must be positive")
    pitch = row_pitch_m()
    extent = float(getattr(physics, "orchard_extent_m", EXTENT_M))
    if not np.isfinite(extent) or extent < 3.0:
        raise ValueError("orchard_extent_m must be finite and at least 3 m")
    n = int(round(2.0 * extent / CELL_M)) + 1
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(xs, ys)
    heights = profile_height_m(xx, pitch)
    heights += _rut_delta_m(xx, pitch, POST_X0_M, rut_depth_m, rut_width_m)
    heights += xx * np.tan(np.deg2rad(slope_deg))
    heights += _value_noise(xs, ys, rng, noise_m)
    if not np.isfinite(heights).all():
        raise ValueError("orchard heightfield contained a non-finite sample")
    return OrchardGround(
        slope_deg=slope_deg, ground_noise_m=noise_m, rut_depth_m=rut_depth_m,
        rut_width_m=rut_width_m, friction=friction, row_pitch_m=pitch,
        extent_m=extent, seed=int(seed), xs_m=xs, ys_m=ys, heights_m=heights,
    )


def add_heightfield(builder, field: OrchardGround):
    """Add the Newton/MuJoCo heightfield. No checkerboard plane at z=0."""
    import newton
    import warp as wp
    hf = newton.Heightfield(
        field.heights_m.astype(np.float32),
        nrow=field.heights_m.shape[0], ncol=field.heights_m.shape[1],
        hx=field.extent_m, hy=field.extent_m,
        min_z=float(field.heights_m.min()), max_z=float(field.heights_m.max()),
    )
    builder.add_shape_heightfield(
        heightfield=hf,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        cfg=builder.ShapeConfig(
            mu=field.friction, restitution=0.0, collision_group=1,
            mu_torsional=FRICTION_TORSIONAL, mu_rolling=FRICTION_ROLLING,
        ),
        color=(0.42, 0.34, 0.18), label="orchard_ground",
    )
    builder.add_ground_plane(
        height=float(field.heights_m.min()) - BASE_THICKNESS_M,
        cfg=builder.ShapeConfig(
            mu=field.friction, restitution=0.0, collision_group=1,
            mu_torsional=FRICTION_TORSIONAL, mu_rolling=FRICTION_ROLLING,
            is_visible=False,
        ),
    )
    return field.height_m
