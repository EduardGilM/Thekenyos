"""Seeded kiwi orchard floor: aisles, planting furrows, slope and noise.

The heightfield is generated in Python and added as one Newton static shape.
It is not a MuJoCo ``terrain.png`` asset. Apple ``--terrain`` keeps the older
value-noise field in ``builder._add_terrain``.

Structured profile (engineering layout, not a surveyed block)::

    ridge     /\\                /\\
    aisle ____/  \\____    ______/  \\____
    furrow           \\____/
              pasillo   surco    pasillo
    ------------------------------------- x

Sampled per seed, assumed domain-randomization ranges (not one measured
orchard-floor survey):

* ``slope_deg`` U(-4, +4)
* ``ground_noise_m`` U(0, 0.04)
* ``rut_depth_m`` U(0, 0.08)
* ``rut_width_m`` U(0.20, 0.60)
* ``friction`` U(0.6, 1.3)  — robot-foot vs dry soil/grass proxy; wet and
  liner friction stay explicit calibration gaps.

Row pitch defaults to 3 m so a surco sits on the pergola post lines
(x = ±1.5 m) and the bay centre remains a pasillo for Spot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Posts of the 3 x 4 m bay. Flatten a small pad under each so a welded foot
# does not fight high-frequency noise.
PERGOLA_POST_XY_M = ((-1.5, -2.0), (1.5, -2.0), (-1.5, 2.0), (1.5, 2.0))
POST_PAD_RADIUS_M = 0.22
POST_EMBED_M = 0.04
GROUND_CLEARANCE_M = 0.004
AISLE_HEIGHT_M = 0.04
AISLE_NOISE_SCALE = 0.35
WHEEL_RUT_FRACTION = 0.35
RIDGE_FRACTION = 0.50


def _as_range(value, default, name: str) -> tuple[float, float]:
    if value is None:
        value = default
    if np.isscalar(value):
        lo = hi = float(value)
    else:
        seq = tuple(value)
        if len(seq) != 2:
            raise ValueError(f"{name} must be a scalar or (low, high)")
        lo, hi = float(seq[0]), float(seq[1])
        if lo > hi:
            lo, hi = hi, lo
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} must be finite")
    return lo, hi


def _sample_unit(rng: np.random.Generator, span: tuple[float, float]) -> float:
    lo, hi = span
    return float(lo if lo == hi else rng.uniform(lo, hi))


def _validate_positive(name: str, value: float, low: float = 0.0) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= low:
        raise ValueError(f"{name} must be finite and > {low}")
    return value


def _value_noise(rng: np.random.Generator, x: np.ndarray, y: np.ndarray,
                 extent_m: float, wavelength_m: float) -> np.ndarray:
    """Three-octave bilinear value noise on a periodic lattice."""
    nrow, ncol = y.size, x.size
    field = np.zeros((nrow, ncol))
    span = 2.0 * extent_m
    for octave, weight in enumerate((1.0, 0.50, 0.25)):
        wl = wavelength_m / (octave + 1)
        cx = max(int(round(span / wl)), 2)
        cy = max(int(round(span / wl)), 2)
        grid = rng.standard_normal((cy, cx))
        u = (x + extent_m) / span * cx
        v = (y + extent_m) / span * cy
        j0 = np.floor(u).astype(int) % cx
        i0 = np.floor(v).astype(int) % cy
        j1, i1 = (j0 + 1) % cx, (i0 + 1) % cy
        fu = (u - np.floor(u))[None, :]
        fv = (v - np.floor(v))[:, None]
        top = grid[np.ix_(i0, j0)] * (1.0 - fu) + grid[np.ix_(i0, j1)] * fu
        bot = grid[np.ix_(i1, j0)] * (1.0 - fu) + grid[np.ix_(i1, j1)] * fu
        field += weight * (top + (bot - top) * fv)
    field -= field.min()
    peak = float(field.max())
    if peak > 1e-12:
        field /= peak
    return field


def _row_relief(x_m: np.ndarray, pitch_m: float, rut_width_m: float,
                rut_depth_m: float) -> np.ndarray:
    """Pasillo at k * pitch; ridge-furrow-ridge surco at the mid-pitch lines."""
    u = np.mod(x_m + 0.5 * pitch_m, pitch_m) - 0.5 * pitch_m
    dist_row = 0.5 * pitch_m - np.abs(u)
    sigma_f = 0.28 * rut_width_m + 1e-6
    sigma_r = 0.22 * rut_width_m + 1e-6
    furrow = -rut_depth_m * np.exp(-0.5 * (dist_row / sigma_f) ** 2)
    dist_ridge = np.abs(dist_row - 0.55 * rut_width_m)
    ridge = RIDGE_FRACTION * rut_depth_m * np.exp(-0.5 * (dist_ridge / sigma_r) ** 2)
    track = min(1.15, 0.38 * pitch_m)
    sigma_t = 0.22 * rut_width_m + 1e-6
    wheels = np.zeros_like(u)
    for side in (-1.0, 1.0):
        wheels += (-WHEEL_RUT_FRACTION * rut_depth_m
                   * np.exp(-0.5 * ((u - side * track) / sigma_t) ** 2))
    crown = 0.01 * np.clip(1.0 - (np.abs(u) / (0.40 * pitch_m + 1e-9)) ** 2, 0.0, 1.0)
    return furrow + ridge + wheels + crown


def _aisle_weight(x_m: np.ndarray, pitch_m: float) -> np.ndarray:
    u = np.mod(x_m + 0.5 * pitch_m, pitch_m) - 0.5 * pitch_m
    return np.clip(1.0 - (np.abs(u) / (0.38 * pitch_m + 1e-9)), 0.0, 1.0)


@dataclass
class OrchardFloor:
    """World-space orchard heightfield and the callables used to plant models."""

    heights_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    half_extent_m: float
    friction: float
    canopy_height_m: float
    reference_z_m: float
    slope_deg: float
    slope_azimuth_rad: float
    sampled: dict = field(default_factory=dict)

    @property
    def nrow(self) -> int:
        return int(self.heights_m.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.heights_m.shape[1])

    def slope_z(self, x, y) -> float:
        angle = np.radians(self.slope_deg)
        return float(np.tan(angle) * (float(x) * np.cos(self.slope_azimuth_rad)
                                      + float(y) * np.sin(self.slope_azimuth_rad)))

    def aisle_plane_z(self, x, y) -> float:
        return self.reference_z_m + self.slope_z(x, y)

    def canopy_z(self, x, y) -> float:
        return self.canopy_height_m + self.aisle_plane_z(x, y)

    def ground_z(self, x, y) -> float:
        x, y = float(x), float(y)
        span = 2.0 * self.half_extent_m
        u = (x + self.half_extent_m) / span * (self.ncol - 1)
        v = (y + self.half_extent_m) / span * (self.nrow - 1)
        if not (0.0 <= u <= self.ncol - 1 and 0.0 <= v <= self.nrow - 1):
            return self.aisle_plane_z(x, y)
        j0 = int(np.floor(u))
        i0 = int(np.floor(v))
        j1 = min(j0 + 1, self.ncol - 1)
        i1 = min(i0 + 1, self.nrow - 1)
        fu, fv = u - j0, v - i0
        g = self.heights_m
        return float((g[i0, j0] * (1.0 - fu) + g[i0, j1] * fu) * (1.0 - fv)
                     + (g[i1, j0] * (1.0 - fu) + g[i1, j1] * fu) * fv)

    def metrics(self) -> dict:
        out = dict(self.sampled)
        out.update(
            kind="kiwi_orchard_floor",
            min_z_m=float(self.heights_m.min()),
            max_z_m=float(self.heights_m.max()),
            provenance=("assumed domain-randomization ranges; not a measured "
                        "orchard-floor survey"),
        )
        return out


def sample_orchard_floor(seed: int = 0, params=None, *,
                         slope_deg=None, noise_m=None, rut_depth_m=None,
                         rut_width_m=None, friction=None,
                         slope_azimuth_deg=None, half_extent_m=None,
                         row_pitch_m=None, cell_m=None,
                         canopy_height_m: float = 1.6) -> OrchardFloor:
    """Sample one seeded orchard floor.

    Omit a keyword to draw it from ``params`` (or the default assumed ranges).
    Pass a scalar to pin a test value. Ranges must stay inside the documented
    physical bounds.
    """
    if not np.isfinite(seed):
        raise ValueError("seed must be finite")
    seed = int(seed)
    ph = params
    half_extent_m = _validate_positive(
        "half_extent_m",
        half_extent_m if half_extent_m is not None else getattr(ph, "orchard_half_extent_m", 15.0))
    row_pitch_m = _validate_positive(
        "row_pitch_m",
        row_pitch_m if row_pitch_m is not None else getattr(ph, "orchard_row_pitch_m", 3.0))
    cell_m = _validate_positive(
        "cell_m",
        cell_m if cell_m is not None else getattr(ph, "orchard_cell_m", 0.05))
    if cell_m > 0.25 * row_pitch_m:
        raise ValueError("cell_m must resolve the row profile (keep well below row_pitch_m)")
    if not np.isfinite(canopy_height_m) or canopy_height_m < 0.3:
        raise ValueError("canopy_height_m must be finite and at least 0.3 m")

    slope_span = _as_range(slope_deg, getattr(ph, "orchard_slope_deg", (-4.0, 4.0)), "slope_deg")
    noise_span = _as_range(noise_m, getattr(ph, "orchard_noise_m", (0.0, 0.04)), "noise_m")
    depth_span = _as_range(rut_depth_m, getattr(ph, "orchard_rut_depth_m", (0.0, 0.08)), "rut_depth_m")
    width_span = _as_range(rut_width_m, getattr(ph, "orchard_rut_width_m", (0.20, 0.60)), "rut_width_m")
    mu_span = _as_range(friction, getattr(ph, "orchard_friction", (0.6, 1.3)), "friction")
    az_span = _as_range(slope_azimuth_deg, (0.0, 360.0), "slope_azimuth_deg")

    if slope_span[0] < -4.0 - 1e-9 or slope_span[1] > 4.0 + 1e-9:
        raise ValueError("slope_deg must stay inside [-4, +4]")
    if noise_span[0] < 0.0 or noise_span[1] > 0.04 + 1e-9:
        raise ValueError("noise_m must stay inside [0, 0.04] m")
    if depth_span[0] < 0.0 or depth_span[1] > 0.08 + 1e-9:
        raise ValueError("rut_depth_m must stay inside [0, 0.08] m")
    if width_span[0] < 0.0 or width_span[1] > 0.60 + 1e-9:
        raise ValueError("rut_width_m must stay inside [0, 0.60] m")
    if width_span[0] < 0.20 - 1e-9 and width_span[1] >= 0.20 and width_span[0] != width_span[1]:
        raise ValueError("sampled rut_width_m range must stay inside [0.20, 0.60] m")
    if mu_span[0] < 0.6 - 1e-9 or mu_span[1] > 1.3 + 1e-9:
        raise ValueError("friction must stay inside [0.6, 1.3]")

    rng = np.random.default_rng((seed * 2654435761) & 0x7FFFFFFF)
    slope = _sample_unit(rng, slope_span)
    noise_amp = _sample_unit(rng, noise_span)
    rut_depth = _sample_unit(rng, depth_span)
    rut_width = _sample_unit(rng, width_span)
    mu = _sample_unit(rng, mu_span)
    azimuth_deg = _sample_unit(rng, az_span)
    if rut_depth > 0.0 and rut_width < 0.20 - 1e-9:
        raise ValueError("rut_width_m must be at least 0.20 m when ruts are present")

    n = int(round(2.0 * half_extent_m / cell_m)) + 1
    n = max(n, 17)
    x = np.linspace(-half_extent_m, half_extent_m, n)
    y = np.linspace(-half_extent_m, half_extent_m, n)
    xx, yy = np.meshgrid(x, y)

    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    meander = 0.12 * np.sin(2.0 * np.pi * yy / 7.0 + phase)
    relief = _row_relief(xx - meander, row_pitch_m, max(rut_width, 1e-6), rut_depth)
    angle = np.radians(slope)
    az = np.radians(azimuth_deg)
    plane = np.tan(angle) * (xx * np.cos(az) + yy * np.sin(az))
    noise = _value_noise(rng, x, y, half_extent_m, wavelength_m=1.8)
    aisle = _aisle_weight(xx - meander, row_pitch_m)
    noise_m_field = noise_amp * noise * (AISLE_NOISE_SCALE * aisle + (1.0 - aisle))
    heights = AISLE_HEIGHT_M + relief + plane + noise_m_field

    for px, py in PERGOLA_POST_XY_M:
        r = np.hypot(xx - px, yy - py)
        pad = np.clip((POST_PAD_RADIUS_M - r) / 0.12, 0.0, 1.0)
        # Sample the unpadded height at the post, then flatten the disc to it.
        iu = int(np.clip(round((px + half_extent_m) / (2.0 * half_extent_m) * (n - 1)), 0, n - 1))
        iv = int(np.clip(round((py + half_extent_m) / (2.0 * half_extent_m) * (n - 1)), 0, n - 1))
        z_post = float(heights[iv, iu])
        heights = (1.0 - pad) * heights + pad * z_post

    lift = GROUND_CLEARANCE_M - float(heights.min())
    heights = heights + lift
    reference_z = AISLE_HEIGHT_M + lift

    sampled = dict(
        slope_deg=slope,
        slope_azimuth_deg=azimuth_deg,
        ground_noise_m=noise_amp,
        rut_depth_m=rut_depth,
        rut_width_m=rut_width,
        friction=mu,
        half_extent_m=float(half_extent_m),
        row_pitch_m=float(row_pitch_m),
        cell_m=float(cell_m),
        aisle_height_m=AISLE_HEIGHT_M,
        canopy_height_m=float(canopy_height_m),
        post_embed_m=POST_EMBED_M,
        lift_m=float(lift),
    )
    return OrchardFloor(
        heights_m=heights.astype(np.float64),
        x_m=x, y_m=y, half_extent_m=float(half_extent_m), friction=mu,
        canopy_height_m=float(canopy_height_m), reference_z_m=float(reference_z),
        slope_deg=slope, slope_azimuth_rad=az, sampled=sampled,
    )


def add_to_builder(builder, floor: OrchardFloor):
    """Add the orchard heightfield as one global static Newton shape."""
    import newton
    import warp as wp

    data = np.asarray(floor.heights_m, dtype=np.float32)
    min_z = float(data.min())
    max_z = float(data.max())
    if max_z - min_z < 1e-6:
        max_z = min_z + 1e-4
    hf = newton.Heightfield(
        data, nrow=data.shape[0], ncol=data.shape[1],
        hx=floor.half_extent_m, hy=floor.half_extent_m,
        min_z=min_z, max_z=max_z,
    )
    builder.add_shape_heightfield(
        heightfield=hf,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        cfg=builder.ShapeConfig(mu=float(floor.friction), restitution=0.0, collision_group=1),
        color=(0.36, 0.42, 0.22), label="orchard_ground",
    )
    return floor.ground_z


def uses_orchard_floor(config) -> bool:
    return (getattr(config.lsystem, "kind", None) == "pergola"
            and bool(getattr(config.physics, "terrain", False)))
