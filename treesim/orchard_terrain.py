"""Seeded kiwi orchard floor: grassed aisles and closer planting strips.

The heightfield is generated in Python and added as one Newton static shape.
It is not a MuJoCo ``terrain.png`` asset. Apple ``--terrain`` keeps the older
value-noise field in ``builder._add_terrain``.

Structured profile (engineering layout, not a surveyed block)::

    ridge     /\\                /\\
    aisle ____/  \\____    ______/  \\____
    furrow           \\____/
              pasillo   surco    pasillo
    ------------------------------------- x

Row pitch defaults to 2.0 m so neighbouring vine strips read as a compact
worked block; the 3 m bay still has its pasillo on centre for Spot. This is
tighter than a typical commercial pergola (often ~4–5 m) and is an assumed
layout, not a measured Hayward-block survey. Sampled per seed, assumed
domain-randomization ranges:

* ``slope_deg`` U(-4, +4) by default; hillside previews may pin a mild residual
  tilt up to ±12°. Rolling ``landform_m`` (value-noise / Perlin-like octaves)
  is an assumed farm landform, not a surveyed DEM; default amplitude is 0.
* ``ground_noise_m`` U(0, 0.04)
* ``rut_depth_m`` U(0, 0.08)
* ``rut_width_m`` U(0.20, 0.60)
* ``friction`` U(0.6, 1.3)  — robot-foot vs dry soil/grass proxy; wet and
  liner friction stay explicit calibration gaps.
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
DEFAULT_ROW_PITCH_M = 2.0
# Visual herbicide/cultivated strip under the vine row (assumed look).
# Narrower than the grass alley so the block reads as sod with vine rows,
# not equal green/brown bars. ~0.64 m of bare earth at 2 m pitch.
PLANTING_STRIP_HALF_M = 0.32
PLANTING_STRIP_EDGE_M = 0.08
APPEARANCE_CELL_M = 0.025
# Visual-only colours (engineering look, not a photographed block).
GRASS_COLOR = (0.22, 0.30, 0.11)
SOIL_COLOR = (0.44, 0.29, 0.15)
FURROW_COLOR = (0.22, 0.14, 0.07)


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


def _row_coords(x_m: np.ndarray, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.mod(x_m + 0.5 * pitch_m, pitch_m) - 0.5 * pitch_m
    return u, 0.5 * pitch_m - np.abs(u)


def _row_relief(x_m: np.ndarray, pitch_m: float, rut_width_m: float,
                rut_depth_m: float) -> np.ndarray:
    """Pasillo on the pitch centres; shallow cultivated trough between.

    The trough is a worked planting strip, not a plough V. Aisle stays
    nearly flat aside from a small drainage crown and wheel tracks.
    """
    u, dist_row = _row_coords(x_m, pitch_m)
    width = max(float(rut_width_m), 1e-6)
    half = 0.5 * pitch_m
    trough_half = float(np.clip(0.55 * width, 0.12, 0.32))
    shoulder = float(np.clip(0.28 * width, 0.08, 0.18))
    knots_d = np.array([0.0, trough_half, trough_half + shoulder,
                        trough_half + shoulder + 0.14, half])
    knots_z = np.array([-rut_depth_m, -0.85 * rut_depth_m,
                        0.08 * rut_depth_m, 0.0, 0.0])
    relief = np.interp(dist_row, knots_d, knots_z)
    relief += 0.008 * np.clip(1.0 - (np.abs(u) / (0.32 * pitch_m + 1e-9)) ** 2, 0.0, 1.0)
    track = min(0.42, 0.22 * pitch_m)
    track_half = 0.11
    for side in (-1.0, 1.0):
        dist = np.abs(u - side * track)
        relief -= (WHEEL_RUT_FRACTION * rut_depth_m
                   * np.clip(1.0 - dist / track_half, 0.0, 1.0) ** 2)
    return relief


def _aisle_weight(x_m: np.ndarray, pitch_m: float, rut_width_m: float) -> np.ndarray:
    """Physical aisle mask used to damp height noise on the walking strip."""
    _, dist_row = _row_coords(x_m, pitch_m)
    soil = float(np.clip(0.30 * rut_width_m + 0.42 * rut_width_m + 0.10, 0.22, 0.55))
    return np.clip((dist_row - 0.6 * soil) / (0.40 * soil + 1e-9), 0.0, 1.0)


def _planting_strip_weight(x_m: np.ndarray, pitch_m: float,
                           edge_noise: np.ndarray | float = 0.0) -> np.ndarray:
    """Visual cultivated strip under the vine row (wider than the physical rut)."""
    _, dist_row = _row_coords(x_m, pitch_m)
    half = PLANTING_STRIP_HALF_M + np.asarray(edge_noise)
    edge = PLANTING_STRIP_EDGE_M
    return np.clip((half + 0.5 * edge - dist_row) / edge, 0.0, 1.0)


def _appearance_rgb(x: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                    pitch_m: float, meander: np.ndarray, rut_width_m: float,
                    extent_m: float, rng: np.random.Generator) -> np.ndarray:
    """Mottled grass alleys and irregular bare vine-row earth.

    Rows stay on the pitch (no lockstep sine, which reads as fabric). Edges
    wander with noise. The bare strip is an assumed herbicide/cultivated
    band, not a photographed spray line. ``rut_width_m`` is unused in the
    colour mix; kept so call sites can pass the sampled value.
    """
    del rut_width_m
    x_eff = xx - meander
    u, dist_row = _row_coords(x_eff, pitch_m)
    clump = _value_noise(rng, x, y, extent_m, wavelength_m=1.20)
    grit = _value_noise(rng, x, y, extent_m, wavelength_m=0.24)
    patch = _value_noise(rng, x, y, extent_m, wavelength_m=2.80)
    broad = _value_noise(rng, x, y, extent_m, wavelength_m=6.50)
    speck = _value_noise(rng, x, y, extent_m, wavelength_m=0.11)
    edge = 0.18 * (2.0 * _value_noise(rng, x, y, extent_m, wavelength_m=2.20) - 1.0)
    soil = np.clip(_planting_strip_weight(x_eff, pitch_m, edge), 0.0, 1.0)
    weed_w = np.clip((patch - 0.82) / 0.18, 0.0, 1.0) * np.clip((clump - 0.58) / 0.32, 0.0, 1.0)
    soil = np.clip(soil * (1.0 - 0.50 * weed_w * soil), 0.0, 1.0)
    scar_w = (1.0 - clump) ** 2 * patch * (1.0 - soil)
    soil = np.clip(soil + 0.22 * scar_w, 0.0, 1.0)
    grass = 1.0 - soil

    dark = np.array([0.13, 0.24, 0.07])
    light = np.array([0.33, 0.50, 0.16])
    dry_col = np.array([0.40, 0.35, 0.15])
    t = np.clip(0.40 * clump + 0.35 * broad + 0.25 * patch, 0.0, 1.0)
    grass_col = (1.0 - t)[..., None] * dark + t[..., None] * light
    grass_col = grass_col * (0.88 + 0.16 * grit[..., None] + 0.12 * speck[..., None])
    dry_w = np.clip((broad - 0.55) / 0.40, 0.0, 1.0) * np.clip((patch - 0.50) / 0.40, 0.0, 1.0) * grass * 0.35
    grass_col = grass_col * (1.0 - dry_w)[..., None] + dry_w[..., None] * dry_col

    wet = np.array([0.20, 0.12, 0.06])
    loam = np.array([0.48, 0.30, 0.15])
    dust = np.array([0.60, 0.40, 0.20])
    s = np.clip(0.45 * grit + 0.35 * patch + 0.20 * broad, 0.0, 1.0)
    soil_col = (1.0 - s)[..., None] * wet + s[..., None] * dust
    soil_col = soil_col * (0.90 + 0.20 * speck[..., None]) + 0.08 * loam
    furrow_w = np.clip(1.0 - dist_row / 0.26, 0.0, 1.0) ** 1.4
    soil_col = soil_col * (1.0 - 0.40 * furrow_w)[..., None] + furrow_w[..., None] * wet

    # RGB lerp of green+brown goes khaki; route the blend through a dark edge.
    edge_col = np.array([0.18, 0.16, 0.07])
    w = np.clip(soil, 0.0, 1.0)
    rgb = np.empty(xx.shape + (3,))
    low = w < 0.5
    hi = ~low
    t_low = (2.0 * w)[..., None]
    t_hi = (2.0 * w - 1.0)[..., None]
    rgb[low] = (1.0 - t_low[low]) * grass_col[low] + t_low[low] * edge_col
    rgb[hi] = (1.0 - t_hi[hi]) * edge_col + t_hi[hi] * soil_col[hi]
    track = min(0.42, 0.22 * pitch_m)
    for side in (-1.0, 1.0):
        worn = np.clip(1.0 - np.abs(u - side * track) / 0.15, 0.0, 1.0) ** 1.6
        worn = worn * (0.25 + 0.75 * grit) * grass
        rgb = rgb * (1.0 - 0.18 * worn)[..., None] + worn[..., None] * np.array(
            [0.34, 0.27, 0.13])
    rgb = np.clip(rgb, 0.0, 1.0)
    tiled = (grass[..., None] * sample_world_tile(grass_tile_rgb(), xx, yy)
             + (1.0 - grass)[..., None] * sample_world_tile(soil_tile_rgb(), xx, yy))
    rgb = 0.18 * rgb + 0.82 * tiled
    return np.clip(rgb, 0.0, 1.0)


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
    colors_rgb: np.ndarray = None
    soil_weight: np.ndarray = None
    landform_m: np.ndarray = None
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

    def _bilinear(self, grid, x, y) -> float:
        grid = np.asarray(grid, dtype=np.float64)
        nr, nc = grid.shape
        span = 2.0 * self.half_extent_m
        u = (float(x) + self.half_extent_m) / span * (nc - 1)
        v = (float(y) + self.half_extent_m) / span * (nr - 1)
        if not (0.0 <= u <= nc - 1 and 0.0 <= v <= nr - 1):
            return 0.0
        j0 = int(np.floor(u))
        i0 = int(np.floor(v))
        j1 = min(j0 + 1, nc - 1)
        i1 = min(i0 + 1, nr - 1)
        fu, fv = u - j0, v - i0
        return float((grid[i0, j0] * (1.0 - fu) + grid[i0, j1] * fu) * (1.0 - fv)
                     + (grid[i1, j0] * (1.0 - fu) + grid[i1, j1] * fu) * fv)

    def landform_z(self, x, y) -> float:
        if self.landform_m is None:
            return 0.0
        return self._bilinear(self.landform_m, x, y)

    def canopy_z(self, x, y) -> float:
        """Leaf-roof height: a constant offset above the local ground.

        The pergola wires and visual foliage follow this surface so the roof
        has the same rolling shape as the floor, not a fitted plane.
        """
        return self.canopy_height_m + self.ground_z(x, y)

    def ground_z(self, x, y) -> float:
        x, y = float(x), float(y)
        span = 2.0 * self.half_extent_m
        u = (x + self.half_extent_m) / span * (self.ncol - 1)
        v = (y + self.half_extent_m) / span * (self.nrow - 1)
        if not (0.0 <= u <= self.ncol - 1 and 0.0 <= v <= self.nrow - 1):
            return self.aisle_plane_z(x, y) + self.landform_z(x, y)
        return self._bilinear(self.heights_m, x, y)

    def color_at(self, x, y) -> np.ndarray:
        """Bilinear sample of the grass/soil map in world metres."""
        rgb = np.asarray(self.colors_rgb, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("colors_rgb must be an (nrow, ncol, 3) map")
        nr, nc = rgb.shape[:2]
        span = 2.0 * self.half_extent_m
        u = (float(x) + self.half_extent_m) / span * (nc - 1)
        v = (float(y) + self.half_extent_m) / span * (nr - 1)
        u = float(np.clip(u, 0.0, nc - 1))
        v = float(np.clip(v, 0.0, nr - 1))
        j0, i0 = int(np.floor(u)), int(np.floor(v))
        j1, i1 = min(j0 + 1, nc - 1), min(i0 + 1, nr - 1)
        fu, fv = u - j0, v - i0
        return ((rgb[i0, j0] * (1.0 - fu) + rgb[i0, j1] * fu) * (1.0 - fv)
                + (rgb[i1, j0] * (1.0 - fu) + rgb[i1, j1] * fu) * fv)

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

    def texture_png_bytes(self, skeleton=None, sun_dir=None, seed: int = 0) -> bytes:
        """RGB PNG of the grass/soil map, origin at the south-west corner."""
        from io import BytesIO
        from PIL import Image
        rgb = np.asarray(self.colors_rgb, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("colors_rgb must be an (nrow, ncol, 3) map")
        if skeleton:
            rgb = shade_under_canopy(rgb, self, skeleton, sun_dir=sun_dir, seed=seed)
        pixels = np.clip(np.flipud(rgb) * 255.0, 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(pixels, mode="RGB").save(buf, format="PNG")
        return buf.getvalue()


def shade_under_canopy(rgb: np.ndarray, floor: OrchardFloor, skeleton,
                       sun_dir=None, seed: int = 0) -> np.ndarray:
    """Bake irregular under-tree shade into a world-mapped albedo.

    Classic-GL shadow maps on thousands of leaf cards alias into a grid, so
    this is a procedural dapple (dark canopy body plus sun flecks), not a
    realtime shadow map. Shifted a little along the sun XY so the pool sits
    slightly off the posts.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    pts = []
    for seg in skeleton:
        if int(getattr(seg, "order", 0)) < 1:
            continue
        mid = 0.5 * (np.asarray(seg.start, dtype=float) + np.asarray(seg.end, dtype=float))
        pts.append(mid[:2])
    if len(pts) < 2:
        return rgb
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(0) - 1.15, pts.max(0) + 1.15
    sun = np.array([0.26, 0.42], dtype=float) if sun_dir is None else np.asarray(sun_dir, dtype=float)[:2]
    nxy = float(np.linalg.norm(sun)) or 1.0
    shift = 0.70 * sun / nxy
    half = float(floor.half_extent_m)
    nr, nc = rgb.shape[:2]
    xt = np.linspace(-half, half, nc)
    yt = np.linspace(-half, half, nr)
    xx, yy = np.meshgrid(xt, yt)
    xs, ys = xx - shift[0], yy - shift[1]
    wx = np.clip(np.minimum(xs - lo[0], hi[0] - xs) / 1.2, 0.0, 1.0)
    wy = np.clip(np.minimum(ys - lo[1], hi[1] - ys) / 1.2, 0.0, 1.0)
    cover = wx * wy
    rng = np.random.default_rng((int(seed) * 7919 + 3) & 0x7FFFFFFF)
    body = _value_noise(rng, xt, yt, half, wavelength_m=1.55)
    fleck = _value_noise(rng, xt, yt, half, wavelength_m=0.42)
    speck = _value_noise(rng, xt, yt, half, wavelength_m=0.16)
    holes = np.clip((fleck * speck - 0.38) / 0.28, 0.0, 1.0) ** 1.35
    dark = 0.28 + 0.20 * body
    bright = 0.78 + 0.16 * fleck
    under = dark * (1.0 - holes) + bright * holes
    factor = (1.0 - cover) + cover * under
    return np.clip(rgb * factor[..., None], 0.0, 1.0)


def earth_cut_png_bytes(seed: int = 0) -> bytes:
    """Small tiled loam albedo for the hillside cut (procedural, not a photo)."""
    from io import BytesIO
    from PIL import Image
    rng = np.random.default_rng((int(seed) * 19 + 5) & 0x7FFFFFFF)
    n = 128
    x = np.linspace(-2.0, 2.0, n)
    y = np.linspace(-2.0, 2.0, n)
    xx, yy = np.meshgrid(x, y)
    field = _value_noise(rng, x, y, 2.0, wavelength_m=0.55)
    grit = _value_noise(rng, x, y, 2.0, wavelength_m=0.14)
    loam = np.array([0.30, 0.19, 0.10])
    dry = np.array([0.50, 0.34, 0.18])
    rgb = (1.0 - field)[..., None] * loam + field[..., None] * dry
    rgb = rgb * (0.86 + 0.20 * grit[..., None])
    pixels = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _png_rgb(rgb: np.ndarray) -> bytes:
    from io import BytesIO
    from PIL import Image
    pixels = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def sample_world_tile(tile: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                      period_m: float = 0.35) -> np.ndarray:
    """Nearest-neighbour wrap of a square RGB tile in world metres."""
    n = int(tile.shape[0])
    j = np.floor(np.mod(xx / period_m, 1.0) * n).astype(np.int32) % n
    i = np.floor(np.mod(yy / period_m, 1.0) * n).astype(np.int32) % n
    return tile[i, j]


def grass_tile_rgb(seed: int = 4, n: int = 256) -> np.ndarray:
    """Tileable lawn: visible blades and litter, not a flat green."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    rgb = np.zeros((n, n, 3), dtype=np.float64)
    rgb[:] = (0.07, 0.13, 0.03)
    for _ in range(2200):
        cx = float(rng.uniform(0.0, n))
        y0 = int(rng.integers(0, n))
        length = int(rng.integers(18, 64))
        lean = float(rng.uniform(-0.55, 0.55))
        hue = float(rng.uniform(0.0, 1.0))
        col = np.array([0.10 + 0.16 * hue, 0.34 + 0.42 * hue, 0.04 + 0.08 * hue])
        width = int(rng.integers(1, 4))
        for t in range(length):
            x = int(cx + lean * t) % n
            y = (y0 + t) % n
            x1 = x + width
            if x1 <= n:
                rgb[y, x:x1] = 0.22 * rgb[y, x:x1] + 0.78 * col
            else:
                rgb[y, x:n] = 0.22 * rgb[y, x:n] + 0.78 * col
                rgb[y, 0:x1 - n] = 0.22 * rgb[y, 0:x1 - n] + 0.78 * col
    for _ in range(180):
        x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
        rgb[y:min(n, y + 3), x:min(n, x + 4)] = (0.32, 0.21, 0.08)
    return np.clip(rgb, 0.0, 1.0)


def grass_tile_png_bytes(seed: int = 4, n: int = 256) -> bytes:
    return _png_rgb(grass_tile_rgb(seed, n))


def soil_tile_rgb(seed: int = 8, n: int = 256) -> np.ndarray:
    """Tileable cultivated earth: crumbs and stones, not a flat brown."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    clump = _value_noise(rng, x, y, 0.5, wavelength_m=0.18)
    grit = _value_noise(rng, x, y, 0.5, wavelength_m=0.05)
    wet = np.array([0.14, 0.08, 0.04])
    dry = np.array([0.58, 0.38, 0.16])
    rgb = (1.0 - clump)[..., None] * wet + clump[..., None] * dry
    rgb = rgb * (0.72 + 0.40 * grit[..., None])
    for _ in range(140):
        px, py = int(rng.integers(2, n - 4)), int(rng.integers(2, n - 4))
        rgb[py:py + 4, px:px + 5] = (0.42, 0.33, 0.20)
    return np.clip(rgb, 0.0, 1.0)


def soil_tile_png_bytes(seed: int = 8, n: int = 256) -> bytes:
    return _png_rgb(soil_tile_rgb(seed, n))


def canopy_dapple_png_bytes(seed: int = 11, n: int = 256) -> bytes:
    """Soft noisy shade card for under the pergola (not a shadow-map grid)."""
    from io import BytesIO
    from PIL import Image
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(u, v)
    blobs = _value_noise(rng, u, v, 0.5, wavelength_m=0.22)
    spec = _value_noise(rng, u, v, 0.5, wavelength_m=0.07)
    edge = np.clip(np.minimum(np.minimum(uu, 1.0 - uu), np.minimum(vv, 1.0 - vv)) / 0.12, 0.0, 1.0)
    alpha = edge * (0.12 + 0.52 * blobs * blobs) * (0.80 + 0.20 * spec)
    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    rgba[..., 0:3] = (18, 22, 12)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def floor_kwargs_for_plantation(rows, columns, spacing, params=None,
                                margin_m: float = 12.0) -> dict:
    """Cover a pergola grid without allocating a fine 0.05 m field over hectares.

    The compact 15 m / 5 cm default stays for a 2×2 fixture. A 45×40 block at
    5 m needs ~122 m half-extent; collision cells coarsen so each side stays
    at most 501 samples. Visual texels coarsen to 0.25 m on fields larger
    than 30 m half-extent. ``row_pitch_m`` follows the structural spacing so
    soil strips line up with posts.
    """
    rows = int(rows)
    columns = int(columns)
    spacing = float(spacing)
    margin_m = float(margin_m)
    if rows < 2 or columns < 2:
        raise ValueError("plantation rows and columns must be at least 2")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be a positive finite length in metres")
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("margin_m must be a finite non-negative length in metres")
    half_extent_m = 0.5 * max(columns - 1, rows - 1) * spacing + margin_m
    default_half = float(getattr(params, "orchard_half_extent_m", 15.0))
    half_extent_m = max(half_extent_m, default_half)
    cell_m = float(getattr(params, "orchard_cell_m", 0.05))
    appearance_cell_m = float(APPEARANCE_CELL_M)
    # Keep the compact 15 m / 5 cm default. Only coarsen a commercial block.
    if half_extent_m > 30.0:
        appearance_cell_m = max(appearance_cell_m, 0.25)
        max_side = 501
        n = int(round(2.0 * half_extent_m / cell_m)) + 1
        if n > max_side:
            cell_m = (2.0 * half_extent_m) / (max_side - 1)
    if cell_m > 0.20 * spacing:
        raise ValueError(
            f"plantation cell_m={cell_m:.4f} is too coarse for spacing={spacing:.3f} m")
    return {
        "half_extent_m": float(half_extent_m),
        "cell_m": float(cell_m),
        "appearance_cell_m": float(appearance_cell_m),
        "row_pitch_m": spacing,
    }


def sample_orchard_floor(seed: int = 0, params=None, *,
                         slope_deg=None, noise_m=None, rut_depth_m=None,
                         rut_width_m=None, friction=None,
                         slope_azimuth_deg=None, half_extent_m=None,
                         row_pitch_m=None, cell_m=None,
                         appearance_cell_m=None,
                         canopy_height_m: float = 1.6,
                         landform_m=None,
                         landform_wavelength_m=None) -> OrchardFloor:
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
        row_pitch_m if row_pitch_m is not None else getattr(ph, "orchard_row_pitch_m", DEFAULT_ROW_PITCH_M))
    cell_m = _validate_positive(
        "cell_m",
        cell_m if cell_m is not None else getattr(ph, "orchard_cell_m", 0.05))
    appearance_cell_m = _validate_positive(
        "appearance_cell_m",
        appearance_cell_m if appearance_cell_m is not None else APPEARANCE_CELL_M)
    if cell_m > 0.25 * row_pitch_m:
        raise ValueError("cell_m must resolve the row profile (keep well below row_pitch_m)")
    if not np.isfinite(canopy_height_m) or canopy_height_m < 0.3:
        raise ValueError("canopy_height_m must be finite and at least 0.3 m")

    slope_span = _as_range(slope_deg, getattr(ph, "orchard_slope_deg", (-4.0, 4.0)), "slope_deg")
    noise_span = _as_range(noise_m, getattr(ph, "orchard_noise_m", (0.0, 0.04)), "noise_m")
    depth_span = _as_range(rut_depth_m, getattr(ph, "orchard_rut_depth_m", (0.0, 0.08)), "rut_depth_m")
    width_span = _as_range(rut_width_m, getattr(ph, "orchard_rut_width_m", (0.20, 0.60)), "rut_width_m")
    mu_span = _as_range(friction, getattr(ph, "orchard_friction", (0.6, 1.3)), "friction")
    az_span = _as_range(slope_azimuth_deg, getattr(ph, "orchard_slope_azimuth_deg", (0.0, 360.0)), "slope_azimuth_deg")

    if slope_span[0] < -12.0 - 1e-9 or slope_span[1] > 12.0 + 1e-9:
        raise ValueError("slope_deg must stay inside [-12, +12]")
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
    landform_amp = float(landform_m if landform_m is not None else getattr(ph, "orchard_landform_m", 0.0))
    landform_wl = float(landform_wavelength_m if landform_wavelength_m is not None else getattr(ph, "orchard_landform_wavelength_m", 22.0))
    if not np.isfinite(landform_amp) or landform_amp < 0.0 or landform_amp > 8.0:
        raise ValueError("landform_m must be finite and inside [0, 8] m")
    if landform_amp > 0.0 and (not np.isfinite(landform_wl) or landform_wl < 4.0):
        raise ValueError("landform_wavelength_m must be finite and at least 4 m")

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

    # Rows stay on the pitch. A shared sine meander shears the block like fabric.
    relief = _row_relief(xx, row_pitch_m, max(rut_width, 1e-6), rut_depth)
    angle = np.radians(slope)
    az = np.radians(azimuth_deg)
    plane = np.tan(angle) * (xx * np.cos(az) + yy * np.sin(az))
    noise = _value_noise(rng, x, y, half_extent_m, wavelength_m=1.8)
    aisle = _aisle_weight(xx, row_pitch_m, max(rut_width, 1e-6))
    noise_m_field = noise_amp * noise * (AISLE_NOISE_SCALE * aisle + (1.0 - aisle))
    heights = AISLE_HEIGHT_M + relief + plane + noise_m_field
    landform = np.zeros_like(heights)
    if landform_amp > 0.0:
        # Long-wavelength value noise (Perlin-like octaves). Assumed rolling
        # farm landform, not a surveyed DEM.
        landform = landform_amp * _value_noise(rng, x, y, half_extent_m, landform_wl)
        landform -= float(landform.mean())
        heights = heights + landform

    for px, py in PERGOLA_POST_XY_M:
        r = np.hypot(xx - px, yy - py)
        pad = np.clip((POST_PAD_RADIUS_M - r) / 0.12, 0.0, 1.0)
        iu = int(np.clip(round((px + half_extent_m) / (2.0 * half_extent_m) * (n - 1)), 0, n - 1))
        iv = int(np.clip(round((py + half_extent_m) / (2.0 * half_extent_m) * (n - 1)), 0, n - 1))
        z_post = float(heights[iv, iu])
        heights = (1.0 - pad) * heights + pad * z_post

    lift = GROUND_CLEARANCE_M - float(heights.min())
    heights = heights + lift
    reference_z = AISLE_HEIGHT_M + lift

    n_tex = int(round(2.0 * half_extent_m / appearance_cell_m)) + 1
    n_tex = max(n_tex, n)
    xt = np.linspace(-half_extent_m, half_extent_m, n_tex)
    yt = np.linspace(-half_extent_m, half_extent_m, n_tex)
    xxt, yyt = np.meshgrid(xt, yt)
    look = np.random.default_rng((seed * 2654435761 + 17) & 0x7FFFFFFF)
    colors = _appearance_rgb(xt, yt, xxt, yyt, row_pitch_m, 0.0,
                             max(rut_width, 1e-6), half_extent_m, look)
    soil = _planting_strip_weight(xx, row_pitch_m, 0.0)

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
        appearance_cell_m=float(appearance_cell_m),
        planting_strip_half_m=float(PLANTING_STRIP_HALF_M),
        aisle_height_m=AISLE_HEIGHT_M,
        canopy_height_m=float(canopy_height_m),
        post_embed_m=POST_EMBED_M,
        lift_m=float(lift),
        landform_m=float(landform_amp),
        landform_wavelength_m=float(landform_wl),
    )
    return OrchardFloor(
        heights_m=heights.astype(np.float64),
        x_m=x, y_m=y, half_extent_m=float(half_extent_m), friction=mu,
        canopy_height_m=float(canopy_height_m), reference_z_m=float(reference_z),
        slope_deg=slope, slope_azimuth_rad=az,
        colors_rgb=colors.astype(np.float64), soil_weight=soil.astype(np.float64),
        landform_m=landform.astype(np.float64),
        sampled=sampled,
    )


def visual_meshes(floor: OrchardFloor, stride: int = 2):
    """Collision-free triangle meshes: grass alleys, vine-row soil, furrow."""
    stride = max(int(stride), 1)
    z = np.asarray(floor.heights_m, dtype=np.float64)[::stride, ::stride]
    soil = np.asarray(floor.soil_weight, dtype=np.float64)[::stride, ::stride]
    x = np.asarray(floor.x_m, dtype=np.float64)[::stride]
    y = np.asarray(floor.y_m, dtype=np.float64)[::stride]
    ny, nx = z.shape
    xx, yy = np.meshgrid(x, y)
    verts = np.stack([xx.ravel(), yy.ravel(), z.ravel() + 0.002], axis=1).astype(np.float32)
    faces = {0: [], 1: [], 2: []}
    for i in range(ny - 1):
        for j in range(nx - 1):
            weight = 0.25 * (soil[i, j] + soil[i, j + 1] + soil[i + 1, j] + soil[i + 1, j + 1])
            kind = 0 if weight < 0.28 else (2 if weight > 0.72 else 1)
            a = i * nx + j
            faces[kind].extend((a, a + 1, a + nx, a + 1, a + nx + 1, a + nx))
    colors = (GRASS_COLOR, SOIL_COLOR, FURROW_COLOR)
    out = []
    for kind, color in enumerate(colors):
        idx = np.asarray(faces[kind], dtype=np.int32)
        if idx.size:
            out.append((verts, idx, color))
    return out


def add_to_builder(builder, floor: OrchardFloor):
    """Add the orchard heightfield plus a grass/soil visual mesh."""
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
        cfg=builder.ShapeConfig(mu=float(floor.friction), restitution=0.0,
                                collision_group=1, is_visible=False),
        color=GRASS_COLOR, label="orchard_ground",
    )
    visual = builder.ShapeConfig(density=0.0, has_shape_collision=False,
                                 has_particle_collision=False)
    for i, (verts, faces, color) in enumerate(visual_meshes(floor)):
        mesh = newton.Mesh(verts, faces)
        builder.add_shape_mesh(
            -1, mesh=mesh, cfg=visual, color=color,
            label=("orchard_grass", "orchard_soil", "orchard_furrow")[i],
        )
    return floor.ground_z


def uses_orchard_floor(config) -> bool:
    return (getattr(config.lsystem, "kind", None) == "pergola"
            and bool(getattr(config.physics, "terrain", False))
            and getattr(config.physics, "terrain_kind", "noise") == "orchard")
