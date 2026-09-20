"""CPU checks for the five-bay kiwi-street still. No GL render."""
from __future__ import annotations

import importlib.util
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_kiwi_street.py"
_SPEC = importlib.util.spec_from_file_location("render_kiwi_street", _SCRIPT)
street = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(street)
from treesim.config import FruitParams
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit


class KiwiStreetRenderTest(unittest.TestCase):
    def test_fruit_hang_matches_place_fruit(self):
        floor = sample_orchard_floor(
            42, canopy_height_m=street.CANOPY_Z_M, slope_deg=0.0,
            noise_m=0.0, rut_depth_m=0.0, **floor_kwargs_for_plantation(
                street.ROWS, street.COLUMNS, street.SPACING_M, margin_m=4.0),
        )
        skeleton = generate(
            height=street.CANOPY_Z_M, seed=42, rows=street.ROWS,
            columns=street.COLUMNS, spacing=street.SPACING_M,
            ground_z=floor.ground_z, canopy_z=floor.canopy_z,
        )
        fruit = place_fruit(skeleton, FruitParams(
            max_count=8, joint="free",
            colors=((0.39, 0.27, 0.12),),
        ), seed=42)
        self.assertTrue(fruit)
        for item in fruit:
            center = street.fruit_center(item)
            np.testing.assert_allclose(
                center,
                item.attach - np.array([0.0, 0.0, STEM_LENGTH + float(item.radii[2])]),
            )
            self.assertGreater(item.attach[2], center[2] + float(item.radii[2]) - 1e-9)

    def test_ground_soil_follows_post_rows(self):
        xs = [-7.5, -2.5, 2.5, 7.5]
        ys = [-2.5, 2.5]
        class _Floor:
            half_extent_m = 12.0
        png = street.ground_texture(_Floor(), xs, ys, seed=42)
        rgb = np.asarray(Image.open(BytesIO(png)))
        # PNG is flipped for MuJoCo; rows are world -Y at the top.
        n = rgb.shape[0]
        half = 12.0
        def sample(x, y):
            j = int(round((x + half) / (2 * half) * (n - 1)))
            i = int(round((half - y) / (2 * half) * (n - 1)))
            return rgb[i, j].astype(np.float64) / 255.0
        aisle = np.mean([sample(x, 0.0) for x in (-4.0, -2.0, 0.0, 2.0, 4.0)], axis=0)
        row = np.mean([sample(x, -2.5) for x in (-4.0, -2.0, 0.0, 2.0, 4.0)], axis=0)
        self.assertGreater(row[0] / max(row[1], 1e-6), aisle[0] / max(aisle[1], 1e-6))
        self.assertGreater(row[0], row[1])

    def test_mjcf_uses_round_posts_and_place_fruit_hang(self):
        floor = sample_orchard_floor(
            42, canopy_height_m=street.CANOPY_Z_M, slope_deg=0.0,
            noise_m=0.0, rut_depth_m=0.0, **floor_kwargs_for_plantation(
                street.ROWS, street.COLUMNS, street.SPACING_M, margin_m=4.0),
        )
        skeleton = generate(
            height=street.CANOPY_Z_M, seed=42, rows=street.ROWS,
            columns=street.COLUMNS, spacing=street.SPACING_M,
            ground_z=floor.ground_z, canopy_z=floor.canopy_z,
        )
        fruit = place_fruit(skeleton, FruitParams(
            max_count=4, joint="free", colors=((0.39, 0.27, 0.12),),
        ), seed=42)
        pads, extents = street.launch_pads(street.ROWS, street.COLUMNS, street.SPACING_M)
        xs = list(np.linspace(extents[0], extents[1], street.COLUMNS))
        ys = list(np.linspace(extents[2], extents[3], street.ROWS))
        xml = street.mjcf(floor, skeleton, fruit, [], pads, xs, ys)
        self.assertIn('type="cylinder"', xml)
        self.assertIn('material="wood"', xml)
        self.assertIn('material="concrete"', xml)
        self.assertGreater(xml.count('type="cylinder"'), street.ROWS * street.COLUMNS)
        center = street.fruit_center(fruit[0])
        self.assertIn(f"{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}", xml)
        self.assertGreaterEqual(street.CAMERA_AZIMUTH_DEG, 4.0)
        self.assertLess(street.CAMERA_AZIMUTH_DEG, 28.0)
        self.assertGreater(street.CAMERA_ELEVATION_DEG, -6.0)
        self.assertLess(street.CAMERA_ELEVATION_DEG, 8.0)


if __name__ == "__main__":
    unittest.main()
