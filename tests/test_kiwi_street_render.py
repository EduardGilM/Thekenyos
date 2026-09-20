"""CPU checks for the kiwi-street render script. No GL, no RELIC."""
from __future__ import annotations

import importlib.util
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from treesim.config import FruitParams
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_kiwi_street.py"
_SPEC = importlib.util.spec_from_file_location("render_kiwi_street", _SCRIPT)
street = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(street)

# RELIC Spot leg vectors (uleg->lleg and lleg->foot origins, x/z), metres.
V1 = (0.025, -0.3205)
V2 = (0.0, -0.3365)


def _small_scene():
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
    return floor, skeleton


class KiwiStreetRenderTest(unittest.TestCase):
    def test_fruit_hang_matches_place_fruit(self):
        _, skeleton = _small_scene()
        fruit = place_fruit(skeleton, FruitParams(
            max_count=8, joint="free", colors=((0.39, 0.27, 0.12),),
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
        xs, ys, _ = street.row_layout()

        class _Floor:
            half_extent_m = 12.0

        png = street.ground_texture(_Floor(), xs, ys, seed=42, n=512)
        rgb = np.asarray(Image.open(BytesIO(png)))
        n = rgb.shape[0]
        half = 12.0

        def sample(x, y):
            j = int(round((x + half) / (2 * half) * (n - 1)))
            i = int(round((half - y) / (2 * half) * (n - 1)))
            return rgb[i, j].astype(np.float64) / 255.0

        probe_x = (-4.0, -2.0, 0.0, 2.0, 4.0)
        aisle = np.mean([sample(x, 0.0) for x in probe_x], axis=0)
        row = np.mean([sample(x, ys[1]) for x in probe_x], axis=0)
        self.assertGreater(row[0] / max(row[1], 1e-6), aisle[0] / max(aisle[1], 1e-6))
        self.assertGreater(row[0], row[1])

    def test_mjcf_uses_round_posts_leaf_meshes_and_place_fruit_hang(self):
        floor, skeleton = _small_scene()
        fruit = place_fruit(skeleton, FruitParams(
            max_count=4, joint="free", colors=((0.39, 0.27, 0.12),),
        ), seed=42)
        xs, ys, aisles = street.row_layout()
        self.assertEqual(len(aisles), street.AISLES)
        xml = street.mjcf(floor, skeleton, fruit, [], xs, ys)
        self.assertIn('type="cylinder"', xml)
        self.assertIn('material="wood"', xml)
        self.assertIn('material="concrete"', xml)
        self.assertIn('<mesh name="leaf0"', xml)
        self.assertNotIn("spotwrap", xml)
        self.assertIn("spotwrap", street.mjcf(floor, skeleton, fruit, [], xs, ys,
                                              with_spot_wrap=True))
        self.assertGreater(xml.count('type="cylinder"'), street.ROWS * street.COLUMNS)
        center = street.fruit_center(fruit[0])
        self.assertIn(f"{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}", xml)
        self.assertGreaterEqual(street.CAMERA_AZIMUTH_DEG, 4.0)
        self.assertLess(street.CAMERA_AZIMUTH_DEG, 28.0)

    def test_leaf_blade_is_a_valid_single_sided_mesh(self):
        verts, uvs, faces = street.leaf_blade(0.22, 0.17)
        self.assertEqual(len(verts), len(uvs))
        self.assertEqual(faces.min(), 0)
        self.assertEqual(faces.max(), len(verts) - 1)
        self.assertTrue(np.isfinite(verts).all())
        self.assertAlmostEqual(float(verts[:, 2].max()), 0.22)
        self.assertLess(float(np.abs(verts[:, 0]).max()), 0.5 * 0.17 * 1.05)

    def test_leg_ik_recovers_forward_kinematics(self):
        hy_range, kn_range = (-0.9, 2.3), (-2.8, -0.25)
        for hy, kn in ((0.5, -1.0), (0.9, -1.6), (0.2, -0.7)):
            target = street.leg_fk(V1, V2, hy, kn)
            q = street.leg_ik(V1, V2, target, (0.5, -1.0), hy_range, kn_range)
            np.testing.assert_allclose(street.leg_fk(V1, V2, *q), target, atol=1e-4)
        with self.assertRaises(ValueError):
            street.leg_ik(V1, V2, (np.nan, 0.0), (0.5, -1.0), hy_range, kn_range)

    def test_trot_foot_offset_is_continuous_and_lifts_only_in_swing(self):
        stride, lift = 0.44, 0.09
        prev = street.trot_foot_offset(0.0, stride, lift)
        for phase in np.linspace(0.0, 1.0, 401)[1:]:
            cur = street.trot_foot_offset(phase, stride, lift)
            self.assertLess(abs(cur[0] - prev[0]), 0.02)
            self.assertLess(abs(cur[1] - prev[1]), 0.01)
            if phase < 0.5:
                self.assertEqual(cur[1], 0.0)
            else:
                self.assertGreaterEqual(cur[1], -1e-9)
            prev = cur
        self.assertAlmostEqual(street.trot_foot_offset(0.75, stride, lift)[1], lift)

    def test_stance_foot_stays_planted_in_world_frame(self):
        speed, freq = 0.72, 1.58
        stride = speed / freq
        world = []
        for t in np.linspace(0.0, 0.5 / freq, 25):
            dx, _dz = street.trot_foot_offset(freq * t, stride, 0.09)
            world.append(speed * t + dx)
        self.assertLess(np.ptp(world), 1e-9)

    def test_camera_path_stays_under_the_roof(self):
        _, _, aisles = street.row_layout()
        for t in np.linspace(0.0, 12.0, 49):
            lookat, distance, _az, elevation = street.camera_path(t, 12.0, aisles)
            height = lookat[2] + distance * np.sin(np.radians(-elevation))
            self.assertLess(height, street.CANOPY_Z_M - 0.15)
            self.assertGreater(height, 0.3)


if __name__ == "__main__":
    unittest.main()
