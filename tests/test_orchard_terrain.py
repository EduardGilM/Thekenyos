"""CPU orchard-floor geometry: python -m unittest discover -s tests."""
import math
import unittest

import numpy as np

from treesim.config import FruitParams, preset
from treesim.orchard_terrain import (
    AISLE_HEIGHT_M, POST_EMBED_M, floor_kwargs_for_plantation,
    sample_orchard_floor,
)
from treesim.pergola import generate, place_fruit


def _pinned(seed=0, **kw):
    defaults = dict(slope_deg=0.0, noise_m=0.0, rut_depth_m=0.08,
                    rut_width_m=0.40, friction=1.0, slope_azimuth_deg=0.0)
    defaults.update(kw)
    return sample_orchard_floor(seed, **defaults)


class OrchardTerrainTest(unittest.TestCase):
    def test_seeded_ranges_and_profile(self):
        a = sample_orchard_floor(42)
        b = sample_orchard_floor(42)
        c = sample_orchard_floor(43)
        np.testing.assert_allclose(a.heights_m, b.heights_m)
        self.assertEqual(a.metrics(), b.metrics())
        self.assertFalse(np.allclose(a.heights_m, c.heights_m))
        m = a.metrics()
        self.assertEqual(m["kind"], "kiwi_orchard_floor")
        self.assertTrue(-4 <= m["slope_deg"] <= 4)
        hill = sample_orchard_floor(0, slope_deg=10.0, noise_m=0.0, rut_depth_m=0.0,
                                    rut_width_m=0.40, friction=1.0, slope_azimuth_deg=0.0)
        self.assertAlmostEqual(hill.slope_deg, 10.0)
        self.assertTrue(0 <= m["ground_noise_m"] <= 0.04)
        self.assertTrue(0 <= m["rut_depth_m"] <= 0.08)
        self.assertTrue(0.20 <= m["rut_width_m"] <= 0.60)
        self.assertTrue(0.6 <= m["friction"] <= 1.3)
        self.assertGreater(m["max_z_m"], m["min_z_m"])
        self.assertGreaterEqual(m["min_z_m"], 0.0)
        self.assertIn("assumed", m["provenance"])

        floor = _pinned()
        pitch = floor.sampled["row_pitch_m"]
        self.assertAlmostEqual(pitch, 2.0)
        aisle = floor.ground_z(0.0, 0.0)
        furrow = floor.ground_z(0.5 * pitch, 0.0)
        ridge = max(floor.ground_z(x, 0.0) for x in np.linspace(0.45, 0.90, 12))
        self.assertGreater(aisle, furrow)
        self.assertGreater(ridge, furrow)
        self.assertLess(aisle - furrow, 0.08 + AISLE_HEIGHT_M)
        # Neighbouring vine rows stay symmetric about the bay centre.
        self.assertAlmostEqual(floor.ground_z(-0.5 * pitch, 1.0),
                               floor.ground_z(0.5 * pitch, 1.0), places=2)
        # Grass alleys stay greener than the bare planting strip.
        aisle_rgb = floor.color_at(0.0, 0.0)
        soil_rgb = floor.color_at(0.5 * pitch, 0.0)
        self.assertGreater(aisle_rgb[1] - aisle_rgb[0], soil_rgb[1] - soil_rgb[0])
        # The cultivated strip is a band, not a hairline: 15 cm off the
        # furrow is still more soil-like than the aisle centre.
        near_rgb = floor.color_at(0.5 * pitch - 0.15, 0.0)
        self.assertGreater(aisle_rgb[1] - aisle_rgb[0], near_rgb[1] - near_rgb[0])
        grass_g = [floor.color_at(0.0, y)[1] for y in np.linspace(-0.8, 0.8, 48)]
        self.assertGreater(float(np.std(grass_g)), 0.003)

    def test_preview_uses_visible_tiles_and_no_shadow_grid(self):
        from io import BytesIO

        from PIL import Image

        from scripts.record_orchard_mujoco import _SUN_DIR, aim_sun, mjcf
        from treesim.orchard_terrain import (
            earth_cut_png_bytes, grass_tile_png_bytes, soil_tile_png_bytes,
        )
        floor = _pinned()
        xml = mjcf(floor, [], [])
        self.assertIn('castshadow="false"', xml)
        self.assertNotIn('castshadow="true"', xml)
        self.assertIn('orchard_ground.png', xml)
        self.assertIn('emission="0.28"', xml)
        self.assertIn('name="earth_mass"', xml)
        self.assertIn('material="earth_cut"', xml)
        self.assertNotIn('mesh="orchard_grass"', xml)
        png = earth_cut_png_bytes(0)
        self.assertGreater(len(png), 64)
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        grass = np.asarray(Image.open(BytesIO(grass_tile_png_bytes(0))))
        soil = np.asarray(Image.open(BytesIO(soil_tile_png_bytes(0))))
        self.assertGreater(float(grass.std()), 18.0)
        self.assertGreater(float(soil.std()), 12.0)
        self.assertGreater(float(floor.colors_rgb.std()), 0.04)

        class Lights:
            light_pos = np.zeros((2, 3))
            light_dir = np.zeros((2, 3))

        model = Lights()
        look = np.array([1.0, -4.0, 1.2])
        aim_sun(model, look, 10.0)
        np.testing.assert_allclose(model.light_dir[0], _SUN_DIR)
        np.testing.assert_allclose(model.light_pos[0], look - 10.0 * _SUN_DIR)

    def test_tree_dapple_darkens_under_the_canopy(self):
        from treesim.orchard_terrain import shade_under_canopy
        floor = _pinned()
        skel = generate(height=1.6, seed=0, rows=2, columns=2, spacing=5.0,
                        ground_z=floor.ground_z, canopy_z=floor.canopy_z)
        shaded = shade_under_canopy(floor.colors_rgb, floor, skel, seed=0)
        n = shaded.shape[0]
        mid, edge = n // 2, 4
        self.assertLess(float(shaded[mid, mid].mean()),
                        0.85 * float(floor.colors_rgb[mid, mid].mean()))
        self.assertGreater(float(shaded[edge, edge].mean()),
                           0.90 * float(floor.colors_rgb[edge, edge].mean()))

    def test_slope_plane_and_canopy(self):
        floor = _pinned(slope_deg=2.0, rut_depth_m=0.0, noise_m=0.0)
        expected = 3.0 * math.tan(math.radians(2.0))
        self.assertAlmostEqual(
            floor.aisle_plane_z(3.0, 0.0) - floor.aisle_plane_z(0.0, 0.0),
            expected, places=9)
        self.assertAlmostEqual(
            floor.canopy_z(0.0, 0.0) - floor.ground_z(0.0, 0.0), 1.6, places=9)
        # With no ruts or noise, ground follows the aisle plane within the 1 cm crown.
        self.assertLess(abs(floor.ground_z(0.0, 0.0) - floor.aisle_plane_z(0.0, 0.0)), 0.015)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, slope_deg=20.0)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, noise_m=0.1)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, rut_depth_m=0.2)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, friction=0.1)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, half_extent_m=0)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, rut_depth_m=0.08, rut_width_m=0.05)
        with self.assertRaises(ValueError):
            sample_orchard_floor(float("nan"))
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, landform_m=9.0)
        with self.assertRaises(ValueError):
            sample_orchard_floor(0, landform_m=2.0, landform_wavelength_m=1.0)

    def test_pergola_posts_sit_on_sampled_floor(self):
        floor = _pinned(seed=42, slope_deg=-3.0, rut_depth_m=0.06, noise_m=0.02)
        skel = generate(height=1.6, seed=42, rows=2, columns=2, spacing=5.0,
                        ground_z=floor.ground_z, canopy_z=floor.canopy_z)
        self.assertEqual(len(skel.roots), 1)
        posts = [s for s in skel if abs(s.axis[2]) > 1.0]
        self.assertEqual(len(posts), 4)
        for post in posts:
            foot = post.start if post.start[2] < post.end[2] else post.end
            top = post.end if post.start[2] < post.end[2] else post.start
            self.assertAlmostEqual(top[2], floor.canopy_z(top[0], top[1]), places=9)
            self.assertAlmostEqual(foot[2], floor.ground_z(foot[0], foot[1]) - POST_EMBED_M,
                                   places=5)
            self.assertGreater(top[2] - foot[2], 1.4)
        for seg in skel:
            if seg.order == 1 or (seg.order == 2 and seg.supported):
                self.assertAlmostEqual(seg.start[2], floor.canopy_z(*seg.start[:2]), places=9)
                self.assertAlmostEqual(seg.end[2], floor.canopy_z(*seg.end[:2]), places=9)
            if seg.order == 2 and not seg.supported:
                self.assertLess(seg.end[2], floor.canopy_z(*seg.end[:2]) - 0.05)
            if seg.parent >= 0:
                np.testing.assert_array_equal(seg.start, skel[seg.parent].end)

        fp = FruitParams(max_count=12, stem_length=0.055,
                         colors=((0.39, 0.27, 0.12),))
        fruit = place_fruit(skel, fp, seed=42)
        self.assertEqual(len(fruit), 12)
        for f in fruit:
            extent = f.radius + f.half_height
            center = f.attach - np.array([0.0, 0.0, fp.stem_length + extent])
            self.assertLess(center[2] + extent, floor.canopy_z(*f.attach[:2]))
            self.assertGreater(center[2] - extent, floor.ground_z(center[0], center[1]))

    def test_plantation_cover_matches_commercial_grid(self):
        small = floor_kwargs_for_plantation(2, 2, 5.0)
        self.assertAlmostEqual(small["half_extent_m"], 15.0)
        self.assertAlmostEqual(small["cell_m"], 0.05)
        self.assertAlmostEqual(small["row_pitch_m"], 5.0)
        cover = floor_kwargs_for_plantation(45, 40, 5.0)
        half = cover["half_extent_m"]
        self.assertAlmostEqual(half, 0.5 * 44 * 5.0 + 12.0)
        self.assertLessEqual(int(round(2.0 * half / cover["cell_m"])) + 1, 501)
        self.assertLessEqual(cover["cell_m"], 0.20 * 5.0)
        self.assertAlmostEqual(cover["appearance_cell_m"], 0.25)
        self.assertAlmostEqual(cover["row_pitch_m"], 5.0)
        # Corner posts of the default 45×40 @ 5 m field stay on the hfield.
        self.assertGreater(half, 110.0)
        floor = _pinned(seed=42, **cover)
        self.assertEqual(floor.heights_m.shape[0],
                         int(round(2.0 * half / cover["cell_m"])) + 1)
        self.assertTrue(np.isfinite(floor.ground_z(97.5, 110.0)))

    def test_landform_rolls_instead_of_a_plane(self):
        roll = _pinned(seed=7, landform_m=2.4, landform_wavelength_m=18.0,
                       slope_deg=3.5, noise_m=0.0, rut_depth_m=0.0)
        self.assertGreater(float(roll.heights_m.max() - roll.heights_m.min()), 1.2)
        xs, ys = np.meshgrid(roll.x_m, roll.y_m)
        plane = np.tan(np.radians(roll.slope_deg)) * (
            xs * np.cos(roll.slope_azimuth_rad) + ys * np.sin(roll.slope_azimuth_rad))
        residual = roll.heights_m - plane
        self.assertGreater(float(residual.max() - residual.min()), 1.0)
        self.assertGreater(float(roll.landform_m.max()), 0.4)
        self.assertLess(float(roll.landform_m.min()), -0.4)
        for x, y in ((0.0, 0.0), (2.0, 3.0), (-5.0, 4.0), (8.0, -6.0)):
            self.assertAlmostEqual(
                roll.canopy_z(x, y) - roll.ground_z(x, y), 1.6, places=9)
        # The roof is not a single inclined plane: landform shows up in canopy_z.
        deltas = [
            abs(roll.canopy_z(x, y) - (roll.aisle_plane_z(x, y) + 1.6))
            for x in (-8.0, 0.0, 5.0, 9.0) for y in (-6.0, 2.0, 7.0)
        ]
        self.assertGreater(max(deltas), 0.4)

    def test_foliage_tracks_ground_offset(self):
        from treesim.config import FoliageParams
        from treesim.foliage import place_canopy_leaves, rotate_xyzw
        floor = _pinned(seed=7, landform_m=2.4, landform_wavelength_m=18.0,
                        slope_deg=3.5, noise_m=0.0, rut_depth_m=0.0)
        skel = generate(height=1.6, seed=7, rows=2, columns=2, spacing=5.0,
                        ground_z=floor.ground_z, canopy_z=floor.canopy_z)
        fp = FoliageParams(enabled=True, leaf_length=.22, leaf_width=.17,
                           canopy_spacing_m=.2)
        leaves = place_canopy_leaves(skel, fp, seed=42, height_z=floor.canopy_z)
        self.assertGreater(len(leaves), 50)
        offsets, zs = [], []
        for p in leaves:
            center = p.attach + rotate_xyzw(p.frame, np.array([0., 0., .11]))
            offsets.append(center[2] - floor.ground_z(center[0], center[1]))
            zs.append(center[2])
        offsets = np.asarray(offsets)
        self.assertTrue(np.all((offsets > 1.62) & (offsets < 1.76)))
        self.assertGreater(float(np.max(zs) - np.min(zs)), 0.6)

    def test_camera_enters_from_the_side_under_the_canopy(self):
        from scripts.record_orchard_mujoco import camera_pose, free_camera_eye

        class Floor:
            def canopy_z(self, x, y):
                return 1.6
            def ground_z(self, x, y):
                return 0.04

        start_eye = free_camera_eye(*camera_pose(0, 100, Floor(), 32.0, 5.0))
        end_eye = free_camera_eye(*camera_pose(99, 100, Floor(), 32.0, 5.0))
        self.assertLess(start_eye[1], -10.0)
        self.assertGreater(end_eye[1], start_eye[1])
        for frame in range(100):
            lookat, distance, azimuth, elevation = camera_pose(
                frame, 100, Floor(), 32.0, 5.0)
            eye = free_camera_eye(lookat, distance, azimuth, elevation)
            canopy = Floor().canopy_z(eye[0], eye[1])
            ground = Floor().ground_z(eye[0], eye[1])
            self.assertLess(eye[2], canopy - 0.25)
            self.assertGreater(eye[2], ground + 0.90)
            self.assertLess(lookat[2], canopy - 0.15)
            self.assertLess(lookat[2], eye[2])

    def test_flat_generate_unchanged(self):
        skel = generate(height=1.6, seed=42, rows=2, columns=2, spacing=5.0)
        posts = [s for s in skel if abs(s.axis[2]) > 1.0]
        for post in posts:
            zs = sorted([post.start[2], post.end[2]])
            self.assertAlmostEqual(zs[0], 0.0)
            self.assertAlmostEqual(zs[1], 1.6)


if __name__ == "__main__":
    unittest.main()
