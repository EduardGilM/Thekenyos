"""CPU-only orchard heightfield checks: python -m unittest discover -s tests."""
import unittest

import numpy as np

from treesim.config import PhysicsParams
from treesim.orchard_ground import (
    AISLE_HEIGHT_M, FURROW_HEIGHT_M, PEAK_HEIGHT_M, generate, nearest_aisle_x_m,
    profile_height_m, row_pitch_m,
)
from treesim.pergola import BAY_X_M, POST_X0_M


class OrchardGroundTest(unittest.TestCase):
    def test_profile_matches_pasillo_surco_pasillo(self):
        pitch = row_pitch_m()
        self.assertEqual(pitch, BAY_X_M)
        self.assertAlmostEqual(profile_height_m(POST_X0_M), PEAK_HEIGHT_M)
        self.assertAlmostEqual(profile_height_m(POST_X0_M + 0.25 * pitch), AISLE_HEIGHT_M)
        self.assertAlmostEqual(profile_height_m(POST_X0_M + 0.5 * pitch), FURROW_HEIGHT_M)
        self.assertAlmostEqual(profile_height_m(POST_X0_M + 0.75 * pitch), AISLE_HEIGHT_M)
        self.assertAlmostEqual(profile_height_m(POST_X0_M + pitch), PEAK_HEIGHT_M)

    def test_robot_snaps_to_aisle_not_furrow(self):
        aisle = nearest_aisle_x_m(-0.65)
        self.assertAlmostEqual(aisle, POST_X0_M + 0.25 * BAY_X_M)
        self.assertAlmostEqual(profile_height_m(aisle), AISLE_HEIGHT_M)
        furrow = POST_X0_M + 0.5 * BAY_X_M
        self.assertAlmostEqual(furrow, 0.0)
        self.assertNotAlmostEqual(aisle, furrow)

    def test_seeded_episode_ranges_and_units(self):
        a = generate(PhysicsParams(), 42)
        b = generate(PhysicsParams(), 42)
        c = generate(PhysicsParams(), 43)
        np.testing.assert_allclose(a.heights_m, b.heights_m)
        self.assertFalse(np.allclose(a.heights_m, c.heights_m))
        for field in (a, c):
            self.assertTrue(-4.0 <= field.slope_deg <= 4.0)
            self.assertTrue(0.0 <= field.ground_noise_m <= 0.04)
            self.assertTrue(0.0 <= field.rut_depth_m <= 0.08)
            self.assertTrue(0.20 <= field.rut_width_m <= 0.60)
            self.assertTrue(0.6 <= field.friction <= 1.3)
            self.assertTrue(np.isfinite(field.heights_m).all())
            self.assertIn("wet/liner", field.metrics()["note"])
            spawn = field.nearest_aisle_x_m(-0.65)
            self.assertAlmostEqual(profile_height_m(spawn), AISLE_HEIGHT_M)
            z = field.height_m(spawn, 0.0)
            self.assertTrue(np.isfinite(z))
            self.assertGreater(field.spawn_height_m(spawn, 0.0), 0.4)

    def test_rejects_inverted_ranges(self):
        ph = PhysicsParams()
        ph.orchard_friction = (1.3, 0.6)
        with self.assertRaises(ValueError):
            generate(ph, 0)
