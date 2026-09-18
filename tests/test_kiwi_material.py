import unittest
import numpy as np
from treesim.kiwi_material import sample_hayward, damage_increment, STEM_AXIAL, STEM_BENDING


class KiwiMaterialTest(unittest.TestCase):
    def test_coupled_mass_size_density(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            radii, mass = sample_hayward(rng)
            self.assertTrue(.0814 <= mass <= .1287)
            self.assertTrue(940 <= mass/(4*np.pi*np.prod(radii)/3) <= 1040)
            self.assertTrue(.04270 <= 2*radii[0] <= .05184)
            self.assertTrue(.04585 <= 2*radii[1] <= .05713)
        self.assertGreater(STEM_AXIAL, STEM_BENDING)

    def test_damage_is_irreversible_and_time_based(self):
        self.assertEqual(damage_increment(.03, 0, .1, 0), 0)
        damaged = damage_increment(.1, 0, .1, 0)
        self.assertGreater(damaged, 0)
        self.assertEqual(damage_increment(0, 0, .1, damaged), damaged)
        a = damage_increment(.1, 0, .05, 0)
        self.assertAlmostEqual(damage_increment(.1, 0, .05, a), damaged)
        self.assertEqual(damage_increment(1, 1e6, 10, 0), 1)
