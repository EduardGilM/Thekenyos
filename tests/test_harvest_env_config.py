import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('newton') and importlib.util.find_spec('gymnasium'), 'Legacy simulation stack required')
class HarvestConfigTest(unittest.TestCase):
    def test_default_fixture_configuration_is_preserved(self):
        from treesim.harvest_env import SpotHarvestEnv
        env = SpotHarvestEnv('.')
        self.assertTrue(env.fixed_base)
        self.assertEqual(env.fruit_count, 1)
        self.assertIsNone(env.foliage_density)
        self.assertEqual(env.action_space.shape, (7,))
        env.close()

    def test_export_options_are_explicit(self):
        from treesim.harvest_env import SpotHarvestEnv
        env = SpotHarvestEnv('.', fixed_base=False, fruit_count=5, foliage_density=.6)
        self.assertFalse(env.fixed_base)
        self.assertEqual(env.fruit_count, 5)
        self.assertEqual(env.foliage_density, .6)
        env.close()

    def test_invalid_generation_options_are_rejected(self):
        from treesim.harvest_env import SpotHarvestEnv
        for options in ({'fixed_base': 'false'}, {'fruit_count': 0}, {'fruit_count': 1.5},
                        {'fruit_count': 129}, {'foliage_density': float('nan')}, {'foliage_density': -1.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SpotHarvestEnv('.', **options)


if __name__ == '__main__':
    unittest.main()
