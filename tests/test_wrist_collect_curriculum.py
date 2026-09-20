import unittest

from scripts.train_wrist_search_harvest import collect_stages, stages


class WristCollectCurriculumTest(unittest.TestCase):
    def test_collect_skips_held_fruit_fixtures_and_mixes_deposits(self):
        lessons = stages()
        self.assertEqual([item['name'] for item in lessons],
                         ['search-grab', 'collect-1', 'collect-3'])
        self.assertEqual([item['name'] for item in collect_stages()],
                         ['collect-1', 'collect-3'])
        for item in collect_stages():
            args = item['args']
            self.assertNotIn('--search-rewards', args)
            self.assertEqual(args[args.index('--deposit-shaping')+1], '1')
            self.assertEqual(args[args.index('--view-weight')+1], '0')
            mix = args[args.index('--deposit-mix')+1]
            self.assertIn('pick:0.5', mix)
            self.assertIn('carry:0.3', mix)
            self.assertIn('release:0.2', mix)
        self.assertEqual(collect_stages()[-1]['args'][collect_stages()[-1]['args'].index('--picks')+1], '3')
        self.assertTrue(all(item['output'].endswith('-03') for item in collect_stages()))


if __name__ == '__main__':
    unittest.main()
