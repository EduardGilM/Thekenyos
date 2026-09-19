import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    'check_deformable_backend',
    Path(__file__).parents[1] / 'scripts' / 'check_deformable_backend.py',
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class PostGroundContactTest(unittest.TestCase):
    def test_latches_only_contact_after_ground_and_keeps_peak(self):
        report = {'first_ground_step': [[10], [20]], 'steps': [10, 20]}
        peak = np.zeros(2)
        first = np.full(2, np.nan)
        _MODULE.update_post_ground_contact(report, 1.0, [[3.0, 0.0], [4.0, 0.0]], peak, first)
        np.testing.assert_array_equal(peak, [0.0, 0.0])
        self.assertTrue(np.isnan(first).all())

        report['steps'] = [11, 21]
        _MODULE.update_post_ground_contact(report, 1.1, [[0.2, 0.0], [0.4, 0.0]], peak, first)
        np.testing.assert_array_equal(peak, [0.2, 0.4])
        np.testing.assert_array_equal(first, [1.1, 1.1])

        report['steps'] = [12, 22]
        _MODULE.update_post_ground_contact(report, 1.2, [[0.1, 0.0], [0.6, 0.0]], peak, first)
        np.testing.assert_array_equal(peak, [0.2, 0.6])
        np.testing.assert_array_equal(first, [1.1, 1.1])


if __name__ == '__main__':
    unittest.main()
