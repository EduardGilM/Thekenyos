import unittest
import numpy as np

from treesim.kiwi_rl.reach_teacher import damped_least_squares, bounded_damped_least_squares


class ReachTeacherMathTest(unittest.TestCase):
    def test_dls_moves_toward_error_and_is_bounded(self):
        J = np.eye(3)
        step = damped_least_squares(J, np.array([2., -2., .01]), damping=.01, max_step=.1)
        np.testing.assert_allclose(step, [.1, -.1, .01], atol=1e-4)

    def test_nonfinite_jacobian_rejected(self):
        with self.assertRaises(ValueError):
            damped_least_squares([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])

    def test_saturated_joint_is_removed_when_error_points_outward(self):
        J = np.eye(3)
        step = bounded_damped_least_squares(
            J, np.array([1., .2, .1]), np.array([1., 0., 0.]),
            np.array([-1., -1., -1.]), np.array([1., 1., 1.]),
            damping=.01, max_step=.1)
        self.assertEqual(float(step[0]), 0.)
        self.assertGreater(float(step[1]), 0.)
        self.assertGreater(float(step[2]), 0.)

    def test_saturated_joint_can_move_inward(self):
        step = bounded_damped_least_squares(
            np.eye(3), np.array([-1., 0., 0.]), np.array([1., 0., 0.]),
            np.array([-1., -1., -1.]), np.array([1., 1., 1.]),
            damping=.01, max_step=.1)
        self.assertLess(float(step[0]), 0.)


if __name__ == '__main__':
    unittest.main()
