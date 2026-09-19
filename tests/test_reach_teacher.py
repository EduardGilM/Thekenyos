import unittest
import numpy as np

from treesim.kiwi_rl.reach_teacher import (
    damped_least_squares, bounded_damped_least_squares, hover_tcp_world_m,
    basket_chassis_aabb_m, easy_start_local_m, tcp_outside_basket,
)


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

    def test_hover_tcp_is_above_basket_not_inside_liner(self):
        from treesim.basket import CENTER, SIZE
        xpos = np.array([1.0, 2.0, 3.0])
        xmat = np.eye(3)
        target = hover_tcp_world_m(xpos, xmat, 0.12)
        local_top = CENTER[2] + SIZE[2]
        self.assertAlmostEqual(float(target[0]), 1.0 + float(CENTER[0]))
        self.assertAlmostEqual(float(target[2]), 3.0 + local_top + 0.12)
        self.assertGreater(float(target[2] - xpos[2]), local_top)

    def test_easy_start_is_outside_crate_and_recedes(self):
        from treesim.basket import CENTER, SIZE
        from treesim.kiwi_rl.reach_teacher import hover_tcp_local_m, push_tcp_outside_basket
        xpos = np.array([1.0, 2.0, 3.0])
        xmat = np.eye(3)
        hover = hover_tcp_world_m(xpos, xmat, 0.28)
        hover_local = hover_tcp_local_m(0.28)
        self.assertAlmostEqual(float(hover[0]), 1.0 + float(CENTER[0]))
        self.assertFalse(tcp_outside_basket(hover_local, margin_m=0.04, above_rim_m=0.0))
        lo, hi = basket_chassis_aabb_m()
        np.testing.assert_allclose(hi[2], float(CENTER[2] + SIZE[2]))
        near = easy_start_local_m(0.0)
        far = easy_start_local_m(1.0)
        self.assertGreaterEqual(float(near[0]), float(hi[0] + 0.40) - 1e-9)
        self.assertGreater(float(far[0]), float(near[0]))
        self.assertTrue(tcp_outside_basket(near, margin_m=0.04, above_rim_m=0.0))
        self.assertTrue(tcp_outside_basket(far, margin_m=0.04, above_rim_m=0.0))
        inside = np.array([float(CENTER[0]), float(CENTER[1]), float(CENTER[2] + SIZE[2] + 0.28)])
        pushed = push_tcp_outside_basket(inside, margin_m=0.40)
        self.assertGreaterEqual(float(pushed[0]), float(hi[0] + 0.40) - 1e-9)
        self.assertTrue(tcp_outside_basket(pushed, margin_m=0.04, above_rim_m=0.0))
        with self.assertRaises(ValueError):
            easy_start_local_m(-0.1)
        with self.assertRaises(ValueError):
            easy_start_local_m(1.1)

    def test_saturated_joint_can_move_inward(self):
        step = bounded_damped_least_squares(
            np.eye(3), np.array([-1., 0., 0.]), np.array([1., 0., 0.]),
            np.array([-1., -1., -1.]), np.array([1., 1., 1.]),
            damping=.01, max_step=.1)
        self.assertLess(float(step[0]), 0.)


if __name__ == '__main__':
    unittest.main()
