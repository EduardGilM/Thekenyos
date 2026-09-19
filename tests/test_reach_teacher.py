import unittest
import numpy as np

from treesim.kiwi_rl.reach_teacher import (
    damped_least_squares, bounded_damped_least_squares, hover_tcp_world_m,
    easy_airdrop_world_m,
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

    def test_easy_airdrop_uses_basket_xy_not_tcp_xy(self):
        from treesim.basket import CENTER, SIZE
        from treesim.kiwi_rl.curriculum import EASY_PRESET
        from treesim.kiwi_rl.reach_teacher import hover_tcp_local_m
        xpos = np.array([1.0, 2.0, 3.0])
        xmat = np.eye(3)
        tcp = hover_tcp_world_m(xpos, xmat, 0.12) + np.array([0.047, -0.02, 0.0])
        pos = easy_airdrop_world_m(tcp, xpos, xmat)
        np.testing.assert_allclose(pos[:2], (xpos + xmat @ CENTER)[:2])
        self.assertAlmostEqual(float(pos[2]), float(tcp[2] - EASY_PRESET['drop_offset_m']))
        self.assertGreater(float(pos[2] - xpos[2]), float(CENTER[2] + SIZE[2]))
        xmat90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        tcp90 = xpos + xmat90 @ hover_tcp_local_m(0.12)
        pos90 = easy_airdrop_world_m(tcp90, xpos, xmat90)
        np.testing.assert_allclose(pos90[:2], (xpos + xmat90 @ CENTER)[:2])
        self.assertAlmostEqual(float(pos90[2]), float(tcp90[2] - 0.10))
        with self.assertRaises(ValueError):
            easy_airdrop_world_m(tcp, xpos, xmat, drop_offset_m=0.0)
        with self.assertRaises(ValueError):
            easy_airdrop_world_m(np.array([np.nan, 0.0, 1.0]), xpos, xmat)

    def test_saturated_joint_can_move_inward(self):
        step = bounded_damped_least_squares(
            np.eye(3), np.array([-1., 0., 0.]), np.array([1., 0., 0.]),
            np.array([-1., -1., -1.]), np.array([1., 1., 1.]),
            damping=.01, max_step=.1)
        self.assertLess(float(step[0]), 0.)


if __name__ == '__main__':
    unittest.main()
