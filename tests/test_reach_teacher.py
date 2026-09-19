import unittest
import numpy as np

from treesim.kiwi_rl.reach_teacher import (
    axial_mouth_local, damped_least_squares, bounded_damped_least_squares,
    fruit_in_release_zone, hover_tcp_local_m, hover_tcp_world_m, basket_chassis_aabb_m,
    easy_over_opening_local_m, easy_start_local_m, grasp_local_near_tcp,
    hold_close_fracs, jaw_hold_q, jaw_open_closed_from_gaps,
    level_wrist_local_m, offset_grasp_local, opening_half_xy_m, over_opening_xy,
    scripted_jaw_target, select_hold_close, tcp_outside_basket,
    tcp_over_opening_above_rim,
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

    def test_low_over_hole_puts_level_wrist_inside_crate(self):
        lo, hi = basket_chassis_aabb_m()
        low_wrist = level_wrist_local_m(hover_tcp_local_m(0.10))
        high_wrist = level_wrist_local_m(hover_tcp_local_m(0.28))
        self.assertTrue(lo[0] < float(low_wrist[0]) < hi[0])
        self.assertTrue(lo[1] < float(low_wrist[1]) < hi[1])
        self.assertLess(float(low_wrist[2]) - float(hi[2]), 0.15)
        self.assertGreater(float(high_wrist[2]) - float(hi[2]), 0.25)
        with self.assertRaises(ValueError):
            level_wrist_local_m([0.0, 0.0, 0.5], tcp_to_wrist_m=0.0)

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
        centered = easy_start_local_m(0.0, side_y_m=0.0)
        self.assertGreaterEqual(float(near[0]), float(hi[0] + 0.32) - 1e-9)
        self.assertAlmostEqual(float(near[1]), float(CENTER[1] + 0.10))
        self.assertAlmostEqual(float(centered[1]), float(CENTER[1]))
        self.assertGreater(float(far[0]), float(near[0]))
        self.assertTrue(tcp_outside_basket(near, margin_m=0.04, above_rim_m=0.0))
        self.assertTrue(tcp_outside_basket(far, margin_m=0.04, above_rim_m=0.0))
        inside = np.array([float(CENTER[0]), float(CENTER[1]), float(CENTER[2] + SIZE[2] + 0.28)])
        pushed = push_tcp_outside_basket(inside, margin_m=0.40)
        self.assertGreaterEqual(float(pushed[0]), float(hi[0] + 0.40) - 1e-9)
        self.assertTrue(tcp_outside_basket(pushed, margin_m=0.04, above_rim_m=0.0))
        over = easy_over_opening_local_m()
        self.assertTrue(tcp_over_opening_above_rim(over))
        self.assertFalse(tcp_outside_basket(over, margin_m=0.04, above_rim_m=0.0))
        self.assertFalse(tcp_over_opening_above_rim(near))
        self.assertFalse(tcp_over_opening_above_rim(inside, min_clearance_m=0.30))
        self.assertTrue(tcp_over_opening_above_rim(inside, min_clearance_m=0.16))
        with self.assertRaises(ValueError):
            easy_start_local_m(-0.1)
        with self.assertRaises(ValueError):
            easy_start_local_m(1.1)
        with self.assertRaises(ValueError):
            tcp_over_opening_above_rim([np.nan, 0.0, 0.0])

    def test_hold_sweep_picks_tightest_contacting_keeper(self):
        fracs = hold_close_fracs()
        self.assertEqual(len(fracs), 10)
        self.assertAlmostEqual(jaw_hold_q(0.0, 0.2, -0.4), 0.2)
        rows = [
            {'close_frac': 0.4, 'slip_m': 0.20, 'max_load_N': 2.0, 'retained': False},
            {'close_frac': 0.6, 'slip_m': 0.03, 'max_load_N': 8.0, 'retained': True},
            {'close_frac': 0.8, 'slip_m': 0.01, 'max_load_N': 12.0, 'retained': True},
            {'close_frac': 1.0, 'slip_m': 0.01, 'max_load_N': 40.0, 'retained': True},
        ]
        chosen = select_hold_close(rows, slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(chosen['close_frac'], 0.8)
        crushed = select_hold_close([
            {'close_frac': 0.5, 'slip_m': 0.30, 'max_load_N': 4.0, 'retained': False},
            {'close_frac': 1.0, 'slip_m': 0.02, 'max_load_N': 22.0, 'retained': True},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(crushed['close_frac'], 1.0)
        over = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.02, 'max_load_N': 19.0, 'retained': True},
            {'close_frac': 1.0, 'slip_m': 0.39, 'max_load_N': 0.0, 'retained': False},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(over['close_frac'], 0.4)
        slipped = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.20, 'max_load_N': 2.0, 'retained': False},
            {'close_frac': 1.0, 'slip_m': 0.16, 'max_load_N': 40.0, 'retained': False},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(slipped['close_frac'], 0.4)
        tied = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.15, 'max_load_N': 2.0, 'retained': False},
            {'close_frac': 0.9, 'slip_m': 0.15, 'max_load_N': 12.0, 'retained': False},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(tied['close_frac'], 0.9)
        dumped = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.10, 'max_load_N': 2.0, 'retained': False},
            {'close_frac': 0.8, 'slip_m': 0.14, 'max_load_N': 10.0, 'retained': False},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(dumped['close_frac'], 0.8)
        empty = select_hold_close([
            {'close_frac': 0.4, 'slip_m': 0.96, 'max_load_N': 0.0, 'retained': False},
            {'close_frac': 0.6, 'slip_m': 0.96, 'max_load_N': 0.0, 'retained': False},
            {'close_frac': 1.0, 'slip_m': 0.96, 'max_load_N': 0.0, 'retained': False},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(empty['close_frac'], 0.6)
        live_like = select_hold_close([
            {'close_frac': 0.25, 'slip_m': 0.033, 'max_load_N': 10.9, 'retained': True},
            {'close_frac': 0.30, 'slip_m': 0.030, 'max_load_N': 13.3, 'retained': True},
            {'close_frac': 0.55, 'slip_m': 0.024, 'max_load_N': 0.0, 'retained': True},
            {'close_frac': 0.65, 'slip_m': 0.022, 'max_load_N': 12.6, 'retained': True},
        ], slip_ok_m=0.04, load_limit_n=15.0)
        self.assertAlmostEqual(live_like['close_frac'], 0.65)
        from treesim.kiwi_rl.reach_teacher import adapt_scripted_hold_q
        held = adapt_scripted_hold_q(-1.10, -1.57, 0.0, slip_m=0.01, over_basket=False)
        self.assertAlmostEqual(held, -1.10)
        tighter = adapt_scripted_hold_q(-1.10, -1.57, 0.0, slip_m=0.08, over_basket=False)
        self.assertGreater(tighter, -1.10)
        self.assertLessEqual(tighter, -1.57 + 0.70 * 1.57 + 1e-9)
        opened = adapt_scripted_hold_q(-1.10, -1.57, 0.0, slip_m=0.08, over_basket=True)
        self.assertAlmostEqual(opened, -1.10)
        with self.assertRaises(ValueError):
            select_hold_close([])

    def test_jaw_open_closed_follows_pad_gap_not_range_sign(self):
        opened, closed = jaw_open_closed_from_gaps(-1.5708, 0.0, 0.143, 0.024)
        self.assertAlmostEqual(opened, -1.5708)
        self.assertAlmostEqual(closed, 0.0)
        swapped_open, swapped_closed = jaw_open_closed_from_gaps(-1.5708, 0.0, 0.024, 0.143)
        self.assertAlmostEqual(swapped_open, 0.0)
        self.assertAlmostEqual(swapped_closed, -1.5708)
        with self.assertRaises(ValueError):
            jaw_open_closed_from_gaps(-1.5708, 0.0, 0.10, 0.10)
        with self.assertRaises(ValueError):
            jaw_open_closed_from_gaps(0.0, -1.5708, 0.14, 0.02)

    def test_scripted_jaw_holds_away_and_opens_over_basket(self):
        from treesim.kiwi_rl.reach_teacher import fruit_in_release_zone
        self.assertAlmostEqual(scripted_jaw_target([1.0, 0.0], [0.0, 0.0], -1.2, 0.0, open_xy_m=0.15), -1.2)
        self.assertAlmostEqual(scripted_jaw_target([0.05, 0.04], [0.0, 0.0], -1.2, 0.0, open_xy_m=0.15), 0.0)
        # Hover-high over the opening must keep holding; open only below the rim.
        self.assertAlmostEqual(
            scripted_jaw_target([0.05, 0.04, 0.70], [0.0, 0.0, 0.145], -1.2, 0.0,
                                open_xy_m=0.15, rim_z_m=0.28),
            -1.2)
        self.assertAlmostEqual(
            scripted_jaw_target([0.05, 0.04, 0.30], [0.0, 0.0, 0.145], -1.2, 0.0,
                                open_xy_m=0.15, rim_z_m=0.28),
            0.0)
        self.assertFalse(fruit_in_release_zone([0.05, 0.04, 0.70], [0.0, 0.0, 0.145],
                                               open_xy_m=0.15, rim_z_m=0.28))
        self.assertTrue(fruit_in_release_zone([0.05, 0.04, 0.30], [0.0, 0.0, 0.145],
                                              open_xy_m=0.15, rim_z_m=0.28))
        # Force-open at centre: hover height is enough if the hand is there too.
        self.assertAlmostEqual(
            scripted_jaw_target([0.05, 0.04, 0.70], [0.0, 0.0, 0.145], -1.2, 0.0,
                                open_xy_m=0.15, rim_z_m=0.28, tcp_xy=[0.02, 0.01, 0.70],
                                release_at_center=True),
            0.0)
        self.assertAlmostEqual(
            scripted_jaw_target([0.05, 0.04, 0.70], [0.0, 0.0, 0.145], -1.2, 0.0,
                                open_xy_m=0.15, rim_z_m=0.28, tcp_xy=[1.0, 0.0, 0.70],
                                release_at_center=True),
            -1.2)
        self.assertTrue(fruit_in_release_zone(
            [0.05, 0.04, 0.70], [0.0, 0.0, 0.145], open_xy_m=0.15, rim_z_m=0.28,
            tcp_xyz=[0.02, 0.01, 0.70], release_at_center=True))
        self.assertFalse(fruit_in_release_zone(
            [0.05, 0.04, 0.70], [0.0, 0.0, 0.145], open_xy_m=0.15, rim_z_m=0.28,
            tcp_xyz=[1.0, 0.0, 0.70], release_at_center=True))
        with self.assertRaises(ValueError):
            scripted_jaw_target([np.nan, 0.0], [0.0, 0.0], -1.2, 0.0, open_xy_m=0.15)

    def test_release_over_opening_is_wider_than_center_disk(self):
        hx, hy = opening_half_xy_m(inset_m=0.04)
        self.assertGreater(hx, 0.20)
        self.assertLess(hx, 0.27)
        self.assertGreater(hy, 0.12)
        self.assertLess(hy, 0.19)
        basket = [0.0, 0.0, 0.145]
        front = [0.20, 0.05, 0.70]
        tcp = [0.18, 0.04, 0.70]
        self.assertFalse(fruit_in_release_zone(
            front, basket, open_xy_m=0.15, rim_z_m=0.28,
            tcp_xyz=tcp, release_at_center=True))
        self.assertTrue(over_opening_xy(front, tcp, basket, inset_m=0.04))
        self.assertTrue(fruit_in_release_zone(
            front, basket, open_xy_m=0.15, rim_z_m=0.28,
            tcp_xyz=tcp, release_over_opening=True, inset_m=0.04))
        self.assertAlmostEqual(
            scripted_jaw_target(front, basket, -1.2, 0.0, open_xy_m=0.15, rim_z_m=0.28,
                                tcp_xy=tcp, release_over_opening=True, inset_m=0.04),
            0.0)
        side = [0.0, 0.18, 0.70]
        self.assertFalse(over_opening_xy(side, side, basket, inset_m=0.04))
        yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertFalse(over_opening_xy(front, tcp, basket, yaw, inset_m=0.04))
        self.assertTrue(over_opening_xy([0.05, 0.20, 0.70], [0.05, 0.20, 0.70],
                                       basket, yaw, inset_m=0.04))
        with self.assertRaises(ValueError):
            opening_half_xy_m(inset_m=0.20)
        with self.assertRaises(ValueError):
            fruit_in_release_zone(front, basket, open_xy_m=0.15, rim_z_m=0.28,
                                 release_over_opening=True)

    def test_grasp_offset_stays_between_pads_not_at_tcp(self):
        tcp = np.array([0.0, 0.0, 0.10])
        np.testing.assert_allclose(offset_grasp_local(tcp, tcp), tcp, atol=1e-9)
        np.testing.assert_allclose(
            offset_grasp_local(tcp, tcp, prefer_m=0.03, min_m=0.0, max_m=0.06),
            [0.0, 0.0, 0.07], atol=1e-9)
        pulled = offset_grasp_local(tcp, np.array([0.0, 0.0, 0.0]), max_m=0.06)
        np.testing.assert_allclose(pulled, [0.0, 0.0, 0.04], atol=1e-9)
        near = offset_grasp_local(tcp, np.array([0.0, 0.0, 0.09]))
        np.testing.assert_allclose(near, [0.0, 0.0, 0.09], atol=1e-9)
        tip = np.array([0.16, 0.0, 0.0])
        np.testing.assert_allclose(offset_grasp_local(tip, np.array([0.21, 0.0, 0.0])), tip)
        np.testing.assert_allclose(
            offset_grasp_local(tip, np.array([0.21, 0.0, 0.0]), prefer_m=0.02),
            [0.14, 0.0, 0.0], atol=1e-9)
        with self.assertRaises(ValueError):
            offset_grasp_local(tcp, [np.nan, 0.0, 0.0])

    def test_grasp_local_gate_uses_tcp_offset_not_body_origin(self):
        tcp = np.array([0.165, 0.0, 0.004])
        pad = np.array([0.140, 0.0, 0.010])
        knuckle = np.zeros(3)
        self.assertTrue(grasp_local_near_tcp(tcp, tcp))
        self.assertTrue(grasp_local_near_tcp(pad, tcp))
        self.assertFalse(grasp_local_near_tcp(knuckle, tcp))
        self.assertGreater(float(np.linalg.norm(pad)), 0.12)
        self.assertFalse(grasp_local_near_tcp([np.nan, 0.0, 0.0], tcp))
        with self.assertRaises(ValueError):
            grasp_local_near_tcp(tcp, tcp, max_offset_m=0.0)

    def test_axial_mouth_stays_on_tcp_axis_and_insets(self):
        tcp = np.array([0.20, 0.0, 0.0])
        np.testing.assert_allclose(axial_mouth_local(tcp), [0.18, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            axial_mouth_local(tcp, np.array([0.26, 0.04, 0.03])),
            [0.18, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            axial_mouth_local(tcp, np.array([0.16, 0.05, 0.0])),
            [0.16, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            axial_mouth_local(tcp, np.array([0.02, 0.0, 0.0])),
            [0.15, 0.0, 0.0], atol=1e-9)
        with self.assertRaises(ValueError):
            axial_mouth_local(np.zeros(3))

    def test_saturated_joint_can_move_inward(self):
        step = bounded_damped_least_squares(
            np.eye(3), np.array([-1., 0., 0.]), np.array([1., 0., 0.]),
            np.array([-1., -1., -1.]), np.array([1., 1., 1.]),
            damping=.01, max_step=.1)
        self.assertLess(float(step[0]), 0.)


if __name__ == '__main__':
    unittest.main()
