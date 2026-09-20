import math
import unittest

import numpy as np

from treesim.six_seven import (
    ARM, DANCE_HZ, HIND_HIP_X_M, LEGS, REAR_PITCH_RAD, REAR_S, STAND_S, STAND_Z_M,
    caption_for_phase, chassis_xyz_m, clamp_joint, duration_s, pose_at,
    rear_joints, rpy_to_wxyz, stand_joints,
)


HOME = {
    'arm_sh0': 0.0, 'arm_sh1': -0.9, 'arm_el0': 1.8, 'arm_el1': 0.0,
    'arm_wr0': -0.9, 'arm_wr1': 0.0, 'arm_f1x': -1.54,
    'fl_hx': 0.12, 'fr_hx': -0.12, 'hl_hx': 0.12, 'hr_hx': -0.12,
    'fl_hy': 0.5, 'fr_hy': 0.5, 'hl_hy': 0.5, 'hr_hy': 0.5,
    'fl_kn': -1.0, 'fr_kn': -1.0, 'hl_kn': -1.0, 'hr_kn': -1.0,
}


class SixSevenTest(unittest.TestCase):
    def test_stand_then_rear_then_six_seven(self):
        stand = pose_at(0.0, HOME)
        self.assertEqual(stand['rpy_rad'][1], 0.0)
        self.assertAlmostEqual(stand['xyz_m'][2], STAND_Z_M)
        self.assertEqual(stand['caption'], '')
        self.assertFalse(stand['reared'])
        self.assertFalse(stand['hind_support'])
        mid = pose_at(1.0 + 1.1, HOME)
        self.assertLess(mid['rpy_rad'][1], -0.3)
        self.assertGreater(mid['rpy_rad'][1], -REAR_PITCH_RAD)
        self.assertGreater(mid['xyz_m'][2], stand['xyz_m'][2])
        self.assertTrue(mid['hind_support'])
        reared = pose_at(STAND_S + REAR_S + 0.05, HOME)
        self.assertTrue(reared['reared'])
        self.assertAlmostEqual(reared['rpy_rad'][1], -REAR_PITCH_RAD)
        self.assertIn(reared['caption'], {'SIX', 'SEVEN'})
        # Front legs are the two "hands": higher hy than standing, spread hx.
        self.assertGreater(reared['joints']['fl_hy'], stand['joints']['fl_hy'])
        self.assertGreater(reared['joints']['fl_hx'], stand['joints']['fl_hx'])
        self.assertLess(reared['joints']['fr_hx'], stand['joints']['fr_hx'])
        # Hind hips swing back so the calves stand under the pelvis, not out behind a tilted chassis.
        self.assertGreater(reared['joints']['hl_hy'], 1.5)
        self.assertGreater(reared['joints']['hl_hy'], stand['joints']['hl_hy'] + 0.8)
        # Front knees fold so the paws sit at shoulder height, not on the floor.
        self.assertLess(reared['joints']['fl_kn'], stand['joints']['fl_kn'] - 0.8)
        six = pose_at(STAND_S + REAR_S, HOME)
        seven = pose_at(STAND_S + REAR_S + 0.6 / DANCE_HZ, HOME)
        self.assertEqual(six['caption'], 'SIX')
        self.assertEqual(seven['caption'], 'SEVEN')
        self.assertNotAlmostEqual(six['joints']['arm_sh0'], seven['joints']['arm_sh0'])
        self.assertLess(duration_s(), 12.0)

    def test_rear_pitch_is_nose_up(self):
        # R_y maps body +X to (cos, 0, -sin). Negative pitch => world z > 0.
        pitch = pose_at(STAND_S + REAR_S, HOME)['rpy_rad'][1]
        self.assertLess(pitch, 0.0)
        self.assertGreater(-math.sin(pitch), 0.8)
        xyz = chassis_xyz_m(pitch)
        self.assertLess(xyz[0], 0.0)
        self.assertGreater(xyz[2], STAND_Z_M)
        self.assertAlmostEqual(HIND_HIP_X_M, -0.29785)

    def test_limits_and_finite_quat(self):
        with self.assertRaises(ValueError):
            pose_at(-1.0, HOME)
        self.assertEqual(clamp_joint('fl_hx', 9.0), 0.785398)
        quat = rpy_to_wxyz(0.1, -REAR_PITCH_RAD, -0.05)
        self.assertTrue(np.isfinite(quat).all())
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=6)
        for name in (*LEGS, *ARM):
            self.assertIn(name, stand_joints(HOME))
            self.assertIn(name, rear_joints(HOME))
        self.assertEqual(caption_for_phase(0.0), 'SIX')
        self.assertEqual(caption_for_phase(math.pi), 'SEVEN')
