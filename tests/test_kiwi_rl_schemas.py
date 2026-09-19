"""Obs/v3 + Action/v3 layout contracts (numpy only)."""
import unittest

import numpy as np

from treesim.kiwi_rl import schemas as S


def _raw_p85():
    return {
        "q_rel": np.zeros(19), "qd": np.zeros(19),
        "v_body_mps": np.zeros(3), "omega_body_rps": np.zeros(3),
        "gravity_body": np.array([0.0, 0.0, -1.0]),
        "arm_target": np.zeros(7),
        "tcp_pos_body_m": np.zeros(3),
        "tcp_rot6d_body": np.array([1, 0, 0, 0, 1, 0], dtype=float),
        "tcp_vel_body": np.zeros(6),
        "gripper_open_01": np.array([1.0]),
        "force_hand_n": np.zeros(3), "force_valid": np.array([0.0]),
        "height_m": np.array([0.5]), "foot_contact": np.ones(4),
        "base_command_prev": np.zeros(3),
        "hand_image_age_valid": np.array([0.1, 1.0]),
        "grip_contact_score": np.array([0.0]),
    }


class SchemaTest(unittest.TestCase):
    def test_layout_dims_sum_exactly(self):
        self.assertEqual(sum(b - a for _, a, b in S.R84_SLICES), 84)
        self.assertEqual(sum(b - a for _, a, b, *_ in S.P85_SLICES), 85)
        self.assertEqual(256 + 128 + 85 + 14 + 3 + 2, S.N3_INPUT_DIM)
        self.assertEqual(256 + 85 + 14 + 16 + 8 + 3, S.M3_INPUT_DIM)
        self.assertEqual(84 + 85 + 14 + 24 + 256, S.CRITIC_INPUT_DIM)

    def test_p85_slices_cover_every_dim_once(self):
        covered = []
        for _, a, b, *_ in S.P85_SLICES:
            covered.extend(range(a, b))
        self.assertEqual(sorted(covered), list(range(85)))

    def test_normalise_p85_finite_and_scaled(self):
        out, clipped = S.normalise_p85(_raw_p85())
        self.assertEqual(out.shape, (85,))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(clipped, 0)

    def test_normalise_rejects_bad_inputs(self):
        bad = _raw_p85()
        bad["height_m"] = np.array([np.inf])
        with self.assertRaises(ValueError):
            S.normalise_p85(bad)
        bad = _raw_p85()
        bad["height_m"] = np.array([99.0])
        with self.assertRaises(ValueError):
            S.normalise_p85(bad)
        bad = _raw_p85()
        bad["q_rel"] = np.zeros(18)
        with self.assertRaises(ValueError):
            S.normalise_p85(bad)

    def test_context14_onehots_and_ranges(self):
        c = S.build_context14("EXPLORE", "HARVEST", 1.0, 30.0, 0, 3, 0.2, 1.0,
                              0.01)
        self.assertEqual(c.shape, (14,))
        self.assertEqual(c[:6].sum(), 1.0)
        self.assertEqual(c[6:9].sum(), 1.0)
        with self.assertRaises(ValueError):
            S.build_context14("NOPE", "HARVEST", 0, 1, 0, 3, 0, 0, 0)

    def test_schema_hash_stable(self):
        self.assertEqual(S.schema_hash(), S.schema_hash())
        self.assertEqual(len(S.schema_hash()), 16)


if __name__ == "__main__":
    unittest.main()
