"""CPU-only geometry checks: python -m unittest discover -s tests."""
import unittest

import numpy as np

from treesim.config import FruitParams, preset
from treesim.lsystem import generate
from treesim.pergola import place_fruit


class PergolaTest(unittest.TestCase):
    def test_seeded_connected_canopy_and_hanging_fruit(self):
        params = preset("pergola")
        params.pergola_rows = 3
        params.pergola_columns = 40
        params.pergola_spacing = 5.0
        a, b = generate(params, seed=42), generate(params, seed=42)
        c = generate(params, seed=43)
        self.assertEqual(len(a.roots), 1)
        self.assertEqual(len(a.descendants(0)), len(a) - 1)
        np.testing.assert_array_equal([s.end for s in a], [s.end for s in b])
        self.assertFalse(np.array_equal([s.end for s in a], [s.end for s in c]))
        posts = [s for s in a if abs(s.axis[2]) > 1.0]
        self.assertEqual(len(posts), params.pergola_rows * params.pergola_columns)
        self.assertAlmostEqual(a.bounds()[1][0] - a.bounds()[0][0], 39 * 5.0, places=1)
        for seg in a:
            self.assertGreater(seg.length, 0)
            self.assertTrue(np.isfinite(seg.frame).all())
            self.assertAlmostEqual(np.linalg.norm(seg.frame), 1)
            q, w = seg.frame[:3], seg.frame[3]
            z = np.array([0., 0., 1.])
            rotated = z + 2 * np.cross(q, np.cross(q, z) + w * z)
            np.testing.assert_allclose(rotated, seg.direction, atol=1e-12)
            if seg.parent >= 0:
                self.assertLess(seg.parent, seg.index)
                np.testing.assert_array_equal(seg.start, a[seg.parent].end)
            if seg.order > 0:
                self.assertAlmostEqual(seg.start[2], 1.6)
                self.assertAlmostEqual(seg.end[2], 1.6)

        fp = FruitParams(max_count=96, radius=(0.024, 0.028), stem_length=0.055,
                         colors=((0.39, 0.27, 0.12),))
        fruit = place_fruit(a, fp, seed=42)
        self.assertEqual(len(fruit), 96)
        np.testing.assert_array_equal([f.attach for f in fruit],
                                      [f.attach for f in place_fruit(a, fp, seed=42)])
        centers = []
        for f in fruit:
            self.assertEqual(a[f.parent_seg].order, 2)
            extent = f.radius + f.half_height
            center = f.attach - [0, 0, fp.stem_length + extent]
            self.assertLess(center[2] + extent, 1.6)
            self.assertGreater(center[2] - extent, 0)
            self.assertGreater(center[0], a.bounds()[0][0])
            self.assertLess(center[0], a.bounds()[1][0])
            self.assertGreater(center[1], a.bounds()[0][1])
            self.assertLessEqual(center[1], a.bounds()[1][1])
            centers.append(center)
        # Conservative bounding-sphere separation also proves capsule pairs
        # do not intersect in their initial pose.
        self.assertTrue(np.isfinite(centers).all())
        fp.max_count = 3
        self.assertEqual(len(place_fruit(a, fp)), 3)
        fp.max_count = 0
        self.assertEqual(place_fruit(a, fp), [])
        fp.max_count = -1
        with self.assertRaises(ValueError):
            place_fruit(a, fp)

    def test_rigid_pergola_still_has_free_fruit(self):
        try:
            import newton
        except ImportError:
            self.skipTest('Requires Newton')
        from treesim import builder
        from treesim.config import TreeConfig
        cfg = TreeConfig(lsystem=preset('pergola'), device='cpu')
        cfg.fruit.enabled, cfg.fruit.max_count = True, 2
        tree = builder.generate_and_build(cfg)
        self.assertEqual(len(tree.apple_bodies), 2)
        mass = tree.model.body_mass.numpy()[tree.apple_bodies]
        self.assertTrue(np.all((mass >= .0814) & (mass <= .1287)))
        for body in tree.apple_bodies:
            joint = list(tree.model.joint_child.numpy()).index(body)
            self.assertEqual(tree.model.joint_type.numpy()[joint], newton.JointType.FREE)

    def test_height_and_existing_presets(self):
        params = preset("pergola")
        params.target_height = 1.8
        self.assertAlmostEqual(generate(params).height(), 1.8)
        for height in (0, -1, float("nan"), float("inf")):
            params.target_height = height
            with self.assertRaises(ValueError):
                generate(params)
        self.assertEqual(preset("apple").kind, "apple")
        self.assertEqual(preset("tb").kind, "ternary")


if __name__ == "__main__":
    unittest.main()
