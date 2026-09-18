"""CPU-only vineyard checks: python -m unittest discover -s tests."""
import unittest

import numpy as np

from treesim.config import FruitParams, preset
from treesim.lsystem import generate
from treesim.vineyard import generate_vsp as gen_vineyard, generate as gen_canopy, place_fruit
from treesim.grape_material import CLUSTER_MASS_RANGE


def _fruit_params(n):
    return FruitParams(max_count=n,
                       colors=((0.35, 0.12, 0.30), (0.28, 0.08, 0.22),
                               (0.55, 0.70, 0.35)))


class VineyardGeometryTest(unittest.TestCase):
    def test_seeded_connected_skeleton(self):
        a = gen_vineyard(seed=42)
        b = gen_vineyard(seed=42)
        c = gen_vineyard(seed=43)
        self.assertEqual(len(a.roots), 1)
        self.assertEqual(len(a.descendants(0)), len(a) - 1)
        np.testing.assert_array_equal([s.end for s in a], [s.end for s in b])
        self.assertFalse(np.array_equal([s.end for s in a], [s.end for s in c]))
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

        # posts: order-0 vertical segments reaching the ground, radius 0.05
        posts = [s for s in a if s.order == 0 and abs(s.direction[2]) > 0.9
                 and s.radius_start > 0.04 and min(s.start[2], s.end[2]) < 0.01]
        self.assertEqual(len(posts), 3)          # x = -3, 0, 3 for the 6 m row
        # trunks: order-1 vertical segments reaching the ground
        trunks = [s for s in a if s.order == 1 and abs(s.direction[2]) > 0.9]
        self.assertEqual(len(trunks), 4)         # 4 vines at spacing 1.5, offset
        # order-0/1 horizontal segments live at the three wire heights
        heights = {round(float(s.midpoint[2]), 3)
                   for s in a if s.order < 2 and abs(s.direction[2]) < 0.1}
        self.assertTrue(heights <= {0.9, 1.25, 1.6})
        self.assertEqual(heights, {0.9, 1.25, 1.6})
        for s in a:
            if s.order == 2:
                self.assertAlmostEqual(s.start[2], 0.9)
                self.assertGreaterEqual(s.end[2], 1.8)

    def test_rows_stay_connected(self):
        sk = gen_vineyard(rows=2, seed=1)
        self.assertEqual(len(sk.roots), 1)
        self.assertEqual(len(sk.descendants(0)), len(sk) - 1)
        ys = {round(float(s.midpoint[1]), 3) for s in sk}
        self.assertIn(-1.2, ys)
        self.assertIn(1.2, ys)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            gen_vineyard(rows=0)
        with self.assertRaises(ValueError):
            gen_vineyard(row_length=float("nan"))
        with self.assertRaises(ValueError):
            gen_vineyard(cordon_height=0.2)

    def test_preset(self):
        params = preset("vineyard")
        self.assertEqual(params.kind, "vineyard")
        self.assertAlmostEqual(params.target_height, 2.3)
        sk = generate(params, seed=7)
        self.assertEqual(len(sk.roots), 1)


class VineyardFruitTest(unittest.TestCase):
    def test_cluster_placement(self):
        sk = gen_vineyard(seed=42)
        fp = _fruit_params(16)
        fruit = place_fruit(sk, fp, seed=42)
        self.assertEqual(len(fruit), 16)
        np.testing.assert_array_equal(
            [f.attach for f in fruit],
            [f.attach for f in place_fruit(sk, fp, seed=42)])
        centres = []
        for f in fruit:
            self.assertEqual(sk[f.parent_seg].order, 2)
            centre = f.attach - np.array([0., 0., f.peduncle_length + f.radii[2]])
            self.assertTrue(0.55 < centre[2] < 1.15)
            self.assertGreater(centre[2] - f.radii[2], 0.4)
            self.assertTrue(CLUSTER_MASS_RANGE[0] <= f.mass <= CLUSTER_MASS_RANGE[1])
            self.assertGreater(len(f.berries), 20)
            self.assertTrue(np.all(np.abs(f.berries) / f.radii <= 1.0 + 1e-9))
            centres.append((centre, float(f.radii[2])))
        for i in range(len(centres)):
            for j in range(i):
                self.assertGreater(
                    np.linalg.norm(centres[i][0] - centres[j][0]),
                    centres[i][1] + centres[j][1])
        fp.max_count = 0
        self.assertEqual(place_fruit(sk, fp), [])
        fp.max_count = -1
        with self.assertRaises(ValueError):
            place_fruit(sk, fp)


class OverheadCanopyTest(unittest.TestCase):
    def test_walkable_canopy(self):
        a, b = gen_canopy(seed=42), gen_canopy(seed=42)
        self.assertEqual(len(a.roots), 1)
        self.assertEqual(len(a.descendants(0)), len(a) - 1)
        np.testing.assert_array_equal([s.end for s in a], [s.end for s in b])
        for s in a:
            self.assertGreater(s.length, 0)
            self.assertTrue(np.isfinite(s.frame).all())
            if s.parent >= 0:
                np.testing.assert_array_equal(s.start, a[s.parent].end)
            if min(s.start[2], s.end[2]) < 1.8:
                self.assertGreaterEqual(abs(s.start[1]), 1.1)
                self.assertGreaterEqual(abs(s.end[1]), 1.1)
        fruit = place_fruit(a, _fruit_params(80), seed=42)
        self.assertEqual(len(fruit), 80)
        for f in fruit:
            bottom = f.attach[2] - f.peduncle_length - 2 * f.radii[2]
            self.assertGreater(bottom, 1.8)
            top = f.berries[f.berries[:, 2] > 0, :2]
            tip = f.berries[f.berries[:, 2] < -.5 * f.radii[2], :2]
            self.assertGreater(np.max(np.linalg.norm(top, axis=1)),
                               np.max(np.linalg.norm(tip, axis=1)))
        for kwargs in ({'rows': 1}, {'cordon_height': 1.2}, {'row_spacing': float('nan')}):
            with self.assertRaises(ValueError):
                gen_canopy(**kwargs)


class VineyardNewtonTest(unittest.TestCase):
    def _build(self, deformable=False):
        try:
            import newton  # noqa: F401
        except ImportError:
            self.skipTest('Requires Newton')
        from treesim import builder
        from treesim.config import TreeConfig
        cfg = TreeConfig(lsystem=preset('vineyard'), device='cpu')
        cfg.deformable = deformable
        cfg.physics.damping_ratio = .3
        cfg.fruit.enabled = True
        cfg.fruit.max_count = 3
        cfg.fruit.joint = "free"
        tree = builder.generate_and_build(cfg)
        tree._placements = place_fruit(tree.skeleton, cfg.fruit, seed=cfg.seed)
        return tree

    def test_cluster_bodies(self):
        import newton
        tree = self._build()
        self.assertEqual(len(tree.apple_bodies), 3)
        mass = tree.model.body_mass.numpy()[tree.apple_bodies]
        # body mass = sampled cluster mass (visual shapes add none), 1%
        expected = np.array([p.mass for p in tree._placements])
        np.testing.assert_allclose(mass, expected, rtol=0.01)
        for body in tree.apple_bodies:
            joint = list(tree.model.joint_child.numpy()).index(body)
            self.assertEqual(tree.model.joint_type.numpy()[joint],
                             newton.JointType.FREE)
        flags = tree.model.shape_flags.numpy()
        sbody = tree.model.shape_body.numpy()
        for body in tree.apple_bodies:
            ids = np.flatnonzero(sbody == body)
            self.assertTrue(flags[ids[0]] & newton.ShapeFlags.COLLIDE_SHAPES)
            self.assertFalse(flags[ids[0]] & newton.ShapeFlags.VISIBLE)
            for s in ids[1:]:
                self.assertFalse(flags[s] & newton.ShapeFlags.COLLIDE_SHAPES)

    def test_sim_cut(self):
        try:
            import newton  # noqa: F401
        except ImportError:
            self.skipTest('Requires Newton')
        import time
        from treesim.sim import Sim
        tree = self._build(deformable=True)
        sim = Sim(tree, solver="mujoco", substeps=3)
        z0 = sim.state_0.body_q.numpy()[tree.apple_bodies, 2].copy()
        t0 = time.time()
        for _ in range(30):
            sim.step()
        q = sim.state_0.body_q.numpy()
        self.assertTrue(np.isfinite(q).all())
        self.assertEqual(sim.apples.broken_count, 0)
        z1 = q[tree.apple_bodies, 2]
        np.testing.assert_allclose(z1, z0, atol=0.03)
        self.assertTrue(sim.apples.cut(0))
        for _ in range(40):
            sim.step()
        q = sim.state_0.body_q.numpy()
        self.assertTrue(np.isfinite(q).all())
        self.assertLess(q[tree.apple_bodies[0], 2], z1[0] - 0.1)
        self.assertEqual(sim.apples.broken_count, 1)
        self.assertEqual(list(sim.apples.attached_indices()), [1, 2])
        print(f"[vineyard-sim] {time.time()-t0:.1f}s wall")


if __name__ == "__main__":
    unittest.main()
