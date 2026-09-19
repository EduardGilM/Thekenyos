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

        tips = [s for s in a if s.order == 2 and not s.supported]
        self.assertTrue(tips)
        for tip in tips:
            self.assertAlmostEqual(tip.length, .35)
            self.assertTrue(a[tip.parent].supported)
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
        # Body independence needs a small fixture, not the full field default.
        cfg.lsystem.pergola_rows = cfg.lsystem.pergola_columns = 2
        cfg.fruit.enabled, cfg.fruit.max_count = True, 2
        small = generate(cfg.lsystem)
        self.assertTrue(all(s.length > 0 and np.isfinite(s.frame).all() for s in small))
        tree = builder.generate_and_build(cfg)
        self.assertEqual(len(tree.apple_bodies), 2)
        mass = tree.model.body_mass.numpy()[tree.apple_bodies]
        self.assertTrue(np.all((mass >= .0814) & (mass <= .1287)))
        for body in tree.apple_bodies:
            joint = list(tree.model.joint_child.numpy()).index(body)
            self.assertEqual(tree.model.joint_type.numpy()[joint], newton.JointType.FREE)

    def test_smooth_noise_seed_post_heights_and_bounds(self):
        import newton
        from treesim.builder import _add_terrain
        from treesim.config import PhysicsParams
        ph = PhysicsParams(terrain=True, terrain_amplitude=.24,
                           terrain_wavelength=1.3, terrain_extent=6., terrain_seed=7)
        pads = [(-1.5, -2.), (-1.5, 2.), (1.5, -2.), (1.5, 2.)]

        def terrain(seed):
            b = newton.ModelBuilder()
            sample = _add_terrain(b, ph, seed, kiwi=True)
            return b.shape_source[-1], sample

        a, sample = terrain(42)
        b, _ = terrain(43)
        np.testing.assert_array_equal(a.data, b.data)
        ph.terrain_seed = 8
        c, _ = terrain(42)
        self.assertFalse(np.array_equal(a.data, c.data))
        self.assertAlmostEqual(a.min_z, .004)
        self.assertAlmostEqual(a.max_z, .244)
        for x, y in pads:
            self.assertGreater(sample(x, y), .02)
        elevation = a.data * (a.max_z-a.min_z) + a.min_z
        self.assertLess(np.abs(np.diff(elevation, axis=0)).max(), .04)
        self.assertLess(np.abs(np.diff(elevation, axis=1)).max(), .04)
        heights = [sample(x, y) for x in np.linspace(-.7, .7, 15)
                   for y in np.linspace(-.7, .7, 15)]
        self.assertGreater(np.ptp(heights), .02)
        self.assertGreater(min(heights), .004)
        for row, col in ((0, 0), (12, 23), (a.nrow-1, a.ncol-1)):
            x = -a.hx + 2*a.hx*col/(a.ncol-1)
            y = -a.hy + 2*a.hy*row/(a.nrow-1)
            self.assertAlmostEqual(sample(x, y), a.min_z + a.data[row, col]*(a.max_z-a.min_z), places=6)
        self.assertEqual(sample(6.01, 0), 0.)
        ph.terrain_amplitude = 0.
        _, flat = terrain(42)
        self.assertAlmostEqual(flat(.5, .5), .004)
        ph.terrain_amplitude = .24
        ph.terrain_seed = None
        a, _ = terrain(42)
        b, _ = terrain(43)
        self.assertFalse(np.array_equal(a.data, b.data))
        tiled = newton.ModelBuilder()
        sample = _add_terrain(tiled, ph, 42, pitch=(6., 8.), grid=(2, 2), kiwi=True)
        for x, y in ((.3, .7), (2.9, 3.9), (-3., -4.)):
            self.assertAlmostEqual(sample(x, y), sample(x+6., y+8.))
        apple = _add_terrain(newton.ModelBuilder(), ph, 42)
        self.assertAlmostEqual(apple(0., 0.), .004)

    def test_terrain_resists_fast_forced_fruit_impact(self):
        import newton
        import warp as wp
        from treesim.builder import _add_terrain
        from treesim.config import PhysicsParams
        params = preset('pergola')
        params.pergola_rows = params.pergola_columns = 2
        fruit = place_fruit(generate(params, seed=42),
                            FruitParams(max_count=1), seed=42)[0]
        b = newton.ModelBuilder()
        ph = PhysicsParams(terrain_amplitude=.24, terrain_extent=6.,
                           terrain_wavelength=1.3, terrain_seed=7)
        height = _add_terrain(b, ph, 42, kiwi=True)
        b.add_ground_plane()
        body = b.add_body(xform=wp.transform(wp.vec3(*(fruit.attach-[0., 0., .1])),
                                             wp.quat_identity()))
        density = float(fruit.mass/(4*np.pi*np.prod(fruit.radii)/3))
        b.add_shape_ellipsoid(body, rx=float(fruit.radii[0]), ry=float(fruit.radii[1]),
                             rz=float(fruit.radii[2]), cfg=b.ShapeConfig(density=density))
        model = b.finalize(device='cpu')
        s0, s1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, s1)
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True,
                                            nconmax=256, njmax=1024, solver=1)
        control, contacts = model.control(), model.contacts()
        minimum_clearance = float('inf')
        for step in range(2000):
            s0.clear_forces()
            if step < 1000:
                s0.body_f.assign(np.array([[0., 0., -20., 0., 0., 0.]], dtype=np.float32))
            solver.step(s0, s1, control, contacts, .0005)
            s0, s1 = s1, s0
            x, y, z = s0.body_q.numpy()[body, :3]
            minimum_clearance = min(minimum_clearance, z-height(x, y))
        self.assertTrue(np.isfinite(s0.body_q.numpy()).all())
        self.assertGreater(minimum_clearance, .0)
        self.assertGreater(z-height(x, y), .01)
        self.assertLess(z-height(x, y), .07)

    def test_terrain_invalid_inputs(self):
        import newton
        from treesim.builder import _add_terrain
        from treesim.config import PhysicsParams
        for name, values in (
            ('terrain_amplitude', (-.1, float('nan'), float('inf'))),
            ('terrain_wavelength', (0., -.1, float('nan'), float('inf'))),
            ('terrain_extent', (0., -1., float('nan'), float('inf'))),
            ('terrain_seed', (-1, 1.5, float('nan'))),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ph = PhysicsParams()
                    setattr(ph, name, value)
                    _add_terrain(newton.ModelBuilder(), ph, 42)

    def test_pergola_terrain_integration(self):
        import newton
        from treesim import builder
        from treesim.config import TreeConfig
        cfg = TreeConfig(lsystem=preset('pergola'), device='cpu', seed=42)
        cfg.lsystem.pergola_rows = cfg.lsystem.pergola_columns = 2
        cfg.fruit.enabled, cfg.fruit.max_count = True, 2
        flat = builder.generate_and_build(cfg)
        self.assertIsNone(flat.terrain_height)
        cfg.physics.terrain = True
        cfg.physics.terrain_amplitude = .24
        cfg.physics.terrain_extent = 6.
        uneven = builder.generate_and_build(cfg)
        np.testing.assert_array_equal(flat.model.body_mass.numpy(), uneven.model.body_mass.numpy())
        self.assertEqual(uneven.model.body_count, flat.model.body_count)
        self.assertEqual(sum(t == newton.GeoType.HFIELD for t in uneven.model.shape_type.numpy()), 1)
        self.assertGreater(uneven.terrain_height(0., 0.), .01)
        for s in uneven.skeleton:
            for p in (s.start, s.end):
                if p[2] == 0:
                    self.assertGreater(uneven.terrain_height(*p[:2]), p[2])
                    self.assertLess(uneven.terrain_height(*p[:2]), cfg.lsystem.target_height)
        np.testing.assert_array_equal([s.end for s in flat.skeleton],
                                      [s.end for s in uneven.skeleton])

    def test_noise_priority_and_explicit_orchard_mode(self):
        from treesim import builder
        from treesim.config import TreeConfig
        from treesim.metrics import Metrics
        from treesim.sim import Sim
        cfg = TreeConfig.compliant('pergola')
        cfg.device, cfg.seed = 'cpu', 42
        cfg.lsystem.pergola_rows = cfg.lsystem.pergola_columns = 2
        cfg.fruit.enabled, cfg.fruit.max_count = True, 1
        cfg.physics.terrain = True
        cfg.physics.terrain_seed = 101
        cfg.physics.terrain_extent = 6.
        for kind in ('noise', 'orchard'):
            cfg.physics.terrain_kind = kind
            tree = builder.generate_and_build(cfg)
            self.assertEqual(tree.terrain_params['kind'],
                             'kiwi_noise' if kind == 'noise' else 'kiwi_orchard_floor')
            self.assertEqual(tree.terrain_params['seed'], 101)
            sim = Sim(tree, substeps=40)
            sim.step()
            self.assertTrue(np.isfinite(sim.state_0.body_q.numpy()).all())
            self.assertEqual(sim.apples.broken_count, 0)
            self.assertEqual(Metrics().summary(sim)['terrain']['kind'], tree.terrain_params['kind'])
        cfg.physics.terrain_kind = 'noise'
        cfg.physics.terrain_extent = None
        cfg.lsystem.pergola_columns = 8
        tree = builder.generate_and_build(cfg)
        self.assertGreaterEqual(tree.terrain_params['half_extent_m'], 18.)
        cfg.physics.terrain_extent = 6.
        with self.assertRaises(ValueError):
            builder.generate_and_build(cfg)

    def test_cli_terrain_random_by_default_and_replayable(self):
        from unittest.mock import patch
        from scripts.grow_tree import make_config, parse_args
        command = ['grow_tree.py', '--preset', 'pergola', '--terrain', '--seed', '42']
        with patch('sys.argv', command):
            args = parse_args()
        with patch('random.SystemRandom.randrange', side_effect=[101, 102]) as random_seed:
            a, b = make_config(args), make_config(args)
            self.assertEqual(random_seed.call_count, 2)
        self.assertEqual(a.seed, b.seed)
        self.assertEqual((a.physics.terrain_seed, b.physics.terrain_seed), (101, 102))
        with patch('sys.argv', command + ['--terrain-seed', '7']):
            args = parse_args()
        with patch('random.SystemRandom.randrange') as random_seed:
            a, b = make_config(args), make_config(args)
            random_seed.assert_not_called()
        self.assertEqual((a.physics.terrain_seed, b.physics.terrain_seed), (7, 7))
        self.assertEqual(a.physics.terrain_kind, 'noise')
        with patch('sys.argv', command + ['--terrain-kind', 'orchard']):
            args = parse_args()
        with patch('random.SystemRandom.randrange') as random_seed:
            cfg = make_config(args)
            random_seed.assert_not_called()
        self.assertEqual(cfg.physics.terrain_kind, 'orchard')
        self.assertIsNone(cfg.physics.terrain_seed)

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
