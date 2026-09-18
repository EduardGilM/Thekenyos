"""Payload geometry/inertia inputs; no GPU required."""
import unittest
import numpy as np
from treesim.basket import payload_layout, CENTER, SIZE, WALL


class BasketTest(unittest.TestCase):
    def test_payload_mass_bounds_and_reproducibility(self):
        for mass in (0., .01, 3., 6.):
            pos, weights, radii = payload_layout(mass, 42)
            np.testing.assert_array_equal(pos, payload_layout(mass, 42)[0])
            self.assertAlmostEqual(float(weights.sum()), mass)
            if not mass:
                continue
            self.assertTrue(np.all(np.abs(pos[:, :2]-CENTER[:2])+radii[:2] < SIZE[:2]/2-WALL))
            self.assertTrue(np.all(pos[:, 2]-radii[2] > CENTER[2]+WALL/2))
            self.assertTrue(np.all(pos[:, 2]+radii[2] < CENTER[2]+SIZE[2]))
            delta = (pos[:, None]-pos[None, :])/(2*radii)
            separation = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(separation, np.inf)
            self.assertGreater(float(separation.min()), 1.)
        a, weights, _ = payload_layout(6., 0)
        b, _, _ = payload_layout(6., 1)
        self.assertFalse(np.allclose(np.average(a, axis=0, weights=weights), np.average(b, axis=0, weights=weights)))
        for invalid in (-1., 6.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                payload_layout(invalid, 0)

    def test_newton_fruit_are_independent(self):
        try:
            import newton
            import warp as wp
        except ImportError:
            self.skipTest('Newton integration check runs in the simulation environment')
        from treesim.basket import add_basket
        b = newton.ModelBuilder()
        chassis = b.add_body(mass=2., inertia=wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.))
        data = add_basket(b, chassis, 1.2, 6., 42)
        self.assertAlmostEqual(float(b.body_mass[chassis]), 3.2, places=5)
        self.assertAlmostEqual(sum(float(b.body_mass[i]) for i in data['fruit_bodies']), 6., places=5)
        self.assertEqual(len(set(data['fruit_bodies'])), 60)
        for body in data['fruit_bodies']:
            joint = b.joint_child.index(body)
            self.assertEqual(b.joint_type[joint], newton.JointType.FREE)
            self.assertTrue(np.all(np.linalg.eigvalsh(np.array(b.body_inertia[body]).reshape(3,3)) > 0))
        # Chassis carries only basket inertia; payload transfers weight by contact.
        self.assertEqual(b.body_count, 61)

    def test_spill_penalty_is_once_per_fruit(self):
        from treesim.basket import SpillTracker
        tracker = SpillTracker({'fruit_bodies': [1, 2]})
        poses = np.zeros((3, 7)); poses[:, 6] = 1
        poses[1, :3] = CENTER + [0, 0, .1]
        poses[2, :3] = CENTER + [.02, .02, .1]
        self.assertEqual(tracker.update(poses, 0), 0)
        poses[1, 0] += 1
        self.assertEqual(tracker.update(poses, 0), -1)
        self.assertEqual(tracker.update(poses, 0), 0)
        poses[2, 2] -= 1
        self.assertEqual(tracker.update(poses, 0), -1)
        self.assertEqual(tracker.total_penalty, -2)
