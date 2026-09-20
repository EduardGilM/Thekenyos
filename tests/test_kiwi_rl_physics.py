import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('warp') and importlib.util.find_spec('mujoco_warp'), 'GPU physics stack required')
class MonitorTest(unittest.TestCase):
    def setUp(self):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest('CUDA required')
        from treesim.kiwi_rl.physics import DeformableMonitor
        self.wp = wp
        model = SimpleNamespace(nflex=1, flex_dim=[3], flex_elemdataadr=[0], flex_elemnum=[1],
                                flex_elem=np.array([0, 1, 2, 3]), flex_vertadr=[0])
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        reference = SimpleNamespace(flexvert_xpos=self.vertices)
        with wp.ScopedDevice('cuda:0'):
            self.data = SimpleNamespace(qpos=wp.zeros((2, 4)), qvel=wp.zeros((2, 4)),
                overflow=wp.zeros(2, dtype=int), solver_niter=wp.zeros(2, dtype=int),
                flexvert_xpos=wp.array(np.stack([self.vertices, self.vertices]), dtype=wp.vec3))
            self.monitor = DeformableMonitor(model, reference, self.data)

    def test_contact_geom_takes_precedence_over_stale_flex_ids(self):
        from treesim.kiwi_rl.physics import contact_flex_ids
        np.testing.assert_array_equal(contact_flex_ids([5, 7], [-1, 0]), [-1, -1])
        np.testing.assert_array_equal(contact_flex_ids([5, -1], [-1, 0]), [-1, 0])
        np.testing.assert_array_equal(contact_flex_ids([-1, -1], [0, 1]), [0, 1])

    def test_substep_contact_forces_and_first_ground_event(self):
        from treesim.kiwi_rl.physics import FlexContactObserver
        wp = self.wp
        model = SimpleNamespace(nflex=1, ngeom=2, geom_type=[7, 0], opt=SimpleNamespace(cone=0),
            geom=lambda g: SimpleNamespace(name=('arm_link_jaw_collision_0', 'floor')[g]))
        with wp.ScopedDevice('cuda:0'):
            contact = SimpleNamespace(geom=wp.array([[0, -1], [1, -1], [0, 1]], dtype=wp.vec2i),
                flex=wp.array([[-1, 0], [-1, 0], [-1, 0]], dtype=wp.vec2i),
                worldid=wp.array([0, 1, 0], dtype=int), dim=wp.array([3, 1, 1], dtype=int),
                efc_address=wp.array([[0, 1, 2, 3], [0, -1, -1, -1], [0, -1, -1, -1]], dtype=int))
            data = SimpleNamespace(qpos=wp.zeros((2, 4)), naconmax=3,
                nacon=wp.array([3], dtype=int), contact=contact,
                efc=SimpleNamespace(force=wp.array([[1., 2., 3., 4.], [2., 0., 0., 0.]], dtype=float)),
                nefc=wp.array([4, 1], dtype=int))
            observer = FlexContactObserver(model, data)
            observer.record()
            result = observer.check()
            self.assertEqual(result['jaw_palm_load_N'][0][0][0], 10.)
            self.assertEqual(result['first_ground_step'], [[-1], [1]])
            data.nacon.zero_()
            observer.record()
            result = observer.check()
            self.assertEqual(result['jaw_palm_load_N'][0][0][0], 0.)
            self.assertEqual(result['peak_jaw_palm_load_N'][0][0][0], 10.)
            self.assertEqual(result['first_ground_step'], [[-1], [1]])

    def test_overflow_is_latched_and_world_local(self):
        self.data.overflow.assign(np.array([512, 0], dtype=np.int32))
        self.monitor.record()
        self.data.overflow.zero_()
        self.monitor.record()
        self.assertEqual(self.monitor.report()['flags'], [512, 0])
        with self.assertRaises(RuntimeError):
            self.monitor.check()
        self.monitor.reset()
        self.monitor.record()
        self.assertEqual(self.monitor.check()['flags'], [0, 0])

    def test_inversion_and_nonfinite_state_are_latched(self):
        positions = np.stack([self.vertices, self.vertices])
        positions[1, [1, 2]] = positions[1, [2, 1]]
        self.data.flexvert_xpos.assign(positions)
        q = np.zeros((2, 4), dtype=np.float32)
        q[0, 0] = np.nan
        self.data.qpos.assign(q)
        self.monitor.record()
        report = self.monitor.report()
        self.assertIn('NONFINITE_STATE', report['failures'][0])
        self.assertIn('INVALID_ELEMENT', report['failures'][1])
        self.assertLess(report['minimum_volume_ratio'][1], 0.)


if __name__ == '__main__':
    unittest.main()
