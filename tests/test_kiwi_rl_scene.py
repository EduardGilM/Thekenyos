import importlib.util
import unittest

import numpy as np


XML = '''<mujoco>
<option gravity="0 0 0" timestep=".00002" integrator="Euler" solver="CG" jacobian="sparse"/>
<asset><mesh name="offset" vertex=".17 0 0 .23 0 0 .20 .04 0 .20 0 .06" face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset>
<worldbody>
<geom name="jaw" type="mesh" mesh="offset"/>
<flexcomp name="fruit" type="grid" dim="3" count="3 3 3" spacing=".015 .015 .015" pos=".2 .02 .03" mass=".1" radius=".0003">
<elasticity young="1570000" poisson=".4" damping=".00001"/>
<contact selfcollide="none" internal="false"/>
</flexcomp>
</worldbody></mujoco>'''


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo required')
class MeshFrameTest(unittest.TestCase):
    def test_world_geometry_and_mass_are_preserved(self):
        import mujoco
        from treesim.kiwi_rl.scene import normalize_collision_meshes
        result, report = normalize_collision_meshes(XML)
        self.assertEqual(report['changed_geoms'], ['jaw'])
        self.assertLess(report['max_surface_error_m'], 1e-6)
        original, normalized = mujoco.MjModel.from_xml_string(XML), mujoco.MjModel.from_xml_string(result)
        np.testing.assert_allclose(original.body_mass, normalized.body_mass)
        index = normalized.geom_dataid[normalized.geom('jaw').id]
        self.assertLess(np.linalg.norm(normalized.mesh_pos[index]), 1e-8)

    def test_normalization_is_idempotent(self):
        from treesim.kiwi_rl.scene import normalize_collision_meshes
        first, _ = normalize_collision_meshes(XML)
        second, report = normalize_collision_meshes(first)
        self.assertEqual(first, second)
        self.assertEqual(report['changed_geoms'], [])

    @unittest.skipUnless(importlib.util.find_spec('mujoco_warp'), 'MJWarp required')
    def test_normalized_mesh_has_real_gpu_flex_contacts(self):
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from treesim.kiwi_rl.scene import normalize_collision_meshes
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest('CUDA required')
        xml, _ = normalize_collision_meshes(XML)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertGreater(d.ncon, 0)
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(m)
            gd = mw.put_data(m, d, nworld=1, nconmax=1024, njmax=8192)
            mw.fwd_position(gm, gd)
            n = int(gd.nacon.numpy()[0])
            geom, flex = gd.contact.geom.numpy()[:n], gd.contact.flex.numpy()[:n]
            self.assertTrue(np.any((geom < 0) & (flex >= 0)))
            self.assertFalse(gd.overflow.numpy().any())


if __name__ == '__main__':
    unittest.main()
