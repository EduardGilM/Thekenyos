import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
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
    def test_multiple_deformable_fruit_attach_to_rotated_canopy(self):
        import mujoco
        from treesim.kiwi_rl.scene import assemble_deformable_scene
        xml = '''<mujoco><worldbody>
        <body name="canopy" pos="0 0 1.6" quat=".7071067811865476 .7071067811865476 0 0">
        <geom type="capsule" fromto="0 0 0 .3 0 0" size=".005"/>
        <site name="anchor0" pos=".1 0 0"/><site name="anchor1" pos=".2 0 0"/>
        </body><body name="robot" pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
        <body name="arm_link_wr1"/><body name="arm_link_fngr"/><body name="arm_link_jaw"/>
        </worldbody><keyframe><key name="home"/></keyframe></mujoco>'''
        manifest = dict(schema='training-base-scene/v1', model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            robot=dict(prefix='', initial_position_rad={}),
            anchors=[dict(site=f'anchor{i}', parent_body='canopy', world_position_m=[.1 * (i + 1), 0., 1.6]) for i in range(2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'base.xml').write_text(xml)
            (path / 'manifest.json').write_text(json.dumps(manifest))
            assembled, result = assemble_deformable_scene(path, count=5)
            model = mujoco.MjModel.from_xml_string(assembled)
            self.assertEqual(model.nflex, 2)
            self.assertEqual(model.neq, 2)
            self.assertFalse(result['training_ready'])
            self.assertEqual(result['material']['stem_segments'], 4)
            self.assertEqual(len(result['attachments']), 2)
            for attachment in result['attachments']:
                root = model.body(attachment['root_body']).id
                self.assertEqual(model.body_parentid[root], model.body('canopy').id)
                equality = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, attachment['equality'])
                self.assertGreaterEqual(equality, 0)
                self.assertEqual(model.eq_type[equality], mujoco.mjtEq.mjEQ_CONNECT)
            coarse_xml, coarse_manifest = assemble_deformable_scene(path, count=5, stem_segments=1)
            coarse = mujoco.MjModel.from_xml_string(coarse_xml)
            self.assertEqual(coarse_manifest['material']['stem_segments'], 1)
            self.assertEqual(model.nv - coarse.nv, 24)  # twelve fewer stem DOFs per fruit
            original_mass = sum(model.body_mass[i] for i in range(model.nbody) if '_stem_' in model.body(i).name)
            coarse_mass = sum(coarse.body_mass[i] for i in range(coarse.nbody) if '_stem_' in coarse.body(i).name)
            self.assertAlmostEqual(original_mass, coarse_mass, places=12)
            (path / 'base.xml').write_text(xml + ' ')
            with self.assertRaises(ValueError):
                assemble_deformable_scene(path, count=5)

    def test_stem_segment_count_is_explicit_and_bounded(self):
        import mujoco
        from treesim.kiwi_rl.scene import assemble_deformable_scene
        xml = '''<mujoco><worldbody><body name="canopy"><site name="anchor0"/></body>
        <body name="arm_link_wr1"/><body name="arm_link_fngr"/><body name="arm_link_jaw"/>
        </worldbody><keyframe><key name="home"/></keyframe></mujoco>'''
        manifest = dict(schema='training-base-scene/v1', model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            robot=dict(prefix='', initial_position_rad={}),
            anchors=[dict(site='anchor0', parent_body='canopy', world_position_m=[0., 0., 0.])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'base.xml').write_text(xml)
            (path / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                assemble_deformable_scene(path, count=5, stem_segments=0)

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
