import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np


XML = '''<mujoco>
<option timestep=".005"/>
<worldbody>
  <body name="canopy" pos="0 0 1.6">
    <geom name="support" type="box" size=".4 .4 .01" contype="1" conaffinity="1"/>
    <site name="anchor0" pos=".1 0 0"/>
    <site name="anchor1" pos="-.1 0 0"/>
  </body>
  <body name="robot" pos="0 0 .3"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
</worldbody>
<keyframe><key name="home" qpos="0 0 .3 1 0 0 0"/></keyframe>
</mujoco>'''


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo required')
class FastSceneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / 'base'
        self.base.mkdir()
        (self.base / 'base.xml').write_text(XML)
        manifest = {
            'schema': 'training-base-scene/v1',
            'model_sha256': hashlib.sha256(XML.encode()).hexdigest(),
            'robot': {'prefix': '', 'initial_position_rad': {}},
            'anchors': [
                {'site': 'anchor0', 'parent_body': 'canopy', 'world_position_m': [.1, 0, 1.6]},
                {'site': 'anchor1', 'parent_body': 'canopy', 'world_position_m': [-.1, 0, 1.6]},
            ],
        }
        (self.base / 'manifest.json').write_text(json.dumps(manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_is_rigid_free_fruit_with_breakable_connects(self):
        import mujoco
        from treesim.kiwi_rl.fast_scene import assemble_fast_scene

        xml, manifest = assemble_fast_scene(self.base, timestep_s=.005)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(manifest['schema'], 'fast-training-scene/v1')
        self.assertFalse(manifest['training_ready'])
        self.assertFalse(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.assertEqual(len(manifest['fruits']), 2)
        self.assertEqual(model.nflex, 0)
        self.assertEqual(model.nq, 7 + 2 * 7)
        for fruit in manifest['fruits']:
            self.assertAlmostEqual(model.body_mass[model.body(fruit['body']).id], .105, places=8)
            self.assertEqual(model.joint(f'{fruit["body"]}_free').type, mujoco.mjtJoint.mjJNT_FREE)
            eq = model.equality(fruit['equality']).id
            self.assertEqual(model.eq_type[eq], mujoco.mjtEq.mjEQ_CONNECT)
            self.assertTrue(model.eq_active0[eq])
        self.assertEqual(manifest['numerical_profile']['frequency_hz'], 200.)
        self.assertIn('segmented stem springs', manifest['approximation']['omitted'])

    def test_500_hz_and_loader_verify_home(self):
        from treesim.kiwi_rl.fast_scene import assemble_fast_scene, load_fast_scene

        xml, manifest = assemble_fast_scene(self.base, fruit_count=1, timestep_s=.002,
                                            visual_stalk=False)
        out = Path(self.tmp.name) / 'fast'
        out.mkdir()
        (out / 'scene.xml').write_text(xml)
        (out / 'manifest.json').write_text(json.dumps(manifest))
        model, data, loaded = load_fast_scene(out)
        self.assertEqual(model.opt.timestep, .002)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertEqual(len(loaded['fruits']), 1)

        (out / 'scene.xml').write_text(xml + ' ')
        with self.assertRaises(ValueError):
            load_fast_scene(out)

    def test_rejects_flex_and_unsupported_timestep(self):
        from treesim.kiwi_rl.fast_scene import assemble_fast_scene

        with self.assertRaises(ValueError):
            assemble_fast_scene(self.base, timestep_s=.001)
        bad = self.base / 'base.xml'
        bad.write_text(XML.replace('</worldbody>', '<flexcomp name="bad" type="grid" count="2 2 2" spacing=".1 .1 .1"/></worldbody>'))
        manifest = json.loads((self.base / 'manifest.json').read_text())
        manifest['model_sha256'] = hashlib.sha256(bad.read_bytes()).hexdigest()
        (self.base / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            assemble_fast_scene(self.base)


if __name__ == '__main__':
    unittest.main()
