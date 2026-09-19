import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import xml.etree.ElementTree as ET

import numpy as np


class FastSceneXmlTest(unittest.TestCase):
    def test_reattach_moves_liner_off_chassis(self):
        from treesim.kiwi_rl.fast_scene import (
            ARM_CRATE_PAIR_PREFIX, BASKET_SHELL_BODY, prepare_arm_crate_collision,
        )
        xml = '''<mujoco><worldbody><body name="chassis">
          <geom name="basket_floor" type="box" size=".1 .1 .01"/>
          <geom name="basket_liner" type="box" size=".01 .1 .1"/>
          <geom name="basket_visual_1" type="box" size=".02 .02 .02"/>
          <body name="arm_link_wr1"><geom name="hand" type="sphere" size=".05"/></body>
        </body></worldbody></mujoco>'''
        patched = prepare_arm_crate_collision(xml)
        self.assertEqual(prepare_arm_crate_collision(patched), patched)
        root = ET.fromstring(patched)
        shell = root.find(f'.//body[@name="{BASKET_SHELL_BODY}"]')
        self.assertIsNotNone(shell)
        self.assertEqual([g.get('name') for g in shell.findall('geom')],
                         ['basket_floor', 'basket_liner'])
        chassis = root.find('.//body[@name="chassis"]')
        chassis_geoms = [g.get('name') for g in chassis.findall('geom')]
        self.assertIn('basket_visual_1', chassis_geoms)
        self.assertNotIn('basket_floor', chassis_geoms)
        self.assertNotIn('basket_liner', chassis_geoms)
        self.assertIsNotNone(shell.find('inertial'))
        pairs = [p.get('name') for p in root.findall('contact/pair')
                 if (p.get('name') or '').startswith(ARM_CRATE_PAIR_PREFIX)]
        self.assertEqual(len(pairs), 2)


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


    def test_basket_shell_restores_arm_crate_contacts(self):
        import mujoco
        from treesim.kiwi_rl.fast_scene import BASKET_SHELL_BODY, prepare_arm_crate_collision

        xml = '''<mujoco>
        <worldbody>
          <body name="chassis" pos="0 0 0.4">
            <freejoint/>
            <inertial pos="0 0 0" mass="5" diaginertia="0.2 0.3 0.4"/>
            <geom name="basket_floor" type="box" size="0.2 0.15 0.01" pos="0 0 0" contype="1" conaffinity="1"/>
            <geom name="basket_liner" type="box" size="0.01 0.15 0.12" pos="0.19 0 0.12" contype="1" conaffinity="1"/>
            <geom name="basket_visual_1" type="box" size="0.02 0.15 0.02" pos="0.22 0 0.2" contype="0" conaffinity="0"/>
            <body name="arm_link_wr1" pos="0.19 0 0.12">
              <joint name="arm" type="slide" axis="1 0 0" range="-0.5 0.5"/>
              <geom name="hand" type="sphere" size="0.05" mass="0.2" contype="1" conaffinity="1"/>
            </body>
          </body>
        </worldbody>
        <contact>
          <exclude body1="chassis" body2="arm_link_wr1"/>
        </contact>
        </mujoco>'''
        before = mujoco.MjModel.from_xml_string(xml)
        before_data = mujoco.MjData(before)
        mujoco.mj_forward(before, before_data)
        self.assertEqual(_arm_basket_contacts(before, before_data), 0)
        self.assertAlmostEqual(float(before.body_mass[before.body('chassis').id]), 5.0, places=5)

        patched = prepare_arm_crate_collision(xml)
        self.assertEqual(prepare_arm_crate_collision(patched), patched)
        after = mujoco.MjModel.from_xml_string(patched)
        after_data = mujoco.MjData(after)
        mujoco.mj_forward(after, after_data)
        self.assertGreater(_arm_basket_contacts(after, after_data), 0)
        self.assertEqual(after.body(BASKET_SHELL_BODY).parentid, after.body('chassis').id)
        self.assertEqual(int(after.geom_bodyid[after.geom('basket_floor').id]),
                         after.body(BASKET_SHELL_BODY).id)
        self.assertEqual(int(after.geom_bodyid[after.geom('basket_visual_1').id]),
                         after.body('chassis').id)
        self.assertAlmostEqual(float(after.body_mass[after.body('chassis').id]), 5.0, places=5)
        self.assertFalse(any(after.eq_type[i] == mujoco.mjtEq.mjEQ_WELD for i in range(after.neq)))


def _arm_basket_contacts(model, data):
    count = 0
    for i in range(data.ncon):
        names = (model.geom(int(data.contact[i].geom1)).name,
                 model.geom(int(data.contact[i].geom2)).name)
        bodies = (model.body(model.geom_bodyid[int(data.contact[i].geom1)]).name,
                  model.body(model.geom_bodyid[int(data.contact[i].geom2)]).name)
        basket = any(n.startswith(('basket_floor', 'basket_liner')) for n in names)
        arm = any('arm_link' in b for b in bodies)
        if basket and arm:
            count += 1
    return count


if __name__ == '__main__':
    unittest.main()
