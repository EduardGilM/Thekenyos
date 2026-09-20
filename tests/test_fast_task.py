import importlib.util
from pathlib import Path
import unittest

import numpy as np


GPU = importlib.util.find_spec('mujoco_warp') is not None and importlib.util.find_spec('warp') is not None


class FastTaskKernelSourceTest(unittest.TestCase):
    def test_contact_kernels_avoid_warp_break_and_constant_mutation(self):
        text = Path(__file__).resolve().parents[1].joinpath('treesim/kiwi_rl/fast_task.py').read_text()
        self.assertNotIn('\n                break', text)
        self.assertIn('fruit_index = int(-1)', text)
        self.assertIn('next_i = int(-1)', text)
        self.assertIn('rows = int(1)', text)
        self.assertIn('jaw_overload', text)
        self.assertIn('goal[world] != 0', text)


@unittest.skipUnless(GPU, 'MJWarp and Warp required')
class FastTaskGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest('CUDA required')
        cls.wp, cls.mw, cls.mujoco = wp, mw, mujoco
        cls.xml = '''<mujoco><option timestep=".005"/><worldbody>
          <geom name="floor" type="plane" size="2 2 .1"/>
          <body name="robot" pos="0 0 .5"><freejoint/><geom name="jaw" type="box" size=".1 .1 .1" mass="1"/></body>
          <body name="canopy" pos="0 0 1"><site name="anchor"/></body>
          <body name="fruit" pos="0 0 .8"><freejoint/><geom name="fruit_geom" type="sphere" size=".05" mass=".1"/><site name="fruit_site"/></body>
        </worldbody><equality><connect name="fruit_connect" site1="fruit_site" site2="anchor"/></equality></mujoco>'''

    def setUp(self):
        from treesim.kiwi_rl.fast_task import (
            CONTAINMENT_JITTER_TOL_M, CONTAINMENT_TOL_M,
            SETTLE_TIME_EPSILON_STEPS, FastHarvestTask,
        )
        self.assertEqual(CONTAINMENT_TOL_M, 0.012)
        self.assertEqual(CONTAINMENT_JITTER_TOL_M, 0.020)
        self.assertEqual(SETTLE_TIME_EPSILON_STEPS, 0.5)
        self.model = self.mujoco.MjModel.from_xml_string(self.xml)
        native = self.mujoco.MjData(self.model)
        self.mujoco.mj_forward(self.model, native)
        self.manifest = {'robot': {'chassis': 'robot'},
                         'fruits': [{'body': 'fruit', 'geom': 'fruit_geom', 'equality': 'fruit_connect'}]}
        with self.wp.ScopedDevice('cuda:0'):
            self.gpu_model = self.mw.put_model(self.model)
            self.data = self.mw.put_data(self.model, native, nworld=2, nconmax=32, njmax=256)
            self.mw.forward(self.gpu_model, self.data)
            self.task = FastHarvestTask(self.model, self.data, self.manifest)

    def _set_equality_forces(self, values):
        self.data.nacon.assign(np.array([0], np.int32))
        self.data.nefc.assign(np.array([1, 1], np.int32))
        self.data.efc.type.assign(np.zeros((2, 256), np.int32))
        self.data.efc.id.assign(np.zeros((2, 256), np.int32))
        forces = np.zeros((2, 256), np.float32)
        forces[:, 0] = values
        self.data.efc.force.assign(forces)

    def test_gravity_scale_force_does_not_release(self):
        with self.wp.ScopedDevice('cuda:0'):
            self._set_equality_forces([1.03, 1.03])
            self.task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(self.task.detached.numpy(), [0, 0])
            np.testing.assert_array_equal(self.data.eq_active.numpy(), [[True], [True]])

    def test_forced_release_and_masked_reset_are_world_local(self):
        with self.wp.ScopedDevice('cuda:0'):
            self._set_equality_forces([9., 1.])
            self.task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(self.task.detached.numpy(), [1, 0])
            np.testing.assert_array_equal(self.data.eq_active.numpy(), [[False], [True]])
            mask = self.wp.array(np.array([1, 0], np.uint8), dtype=self.wp.uint8, device='cuda:0')
            self.task.reset(mask)
            self.wp.synchronize()
            np.testing.assert_array_equal(self.task.detached.numpy(), [0, 0])
            np.testing.assert_array_equal(self.data.eq_active.numpy(), [[True], [True]])

    def test_detached_fruit_settles_on_basket_and_ground_drop_fails(self):
        from treesim.basket import CENTER, SIZE, WALL
        from treesim.native_kiwi import RADII_M
        from treesim.kiwi_rl.fast_task import FastHarvestTask

        xml = f'''<mujoco><option timestep=".005" gravity="0 0 -9.81"/><worldbody>
          <geom name="floor" type="plane" size="2 2 .1"/>
          <body name="robot" pos="0 0 .5">
            <geom name="chassis" type="box" size=".1 .1 .1" mass="1"/>
            <geom name="basket_floor" pos="{CENTER[0]} {CENTER[1]} {CENTER[2]}" type="box"
                  size="{SIZE[0]/2} {SIZE[1]/2} {WALL/2}"/>
          </body>
          <body name="canopy" pos="0 0 1"><site name="anchor"/></body>
          <body name="fruit" pos="{CENTER[0]} {CENTER[1]} {0.5 + CENTER[2] + WALL/2 + RADII_M[2]}">
            <freejoint/><geom name="fruit_geom" type="ellipsoid" size="{RADII_M[0]} {RADII_M[1]} {RADII_M[2]}" mass=".1"/>
            <site name="fruit_site"/>
          </body>
        </worldbody><equality><connect name="fruit_connect" site1="fruit_site" site2="anchor"/></equality></mujoco>'''
        model = self.mujoco.MjModel.from_xml_string(xml)
        native = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, native)
        manifest = {'robot': {'chassis': 'robot'},
                    'fruits': [{'body': 'fruit', 'geom': 'fruit_geom', 'equality': 'fruit_connect'}]}
        with self.wp.ScopedDevice('cuda:0'):
            gpu_model = self.mw.put_model(model)
            data = self.mw.put_data(model, native, nworld=2, nconmax=64, njmax=256)
            qpos = data.qpos.numpy()
            qpos[0, 0:3] = [CENTER[0], CENTER[1], .5 + CENTER[2] + WALL/2 + RADII_M[2]]
            qpos[1, 0:3] = [.8, 0., .3]
            data.qpos.assign(qpos)
            self.mw.forward(gpu_model, data)
            task = FastHarvestTask(model, data, manifest)
            data.nacon.assign(np.array([0], np.int32))
            data.nefc.assign(np.array([1, 1], np.int32))
            data.efc.type.assign(np.zeros((2, 256), np.int32))
            data.efc.id.assign(np.zeros((2, 256), np.int32))
            force = np.zeros((2, 256), np.float32)
            force[:, 0] = 9.
            data.efc.force.assign(force)
            task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(task.detached.numpy(), [1, 1])
            for _ in range(130):
                self.mw.step(gpu_model, data)
                task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(task.success.numpy()[0], 1)
            np.testing.assert_array_equal(task.failed.numpy()[1], 1)
            np.testing.assert_array_equal(task.basket_contact.numpy()[0], 1)

    def test_second_fruit_stays_free_after_first_deposit(self):
        from treesim.basket import CENTER, SIZE, WALL
        from treesim.native_kiwi import RADII_M
        from treesim.kiwi_rl.fast_task import FastHarvestTask, MAX_FRUITS

        z_in = 0.5 + CENTER[2] + WALL / 2 + RADII_M[2]
        xml = f'''<mujoco><option timestep=".005" gravity="0 0 -9.81"/><worldbody>
          <geom name="floor" type="plane" size="2 2 .1"/>
          <body name="robot" pos="0 0 .5">
            <geom name="chassis" type="box" size=".1 .1 .1" mass="1"/>
            <geom name="basket_floor" pos="{CENTER[0]} {CENTER[1]} {CENTER[2]}" type="box"
                  size="{SIZE[0]/2} {SIZE[1]/2} {WALL/2}"/>
          </body>
          <body name="canopy" pos="0 0 1.2"><site name="anchor0"/><site name="anchor1" pos=".2 0 0"/></body>
          <body name="fruit_0" pos="{CENTER[0]} {CENTER[1]} {z_in}">
            <freejoint/><geom name="fruit_0_geom" type="ellipsoid" size="{RADII_M[0]} {RADII_M[1]} {RADII_M[2]}" mass=".1"/>
            <site name="fruit_0_site"/>
          </body>
          <body name="fruit_1" pos="0.2 0 1.0">
            <freejoint/><geom name="fruit_1_geom" type="ellipsoid" size="{RADII_M[0]} {RADII_M[1]} {RADII_M[2]}" mass=".1"/>
            <site name="fruit_1_site"/>
          </body>
        </worldbody><equality>
          <connect name="fruit_0_connect" site1="fruit_0_site" site2="anchor0"/>
          <connect name="fruit_1_connect" site1="fruit_1_site" site2="anchor1"/>
        </equality></mujoco>'''
        model = self.mujoco.MjModel.from_xml_string(xml)
        native = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, native)
        fruit0 = int(model.body('fruit_0').id)
        fruit1 = int(model.body('fruit_1').id)
        hanging = native.xpos[fruit1].copy()
        manifest = {'robot': {'chassis': 'robot'},
                    'fruits': [
                        {'body': 'fruit_0', 'geom': 'fruit_0_geom', 'equality': 'fruit_0_connect'},
                        {'body': 'fruit_1', 'geom': 'fruit_1_geom', 'equality': 'fruit_1_connect'},
                    ]}
        with self.wp.ScopedDevice('cuda:0'):
            gpu_model = self.mw.put_model(model)
            data = self.mw.put_data(model, native, nworld=1, nconmax=64, njmax=256)
            qpos = data.qpos.numpy()
            qadr0 = int(model.jnt_qposadr[model.body_jntadr[fruit0]])
            qpos[0, qadr0:qadr0 + 3] = [CENTER[0], CENTER[1], z_in]
            data.qpos.assign(qpos)
            self.mw.forward(gpu_model, data)
            task = FastHarvestTask(model, data, manifest)
            self.assertEqual(task.fruit_count, 2)
            self.assertEqual(MAX_FRUITS, 5)
            task.continue_after_success.assign(np.array([1], np.uint8))
            task.required_harvests.assign(np.array([2], np.int32))
            data.nacon.assign(np.array([0], np.int32))
            data.nefc.assign(np.array([1], np.int32))
            data.efc.type.assign(np.zeros((1, 256), np.int32))
            data.efc.id.assign(np.zeros((1, 256), np.int32))
            force = np.zeros((1, 256), np.float32)
            force[:, 0] = 9.
            data.efc.force.assign(force)
            task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(task.detached.numpy(), [1])
            for _ in range(130):
                self.mw.step(gpu_model, data)
                task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(task.success.numpy(), [0])
            np.testing.assert_array_equal(task.harvested.numpy(), [1])
            np.testing.assert_array_equal(task.active_fruit.numpy(), [1])
            np.testing.assert_array_equal(task.deposited.numpy()[0, :2], [1, 0])
            eq = data.eq_active.numpy()[0]
            self.assertFalse(bool(eq[task.equality_id]))
            self.assertTrue(bool(eq[int(model.equality('fruit_1_connect').id)]))
            xpos = data.xpos.numpy()[0]
            np.testing.assert_allclose(xpos[fruit1], hanging, atol=0.08)
            self.assertGreater(float(xpos[fruit0, 2]), 0.35)


if __name__ == '__main__':
    unittest.main()
