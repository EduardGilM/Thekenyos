import importlib.util
import unittest

import numpy as np


GPU = importlib.util.find_spec('mujoco_warp') is not None and importlib.util.find_spec('warp') is not None


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
        from treesim.kiwi_rl.fast_task import FastHarvestTask
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


if __name__ == '__main__':
    unittest.main()
