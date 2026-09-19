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
          <body name="robot" pos="0 0 .5"><freejoint/><geom name="chassis" type="box" size=".1 .1 .1" mass="1"/>
            <body name="arm_link_wr1"><geom name="palm" contype="0" conaffinity="0" type="box" size=".01 .01 .01"/></body>
            <body name="arm_link_fngr"><geom name="finger_pad" contype="0" conaffinity="0" type="box" size=".01 .01 .01"/></body>
            <body name="arm_link_jaw"><geom name="jaw_pad" contype="0" conaffinity="0" type="box" size=".01 .01 .01"/></body>
          </body>
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

    def _set_pad_contacts(self):
        """Inject native-shaped contacts; classification still uses real model body IDs."""
        fruit = self.model.geom('fruit_geom').id
        finger = self.model.geom('finger_pad').id
        jaw = self.model.geom('jaw_pad').id
        geom = np.zeros((32, 2), np.int32)
        worldid = np.zeros(32, np.int32)
        dim = np.ones(32, np.int32)
        address = np.zeros((32, 4), np.int32)
        geom[:8] = [[fruit, finger], [fruit, jaw], [fruit, finger], [fruit, jaw]] * 2
        worldid[:8] = [0, 0, 0, 0, 1, 1, 1, 1]
        self.data.contact.geom.assign(geom)
        self.data.contact.worldid.assign(worldid)
        self.data.contact.dim.assign(dim)
        self.data.contact.efc_address.assign(address)
        self.data.nacon.assign(np.array([8], np.int32))
        self.data.nefc.assign(np.ones(2, np.int32))
        forces = np.zeros((2, 256), np.float32); forces[:, 0] = .3
        self.data.efc.force.assign(forces)

    def test_pad_contacts_require_both_actual_pad_bodies_and_latch_per_world(self):
        with self.wp.ScopedDevice('cuda:0'):
            self._set_pad_contacts()
            for _ in range(20):
                self.task.record()
            self.wp.synchronize()
            np.testing.assert_array_equal(self.task.hand_contact.numpy(), [1, 1])
            np.testing.assert_array_equal(self.task.bilateral_contact.numpy(), [1, 1])
            np.testing.assert_array_equal(self.task.stable_grasp.numpy(), [1, 1])
            np.testing.assert_array_equal(self.task.ever_grasped.numpy(), [1, 1])
            # A masked reset clears all episode-local latches only in world 0.
            mask = self.wp.array(np.array([1, 0], np.uint8), dtype=self.wp.uint8, device='cuda:0')
            self.task.reset(mask)
            self.wp.synchronize()
            np.testing.assert_array_equal(self.task.ever_grasped.numpy(), [0, 1])
            np.testing.assert_array_equal(self.task.stable_grasp.numpy(), [0, 1])

    def test_jaw_load_limit_is_per_pad_and_total_hand_load_is_diagnostic(self):
        with self.wp.ScopedDevice('cuda:0'):
            self._set_pad_contacts()
            forces = np.zeros((2, 256), np.float32)
            forces[:, 0] = 7.0
            self.data.efc.force.assign(forces)
            self.task.record()
            self.wp.synchronize()
            np.testing.assert_allclose(self.task.finger_load.numpy(), [14., 14.])
            np.testing.assert_allclose(self.task.jaw_load.numpy(), [14., 14.])
            np.testing.assert_allclose(self.task.hand_load.numpy(), [28., 28.])
            np.testing.assert_array_equal(self.task.failed.numpy(), [0, 0])

    def test_nonpad_palm_overload_still_fails(self):
        with self.wp.ScopedDevice('cuda:0'):
            fruit = self.model.geom('fruit_geom').id
            palm = self.model.geom('palm').id
            geom = np.zeros((32, 2), np.int32)
            geom[0] = [fruit, palm]
            self.data.contact.geom.assign(geom)
            self.data.contact.worldid.assign(np.zeros(32, np.int32))
            self.data.contact.dim.assign(np.ones(32, np.int32))
            self.data.contact.efc_address.assign(np.zeros((32, 4), np.int32))
            self.data.nacon.assign(np.array([1], np.int32))
            self.data.nefc.assign(np.ones(2, np.int32))
            forces = np.zeros((2, 256), np.float32)
            forces[0, 0] = 16.0
            self.data.efc.force.assign(forces)
            self.task.record()
            self.wp.synchronize()
            np.testing.assert_allclose(self.task.palm_load.numpy(), [16., 0.])
            np.testing.assert_array_equal(self.task.failed.numpy(), [1, 0])

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
        self._settle_and_ground(None)

    def test_graph_two_second_settle_motion_reset_and_intact_ground(self):
        from treesim.kiwi_rl.reward_graph import GRAPH_PROFILE
        self._settle_and_ground(GRAPH_PROFILE)

    def _settle_and_ground(self, profile):
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
            if profile is not None:
                # Rest the long ellipsoid axis horizontally. The legacy upright
                # pose tips over after .5 s; elapsed time is not settled time.
                qpos[0, 2] = .5 + CENTER[2] + WALL/2 + RADII_M[0]
                qpos[0, 3:7] = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
            qpos[1, 0:3] = [.8, 0., .3]
            data.qpos.assign(qpos)
            self.mw.forward(gpu_model, data)
            task = FastHarvestTask(model, data, manifest, task_profile=profile)
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
            if profile is not None:
                self.assertEqual(task.success.numpy()[0], 0)
                self.assertGreater(task.settle_time.numpy()[0], .5)
                # Actual evaluator timer clears on relative motion.
                velocities = data.cvel.numpy()
                velocities[0, model.body('fruit').id, 3] = .1
                data.cvel.assign(velocities)
                task.record()
                self.assertEqual(task.settle_time.numpy()[0], 0.)
                for _ in range(390):
                    self.mw.step(gpu_model, data)
                    task.record()
                self.assertEqual(task.success.numpy()[0], 0)
                for _ in range(25):
                    self.mw.step(gpu_model, data)
                    task.record()
            np.testing.assert_array_equal(task.success.numpy()[0], 1)
            np.testing.assert_array_equal(task.failed.numpy()[1], int(profile is None))
            np.testing.assert_array_equal(task.basket_contact.numpy()[0], 1)


if __name__ == '__main__':
    unittest.main()
