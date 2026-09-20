import importlib.util
import unittest
import xml.etree.ElementTree as ET

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo required')
class ControlTest(unittest.TestCase):
    def setUp(self):
        import mujoco
        from treesim.kiwi_rl.control import NativeSpotControl
        root = ET.fromstring('<mujoco><worldbody><body name="chassis" pos="0 0 .5"><freejoint/><inertial pos=".03 .02 .01" mass="5" diaginertia=".2 .3 .4"/><geom type="box" size=".1 .1 .1" contype="0" conaffinity="0"/></body></worldbody><actuator/></mujoco>')
        parent = root.find('worldbody/body')
        names = [f'j{i}' for i in range(19)]
        for i, name in enumerate(names):
            body = ET.SubElement(parent, 'body', name=f'link{i}', pos=f'{.2 + i * .03} 0 0')
            ET.SubElement(body, 'joint', name=name, type='hinge', axis='0 1 0', range='-180 180')
            ET.SubElement(body, 'geom', type='sphere', size='.01', mass='.1', contype='0', conaffinity='0')
            ET.SubElement(root.find('actuator'), 'motor', name=name, joint=name)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        home = {name: .1 if i >= 12 else 0. for i, name in enumerate(names)}
        self.robot = dict(prefix='', chassis='chassis', legs=names[:12], arm=names[12:],
            observation_joints=names[12:13] + names[:12] + names[13:], home_position_rad=home,
            initial_position_rad=home, kp=[60.] * 19, kd=[1.5] * 19,
            controller_torque_limit_Nm=[45.] * 8 + [113.24] * 4 + [15.] * 7,
            knee_lookup=[[-2., 0., 50.], [0., 0., 100.], [2., 0., 60.]])
        self.control = NativeSpotControl(self.model, self.robot)
        self.data.qpos[self.control.qids] = self.control.home
        mujoco.mj_forward(self.model, self.data)

    @unittest.skipUnless(importlib.util.find_spec('mujoco_warp'), 'GPU physics stack required')
    def test_device_controller_matches_native_observations_and_efforts(self):
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from treesim.kiwi_rl.control_warp import WarpSpotControl
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest('CUDA required')
        self.data.qvel[:] = np.linspace(-.2, .3, self.model.nv)
        self.control.targets += np.linspace(-.1, .1, 19).astype(np.float32)
        mujoco.mj_forward(self.model, self.data)
        expected_obs = self.control.observe(self.data, [.2, -.1, .3])
        expected_effort = self.control.apply(self.data, jaw_cap_Nm=.15)
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(self.model)
            gd = mw.put_data(self.model, self.data, nworld=2, nconmax=64, njmax=256)
            mw.forward(gm, gd)
            control = WarpSpotControl(self.model, gd, self.robot)
            control.set_targets(np.tile(self.control.targets, (2, 1)))
            control.set_commands(np.tile([.2, -.1, .3], (2, 1)))
            control.set_jaw_caps([.15, .15])
            actual_obs = control.observe().numpy()
            control.apply()
            np.testing.assert_allclose(actual_obs, np.tile(expected_obs, (2, 1)), rtol=0, atol=2e-6)
            np.testing.assert_allclose(control.effort.numpy(), np.tile(expected_effort, (2, 1)), rtol=1e-6, atol=2e-6)
            with self.assertRaises(ValueError):
                control.set_jaw_caps([float('nan'), .3])

    def test_body_velocity_matches_independent_com_jacobian(self):
        import mujoco
        from treesim.kiwi_rl.control import body_com_velocity
        self.data.qpos[3:7] = [np.cos(.3), 0., 0., np.sin(.3)]
        self.data.qvel[:] = np.linspace(-.4, .5, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        linear, angular = body_com_velocity(self.model, self.data, self.control.chassis)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jp, jr, self.control.chassis)
        rotation = self.data.xmat[self.control.chassis].reshape(3, 3)
        np.testing.assert_allclose(linear, rotation.T @ jp @ self.data.qvel, atol=1e-12)
        np.testing.assert_allclose(angular, rotation.T @ jr @ self.data.qvel, atol=1e-12)

    def test_r84_retains_raw_legacy_units_and_absolute_arm_targets(self):
        import mujoco
        self.control.targets[12] = .3
        self.data.qvel[self.control.dofs[0]] = 3.
        mujoco.mj_forward(self.model, self.data)
        obs = self.control.observe(self.data, [.2, -.1, .3])
        self.assertEqual(obs.shape, (84,))
        np.testing.assert_allclose(obs[9:12], [.2, -.1, .3])
        self.assertAlmostEqual(float(obs[12]), .3)
        self.assertAlmostEqual(float(obs[34]), 0.)
        self.assertAlmostEqual(float(obs[54]), 3.)

    def test_effort_limits_include_knee_speed_and_jaw_cap(self):
        self.control.targets.fill(100.)
        self.data.qvel[self.control.dofs[8:12]] = [14., 15., -15., -20.]
        torque = self.control.apply(self.data, jaw_cap_Nm=.15)
        np.testing.assert_array_equal(torque[8:10], [0., 0.])
        self.assertLessEqual(abs(torque[-1]), .15)
        self.assertTrue(np.all(np.abs(torque) <= self.control.limits))
        self.control.targets.fill(-100.)
        torque = self.control.apply(self.data)
        np.testing.assert_array_equal(torque[10:12], [0., 0.])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_optimizer_updates_reach_applied_motor_torques(self):
        import torch
        actor = torch.nn.Linear(84, 12)
        torch.nn.init.zeros_(actor.weight)
        torch.nn.init.constant_(actor.bias, .1)
        self.control.gait = actor
        self.control.update_gait(self.data, [0., 0., 0.])
        before = self.control.apply(self.data).copy()
        optimizer = torch.optim.SGD(actor.parameters(), lr=.12)
        (-actor(torch.zeros(16, 84)).mean()).backward()
        optimizer.step()
        self.control.update_gait(self.data, [0., 0., 0.])
        after = self.control.apply(self.data)
        self.assertFalse(np.allclose(before[:12], after[:12]))
        np.testing.assert_allclose(self.data.ctrl[self.control.actuators], after)


if __name__ == '__main__':
    unittest.main()
