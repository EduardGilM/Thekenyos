import importlib.util
import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo required')
class SpotCameraTest(unittest.TestCase):
    def test_installs_nominal_frame_and_intrinsics(self):
        import mujoco
        from treesim.kiwi_rl.spot_camera import install_spot_gripper_camera

        root = ET.fromstring('''<mujoco><worldbody><body name="chassis">
          <body name="wrist"><camera name="body_camera" pos="0 0 0"/>
          <camera name="hand_camera" pos="0 0 0"/></body></body></worldbody></mujoco>''')
        metadata = install_spot_gripper_camera(root, {'wrist': 'wrist'})
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.assertEqual(model.ncam, 1)
        camera = model.camera('hand_camera').id
        np.testing.assert_allclose(model.cam_pos[camera], [.13806, .0202, .02452], atol=1e-12)
        expected_forward = np.array([.987688858, 0., .1564312])
        actual_forward = -model.cam_mat0[camera].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(actual_forward, expected_forward, atol=2e-5)
        self.assertEqual(model.cam_bodyid[camera], model.body('wrist').id)
        focal = model.cam_intrinsic[camera, :2]
        self.assertAlmostEqual(2 * math.degrees(math.atan(1 / (2 * focal[0]))), 60.2, places=4)
        self.assertAlmostEqual(2 * math.degrees(math.atan(1 / (2 * focal[1]))), 46.4, places=4)
        self.assertEqual(tuple(model.cam_resolution[camera]), (640, 480))
        self.assertEqual(metadata[0]['source_commit'], '27f8033c5064d32f049a17accb71cd1091422878')

    def test_parent_motion_moves_camera(self):
        import mujoco
        from treesim.kiwi_rl.spot_camera import install_spot_gripper_camera

        root = ET.fromstring('<mujoco><worldbody><body name="wrist" pos="1 2 3"/></worldbody></mujoco>')
        install_spot_gripper_camera(root, {'wrist': 'wrist'})
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.cam_xpos[0], [1.13806, 2.0202, 3.02452], atol=1e-12)
        model.body_quat[model.body('wrist').id] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.cam_xpos[0], [.9798, 2.13806, 3.02452], atol=1e-12)
        np.testing.assert_allclose(-data.cam_xmat[0].reshape(3, 3)[:, 2],
                                  [0., .987688858, .1564312], atol=2e-5)

    def test_aperture_splits_crossing_triangles_without_removing_housing(self):
        from treesim.kiwi_rl.spot_camera import _open_aperture, _urdf_rpy_matrix, OPTICAL_RPY_RAD, POSITION_M
        optical = _urdf_rpy_matrix(OPTICAL_RPY_RAD)
        local = np.array([[-.02, -.02, .004], [.02, -.02, .004], [0., .03, .004]])
        vertices = local @ optical.T + POSITION_M
        out, faces, _ = _open_aperture(vertices, np.array([[0, 1, 2]]), POSITION_M,
                                      np.zeros(3), [1, 0, 0, 0], np.ones(3))
        area = sum(np.linalg.norm(np.cross(out[f[1]]-out[f[0]], out[f[2]]-out[f[0]]))/2 for f in faces)
        expected_hole = 16 * .008**2 * np.sin(2*np.pi/16) / 2
        self.assertAlmostEqual(.001-area, expected_hole, places=10)
        self.assertGreater(len(faces), 1)


if __name__ == '__main__':
    unittest.main()
