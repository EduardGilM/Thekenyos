import tempfile
import unittest
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from treesim.kiwi_rl.spot_cameras import (
    COLOR_FOV_DEG, DEPTH_FOV_DEG, NOMINAL_OPTICAL_FRAMES,
    attach_mujoco_cameras, load_relic_gripper_cameras,
    mujoco_gripper_cameras, optical_to_mujoco, parse_urdf_frames,
    require_gripper_camera_names, rpy_matrix,
)


URDF = '''
<link name="arm_link_wr1"/>
<!--Hand camera.-->
<!-- <frame link="arm_link_wr1" name="hand_camera_body" rpy="0.0 1.41372 0.0" xyz="0.13495 0.0 0.00799"/>
<frame link="arm_link_wr1" name="hand_color_sensor" rpy="-1.41372 0.0 -1.5708" xyz="0.13806 0.0202 0.02452"/>
<frame link="arm_link_wr1" name="hand_depth_sensor" rpy="0.0 1.41372 0.0" xyz="0.13495 0.0 0.00799"/> -->
'''


class SpotCameraTest(unittest.TestCase):
    def test_urdf_comments_match_nominal_gripper_frames(self):
        frames = parse_urdf_frames(URDF)
        for name, expected in NOMINAL_OPTICAL_FRAMES.items():
            self.assertEqual(frames[name]['urdf_link'], expected['urdf_link'])
            np.testing.assert_allclose(frames[name]['xyz'], expected['xyz'])
            np.testing.assert_allclose(frames[name]['rpy'], expected['rpy'])

    def test_optical_to_mujoco_keeps_look_axis_and_flips_image_up(self):
        rpy, xyz = NOMINAL_OPTICAL_FRAMES['hand_color_sensor']['rpy'], NOMINAL_OPTICAL_FRAMES['hand_color_sensor']['xyz']
        pose = optical_to_mujoco(rpy, xyz)
        optical = rpy_matrix(rpy)
        mujoco = np.asarray(pose['rotation'])
        np.testing.assert_allclose(pose['position_m'], xyz)
        np.testing.assert_allclose(mujoco[:, 0], optical[:, 0], atol=1e-12)
        np.testing.assert_allclose(mujoco[:, 1], -optical[:, 1], atol=1e-12)
        np.testing.assert_allclose(mujoco[:, 2], -optical[:, 2], atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(mujoco), 1.0, atol=1e-12)

    def test_export_uses_published_fov_and_rejects_a_mast(self):
        cameras = mujoco_gripper_cameras()
        self.assertEqual([c['name'] for c in cameras], ['hand_color_sensor', 'hand_depth_sensor'])
        self.assertEqual(cameras[0]['fovy_degrees'], COLOR_FOV_DEG[1])
        self.assertEqual(cameras[1]['fovy_degrees'], DEPTH_FOV_DEG[1])
        self.assertNotIn('body_camera', [c['name'] for c in cameras])
        self.assertTrue(all(max(abs(x) for x in c['position_m']) < 0.2 for c in cameras))
        wrist = Element('body', name='spot_with_arm_arm_link_wr1')
        bodies = {'spot_with_arm_arm_link_wr1': wrist, 'spot_with_arm_body': Element('body', name='spot_with_arm_body')}
        robot = dict(prefix='spot_with_arm_', wrist='spot_with_arm_arm_link_wr1', chassis='spot_with_arm_body')
        attached = attach_mujoco_cameras(bodies, robot, cameras)
        xml = tostring(wrist, encoding='unicode')
        self.assertIn('hand_color_sensor', xml)
        self.assertIn('hand_depth_sensor', xml)
        self.assertNotIn('body_camera', xml)
        self.assertEqual(attached[0]['body'], robot['wrist'])
        mast = dict(cameras[0], name='body_camera', urdf_link='body', position_m=[0.45, 0.0, 0.55])
        with self.assertRaises(ValueError):
            attach_mujoco_cameras(bodies, robot, [mast])
        with self.assertRaises(RuntimeError):
            require_gripper_camera_names(('hand_camera', 'body_camera'))
        require_gripper_camera_names(('hand_color_sensor', 'hand_depth_sensor'))

    def test_pinned_relic_urdf_is_checked_when_present(self):
        relic = Path('/workspace/assets/relic')
        if not (relic / 'source/relic/relic/assets/spot/spot_with_arm.urdf').is_file():
            relic = Path('/tmp/relic')
        if not (relic / 'source/relic/relic/assets/spot/spot_with_arm.urdf').is_file():
            self.skipTest('external RELIC checkout is not available')
        cameras = load_relic_gripper_cameras(relic)
        self.assertEqual(cameras[0]['position_m'], list(NOMINAL_OPTICAL_FRAMES['hand_color_sensor']['xyz']))
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            urdf = fake / 'source/relic/relic/assets/spot/spot_with_arm.urdf'
            urdf.parent.mkdir(parents=True)
            urdf.write_text(URDF.replace('0.13806', '9.0'))
            with self.assertRaises(ValueError):
                load_relic_gripper_cameras(fake)


if __name__ == '__main__':
    unittest.main()
