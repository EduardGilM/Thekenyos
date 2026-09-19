import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mujoco_warp') and importlib.util.find_spec('torch'), 'CUDA sensing stack required')
class RGBDTest(unittest.TestCase):
    def test_spot_housing_is_omitted_only_in_sensor_context(self):
        import mujoco
        import mujoco_warp as mw
        import warp as wp
        from treesim.kiwi_rl.sensors_warp import WarpRGBDRig
        wp.init()
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom type="plane" size="10 10 .1"/>
        <body name="spot_arm_link_wr1" pos="0 0 2">
          <camera name="hand_camera"/>
          <geom type="sphere" size=".01" contype="0" conaffinity="0" group="2"/>
        </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        groups, pos = model.geom_group.copy(), model.cam_pos.copy()
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(model)
            gd = mw.put_data(model, data, nworld=1, nconmax=64, njmax=256)
            mw.forward(gm, gd)
            rig = WarpRGBDRig(model, gd, cameras=('hand_camera',), resolution=(32, 24))
            rig.capture(gm, gd, 0.)
            np.testing.assert_allclose(rig.depth[0].numpy(), 2., atol=2e-5)
        np.testing.assert_array_equal(model.geom_group, groups)
        np.testing.assert_array_equal(model.cam_pos, pos)

    def test_metric_planar_depth_range_masks_and_frame_ownership(self):
        import mujoco
        import mujoco_warp as mw
        import torch
        import warp as wp
        from treesim.kiwi_rl.sensors_warp import WarpRGBDRig
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest('CUDA required')
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
        <worldbody><geom type="plane" size="10 10 .1" rgba=".4 .6 .3 1"/>
        <body pos="10 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
        <camera name="unused" pos="0 0 1"/>
        <camera name="near" pos="0 0 2.5" fovy="60"/>
        <camera name="far" pos="0 0 5" fovy="60"/>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(model)
            gd = mw.put_data(model, data, nworld=2, nconmax=64, njmax=256)
            mw.forward(gm, gd)
            rig = WarpRGBDRig(model, gd, cameras=('near', 'far'), resolution=(32, 24))
            with self.assertRaises(RuntimeError):
                rig.tensor('near')
            rig.capture(gm, gd, 0.)
            np.testing.assert_allclose(rig.depth[0].numpy(), 2.5, rtol=0, atol=2e-5)
            self.assertTrue(np.all(rig.valid[0].numpy() == 1))
            self.assertTrue(np.all(rig.depth[1].numpy() == 0))
            self.assertTrue(np.all(rig.valid[1].numpy() == 0))
            first = rig.tensor('near')
            self.assertEqual(tuple(first.shape), (2, 5, 24, 32))
            torch.testing.assert_close(first[:, 3], torch.full_like(first[:, 3], .625))
            frozen = first.clone()
            rig.rgb[0].zero_()
            wp.synchronize_device(rig.device)
            torch.testing.assert_close(first, frozen)
            frame_id = rig.frame_id
            rig.invalidate()
            with self.assertRaises(RuntimeError):
                rig.tensor('near')
            rig.capture(gm, gd, 0.)
            self.assertGreater(rig.frame_id, frame_id)


if __name__ == '__main__':
    unittest.main()
