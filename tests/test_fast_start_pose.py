import os
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('FAST_SCENE'), 'FAST_SCENE with actual Spot assets required')
class FastStartPoseTest(unittest.TestCase):
    def test_actual_spot_pose_preserves_other_state_and_matches_targets(self):
        from treesim.kiwi_rl.fast_scene import load_fast_scene
        from treesim.kiwi_rl.fast_start_pose import set_camera_start_pose
        model, data, manifest = load_fast_scene(os.environ['FAST_SCENE'])
        before = data.qpos.copy()
        result = set_camera_start_pose(model, data, manifest['robot'], manifest['fruits'][0]['body'])
        ids = [model.jnt_qposadr[model.joint(manifest['robot']['prefix'] + n).id]
               for n in manifest['robot']['arm'][:6]]
        keep = np.ones(model.nq, dtype=bool); keep[ids] = False
        np.testing.assert_array_equal(data.qpos[keep], before[keep])
        self.assertTrue(result['collision_free'])
        self.assertLess(max(result['camera_angles_rad']), .1)
        self.assertGreater(result['tcp_fruit_distance_m'], .1)
        # The exported scene initializes both physical joints and motor targets.
        for name, qid in zip(manifest['robot']['arm'][:6], ids):
            self.assertAlmostEqual(before[qid], manifest['robot']['initial_position_rad'][name])
