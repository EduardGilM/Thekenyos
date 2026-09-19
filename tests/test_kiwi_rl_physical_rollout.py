import unittest
from types import SimpleNamespace

import numpy as np

from treesim.kiwi_rl.physical_rollout import BatchedPhysicalRollout


class _Array:
    def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
    def numpy(self): return self.value


class _Control:
    def __init__(self):
        self.contract = SimpleNamespace(joints=np.arange(19))
        self.targets = _Array(np.zeros((1, 19)))
    def set_targets(self, targets): self.targets = _Array(targets)
    def observe(self): return _Array(np.zeros((1, 84)))


class _Runtime:
    worlds, dt, step_index = 1, .001, 0
    def __init__(self):
        self.control = _Control()
        self.model = SimpleNamespace(jnt_range=np.tile([[-1., 1.]], (19, 1)),
                                     jnt_limited=np.ones(19, dtype=bool))
    def capture(self): return {"hand_color_sensor": np.zeros((1, 5, 16, 16), dtype=np.float32)}
    def advance(self, control_dt_s=.04): self.step_index += round(control_dt_s / self.dt)


class PhysicalRolloutTest(unittest.TestCase):
    def test_action_is_seven_arm_and_jaw_targets_and_is_clamped(self):
        runtime = _Runtime()
        applied = BatchedPhysicalRollout(runtime).apply_action(np.full(7, 2., dtype=np.float32))
        np.testing.assert_array_equal(applied, np.ones((1, 7), dtype=np.float32))
        np.testing.assert_array_equal(runtime.control.targets.numpy()[0, 12:19], np.ones(7))

    def test_runtime_observation_has_explicit_unsupported_fields(self):
        observation = BatchedPhysicalRollout(_Runtime()).observe()
        self.assertEqual(observation.rgbd.shape, (1, 5, 16, 16))
        self.assertIn("fruit_truth", observation.unsupported_fields)

    def test_invented_body_mast_is_not_a_policy_camera(self):
        with self.assertRaises(ValueError):
            BatchedPhysicalRollout(_Runtime(), camera="body_camera")

if __name__ == "__main__": unittest.main()
