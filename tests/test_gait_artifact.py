import unittest

from treesim.kiwi_rl.control import validate_gait_metadata


def _meta(**overrides):
    data = dict(passed=False, device='cpu', samples=10000, weights_identical=True,
                exported_sha256='abc', precision_profile='cuda-fp32',
                jit_vs_onnx_max_abs_diff=3.05e-5, max_scaled_error=1.12)
    data.update(overrides)
    return data


class GaitArtifactTest(unittest.TestCase):
    def test_cpu_profile_rejects_failed_scaled_gate(self):
        with self.assertRaises(ValueError):
            validate_gait_metadata(_meta(), precision_profile='cpu')

    def test_cuda_fp32_accepts_recorded_scaled_failure(self):
        metadata = validate_gait_metadata(_meta(), precision_profile='cuda-fp32', digest='abc')
        self.assertFalse(metadata['passed'])
        self.assertEqual(metadata['precision_profile'], 'cuda-fp32')

    def test_cuda_fp32_rejects_large_absolute_mismatch(self):
        with self.assertRaises(ValueError):
            validate_gait_metadata(_meta(jit_vs_onnx_max_abs_diff=0.2), precision_profile='cuda-fp32')

    def test_cpu_verified_artifact_can_move_to_cuda_profile(self):
        metadata = validate_gait_metadata(
            _meta(passed=True, precision_profile='cpu', max_scaled_error=0.4),
            precision_profile='cuda-fp32')
        self.assertTrue(metadata['passed'])


if __name__ == '__main__':
    unittest.main()
