"""CPU checks for NVIDIA / MuJoCo GL selection."""
import os
import unittest

from treesim.gl_backend import bind_mujoco_gl, nvidia_gpu_present


class GlBackendTest(unittest.TestCase):
    def test_nvidia_gpu_present_is_bool(self):
        self.assertIsInstance(nvidia_gpu_present(), bool)

    def test_require_gpu_refuses_cpu_host(self):
        if nvidia_gpu_present():
            self.skipTest("this host has an NVIDIA GPU")
        previous = os.environ.get("MUJOCO_GL")
        try:
            with self.assertRaises(RuntimeError):
                bind_mujoco_gl("auto", require_gpu=True)
            with self.assertRaises(RuntimeError):
                bind_mujoco_gl("egl", require_gpu=False)
        finally:
            if previous is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = previous


if __name__ == "__main__":
    unittest.main()
