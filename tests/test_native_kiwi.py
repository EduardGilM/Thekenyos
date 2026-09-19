import runpy
from pathlib import Path
import unittest
import mujoco
import numpy as np
from treesim.native_kiwi import flesh_mass, FLESH_DENSITY_KG_M3


class NativeKiwiTest(unittest.TestCase):
    def test_mass_matches_actual_flex_volume(self):
        scene=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/kiwi_compression.py'))['scene']
        for count in (7,9):
            m=mujoco.MjModel.from_xml_string(scene(count=count));d=mujoco.MjData(m)
            mujoco.mj_forward(m,d)
            vertices=d.flexvert_xpos[m.flex_elem.reshape(-1,4)]
            volume=np.abs(np.linalg.det(vertices[:,1:]-vertices[:,:1])/6).sum()
            mass=m.body_mass[np.unique(m.flex_vertbodyid)].sum()
            self.assertAlmostEqual(mass/volume,FLESH_DENSITY_KG_M3,places=5)
            self.assertAlmostEqual(mass,flesh_mass(count),places=10)

    def test_convergence_rejects_failure_and_mesh_force_drift(self):
        compare=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/check_compression_convergence.py'))['compare']
        reference=dict(passed=True,held_force_per_pad_N=[10.,10.],compression_fraction=.03)
        self.assertTrue(compare(reference,reference,.10)['passed'])
        self.assertFalse(compare(dict(reference,held_force_per_pad_N=[12.,12.]),reference,.10)['passed'])
        self.assertFalse(compare(dict(passed=False),reference,.10)['passed'])
