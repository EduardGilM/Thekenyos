import copy
import json
import tempfile
import unittest
from pathlib import Path

from treesim.kiwi_rl.contact_acceptance import read_contact_gate


class ContactAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.report = dict(report_kind='deformable-backend-screen/v2', case='grip', backend='gpu',
            passed=False, source_sha256='a' * 64, scene_sha256='b' * 64,
            timestep_s=.00002, integrator='Euler', solver='CG', iterations=1000,
            tolerance=1e-9, contact_time_s=.002, mesh_count=9,
            mesh_normalization={'max_surface_error_m': 2.7e-10}, worlds=2,
            mesh_flex_contacts=10, minimum_volume_ratio=.923, hold_samples=10,
            sampled_bilateral_fraction=[1., 1.], max_hold_motion_m=[.002, .003],
            post_release_jaw_load_N=[0., 0.], max_hand_penetration_m=.001231,
            finite_all_worlds=True, simulated_seconds=4.,
            contacts={'flags': [0, 0], 'first_ground_step': [[168045], [168548]]},
            gpu_numerical={'flags': [0, 0], 'minimum_volume_ratio': [.92, .93]})
        self.manifest = dict(numerical_profile={key: self.report[key] for key in (
            'timestep_s', 'integrator', 'solver', 'iterations', 'tolerance', 'contact_time_s')},
            material={'mesh_count': 9}, mesh_normalization={'max_surface_error_m': 1.7e-9})

    def check_report(self, report):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probe.json'
            path.write_text(json.dumps(report))
            return read_contact_gate(path, self.manifest)

    def test_legacy_report_passes_explicit_approximation_and_retains_limits(self):
        result = self.check_report(self.report)
        self.assertFalse(result['strict_pass'])
        self.assertTrue(result['accepted'])
        self.assertTrue(result['penetration_sampling_limited'])
        self.assertAlmostEqual(result['observed_ground_times_s'][0], 3.3609)

    def test_numerical_failure_is_not_an_acceptable_approximation(self):
        for section in ('gpu_numerical', 'contacts'):
            report = copy.deepcopy(self.report)
            report[section]['flags'][1] = 1
            with self.assertRaises(ValueError):
                self.check_report(report)

    def test_release_and_collapse_failures_are_rejected(self):
        for kind in ('release', 'early_ground', 'volume'):
            report = copy.deepcopy(self.report)
            if kind == 'release': report['post_release_jaw_load_N'][1] = 18.
            elif kind == 'early_ground': report['contacts']['first_ground_step'][1] = [10000]
            else: report['gpu_numerical']['minimum_volume_ratio'][1] = .01
            self.assertFalse(self.check_report(report)['accepted'])

    def test_profile_mismatch_is_rejected(self):
        self.manifest['numerical_profile']['contact_time_s'] = .001
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            self.check_report(self.report)


if __name__ == '__main__':
    unittest.main()
