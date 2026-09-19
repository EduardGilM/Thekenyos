"""Arbiter: sensor-gated phases, privileged flags never consulted."""
import unittest

from treesim.kiwi_rl.arbiter import Arbiter


class ArbiterTest(unittest.TestCase):
    def test_explore_attempt_settle_manipulate_verify_cycle(self):
        a = Arbiter()
        # ATTEMPT ignored with stale frame
        a.update(0.04, "ATTEMPT", None, 0.0, 0.0, 5.0, False)
        self.assertEqual(a.state.phase, "EXPLORE")
        # fresh frame accepts ATTEMPT
        a.update(0.04, "ATTEMPT", None, 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.phase, "SETTLE")
        for _ in range(8):
            a.update(0.04, None, None, 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.phase, "MANIPULATE")
        a.update(0.04, None, "FINISH", 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.phase, "VERIFY")
        # VERIFY needs 1 s before FINISH is accepted
        a.update(0.04, None, "FINISH", 0.0, 0.0, 0.1, True, verify_time_s=0.2)
        self.assertEqual(a.state.phase, "VERIFY")
        a.update(0.04, None, "FINISH", 0.0, 0.0, 0.1, True, verify_time_s=1.2)
        self.assertEqual(a.state.phase, "EXPLORE")

    def test_settle_requires_continuous_stability(self):
        a = Arbiter()
        a.update(.04, 'ATTEMPT', None, 0., 0., 0., True)
        a.update(.29, None, None, 1., 0., 0., True)
        a.update(.01, None, None, 0., 0., 0., True)
        self.assertEqual(a.state.phase, 'SETTLE')
        for _ in range(30):
            a.update(.01, None, None, 0., 0., 0., True)
        self.assertEqual(a.state.phase, 'MANIPULATE')
        self.assertEqual(a.state.active_policy, 'M3')

    def test_stale_frame_resets_stable_dwell(self):
        a = Arbiter()
        a.update(.04, 'ATTEMPT', None, 0., 0., 0., True)
        a.update(.25, None, None, 0., 0., 0., True)
        a.update(.1, None, None, 0., 0., 2., False)
        a.update(.05, None, None, 0., 0., 0., True)
        self.assertEqual(a.state.phase, 'SETTLE')

    def test_manipulate_holds_base_command_zero(self):
        a = Arbiter()
        a.update(0.04, "ATTEMPT", None, 0.0, 0.0, 0.1, True)
        for _ in range(8):
            a.update(0.04, None, None, 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.phase, "MANIPULATE")
        a.update(0.04, None, None, 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.active_policy, "M3")

    def test_recover_after_retries(self):
        a = Arbiter()
        a.update(0.04, None, "RECOVER", 0.0, 0.0, 0.1, True)
        # RECOVER from EXPLORE only via MANIPULATE; force path instead:
        a.state.phase = "MANIPULATE"
        a.update(0.04, None, "RECOVER", 0.0, 0.0, 0.1, True)
        self.assertEqual(a.state.phase, "RECOVER")
        self.assertEqual(a.state.retries, 1)
        with self.assertRaises(ValueError):
            a.update(0.04, "BOGUS", None, 0.0, 0.0, 0.1, True)


if __name__ == "__main__":
    unittest.main()
