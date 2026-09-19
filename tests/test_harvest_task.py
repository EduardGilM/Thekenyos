import unittest
from dataclasses import replace
import numpy as np
from treesim.harvest_task import TaskDefinition, HarvestObservation, HarvestOracle
from treesim.kiwi_material import detachment_force, fruit_stem_angle


def initial():
    return HarvestObservation(0., 3, True, .3, (0.,0.), 0., False, False, 0.,
                              False, 0., 180., 1., 36.5, 0.)


class HarvestTaskTest(unittest.TestCase):
    def setUp(self):
        self.o = HarvestOracle()
        self.obs = initial()
        self.o.reset(self.obs)

    def step(self, **changes):
        self.obs = replace(self.obs, time_s=round(self.obs.time_s+.02, 6), **changes)
        return self.o.update(self.obs)

    def grasp(self):
        for _ in range(9):
            self.step(jaw_forces_N=(2.,2.), tcp_distance_m=.002)
        self.assertTrue(self.o.grasped)

    def test_success_requires_detachment_release_containment_and_settling(self):
        self.grasp()
        result = self.step(attached=False)
        self.assertIn('retained_detachment', result['events'])
        for _ in range(40):
            result = self.step(in_basket=True, basket_contact=True)
        self.assertFalse(result['terminated'])  # Still gripping.
        for _ in range(15):
            result = self.step(jaw_forces_N=(0.,0.), basket_relative_speed_m_s=.2)
        self.assertFalse(result['terminated'])  # Moving fruit is not deposited.
        for _ in range(28):
            result = self.step(basket_relative_speed_m_s=0.)
        self.assertTrue(result['success'])
        self.assertEqual(self.step()['reward'], 0.)  # No repeated terminal bonus.

    def test_no_reward_for_angle_alone_or_repeated_grasp(self):
        result = self.step(fsa_deg=60.)
        self.assertLess(result['reward'], 0)
        self.grasp()
        self.step(jaw_forces_N=(0.,0.))
        for _ in range(10):
            result = self.step(jaw_forces_N=(2.,2.))
            self.assertNotIn('grasp', result['terms'])

    def test_uncontrolled_detachment_ground_damage_and_timeout(self):
        for change, reason in [({'ground_contact':True},'dropped'),
                               ({'damage':.06},'damage_limit'),
                               ({'jaw_forces_N':(16.,0.)},'jaw_overload'),
                               ({'collateral_losses':1},'collateral_loss'),
                               ({'forbidden_collision':True},'forbidden_collision')]:
            self.setUp()
            self.assertEqual(self.step(**change)['outcome'], reason)
        self.o = HarvestOracle(TaskDefinition(time_limit_s=.1))
        self.obs = initial(); self.o.reset(self.obs)
        for _ in range(6):
            result = self.step()
        self.assertEqual(result['outcome'], 'timeout')

    def test_damage_never_refunds_and_short_grasps_do_not_count(self):
        first = self.step(damage=.01)
        self.assertAlmostEqual(first['terms']['damage'], -.2)
        self.assertEqual(self.step(damage=0.)['terms']['damage'], 0.)
        self.assertEqual(self.step(damage=.01)['terms']['damage'], 0.)
        self.grasp()
        self.step(jaw_forces_N=(0.,0.))
        result = self.step(jaw_forces_N=(2.,2.), attached=False)
        self.assertFalse(result['terminated'])
        self.assertNotIn('detach', result['terms'])

    def test_alternative_strategy_can_succeed_without_grasp_or_angle(self):
        self.o = HarvestOracle(TaskDefinition(guidance_weight=0.))
        self.o.reset(self.obs)
        self.step(attached=False, fsa_deg=145.)
        for _ in range(28):
            result = self.step(in_basket=True, basket_contact=True)
            self.assertEqual(result['terms']['reach_potential'], 0.)
            if result['terminated']:
                break
        self.assertTrue(result['success'])
        self.assertFalse(self.o.grasped)
        self.assertEqual(result['terms']['reach_potential'], 0.)

    def test_invalid_inputs_and_sparse_observations(self):
        for action in ([0]*6, [2]*7, [float('nan')]*7):
            with self.assertRaises(ValueError): TaskDefinition().action_velocity(action)
        np.testing.assert_allclose(TaskDefinition().action_velocity([1]*7), [.5]*7)
        with self.assertRaises(ValueError): self.o.update(replace(self.obs, time_s=1.))
        with self.assertRaises(ValueError): self.o.update(replace(self.obs, time_s=.02, fruit_id=4))


class DetachmentModelTest(unittest.TestCase):
    def test_angle_convention_and_curve(self):
        self.assertEqual(fruit_stem_angle([0,0,-1], [0,0,1]), 180)
        self.assertAlmostEqual(fruit_stem_angle([np.sqrt(3)/2,0,.5], [0,0,1]),60)
        self.assertAlmostEqual(detachment_force(60),5.98)
        self.assertAlmostEqual(detachment_force(160),40.27)
        self.assertLess(detachment_force(180),detachment_force(160))
        self.assertEqual(detachment_force(0),detachment_force(60))
        self.assertAlmostEqual(detachment_force(70),(5.98+6.3)/2)
        for angle in (-1,181,float('nan')):
            with self.assertRaises(ValueError): detachment_force(angle)
        with self.assertRaises(ValueError): fruit_stem_angle([0,0,0],[0,0,1])
