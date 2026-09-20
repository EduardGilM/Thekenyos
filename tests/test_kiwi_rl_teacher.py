import unittest

try:
    import torch
except ImportError:
    torch = None

from treesim.kiwi_rl.models_torch import build_student
from treesim.kiwi_rl.teacher import build_teacher, distillation_loss


@unittest.skipUnless(torch is not None, 'Torch required')
class TeacherTest(unittest.TestCase):
    def test_privileged_teacher_and_sensor_student_have_separate_parameters(self):
        torch.manual_seed(42)
        teacher = build_teacher(32, hidden_size=64)
        student = build_student(hidden_size=64, intent_dim=8)
        proprio, context, basket = torch.zeros(2, 85), torch.zeros(2, 14), torch.zeros(2, 16)
        target = teacher(torch.randn(2, 32), proprio, context, basket, torch.zeros(2, 64))
        prediction = student(rgbd=torch.randn(2, 5, 64, 64), proprio=proprio, context=context,
            basket=basket, local_map=torch.zeros(2, 3, 64, 64), previous_actions=torch.zeros(2, 11),
            previous_events=torch.zeros(2, 5), memory=torch.zeros(2, 64))
        loss = distillation_loss(prediction, target, torch.tensor([[False, True], [True, False]]))['loss']
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in student.vision.parameters() if p.grad is not None), 0.)
        self.assertFalse(teacher.config['deployable'])

    def test_unobservable_teacher_label_can_abstain(self):
        student = {f'{name}_{part}': torch.zeros(2, width, requires_grad=True)
                   for name, continuous, events in [('n', 3, 2), ('m', 8, 3)]
                   for part, width in [('mu', continuous), ('events', events)]}
        teacher = {k: torch.ones_like(v) for k, v in student.items()}
        result = distillation_loss(student, teacher, torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 2))
        self.assertEqual(result['loss'].item(), 0.)
        result['loss'].backward()
        for value in student.values():
            self.assertEqual(value.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
