import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from treesim.kiwi_rl.fast_teacher import (
    PRIVILEGED_DIM, TEACHER_SCHEMA, build_privileged_policy,
    privileged_observation, validate_teacher_report,
)


class FastTeacherTest(unittest.TestCase):
    def test_privileged_observation_is_compact_relative_state(self):
        import warp

        class Model:
            body_rootid = np.array([0, 0, 0])
            def site(self, site_id):
                return types.SimpleNamespace(bodyid=0)

        class Data:
            pass

        data = Data()
        data.xpos = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.2, 0., .1]]])
        data.xipos = data.xpos.clone()
        data.xmat = torch.eye(3).repeat(1, 3, 1, 1)
        data.site_xpos = torch.zeros((1, 1, 3))
        data.cvel = torch.zeros((1, 3, 6))
        data.subtree_com = torch.zeros((1, 1, 3))
        task = types.SimpleNamespace(**{name: torch.zeros(1) for name in (
            'hand_contact', 'bilateral_contact', 'stable_grasp', 'ever_grasped',
            'detached', 'hand_load', 'finger_load', 'jaw_load', 'palm_load',
            'stem_force', 'damage_proxy')})
        runtime = types.SimpleNamespace(worlds=1, data=data, model=Model(),
            chassis=1, fruit_body=2, tcp_site=0, task=task)
        r84 = torch.zeros((1, 84))
        with patch.object(warp, 'to_torch', side_effect=lambda value: value):
            observed = privileged_observation(runtime, r84)
        self.assertEqual(tuple(observed.shape), (1, PRIVILEGED_DIM))
        self.assertTrue(torch.equal(observed[0, :3], torch.tensor([.2, 0., .1])))
        self.assertEqual(observed[0, -1].item(), 0.)

    def test_privileged_policy_uses_same_recurrent_actor_interface(self):
        policy = build_privileged_policy()
        privileged = torch.randn(3, PRIVILEGED_DIM)
        r84 = torch.randn(3, 84)
        memory = torch.zeros(3, 64)
        mean, logstd, value, next_memory = policy(privileged, r84, memory)
        self.assertEqual(tuple(mean.shape), (3, 7))
        self.assertEqual(tuple(logstd.shape), (7,))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertEqual(tuple(next_memory.shape), (3, 64))
        mean.square().mean().backward()
        self.assertIsNotNone(policy.mean.weight.grad)

    def test_teacher_evidence_must_match_checkpoint_and_holdout_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'teacher.pt'
            checkpoint.write_bytes(b'learned teacher weights')
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report_path = Path(directory) / 'report.json'
            report = dict(role='teacher', schema=TEACHER_SCHEMA,
                teacher_checkpoint_sha256=digest, training_model_sha256='train',
                evaluation_model_sha256='heldout', guidance_enabled=False,
                numerical_failures=0,
                evaluation=dict(episodes=32, success=.5))
            report_path.write_text(json.dumps(report))
            validate_teacher_report(report_path, checkpoint,
                training_model_sha256='train', evaluation_model_sha256='heldout')
            report['evaluation']['success'] = .49
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'success gate'):
                validate_teacher_report(report_path, checkpoint,
                    training_model_sha256='train', evaluation_model_sha256='heldout')

    def test_ppo_rejects_counterfactual_rows(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_fast import update
        with self.assertRaisesRegex(ValueError, 'factual on-policy'):
            update(None, None, [dict(kind='counterfactual')], None)


@unittest.skipUnless(os.environ.get('FAST_SCENE') and os.environ.get('GAIT_CHECKPOINT'),
                     'GPU fast scene and verified gait required')
class DistillationIntegrationTest(unittest.TestCase):
    def test_student_updates_from_frozen_teacher_without_privileged_inputs(self):
        import warp as wp
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.harvest_training import HarvestCollector
        from treesim.kiwi_rl.control import load_gait_artifact
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_physical_smoke import build_policy
        from train_fast import update
        wp.init()
        torch.set_num_threads(1)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            rt = FastRuntime(os.environ['FAST_SCENE'], worlds=2, camera='hand_camera')
            teacher = build_privileged_policy().cuda().eval().requires_grad_(False)
            student = build_policy().cuda()
            gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
            collector = HarvestCollector(rt, role='student', teacher_policy=teacher)
            old_teacher = {k:v.clone() for k,v in teacher.state_dict().items()}
            old_student = [p.detach().clone() for p in student.parameters()]
            rows, bootstrap, _ = collector.collect(student, gait, 4)
            self.assertTrue(all('privileged' not in row for row in rows))
            self.assertTrue(all(not row['teacher_action'].requires_grad for row in rows))
            metrics = update(student, torch.optim.Adam(student.parameters(), lr=1e-4),
                             rows, bootstrap, 2, teacher_coef=1.)
            self.assertGreater(metrics['teacher_mse'], 0.)
            self.assertTrue(any(not torch.equal(before, after) for before, after in
                                zip(old_student, student.parameters())))
            for key, before in old_teacher.items():
                torch.testing.assert_close(before, teacher.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            rt.check()


if __name__ == '__main__':
    unittest.main()
