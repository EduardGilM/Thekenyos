import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from treesim.kiwi_rl.training_monitor import (
    LiveDashboard, compose_progress_frame, due_checkpoints, read_jsonl,
    render_dashboard_html, series_from_rows, svg_chart, write_dashboard,
)


class TrainingMonitorTests(unittest.TestCase):
    def test_jsonl_skips_truncated_line_and_charts_all_numeric_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.jsonl'
            path.write_text(
                '{"step": 0, "evaluation/mean_closest_distance_m": 0.8, "evaluation/harvest_successes": 0}\n'
                '{"step": 5, "loss": 0.12, "reward_mean": 0.01, "distance_mean_closest_m": 0.4}\n'
                '{"step": 5, "loss":\n',
                encoding='utf-8')
            rows = read_jsonl(path)
            series = series_from_rows(rows)
            self.assertEqual([row['step'] for row in rows], [0, 5])
            self.assertEqual(series['loss']['values'], [0.12])
            self.assertEqual(series['evaluation/mean_closest_distance_m']['values'], [0.8])
            self.assertNotIn('step', series)
            svg = svg_chart(series['loss']['steps'], series['loss']['values'], 'loss')
            self.assertIn('loss', svg)
            self.assertIn('polyline', svg)
            payload = write_dashboard(rows, Path(tmp) / 'monitor', run=tmp)
            html = (Path(tmp) / 'monitor' / 'index.html').read_text(encoding='utf-8')
            self.assertEqual(payload['training_ready'], False)
            self.assertIn('Monitor de entrenamiento', html)
            self.assertIn('no demuestran cosecha', html)
            self.assertIn('evaluation/mean_closest_distance_m', html)
            self.assertNotIn('http-equiv="refresh"', html)
            self.assertIn('metrics.json', html)
            self.assertIn('Último vídeo de progreso', html)

    def test_compose_overlay_keeps_scene_and_nearest_gripper_patch(self):
        scene = np.zeros((180, 320, 3), dtype=np.uint8)
        scene[:] = (10, 20, 30)
        gripper = np.zeros((16, 16, 3), dtype=np.uint8)
        gripper[:] = (200, 40, 10)
        frame = compose_progress_frame(scene, gripper, overlay_px=32, inset=4)
        self.assertEqual(frame.shape, scene.shape)
        np.testing.assert_array_equal(frame[40, 40], (10, 20, 30))
        np.testing.assert_array_equal(frame[4 + 16, 320 - 4 - 16], (200, 40, 10))

    def test_due_checkpoints_include_zero_and_skip_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for update in (0, 4, 5, 10):
                (root / f'checkpoint-{update:04d}.pt').write_bytes(b'x')
            self.assertEqual([path.name for path in due_checkpoints(root, 5)],
                             ['checkpoint-0000.pt', 'checkpoint-0005.pt', 'checkpoint-0010.pt'])
            self.assertEqual(due_checkpoints(root, 0), [])

    def test_live_dashboard_copies_hub_and_lists_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, hub = Path(tmp) / 'run', Path(tmp) / 'hub'
            run.mkdir()
            (run / 'training.jsonl').write_text('{"step": 1, "loss": 0.5}\n', encoding='utf-8')
            dashboard = LiveDashboard(run, hub=hub)
            video = dashboard.video_path(10)
            video.write_bytes(b'\x00')
            video.with_suffix('.json').write_text(json.dumps({
                'label': 'CPU native progress preview; not a harvest demonstration',
                'min_tcp_fruit_distance_m': 0.21,
            }), encoding='utf-8')
            payload = dashboard.refresh()
            html = (hub / 'index.html').read_text(encoding='utf-8')
            self.assertEqual(payload['latest']['loss'], 0.5)
            self.assertEqual(payload['videos'][0]['step'], 10)
            self.assertIn('progress-0010.mp4', html)
            self.assertTrue((hub / 'videos').is_symlink())
            html_two = render_dashboard_html(dict(
                schema='training-monitor/v1', training_ready=False, run=str(run), rows=1,
                latest=payload['latest'], series=payload['series'], generated_at='now',
                videos=[
                    dict(step=0, file='progress-0000.mp4', url='videos/progress-0000.mp4', label='old'),
                    dict(step=50, file='progress-0050.mp4', url='videos/progress-0050.mp4',
                         label='CPU native progress preview; not a harvest demonstration',
                         min_tcp_fruit_distance_m=0.21),
                ]))
            self.assertIn('progress-0050.mp4', html_two)
            self.assertNotIn('progress-0000.mp4', html_two)
            self.assertNotIn('http-equiv="refresh"', html_two)
            self.assertIn('training_ready', html_two)


if __name__ == '__main__':
    unittest.main()
