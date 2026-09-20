import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from treesim.kiwi_rl.training_monitor import (
    LiveDashboard, compose_progress_frame, curriculum_preview_from_checkpoint,
    due_checkpoints, due_latest_checkpoint, read_jsonl, render_dashboard_html,
    serve_monitor, series_from_rows, svg_chart, write_dashboard,
)


class TrainingMonitorTests(unittest.TestCase):
    def test_jsonl_skips_truncated_line_and_charts_all_numeric_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.jsonl'
            path.write_text(
                '{"step": 0, "evaluation/mean_closest_distance_m": 0.8, "evaluation/harvest_successes": 0, "curriculum_stage": "deposit_pixels"}\n'
                '{"step": 5, "loss": 0.12, "reward_mean": 0.01, "distance_mean_closest_m": 0.4, "curriculum_stage": "deposit_pixels"}\n'
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
            self.assertIn('Curriculum TK-RL-003', html)
            self.assertIn('entropía es diferencial', html)
            self.assertIn('overflow', html)
            self.assertIn('training_ready', html)
            self.assertIn('evaluation/mean_closest_distance_m', html)
            self.assertNotIn('http-equiv="refresh"', html)
            self.assertIn('metrics.json', html)
            self.assertIn('withAuth', html)
            self.assertIn('queryToken', html)
            self.assertIn('credentials', html)
            self.assertIn('/files/workspace/training/monitor-live/index.html', html)
            self.assertIn('Último vídeo de progreso', html)
            self.assertIn('deposit_pixels', html)
            self.assertIn('001 · deposit_pixels', html)

    def test_compose_overlay_keeps_scene_and_nearest_gripper_patch(self):
        scene = np.zeros((180, 320, 3), dtype=np.uint8)
        scene[:] = (10, 20, 30)
        gripper = np.zeros((16, 16, 3), dtype=np.uint8)
        gripper[:] = (200, 40, 10)
        frame = compose_progress_frame(scene, gripper, overlay_px=32, inset=4)
        self.assertEqual(frame.shape, scene.shape)
        np.testing.assert_array_equal(frame[40, 40], (10, 20, 30))
        np.testing.assert_array_equal(frame[4 + 16, 320 - 4 - 16], (200, 40, 10))

    def test_due_checkpoints_start_at_first_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for update in (0, 1, 4, 5, 10):
                (root / f'checkpoint-{update:04d}.pt').write_bytes(b'x')
            self.assertEqual([path.name for path in due_checkpoints(root, 5)],
                             ['checkpoint-0001.pt', 'checkpoint-0005.pt', 'checkpoint-0010.pt'])
            self.assertEqual(due_checkpoints(root, 0), [])

    def test_due_latest_checkpoint_follows_completed_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'latest.pt').write_bytes(b'x')
            (root / 'latest.pt.json').write_text(json.dumps({'completed_updates': 20}), encoding='utf-8')
            self.assertEqual(due_latest_checkpoint(root, 10).name, 'latest.pt')
            self.assertIsNone(due_latest_checkpoint(root, 50))
            (root / 'latest.pt.json').write_text(json.dumps({'completed_updates': 0}), encoding='utf-8')
            self.assertIsNone(due_latest_checkpoint(root, 10))
            self.assertIsNone(due_latest_checkpoint(root, 0))

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
            link = (hub / 'videos').readlink()
            dashboard.refresh()
            self.assertTrue((hub / 'videos').is_symlink())
            self.assertEqual((hub / 'videos').readlink(), link)
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

    def test_dashboard_shows_stage_001_before_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps({
                'stage': 'deposit_pixels',
                'curriculum': {'name': 'deposit_pixels', 'index': 1},
            }), encoding='utf-8')
            payload = write_dashboard([], Path(tmp) / 'monitor', run=run)
            html = (Path(tmp) / 'monitor' / 'index.html').read_text(encoding='utf-8')
            self.assertEqual(payload['curriculum_label'], '001 · deposit_pixels')
            self.assertEqual(payload['curriculum_index'], 1)
            self.assertIn('001 · deposit_pixels', html)
            self.assertNotIn('sin etapa', html)
            html_easy = render_dashboard_html(dict(
                schema='training-monitor/v1', training_ready=False, run=str(run), rows=1,
                latest=dict(step=1, harvest_successes=0, basket_distance_mean_m=0.41,
                            ground_contact_worlds=12, reward_mean=-0.2),
                series={}, generated_at='now', videos=[],
                curriculum_label='001 · deposit_pixels'))
            self.assertIn('basket_distance_mean_m', html_easy)
            self.assertIn('ground_contact_worlds', html_easy)
            self.assertIn('harvest_successes', html_easy)
        from treesim.kiwi_rl import training_monitor as mon
        mon_src = inspect.getsource(mon)
        self.assertIn('deposit_return_sum', mon_src)
        self.assertIn('harvest_jackpot_sum', mon_src)
        self.assertIn('success_window_return_mean', mon_src)
        self.assertIn('reward_window_mean', mon_src)
        self.assertIn('fail_return_sum', mon_src)

    def test_curriculum_preview_matches_stage_reset(self):
        deposit = curriculum_preview_from_checkpoint({
            'meta': {'curriculum_stage': 'deposit_pixels'}, 'config': {},
        })
        self.assertEqual(deposit['stage'], 'deposit_pixels')
        self.assertEqual(deposit['reset_mode'], 1)
        self.assertFalse(deposit['allow_locomotion'])
        detach = curriculum_preview_from_checkpoint({
            'meta': {'curriculum_stage': 'grasp_detach'}, 'config': {},
        })
        self.assertEqual(detach['reset_mode'], 2)
        approach = curriculum_preview_from_checkpoint({
            'meta': {}, 'config': {'stage': 'visual_approach'},
        })
        self.assertEqual(approach['reset_mode'], 3)
        self.assertTrue(approach['allow_locomotion'])
        hanging = curriculum_preview_from_checkpoint({'meta': {}, 'config': {}})
        self.assertEqual(hanging['reset_mode'], 0)
        self.assertFalse(hanging['allow_locomotion'])
        self.assertFalse(hanging['easy'])
        easy = curriculum_preview_from_checkpoint({
            'meta': {'curriculum_stage': 'deposit_pixels'},
            'config': {'easy': True, 'easy_far_frac': 0.25},
        })
        self.assertTrue(easy['easy'])
        self.assertEqual(easy['reset_mode'], 1)
        self.assertEqual(easy['easy_far_frac'], 0.25)
        self.assertAlmostEqual(easy['hold_close_frac'], 0.75)
        held = curriculum_preview_from_checkpoint({
            'meta': {'curriculum_stage': 'deposit_pixels'},
            'config': {'easy': True, 'easy_far_frac': 0.25, 'hold_close_frac': 0.9},
        })
        self.assertAlmostEqual(held['hold_close_frac'], 0.9)
        from treesim.kiwi_rl.training_monitor import (
            apply_native_easy_hover, apply_native_easy_start, apply_native_skill_reset, _n3_command,
        )
        self.assertIn('hover_tcp_world_m', inspect.getsource(apply_native_easy_hover))
        self.assertIn('easy_over_opening_local_m', inspect.getsource(apply_native_easy_start))
        self.assertIn('tcp_over_opening_above_rim', inspect.getsource(apply_native_easy_start))
        self.assertIn('start_over_opening', inspect.getsource(apply_native_easy_start))
        self.assertIn('apply_native_easy_hover', inspect.getsource(apply_native_easy_start))
        from treesim.kiwi_rl.training_monitor import apply_native_carry_start
        self.assertIn('random_carry_start_local_m', inspect.getsource(apply_native_carry_start))
        self.assertIn('ik_demo', inspect.getsource(apply_native_skill_reset))
        self.assertIn('random_grasp_offset_local_m', inspect.getsource(apply_native_skill_reset))
        self.assertIn('tcp_world', inspect.getsource(apply_native_skill_reset))
        from treesim.kiwi_rl.training_monitor import _record_progress_video_locked as _rec
        rec_src = inspect.getsource(_rec)
        self.assertIn('release_at_center', rec_src)
        self.assertIn('release_over_opening', rec_src)
        self.assertIn('max_above_rim_m', rec_src)
        self.assertIn('tcp_xy', rec_src)
        self.assertIn('easy and reset_mode == 1', inspect.getsource(apply_native_skill_reset))
        self.assertIn('apply_native_easy_start', inspect.getsource(apply_native_skill_reset))
        self.assertIn('jaw_hold_q', inspect.getsource(apply_native_skill_reset))
        self.assertIn('jaw_open_closed_q', inspect.getsource(apply_native_skill_reset))
        self.assertIn('grasp_pocket_world_m', inspect.getsource(apply_native_skill_reset))
        self.assertNotIn('model.jnt_range[jaw_joint, 1]', inspect.getsource(apply_native_skill_reset))
        self.assertIn('hold_close_frac', inspect.getsource(apply_native_skill_reset))
        self.assertNotIn('easy_airdrop_world_m', inspect.getsource(apply_native_skill_reset))
        from treesim.kiwi_rl.training_monitor import _record_progress_video_locked
        self.assertIn('scripted_jaw_target', inspect.getsource(_record_progress_video_locked))
        self.assertIn('fruit_in_release_zone', inspect.getsource(_record_progress_video_locked))
        self.assertIn('rim_z_m', inspect.getsource(_record_progress_video_locked))
        self.assertIn('adapt_scripted_hold_q', inspect.getsource(_record_progress_video_locked))
        self.assertIn('jaw_open_closed_q', inspect.getsource(_record_progress_video_locked))
        self.assertIn('controller.qids[18]', inspect.getsource(_record_progress_video_locked))
        html_easy_video = render_dashboard_html(dict(
            schema='training-monitor/v1', training_ready=False, run='x', rows=1,
            latest=dict(step=1), series={}, generated_at='now',
            videos=[dict(step=10, file='progress-0010.mp4', url='videos/progress-0010.mp4',
                         label='CPU native curriculum preview; not a harvest demonstration',
                         easy=True, easy_far_frac=0.0, min_basket_distance_m=0.39)]))
        self.assertIn('easy-carry', html_easy_video)
        self.assertNotIn('easy-airdrop', html_easy_video)
        first = _n3_command(np.zeros(3), np.array([1.0, 0.0, -1.0]))
        np.testing.assert_allclose(first, [0.02, 0.0, -0.04], atol=1e-6)

    def test_serve_monitor_exposes_hub_without_cache(self):
        import urllib.request
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            (hub / 'index.html').write_text('<html>monitor</html>', encoding='utf-8')
            (hub / 'metrics.json').write_text(json.dumps({'rows': 1}) + '\n', encoding='utf-8')
            server = serve_monitor(hub, 0)
            try:
                host, port = server.server_address[:2]
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics.json', timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn('no-store', response.headers.get('Cache-Control', ''))
                    self.assertEqual(response.headers.get('Access-Control-Allow-Origin'), '*')
                    self.assertEqual(json.loads(response.read().decode()), {'rows': 1})
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
