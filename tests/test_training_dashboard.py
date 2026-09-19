import json
from collections import deque
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_dashboard import Dashboard, REMOTE_PROBE, _decode_jsonl_tail, _phase, handler_for, select_checkpoint
import training_dashboard as dashboard_module

class DashboardTest(unittest.TestCase):
    def test_active_pid_and_phase_fallbacks(self):
        self.assertIn("active=read('active-process.json')", REMOTE_PROBE)
        self.assertIn("active.get('pid',1545354)", REMOTE_PROBE)
        self.assertIn("'train_harvest_fast.py' in cmd", REMOTE_PROBE)
        self.assertEqual(_phase({'phase_events': [{'phase': 'ppo'}, {'phase': 'cti-v2'}]}), 'cti-v2')
        self.assertEqual(_phase({'rows': [{'cti/version': 2}]}), 'cti-v2')
        self.assertEqual(_phase({'rows': [{}]}), 'ppo')
        self.assertEqual(_phase({'rows': [{'cti/version': 7}]}), 'cti-v7')

    def test_jsonl_tail_keeps_large_records_and_latest_complete_rows(self):
        stream = io.BytesIO((json.dumps({'payload': 'x' * 70000}) + '\n'
                             + '{"round":1}\n{"round":').encode())
        records = _decode_jsonl_tail(deque(stream, maxlen=3))
        self.assertEqual([record.get('round') for record in records], [None, 1])
        self.assertEqual(len(records[0]['payload']), 70000)
        self.assertIn('deque(f,maxlen=limit)', REMOTE_PROBE)

    def test_remote_probe_executes_with_branch_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'phase-events.jsonl').write_text('{"phase":"cti-v2"}\n')
            (root/'cti-branches.jsonl').write_text(json.dumps({'records': ['x'*70000]})+'\n')
            probe=REMOTE_PROBE.replace('/mnt/ssd/experiments/kiwi-pergola/training/runs/teacher-reward-cti-001',tmp)
            result=subprocess.run([sys.executable,'-c',probe],capture_output=True,text=True,check=True)
            state=json.loads(result.stdout)
            self.assertEqual(state['phase'],'cti-v2')
            self.assertEqual(len(state['branch_diagnostics'][0]['records'][0]),70000)

    def test_run_configuration_drives_paths_probe_and_wandb_metadata(self):
        run = 'teacher-curriculum-cti-001'
        url = 'https://wandb.ai/example/project/runs/newrun'
        try:
            dashboard_module._configure_run(run, url)
            self.assertEqual(dashboard_module.RUN, run)
            self.assertEqual(dashboard_module.REMOTE_RUN,
                             f'{dashboard_module.REMOTE_ROOT}/training/runs/{run}')
            self.assertEqual(dashboard_module.WANDB_URL, url)
            self.assertIn(f"root = pathlib.Path('{dashboard_module.REMOTE_RUN}')",
                          dashboard_module.REMOTE_PROBE)
            self.assertIn(f"run = '{run}'", dashboard_module.REMOTE_PROBE)
            self.assertIn("and run in cmd", dashboard_module.REMOTE_PROBE)
            with tempfile.TemporaryDirectory() as tmp:
                instance = Dashboard(Path(tmp), no_render=True, run=run, wandb_url=url)
                self.assertEqual(instance.snapshot()['run'], run)
                self.assertEqual(instance.snapshot()['wandb_url'], url)
                self.assertEqual(instance.remote_run, dashboard_module.REMOTE_RUN)
        finally:
            dashboard_module._configure_run(dashboard_module.DEFAULT_RUN)

    def test_remote_report_can_supply_missing_wandb_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = Dashboard(Path(tmp), no_render=True, run='teacher-curriculum-cti-001')
            instance._remote_snapshot = lambda: {
                'rows': [], 'report': {'wandb_url': 'https://wandb.ai/project/run'},
                'failure': None, 'process_alive': False,
            }
            instance.poll()
            self.assertEqual(instance.snapshot()['wandb_url'], 'https://wandb.ai/project/run')

    def test_best_uses_physical_outcomes_and_retains_earliest_tie(self):
        def row(step,success=0,failure=0,grasp=0,distance=.1):
            return {'step':step,'evaluation/success':success,'evaluation/physical_failure':failure,
                    'evaluation/grasp':grasp,'evaluation/closest_distance_m':distance}
        rows=[row(0),row(1,failure=1,grasp=1,distance=.001),row(2),{'step':3,'evaluation/success':1}]
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000000.pt')
        rows.append(row(4,success=.5))
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000004.pt')
        self.assertEqual(select_checkpoint(rows,{'best_checkpoint':'/run/checkpoint-000009.pt'})[0],'checkpoint-000009.pt')

    def test_graph_best_uses_current_stage_and_physical_failure(self):
        rows=[dict(step=i, **{'evaluation/success':0.,'evaluation/physical_failure':0.,
              'evaluation/grasp':1.,'evaluation/closest_distance_m':.02,
              'evaluation/graph_score':grade,'evaluation/held_detach':0.})
              for i,grade in enumerate([1.5,3.6,2.])]
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000001.pt')
        rows[1]['evaluation/physical_failure']=1.
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000002.pt')

    def test_api_media_range_and_path_confinement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'rollout.mp4').write_bytes(b'0123456789')
            html=root/'index.html';html.write_text('dashboard')
            d=Dashboard(root,no_render=True)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(d,html))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/api/state') as r:self.assertIsNone(json.load(r)['video'])
                with urlopen(Request(base+'/media/rollout.mp4',headers={'Range':'bytes=2-5'})) as r:
                    self.assertEqual(r.status,206);self.assertEqual(r.read(),b'2345')
                with self.assertRaises(HTTPError) as ctx:urlopen(base+'/media/%2e%2e/rollout.mp4')
                self.assertEqual(ctx.exception.code,404)
                with self.assertRaises(HTTPError) as ctx:urlopen(Request(base+'/media/rollout.mp4',headers={'Range':'bytes=99-100'}))
                self.assertEqual(ctx.exception.code,416)
            finally:
                server.shutdown();server.server_close();thread.join()


class LatestVideoTest(unittest.TestCase):
    def test_later_tied_evaluation_renders_even_when_best_is_initial(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            d=Dashboard(Path(tmp),run='test')
            d._remote_snapshot=lambda:dict(rows=[dict(step=i,**{'evaluation/success':0.,
                'evaluation/physical_failure':0.,'evaluation/grasp':0.,'evaluation/closest_distance_m':.2})
                for i in (0,25)],report=None,failure=None,process_alive=True)
            with patch('training_dashboard.threading.Thread') as thread:
                d.poll()
                self.assertEqual(thread.call_args.kwargs['args'],('checkpoint-000025.pt','checkpoint-000000.pt'))


class AcceptedCheckpointTest(unittest.TestCase):
    def test_rejected_candidate_is_not_displayed_as_accepted(self):
        rows=[dict(step=i, **{'evaluation/success':0.,'evaluation/physical_failure':0.,
            'evaluation/grasp':1.,'evaluation/closest_distance_m':.01,
            'evaluation/completed/position':1.,'evaluation/completed/grip':grip,
            'acceptance/checkpoint':'/run/checkpoint-000000.pt'}) for i,grip in ((0,.9),(1,1.))]
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000000.pt')
