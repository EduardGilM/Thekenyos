import json
from collections import deque
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_dashboard import Dashboard, REMOTE_PROBE, _decode_jsonl_tail, _phase, handler_for, select_checkpoint

class DashboardTest(unittest.TestCase):
    def test_active_pid_and_phase_fallbacks(self):
        self.assertIn("active=read('active-process.json')", REMOTE_PROBE)
        self.assertIn("active.get('pid',1545354)", REMOTE_PROBE)
        self.assertIn("'train_harvest_fast.py' in cmd", REMOTE_PROBE)
        self.assertEqual(_phase({'phase_events': [{'phase': 'ppo'}, {'phase': 'cti-v2'}]}), 'cti-v2')
        self.assertEqual(_phase({'rows': [{'cti/version': 2}]}), 'cti-v2')
        self.assertEqual(_phase({'rows': [{}]}), 'ppo')

    def test_jsonl_tail_keeps_large_records_and_latest_complete_rows(self):
        stream = io.BytesIO((json.dumps({'payload': 'x' * 70000}) + '\n'
                             + '{"round":1}\n{"round":').encode())
        records = _decode_jsonl_tail(deque(stream, maxlen=3))
        self.assertEqual([record.get('round') for record in records], [None, 1])
        self.assertEqual(len(records[0]['payload']), 70000)
        self.assertIn('deque(f,maxlen=limit)', REMOTE_PROBE)

    def test_best_uses_physical_outcomes_and_retains_earliest_tie(self):
        def row(step,success=0,failure=0,grasp=0,distance=.1):
            return {'step':step,'evaluation/success':success,'evaluation/physical_failure':failure,
                    'evaluation/grasp':grasp,'evaluation/closest_distance_m':distance}
        rows=[row(0),row(1,failure=1,grasp=1,distance=.001),row(2),{'step':3,'evaluation/success':1}]
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000000.pt')
        rows.append(row(4,success=.5))
        self.assertEqual(select_checkpoint(rows,None)[0],'checkpoint-000004.pt')
        self.assertEqual(select_checkpoint(rows,{'best_checkpoint':'/run/checkpoint-000009.pt'})[0],'checkpoint-000009.pt')

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
