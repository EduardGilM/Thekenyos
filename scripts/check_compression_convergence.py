"""Compare native kiwi compression across mesh resolutions and timesteps.

Numerical tolerances are engineering acceptance criteria, not fruit calibration.
Each case runs in a child process so a native failure remains in the report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def compare(coarse, fine, force_tolerance):
    if not (coarse.get('passed') and fine.get('passed')):
        return dict(passed=False, reason='A numerical case failed')
    reference = np.asarray(fine['held_force_per_pad_N'])
    force_error = float(np.max(np.abs(np.asarray(coarse['held_force_per_pad_N'])-reference)/np.maximum(reference, 1e-9)))
    strain_error = abs(coarse['compression_fraction']-fine['compression_fraction'])
    return dict(force_relative_error=force_error, compression_absolute_error=strain_error,
                force_tolerance=force_tolerance, compression_tolerance=.005,
                passed=bool(force_error <= force_tolerance and strain_error <= .005))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/compression-convergence'))
    p.add_argument('--counts',type=int,nargs=2,default=[9,11])
    p.add_argument('--timestep',type=float,default=2e-5)
    p.add_argument('--compression',type=float,default=.03)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--video',action='store_true',help='Record the finer mesh at the smaller timestep')
    a=p.parse_args()
    if not 4 <= a.counts[0] < a.counts[1] or not 0<a.compression<.3 or not 1<=a.workers<=4 or not 0<a.timestep<=1e-4:
        p.error('Use increasing mesh counts >=4, compression in (0,.3), and 1..4 workers')
    a.output.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('kiwi_compression.py').resolve()
    material=script.parents[1]/'treesim/kiwi_material.py'
    native=script.parents[1]/'treesim/native_kiwi.py'
    hashes={str(f.relative_to(script.parents[1])):hashlib.sha256(f.read_bytes()).hexdigest() for f in (script,material,native)}
    cases=[(count,dt) for count in a.counts for dt in (a.timestep,a.timestep/2)]
    def run(case):
        count,dt=case;key=f'count-{count}-dt-{dt:g}';out=a.output/key
        out.mkdir(exist_ok=True)
        # Do not accept a prior metrics file if this execution crashes.
        (out/'metrics.json').unlink(missing_ok=True)
        cmd=[sys.executable,str(script),'--output',str(out),'--count',str(count),
             '--timestep',str(dt),'--compression',str(a.compression)]
        if a.video and case==cases[-1]: cmd+=['--video',str(a.output/'fine-mesh.mp4')]
        with (out/'execution.log').open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        metrics=json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else {}
        metrics.update(execution_returncode=proc.returncode,mesh_count=count,timestep_s=dt)
        metrics['passed']=bool(proc.returncode==0 and metrics.get('passed',False))
        print(key,metrics['passed'],flush=True)
        return key,metrics
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=dict(pool.map(run,cases))
    get=lambda n,dt: results[f'count-{n}-dt-{dt:g}']
    checks={f'timestep-count-{n}':compare(get(n,a.timestep),get(n,a.timestep/2),.02) for n in a.counts}
    checks['mesh-halfstep']=compare(get(a.counts[0],a.timestep/2),get(a.counts[1],a.timestep/2),.10)
    passed=all(x['passed'] for x in checks.values())
    report=dict(numerical_convergence_passed=passed,real_fruit_calibrated=False,
                scope='Homogeneous Xuxiang flesh proxy, ideal pads, zero gravity; no skin, core or plasticity',
                source_sha256=hashes,
                cases=results,comparisons=checks)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
