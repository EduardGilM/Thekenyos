"""Matched pose/torque screen; agreement is not real-fruit calibration."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('output/contact-transfer'))
    p.add_argument('--resume', action='store_true', help='Reuse matching recorded cases, including failures')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 4: p.error('Use 1..4 workers')
    a.output.mkdir(parents=True, exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    fingerprint={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                 ('scripts/check_spot_gripper.py','treesim/native_kiwi.py','treesim/kiwi_material.py')}
    poses = [('center', [0,0,0], 0), ('offset_plus', [4,0,0], 0),
             ('offset_minus', [-4,0,0], 0), ('tilt', [0,0,0], 15)]
    cases = [(name, offset, tilt, rigid, .3, .00002)
             for name,offset,tilt in poses for rigid in (True,False)]
    cases += [('strong', [0,0,0], 0, rigid, 1., .00002) for rigid in (True,False)]
    cases += [('halfstep', [0,0,0], 0, False, .3, .00001),
              ('tilt_halfstep', [0,0,0], 15, False, .3, .00001),
              ('offset_plus_halfstep', [4,0,0], 0, False, .3, .00001)]
    def run(case):
        name, offset, tilt, rigid, torque, dt = case
        key = f'{name}-{"rigid" if rigid else "flex"}'
        out = a.output/key
        out.mkdir(exist_ok=True)
        if a.resume and (out/'metrics.json').exists():
            previous=json.loads((out/'metrics.json').read_text())
            if (previous['offset_mm']==offset and previous['tilt_deg']==tilt
                and previous['timestep_s']==dt and previous['torque_limit_Nm']==torque
                and previous['rigid']==rigid and previous.get('source_sha256')==fingerprint):
                return key,previous
        (out/'metrics.json').unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name('check_spot_gripper.py')),
               '--relic', str(a.relic.resolve()), '--output', str(out),
               '--offset-mm', *map(str,offset), '--tilt-deg', str(tilt),
               '--torque', str(torque), '--timestep', str(dt)]
        if rigid: cmd.append('--rigid')
        with (a.output/f'{key}.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failure = dict(rigid=rigid, execution_returncode=result.returncode, source_sha256=fingerprint,
                           passed_contact_check=False, numerical_success=False,
                           timestep_s=dt, offset_mm=offset, tilt_deg=tilt,
                           torque_limit_Nm=torque)
            (out/'metrics.json').write_text(json.dumps(failure,indent=2)+'\n')
            return key, failure
        metrics = json.loads((out/'metrics.json').read_text())
        metrics['source_sha256']=fingerprint
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(key, metrics['passed_contact_check'], metrics['max_hold_displacement_m'], flush=True)
        return key, metrics
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = dict(pool.map(run, cases))
    comparisons = {}
    for name,_,_ in poses:
        rigid, flex = results[name+'-rigid'], results[name+'-flex']
        valid = not (rigid.get('execution_returncode',0) or flex.get('execution_returncode',0))
        comparison = dict(numerical_success=valid, outcome_agreement=False,
                          force_relative_error=None, displacement_difference_m=None)
        if valid:
            comparison.update(outcome_agreement=rigid['passed_contact_check']==flex['passed_contact_check'],
                force_relative_error=float(np.max(np.abs(np.array(rigid['peak_jaw_force_N'])-flex['peak_jaw_force_N'])/
                                                   np.maximum(flex['peak_jaw_force_N'], .1))),
                displacement_difference_m=abs(rigid['max_hold_displacement_m']-flex['max_hold_displacement_m']))
        comparisons[name]=comparison
    # Engineering screening thresholds, not empirical calibration tolerances.
    accepted = all(v['numerical_success'] and v['outcome_agreement'] and
                   v['force_relative_error']<=.10 and v['displacement_difference_m']<=.005
                   for v in comparisons.values())
    accepted &= any(results[name+'-rigid']['passed_contact_check'] and results[name+'-flex']['passed_contact_check']
                    for name,_,_ in poses)
    summary = dict(cases=results, comparisons=comparisons,
                   scope='Synthetic contact-model comparison; not hardware calibration',
                   transfer_accepted=accepted)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}, indent=2))


if __name__ == '__main__': main()
