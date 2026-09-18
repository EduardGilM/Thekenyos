"""Check fixed-base actions, seeded reset, substep outcomes and Gymnasium API.

Optional video shows scripted joint actions, not a harvesting policy.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from treesim.harvest_env import SpotHarvestEnv
from treesim.harvest_task import TaskDefinition

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--relic',required=True,type=Path)
p.add_argument('--video',type=Path)
p.add_argument('--device', default=None)
p.add_argument('--physics-hz',type=int,default=1000)
p.add_argument('--output',type=Path,default=Path('output/harvest-env-check.json'))
a=p.parse_args()
e=gym.make('Thekenyos/SpotHarvest-v0', relic=a.relic,physics_hz=a.physics_hz,device=a.device,
           render_mode='rgb_array' if a.video else None).unwrapped
check_env(e,skip_render_check=True)
o,_=e.reset(seed=42)
for action in ([0]*6, [2]*7, [np.nan]*7):
    try: e.step(action)
    except ValueError: pass
    else: raise AssertionError('Invalid action accepted')
assert e.sim.sim_time==0
initial={k:v.copy() for k,v in o.items()}
first=e.step(np.ones(7)*.1)
o,_=e.reset(seed=42)
for k in o: np.testing.assert_array_equal(o[k],initial[k])
assert e.work==0 and not e.oracle.grasped and not e.sim.apples.detached.any()
second=e.step(np.ones(7)*.1)
# GPU solve/reduction order is nondeterministic. Keep pose/velocity bounds
# explicit: 10 micrometres, 1e-4 quaternion components, 1 mm/s or mrad/s.
repeat_drift = {}
for key in first[0]:
    tolerance = 1e-3 if key.endswith('velocity') else 1e-4 if key.endswith('quaternion') else 1e-5
    repeat_drift[key] = float(np.max(np.abs(first[0][key]-second[0][key])))
    np.testing.assert_allclose(first[0][key],second[0][key],atol=tolerance,rtol=0)
assert abs(first[1]-second[1])<1e-6
o,_=e.reset(seed=42)
base=e.base_pose.copy(); start_joint=o['joint_position'].copy()
encoder=None
from treesim.builder import _qrot
max_anchor_gap = 0.
max_tip_sag = 0.
try:
    for i in range(100):
        # Exercise shoulder yaw and wrist without attempting a pick.
        action=np.zeros(7,np.float32); action[0]=.4 if i<50 else -.4
        action[5]=.2*np.sin(i/20)
        o,r,done,cut,info=e.step(action)
        assert np.isfinite(r) and not cut
        poses = e.sim.body_q_np(); tree = e.sim.tree
        for seg in tree.skeleton:
            body = tree.seg_to_body[seg.index]
            tip = poses[body,:3] + _qrot(poses[body,3:], np.array([0.,0.,seg.length]))
            if seg.parent >= 0:
                parent = tree.skeleton[seg.parent]; pb = tree.seg_to_body[parent.index]
                anchor = poses[pb,:3] + _qrot(poses[pb,3:], np.array([0.,0.,parent.length]))
                max_anchor_gap = max(max_anchor_gap, float(np.linalg.norm(poses[body,:3]-anchor)))
            if seg.order == 2:
                max_tip_sag = max(max_tip_sag, float(seg.end[2]-tip[2]))
            if seg.order < 2 or seg.supported:
                np.testing.assert_allclose(tip, seg.end, atol=1e-4)
        assert max_anchor_gap < 1e-4, max_anchor_gap
        assert max_tip_sag < .05, max_tip_sag
        assert e.physical.attached

        np.testing.assert_allclose(e.sim.body_q_np()[e.observer.chassis],base,atol=1e-6)
        assert np.all(e.controller.targets[12:]>=e.lower-1e-6)
        assert np.all(e.controller.targets[12:]<=e.upper+1e-6)
        if a.video:
            from PIL import Image,ImageDraw,ImageFont
            frame=e.render(); h,w=frame.shape[:2]
            canvas=Image.fromarray(frame); draw=ImageDraw.Draw(canvas)
            font=ImageFont.truetype('DejaVuSans.ttf',24)
            draw.rectangle((0,0,w,95),fill=(18,24,32))
            draw.text((20,12),'FIXED-BASE ACTION TEST - scripted, not a trained picker',font=font,fill='white')
            draw.text((20,48),f't={e.sim.sim_time:.2f}s | rigid fruit | 2x slow motion | reward {r:.4f}',font=font,fill='white')
            if encoder is None:
                a.video.parent.mkdir(parents=True,exist_ok=True)
                encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i','-','-vf','scale=1280:-2','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.video)],stdin=subprocess.PIPE)
            encoder.stdin.write(np.asarray(canvas).tobytes())
        assert not done, info
    assert abs(o['joint_position'][0]-start_joint[0])>.005 or e.work>.001
    work=e.work
    # Check native solver body poses agree with Newton forward kinematics.
    import newton
    fk=e.sim.model.state()
    newton.eval_fk(e.sim.model,e.sim.state_0.joint_q,e.sim.state_0.joint_qd,fk)
    ids=np.arange(e.sim.model.body_count)
    kinematic_error=float(np.max(np.abs(fk.body_q.numpy()[ids,:3]-e.sim.body_q_np()[ids,:3])))
    assert kinematic_error<1e-4, kinematic_error
finally:
    if encoder:
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError('ffmpeg failed')
    e.close()

# Integration fault injection: a real external load, not a fake detached flag.
e=SpotHarvestEnv(a.relic,task=TaskDefinition(guidance_weight=0.),physics_hz=a.physics_hz,device=a.device)
e.reset(seed=42)
e.sim.set_external_force(e.observer.body,force=(30,0,-50))
for i in range(150):
    _,_,done,_,info=e.step(np.zeros(7))
    if not info['physical']['attached']: e.sim.clear_external_forces()
    if done: break
assert done and not info['success'] and info['outcome'] in ('dropped','damage_limit','forbidden_collision'), info
assert not info['physical']['attached']
try: e.step(np.zeros(7))
except RuntimeError: pass
else: raise AssertionError('Stepped a terminal episode')
e.reset(seed=42)
assert e.physical.attached and e.physical.damage==0 and e.work==0
e.close()

# Outcome fixtures: detach through physical load, then INITIALIZE the detached
# fruit at a known drop/basket location. Neither fixture is a robot pick.
from treesim.basket import CENTER, WALL
from treesim.harvest_task import rotation
fixture_outcomes = {}
for destination in ('ground', 'basket'):
    e=SpotHarvestEnv(a.relic,task=TaskDefinition(guidance_weight=0.),physics_hz=a.physics_hz,device=a.device)
    e.reset(seed=42)
    e.sim.set_external_force(e.observer.body,force=(0,0,-50))
    e.step(np.zeros(7))
    assert not e.physical.attached and not e.done
    e.sim.clear_external_forces()
    sim=e.sim
    joint=list(sim.model.joint_child.numpy()).index(e.observer.body)
    qidx=int(sim.model.joint_q_start.numpy()[joint])
    base=sim.body_q_np()[e.observer.chassis]
    local = CENTER+[0,0,WALL/2+e.observer.radii[2]+.001] if destination=='basket' else np.array([0,1.2,.3])
    position=base[:3]+rotation(base[3:])@local
    for state in (sim.state_0,sim.state_1):
        q=state.joint_q.numpy(); q[qidx:qidx+3]=position; q[qidx+3:qidx+7]=[0,0,0,1]
        state.joint_q.assign(q); state.joint_qd.zero_()
        newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
    sim._host_step+=1
    for _ in range(100):
        _,_,done,_,fixture_info=e.step(np.zeros(7))
        if done: break
    expected='success' if destination=='basket' else 'dropped'
    assert done and fixture_info['outcome']==expected, fixture_info
    fixture_outcomes[destination]=fixture_info['outcome']
    e.close()

# Inject one transient overload in the real observer stream; ensure the wrapper
# evaluates every physics step, rather than losing it at the policy boundary.
e=SpotHarvestEnv(a.relic,physics_hz=a.physics_hz,device=a.device); e.reset(seed=42)
observe=e.observer.observe
samples=[0]
def spike(work):
    samples[0]+=1
    obs=observe(work)
    return replace(obs,jaw_forces_N=(16.,0.)) if samples[0]==3 else obs
e.observer.observe=spike
_,_,done,_,spike_info=e.step(np.zeros(7))
assert done and spike_info['outcome']=='jaw_overload' and samples[0]==3
assert spike_info['elapsed_s']<.02
e.close()
e=SpotHarvestEnv(a.relic,task=TaskDefinition(time_limit_s=.03),physics_hz=a.physics_hz,device=a.device); e.reset(seed=42)
e.step(np.zeros(7)); _,_,done,cut,timeout_info=e.step(np.zeros(7))
assert done and not cut and timeout_info['outcome']=='timeout'
e.close()
metrics=dict(passed=True,gymnasium_check=True,seeded_reset=True,seeded_step_drift=repeat_drift,fixed_base=True,
             actuator_work_J=work,all_body_fk_error_m=kinematic_error, max_canopy_anchor_gap_m=max_anchor_gap, max_tip_sag_m=max_tip_sag,forced_loss=info['outcome'],
             transient_overload_detected_at_s=spike_info['elapsed_s'],physics_hz=a.physics_hz,device=a.device,
             timeout_is_task_failure=True,fixture_outcomes=fixture_outcomes,policy_trained=False,fruit_model='rigid surrogate')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2))
