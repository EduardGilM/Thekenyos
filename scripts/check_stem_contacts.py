"""Adversarial native contact checks against a generated grasp/pull scene.

The hand sweeps the stalk; a detached fruit then strikes it. Neither test is a
picking policy. The second case disables the abscission joint at reset only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from treesim.kiwi_material import detachment_force, fruit_stem_angle
import mujoco
import numpy as np


def run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'check_stem_contacts.py').write_text(Path(__file__).read_text())
    (a.output/'scene.xml').write_text(a.scene.read_text())
    results=[]; encoder=None; renderer=None
    try:
        for case in ('hand_stem', 'fruit_stem'):
            m=mujoco.MjModel.from_xml_path(str(a.scene.resolve()));m.opt.timestep=a.timestep
            d=mujoco.MjData(m);d.qpos[0]=-1;d.ctrl[0]=-1;mujoco.mj_forward(m,d)
            flex=m.nflex>0
            if flex:
                fruit_bodies=np.unique(m.flex_vertbodyid)
                joints=np.concatenate([np.arange(m.body_jntadr[b],m.body_jntadr[b]+3) for b in fruit_bodies])
                fruit_q=m.jnt_qposadr[joints].reshape(-1,3);fruit_v=m.jnt_dofadr[joints].reshape(-1,3)
            else:
                fruit=m.body('kiwi').id;fruit_q=m.jnt_qposadr[m.body_jntadr[fruit]]
                fruit_v=m.jnt_dofadr[m.body_jntadr[fruit]]
            stems={m.geom(f'stem_collision_{i}').id for i in range(4)}
            hand={i for i in range(m.ngeom) if m.geom(i).name.startswith('arm_link_') and 'collision' in m.geom(i).name}
            fg=-2 if flex else m.geom('kiwi').id;eq=m.equality('abscission').id;stem_body=m.body('stem_3').id
            target=d.geom_xpos[m.geom('stem_collision_1').id].copy()
            if case=='hand_stem':
                palm=m.geom('arm_link_wr1_collision_0' if a.hand_surface=='palm' else 'arm_link_jaw_collision_2').id
                start=d.mocap_pos[0]+target-d.geom_xpos[palm]+[0,-.14,0]
                d.mocap_pos[0]=start
            else:
                d.eq_active[eq]=False;d.mocap_pos[0]=[0,0,2]
                if flex:
                    shift=target+[.075,0,0]-d.flexvert_xpos.mean(axis=0)
                    d.qpos[fruit_q]+=shift;d.qvel[fruit_v[:,0]]=-.12
                else:
                    d.qpos[fruit_q:fruit_q+3]=target+[.075,0,0]
                    d.qvel[fruit_v]=-.12
            mujoco.mj_forward(m,d)
            apex=int(np.argmax(d.flexvert_xpos[:,2])) if flex else None
            min_volume=1.
            if flex:
                elems=m.flex_elem.reshape(-1,4);tet=d.flexvert_xpos[elems]
                volumes=np.linalg.det(tet[:,1:]-tet[:,:1])
            print(f'Starting {case}, elastic={flex}',flush=True)
            peak=penetration=deflection=0.;contact_steps=0;detached=case=='fruit_stem';force=np.zeros(6)
            touch_time=None
            initial_tip=d.site_xpos[m.site('stem_tip').id].copy();next_frame=0.
            if a.video:
                from PIL import Image,ImageDraw,ImageFont
                font=ImageFont.truetype('DejaVuSans.ttf',22)
                renderer=mujoco.Renderer(m,720,1280)
                cam=mujoco.MjvCamera();cam.lookat[:]=target;cam.distance=.48;cam.azimuth=0;cam.elevation=0
                if encoder is None:
                    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'stem-contacts.mp4')],stdin=subprocess.PIPE)
            for step in range(round(2/a.timestep)):
                if case=='hand_stem':
                    travel=.10*min(d.time,1.8)
                    if touch_time is not None:
                        travel=.10*touch_time + min(.10*(d.time-touch_time),.0005) - max(0.,d.time-touch_time-.05)*.04
                    d.mocap_pos[0]=start+[0,travel,0]
                    rows=np.flatnonzero((d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id[:d.nefc]==eq))
                    if len(rows)==3 and not detached:
                        direction=d.xmat[stem_body].reshape(3,3)[:,2]
                        axis=(d.flexvert_xpos[apex]-d.flexvert_xpos.mean(axis=0)) if flex else d.xmat[fruit].reshape(3,3)[:,2]
                        threshold=detachment_force(fruit_stem_angle(-axis,direction))
                        if d.efc_force[rows]@direction>=threshold:d.eq_active[eq]=False;detached=True
                mujoco.mj_step(m,d)
                touched=False
                for j in range(d.ncon):
                    c=d.contact[j];pair=set(c.geom)
                    if not pair&stems or not (pair&hand if case=='hand_stem' else (0 in c.flex if flex else fg in pair)):continue
                    mujoco.mj_contactForce(m,d,j,force);peak=max(peak,float(force[0]));penetration=max(penetration,-float(c.dist));touched|=force[0]>.001
                if touched and touch_time is None:touch_time=d.time
                contact_steps+=int(touched)
                deflection=max(deflection,float(np.linalg.norm(d.site_xpos[m.site('stem_tip').id]-initial_tip)))
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):raise RuntimeError(f'{case}: numerical failure')
                if flex and step % max(1,round(.01/a.timestep))==0:
                    tet=d.flexvert_xpos[elems];min_volume=min(min_volume,float((np.linalg.det(tet[:,1:]-tet[:,:1])/volumes).min()))
                    if min_volume<=.1:raise RuntimeError(f'{case}: collapsed tissue')
                if step % round(.5/a.timestep)==0:print(f'{case} {d.time:.2f}s peak={peak:.3f} N',flush=True)
                if renderer and d.time>=next_frame:
                    renderer.update_scene(d,cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,1280,85),fill=(18,24,32))
                    draw.text((16,10),f'PHYSICAL STEM CONTACT TEST | {case} | {d.time:.2f}s | {"elastic proxy" if flex else "rigid fruit"}',font=font,fill='white')
                    draw.text((16,44),f'Peak contact {peak:.2f} N | stem displacement {deflection*1000:.2f} mm | scripted fixture',font=font,fill='white')
                    encoder.stdin.write(np.asarray(im).tobytes());next_frame+=1/30
            if renderer:renderer.close();renderer=None
            result=dict(case=case,minimum_volume_ratio=min_volume,peak_contact_N=peak,contact_steps=contact_steps,max_penetration_m=penetration,max_stem_displacement_m=deflection,
                        final_fruit_vx_m_s=float(d.qvel[fruit_v[:,0]].mean() if flex else d.qvel[fruit_v]),stem_geoms_present=len(stems),detached=detached,passed=bool(peak>.01 and contact_steps>0 and deflection>1e-6 and penetration<.001))
            results.append(result)
        report=dict(hand_surface=a.hand_surface,elastic_fruit=flex,timestep_s=a.timestep,cases=results,passed=all(r['passed'] for r in results))
        (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        if not report['passed']:raise RuntimeError('Stem contact check failed')
    finally:
        if renderer:renderer.close()
        if encoder:
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('Video encoding failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timestep',type=float,default=.00002)
    p.add_argument('--hand-surface',choices=('palm','jaw'),default='palm')
    p.add_argument('--video',action='store_true')
    a=p.parse_args()
    if not np.isfinite(a.timestep) or not 0<a.timestep<=.00002:p.error('Timestep must be in (0,20us]')
    run(a)
