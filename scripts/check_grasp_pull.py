"""Native scripted grasp/pull fixture; optional explicitly labelled ideal grip.

A prescribed wrist pulls at 9 mm/s. A collidable stalk joins the fruit surface;
its reaction goes to the fixed world anchor. Tissue and damage are uncalibrated.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from check_spot_gripper import scene
from treesim.kiwi_material import STEM_LENGTH, detachment_force, fruit_stem_angle
from treesim.native_stem import add_stem


def run(a):
    root=ET.fromstring(scene(a.relic.resolve()/'source/relic/relic/assets/spot',a.timestep,a.rigid,a.torque,count=9))
    # MPR avoids the native CCD path that has intermittently crashed on flex/mesh.
    # This is a scoped bench workaround, not proof of a MuJoCo root cause.
    flag=root.find('option/flag')
    if flag is None: flag=ET.SubElement(root.find('option'),'flag')
    flag.set('nativeccd','disable')
    for contact in (root.find('default/geom'), root.find('worldbody/flexcomp/contact')):
        if contact is not None: contact.set('solref',f'{a.contact_time} 1')
    motor=root.find('actuator/position');motor.set('kp','20');motor.set('kv','.2')
    wrist=root.find("worldbody/body[@name='wrist']");wrist.set('mocap','true')
    fruit=root.find("worldbody/body[@name='kiwi']" if a.rigid else "worldbody/flexcomp[@name='kiwi']")
    fruit.set('quat','1 0 0 0');fruit.set('pos',f'{a.grasp_x} .004 .24')
    ET.SubElement(root.find('worldbody'),'geom',name='anchor',type='sphere',size='.005',pos=f'{a.grasp_x} .004 {.276+STEM_LENGTH}',contype='0',conaffinity='0',rgba='.2 .7 .2 1')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'check_grasp_pull.py').write_text(Path(__file__).read_text())
    (a.output/'native_stem.py').write_text((Path(__file__).resolve().parents[1]/'treesim/native_stem.py').read_text())
    (a.output/'metrics.json').unlink(missing_ok=True)
    # Compile once to locate the actual flex surface node, then attach the stalk.
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    if a.rigid:
        junction=np.array([a.grasp_x,.004,.276]);fruit_body='kiwi';fruit_anchor=[0,0,.036]
    else:
        apex=int(np.argmax(d.flexvert_xpos[:,2]));body=int(m.flex_vertbodyid[apex])
        junction=d.flexvert_xpos[apex].copy();fruit_body=m.body(body).name
        fruit_anchor=d.xmat[body].reshape(3,3).T@(junction-d.xpos[body])
    add_stem(root,fruit_body,fruit_anchor,junction)
    if a.ideal_grip:
        ET.SubElement(wrist,'site',name='ideal_hand',size='.0001',rgba='0 0 0 0')
        ET.SubElement(fruit,'site',name='ideal_fruit',size='.0001',rgba='0 0 0 0')
        ET.SubElement(root.find('equality'),'weld',name='ideal_grip',site1='ideal_hand',site2='ideal_fruit',active='false',solref='.0002 1',solimp='.99 .999 .0001')
    xml=ET.tostring(root,encoding='unicode');(a.output/'scene.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qpos[0]=-1;d.ctrl[0]=-1;mujoco.mj_forward(m,d)
    ids=np.array([m.body('kiwi').id]) if a.rigid else np.unique(m.flex_vertbodyid)
    stem_bodies=np.array([m.body(f'stem_{i}').id for i in range(4)])
    stem_geoms={m.geom(f'stem_collision_{i}').id for i in range(4)}
    abscission=m.equality('abscission').id
    tip_id=m.site('stem_tip').id
    anchor=junction+[0,0,STEM_LENGTH]
    if not a.rigid:
        rest=d.flexvert_xpos.copy();patch=np.array([apex])
        elems=m.flex_elem.reshape(-1,4);tet=rest[elems];vol0=np.linalg.det(tet[:,1:]-tet[:,:1])/6
    fixed={i for i in range(m.ngeom) if 'arm_link_jaw_collision' in m.geom(i).name}
    moving={i for i in range(m.ngeom) if 'arm_link_fngr_collision' in m.geom(i).name}
    palm={i for i in range(m.ngeom) if 'arm_link_wr1_collision' in m.geom(i).name}
    floor=m.geom('floor').id; fg=m.geom('kiwi').id if a.rigid else -2
    cam=mujoco.MjvCamera();cam.lookat[:]=[.18,0,.24];cam.distance=.48;cam.azimuth=125;cam.elevation=-20
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    for g in fixed:m.geom_group[g]=0
    renderer=mujoco.Renderer(m,720,1280) if a.video else None;encoder=None
    if renderer:
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('DejaVuSans.ttf',22)
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'grasp-pull.mp4')],stdin=subprocess.PIPE)
    detached=False;break_load=break_angle=break_time=None;ground=False;minvol=1.;peak=np.zeros(3);rows=[];next_sample=next_frame=0.;force6=np.zeros(6);max_slip=0.; max_penetration=0.;max_torque=0.; max_hand_penetration=0.; retention_origin=None; peak_stem=0.
    abort_reason=None;assist_time=None;loads=np.zeros(3)
    filtered_load=0.;jaw_target=-1.;peak_stem_hand=0.;peak_stem_fruit=0.;max_attachment_error=0.;max_stem_penetration=0.
    rotation_pivot=None;feedback_roll=0.;angle_hold=0.;pull_start=None;angle_at_pull=None
    duration=a.duration if a.duration is not None else (10. if a.target_fsa is not None else 6.)
    try:
        for step in range(round(duration/a.timestep)):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_flex(m,d)
            t=d.time;u=np.clip(t/2,0,1);d.ctrl[0]=-1+u*u*(3-2*u)
            if a.grip_force is not None:
                jaw_target=float(np.clip(jaw_target+np.clip(.03*(a.grip_force-filtered_load),-.6,.6)*a.timestep,-1.,0.))
                d.ctrl[0]=jaw_target
            if a.ideal_grip and assist_time is None and t>=2 and min(loads[:2])>.1:
                hand=m.body('wrist').id;fb=m.body('kiwi').id;hs=m.site('ideal_hand').id
                m.site_pos[hs]=d.xmat[hand].reshape(3,3).T@(d.xpos[fb]-d.xpos[hand])
                inverse=d.xquat[hand].copy();inverse[1:]*=-1
                mujoco.mju_mulQuat(m.site_quat[hs],inverse,d.xquat[fb])
                m.site_sameframe[hs]=0  # Site pose is now dynamic, not the compiled body frame.
                mujoco.mj_kinematics(m,d)
                assert np.linalg.norm(d.site_xpos[hs]-d.xpos[fb])<1e-9
                assert np.allclose(d.site_xmat[hs],d.xmat[fb],atol=1e-9)
                d.eq_active[m.equality('ideal_grip').id]=True;assist_time=t
                print(f'{t:.3f}s IDEAL GRIP enabled after bilateral jaw contact',flush=True)
            pull=.009*np.clip(t-3,0,2)
            roll=np.deg2rad(a.roll_deg)*np.clip((t-3)/2,0,1)
            d.mocap_quat[0]=[np.cos((np.pi/2+roll)/2),np.sin((np.pi/2+roll)/2),0,0]
            d.mocap_pos[0]=[0,.004*(1-np.cos(roll)),.24-pull-.004*np.sin(roll)]
            phase='CLOSE' if t<2 else ('SETTLE' if t<3 else ('PULL' if t<5 else 'RETAIN'))
            d.xfrc_applied[:]=0;d.qfrc_applied[:]=0;d.xfrc_applied[ids,2]=-9.81*m.body_mass[ids];d.xfrc_applied[stem_bodies,2]=-9.81*m.body_mass[stem_bodies]
            if a.rigid:
                body=ids[0];center=d.xpos[body].copy();axis=d.xmat[body].reshape(3,3)[:,2];site=center+.036*axis
            else:
                center=d.flexvert_xpos.mean(axis=0);site=d.flexvert_xpos[patch].mean(axis=0);axis=site-center;axis/=np.linalg.norm(axis)
            stem_direction=d.xmat[stem_bodies[-1]].reshape(3,3)[:,2]
            angle=fruit_stem_angle(-axis,stem_direction) if not detached else break_angle;threshold=float(detachment_force(angle));load=0.
            if a.target_fsa is not None and t>=2:
                if rotation_pivot is None: rotation_pivot=site.copy()
                if not detached and pull_start is None:
                    error=angle-a.target_fsa
                    feedback_roll=float(np.clip(feedback_roll+np.clip(3*np.deg2rad(error),-a.roll_speed,a.roll_speed)*a.timestep,0,np.pi))
                    angle_hold=angle_hold+a.timestep if abs(error)<=3 else 0.
                    if (a.pull_after is not None and t>=a.pull_after) or (a.pull_after is None and angle_hold>=.1):
                        pull_start=t;angle_at_pull=angle
                roll=feedback_roll
                cosine,sine=np.cos(roll),np.sin(roll)
                rotation=np.array([[1,0,0],[0,cosine,-sine],[0,sine,cosine]])
                pull=0. if pull_start is None else .009*np.clip(t-pull_start,0,3)
                d.mocap_pos[0]=rotation_pivot+rotation@(np.array([0,0,.24])-rotation_pivot)+[0,0,-pull]
                d.mocap_quat[0]=[np.cos((np.pi/2+roll)/2),np.sin((np.pi/2+roll)/2),0,0]
                phase='RETAIN' if detached else ('ROTATE' if pull_start is None else 'PULL VERTICALLY')
            f=np.zeros(3)
            if not detached:
                # body1 is fruit: connect rows are the world-space force on fruit.
                eq_rows=np.flatnonzero((d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id[:d.nefc]==abscission))
                if len(eq_rows)==3: f=d.efc_force[eq_rows].copy()
                load=max(0.,float(f@stem_direction))
                max_attachment_error=max(max_attachment_error,float(np.linalg.norm(site-d.site_xpos[tip_id])))
                if load>=threshold:
                    detached=True;break_load=load;break_angle=angle;break_time=t
                    d.eq_active[abscission]=False
            peak_stem=max(peak_stem,load)
            mujoco.mj_step(m,d)
            loads=np.zeros(3)
            for j in range(d.ncon):
                con=d.contact[j];isfruit=fg in con.geom if a.rigid else 0 in con.flex
                gs=set(con.geom)
                if gs&stem_geoms:
                    max_stem_penetration=max(max_stem_penetration,max(0.,-float(con.dist)))
                    mujoco.mj_contactForce(m,d,j,force6)
                    if gs&(fixed|moving|palm): peak_stem_hand=max(peak_stem_hand,abs(float(force6[0])))
                    if isfruit: peak_stem_fruit=max(peak_stem_fruit,abs(float(force6[0])))
                if not isfruit:continue
                max_penetration=max(max_penetration,max(0.,-float(con.dist)))
                mujoco.mj_contactForce(m,d,j,force6);gs=set(con.geom)
                if gs&(fixed|moving|palm): max_hand_penetration=max(max_hand_penetration,max(0.,-float(con.dist)))
                if gs&fixed:loads[0]+=abs(force6[0])
                if gs&moving:loads[1]+=abs(force6[0])
                if gs&palm:loads[2]+=abs(force6[0])
                if floor in gs and force6[0]>.01:ground=True
            filtered_load+=a.timestep/(.005+a.timestep)*(max(loads)-filtered_load)
            peak=np.maximum(peak,loads);max_torque=max(max_torque,abs(float(d.actuator_force[0])))
            if detached:
                wrist_id=m.body('wrist').id
                local=d.xmat[wrist_id].reshape(3,3).T@(center-d.xpos[wrist_id])
                if retention_origin is None: retention_origin=local.copy()
                max_slip=max(max_slip,float(np.linalg.norm(local-retention_origin)))
            if step % round(1/a.timestep)==0: print(f'{t:.1f}s {phase} stem={load:.2f}N detached={detached}',flush=True)
            if t>=next_sample:
                if not a.rigid:
                    tet=d.flexvert_xpos[elems];minvol=min(minvol,float(np.min(np.linalg.det(tet[:,1:]-tet[:,:1])/6/vol0)))
                    if minvol<=.1:raise RuntimeError('Collapsed tissue element')
                rows.append(dict(wrist_position_m=d.mocap_pos[0].tolist(),wrist_quaternion=d.mocap_quat[0].tolist(),time_s=t,stem_load_N=load,threshold_N=threshold,angle_deg=angle,site_m=site.tolist(),anchor_m=anchor.tolist(),stem_force_N=f.tolist() if not detached else [0.,0.,0.],wrist_roll_deg=float(np.rad2deg(roll)),jaw_force_N=loads[:2].tolist(),palm_force_N=float(loads[2]),detached=detached,center_m=center.tolist()))
                next_sample+=.01
            if renderer and t>=next_frame:
                renderer.update_scene(d,cam,scene_option=opt)
                img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img);draw.rectangle((0,0,1280,105),fill=(18,24,32))
                draw.text((16,10),f'SCRIPTED GRASP / PULL | {"RIGID SURROGATE" if a.rigid else "ELASTIC PROXY"} | {phase} | {t:.2f}s',font=font,fill='white')
                draw.text((16,42),f'Stem {load:.1f}/{threshold:.1f} N | jaws {loads[0]:.1f}/{loads[1]:.1f} N | detached: {detached}',font=font,fill='white')
                draw.text((16,74),f'{"FSA at detachment" if detached else "Fruit-stem angle"} {angle:.1f} deg | wrist roll {np.rad2deg(roll):.1f} deg | {"IDEAL GRIP ASSIST" if a.ideal_grip else "no grasp weld"}',font=font,fill='white');encoder.stdin.write(np.asarray(img).tobytes());next_frame+=1/30
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning):raise RuntimeError('Numerical failure')
            if max_hand_penetration>.001 or max_stem_penetration>.001 or max_attachment_error>.001:
                abort_reason='Stopped: hand/stalk contact penetration or attachment error exceeded 1 mm'
                print(abort_reason,flush=True)
                break
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait(): raise RuntimeError('Video encoding failed')
        if renderer:renderer.close()
    result = dict(
        ideal_grip_assist=a.ideal_grip,ideal_grip_activation_s=assist_time,
        stem_model='four collidable beam segments with breakable fruit connection',
        peak_stem_hand_contact_N=peak_stem_hand,peak_stem_fruit_contact_N=peak_stem_fruit,
        max_attachment_error_m=max_attachment_error,max_stem_contact_penetration_m=max_stem_penetration,
        numerically_completed=abort_reason is None, abort_reason=abort_reason, executed_duration_s=float(d.time), duration_s=duration, target_fsa_deg=a.target_fsa,scheduled_pull_after_s=a.pull_after, angle_at_pull_deg=angle_at_pull, pull_start_s=pull_start,
        grip_force_target_N=a.grip_force,roll_speed_rad_s=a.roll_speed,rigid=a.rigid, grasp_x_m=a.grasp_x, roll_deg=a.roll_deg, contact_time_s=a.contact_time,
        jaw_kp=20., jaw_kv=.2, timestep_s=a.timestep,
        max_contact_penetration_m=max_penetration,
        max_hand_penetration_m=max_hand_penetration,
        torque_limit_Nm=a.torque, peak_actuator_torque_Nm=max_torque,
        detached=detached, break_load_N=break_load, break_angle_deg=break_angle,
        break_angle_in_measured_range=bool(break_angle is not None and 60<=break_angle<=180),
        break_time_s=break_time, ground_contact=ground,
        peak_jaw_force_N=peak[:2].tolist(),peak_palm_force_N=float(peak[2]), peak_stem_load_N=peak_stem,
        retention_slip_m=max_slip if detached else None,
        minimum_volume_ratio=minvol,
        passed=bool(abort_reason is None and (not a.ideal_grip or assist_time is not None) and detached and (pull_start is not None and break_time>=pull_start if a.target_fsa is not None else 3<=break_time<=5) and d.time-break_time>=1 and not ground
                    and max_slip<.02 and max_hand_penetration<.001 and max_stem_penetration<.001 and max_attachment_error<.001),
        scope=('IDEAL GRIP ASSIST; ' if a.ideal_grip else '')+'Scripted fixture; fixed world stem reaction, prescribed wrist; '
              'collidable segmented stalk; mixed-source elasticity/abscission diagnostic, '
              'not calibrated cultivar or damage model',
    )
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'measurements.json').write_text(json.dumps(rows));print(json.dumps(result,indent=2))
    if a.check and not result['passed']: raise RuntimeError('Grasp/pull failed; inspect metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,help='Optional shorter numerical smoke check; does not imply harvest success')
    p.add_argument('--ideal-grip',action='store_true',help='Diagnostic hand-fruit weld after closure; requires rigid fruit, not a physical-grasp validation')
    p.add_argument('--rigid',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--grip-force',type=float,help='Optional force-feedback setpoint in N; not a calibrated damage limit')
    p.add_argument('--roll-speed',type=float,default=1.2,help='Maximum ideal-controller roll speed, rad/s')
    p.add_argument('--grasp-x',type=float,default=.195,help='Fruit centre along the jaw, metres')
    p.add_argument('--pull-after',type=float,help='Start vertical pull at this time and freeze wrist orientation; bypasses the FSA gate')
    p.add_argument('--target-fsa',type=float,help='Privileged feedback target for this scripted ideal test only')
    p.add_argument('--roll-deg',type=float,default=0.)
    p.add_argument('--contact-time',type=float,default=.002)
    p.add_argument('--torque',type=float,default=1.)
    p.add_argument('--timestep',type=float,default=.00002)
    a=p.parse_args()
    if not np.isfinite([a.torque,a.timestep,a.contact_time,a.roll_deg]).all() or abs(a.roll_deg)>90 or not .0001<=a.contact_time<=.01 or not 0<a.torque<=15.32 or not 0<a.timestep<=(.001 if a.rigid else .00002):p.error('Use finite torque in (0,15.32], timestep in (0,20us], contact time in [0.1,10]ms and roll within +/-90deg')
    if a.target_fsa is not None and (not np.isfinite(a.target_fsa) or not 60<=a.target_fsa<=180 or a.roll_deg!=0): p.error('Target FSA must be in [60,180]; do not combine with open-loop roll')
    if not np.isfinite(a.grasp_x) or not .16<=a.grasp_x<=.23:p.error('Grasp x must be in [.16,.23] m')
    if not np.isfinite(a.roll_speed) or not 0<a.roll_speed<=1.2:p.error('Roll speed must be in (0,1.2] rad/s')
    if a.grip_force is not None and (not np.isfinite(a.grip_force) or not 0<a.grip_force<=50):p.error('Grip-force target must be in (0,50] N')
    if a.duration is not None and (not np.isfinite(a.duration) or a.duration<a.timestep):p.error('Duration must be finite and at least one timestep')
    if a.pull_after is not None and (not np.isfinite(a.pull_after) or not 2<a.pull_after< (a.duration or 10)-1 or a.target_fsa is None):p.error('pull-after requires target-fsa and time after closure with at least 1s remaining')
    if a.ideal_grip and not a.rigid:p.error('Ideal-grip diagnostic currently requires --rigid')
    run(a)
