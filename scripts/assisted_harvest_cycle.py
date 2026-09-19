"""Fixed-base Spot assisted pick and gravity deposit; not an RL policy.

Reuse the orchard robot/basket geometry and native collidable stalk. Only the
explicit ideal grip is artificial; release disables it before the basket drop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from treesim.native_stem import add_stem
from treesim.kiwi_material import STEM_LENGTH, detachment_force, fruit_stem_angle
from treesim.basket import CENTER, SIZE, WALL

ARM=['arm_sh0','arm_sh1','arm_el0','arm_el1','arm_wr0','arm_wr1']
PREFIX='spot_with_arm_'


def build(a):
    from treesim.harvest_env import SpotHarvestEnv
    import newton
    e=SpotHarvestEnv(a.relic,device='cpu');e.reset(seed=42)
    solver=newton.solvers.SolverMuJoCo(e.sim.model,use_mujoco_cpu=True,save_to_mjcf=str(a.output/'export.xml'))
    root=ET.parse(a.output/'export.xml').getroot();model=e.sim.model
    geoms={g.get('name'):g for g in root.findall('.//geom')}
    for i,shape in enumerate(solver.mjc_geom_to_newton_shape.numpy()[0]):
        if shape>=0:geoms[solver.mj_model.geom(i).name].set('rgba',' '.join(map(str,[*model.shape_color.numpy()[shape],1.])))
    # Freeze only the canopy for this fixture. The arm and fruit remain dynamic.
    for b in root.findall('.//body'):
        if b.get('name','').startswith('seg'):
            for j in list(b.findall('joint')):b.remove(j)
    body=root.find(f'.//body[@name="{PREFIX}body"]');base=np.fromstring(body.get('pos'),sep=' ')
    asset=a.relic.resolve()/'source/relic/relic/assets/spot'
    constants=runpy.run_path(str(asset/'constants.py'))
    # Restore URDF visuals omitted by Newton's physics-only MJCF exporter.
    urdf=ET.parse(asset/'spot_with_arm.urdf').getroot();assets=root.find('asset')
    for link in urdf.findall('link'):
        b=root.find(f'.//body[@name="{PREFIX}{link.get("name")}"]')
        if b is None:continue
        for g in b.findall('geom'):
            g.set('group','0' if link.get('name')=='arm_link_jaw' else '3')
            if link.get('name') in ('arm_link_wr1','arm_link_fngr','arm_link_jaw'):g.set('solref','.002 1')
        color=(.96,.73,.04,1) if link.get('name')=='body' or 'uleg' in link.get('name') or link.get('name') in ('arm_link_sh0','arm_link_sh1') else (.18,.2,.23,1)
        for n,v in enumerate(link.findall('visual')):
            mesh=v.find('geometry/mesh')
            if mesh is None:continue
            name=f'visual_{link.get("name")}_{n}';origin=v.find('origin')
            ET.SubElement(assets,'mesh',name=name,file=str((asset/mesh.get('filename')).resolve()))
            attrs=dict(type='mesh',mesh=name,contype='0',conaffinity='0',group='2',rgba=' '.join(map(str,color)))
            if origin is not None:
                attrs['pos']=origin.get('xyz','0 0 0');r=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_quat();attrs['quat']=' '.join(map(str,r[[3,0,1,2]]))
            ET.SubElement(b,'geom',**attrs)
    # Reuse the exact decorative basket parts, including vents and mounts.
    basket=e.sim.tree.robot_data['basket'];transforms=model.shape_transform.numpy();scales=model.shape_scale.numpy()
    for i in range(basket['first_shape']+5,basket['shape_end']):
        t=transforms[i];attrs=dict(name=f'basket_visual_{i}',pos=' '.join(map(str,t[:3])),quat=' '.join(map(str,t[[6,3,4,5]])),contype='0',conaffinity='0',group='2',rgba=' '.join(map(str,[*model.shape_color.numpy()[i],1.])))
        if model.shape_type.numpy()[i]==newton.GeoType.BOX:attrs.update(type='box',size=' '.join(map(str,scales[i])))
        else:
            mesh=model.shape_source[i];name=f'basket_mesh_{i}'
            mesh_element=ET.SubElement(assets,'mesh',name=name,inertia='shell',vertex=' '.join(map(str,np.asarray(mesh.vertices).ravel())),face=' '.join(map(str,np.asarray(mesh.indices).ravel())))
            attrs.update(type='mesh',mesh=name)
            if getattr(mesh,'texture',None) is not None:
                from PIL import Image
                texture_path=(a.output/f'{name}.png').resolve();Image.fromarray(np.asarray(mesh.texture)).save(texture_path)
                mesh_element.set('texcoord',' '.join(map(str,np.asarray(mesh.uvs).ravel())))
                ET.SubElement(assets,'texture',name=name,type='2d',file=str(texture_path))
                ET.SubElement(assets,'material',name=name,texture=name)
                attrs['material']=name
        ET.SubElement(body,'geom',**attrs)
    # Make the liner visually unobtrusive while retaining its collisions.
    for g in body.findall('geom'):
        if g.get('name','').startswith('basket_floor'):g.set('group','0')
        if g.get('name','').startswith(('basket_floor','basket_liner')):
            # Numerical rigid-liner contact, not a calibrated cushioning model.
            g.set('solref','.0005 1');g.set('solimp','.99 .999 .0001');g.set('priority','1');g.set('condim','6')
    fruit=root.find('.//body[@name="apple0"]');fruit.set('name','kiwi')
    for element in root.iter():
        for key,value in list(element.attrib.items()):
            if value=='apple0':element.set(key,'kiwi')
    fp=np.fromstring(fruit.get('pos'),sep=' ');fp[2]=1.6-STEM_LENGTH-.036;fruit.set('pos',' '.join(map(str,fp)))
    inert=fruit.find('inertial');fruit.remove(inert)
    fg=fruit.find('geom');fg.set('name','kiwi_collision');fg.set('size','.027 .027 .036');fg.set('mass','.1111');fg.set('friction','.44 .005 .0001');fg.set('solref','.002 1')
    fg.set('rgba','.45 .29 .11 1');fg.set('group','0')
    add_stem(root,'kiwi',[0,0,.036],fp+[0,0,.036])
    for g in root.findall('.//geom'):
        if g.get('name','').startswith('stem_collision'):g.set('solref','.002 1')
    # The clamped stalk root shares material with the two fixed canopy canes.
    # Exclude only this mounting seam; stalk/hand and stalk/fruit still collide.
    contact=root.find('contact')
    for cane in ('seg4','seg17'):
        ET.SubElement(contact,'exclude',body1='stem_0',body2=cane)
    for ex in list(contact.findall('exclude')):
        if 'kiwi' in ex.attrib.values():contact.remove(ex)
    wrist=root.find(f'.//body[@name="{PREFIX}arm_link_wr1"]')
    ET.SubElement(wrist,'site',name='grip_frame',size='.001',rgba='0 0 0 0')
    ET.SubElement(fruit,'site',name='fruit_frame',size='.001',rgba='0 0 0 0')
    ET.SubElement(root.find('equality'),'weld',name='grip_assist',site1='grip_frame',site2='fruit_frame',active='false',solref='.0002 1',solimp='.99 .999 .0001')
    option=root.find('option');option.set('timestep',str(a.timestep));option.set('solver','Newton');option.set('iterations','100');option.set('tolerance','1e-9')
    flag=option.find('flag')
    if flag is None:flag=ET.SubElement(option,'flag')
    flag.set('midphase','disable');flag.set('nativeccd','disable')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720');ET.SubElement(visual,'headlight',ambient='.45 .45 .45')
    ET.SubElement(root.find('worldbody'),'light',pos='0 -1 4',dir='0 0 -1')
    actuators=root.find('actuator')
    if actuators is not None:root.remove(actuators)
    actuators=ET.SubElement(root,'actuator')
    for name,home in constants['SPOT_DEFAULT_JOINT_POS'].items():
        j=root.find(f'.//joint[@name="{PREFIX}{name}"]')
        if j is None:continue
        arm=name in ARM or name=='arm_f1x';idx=(ARM+['arm_f1x']).index(name) if arm else 0
        kp=constants['ARM_STIFFNESS'][idx] if arm else 120
        kd=constants['ARM_DAMPING'][idx] if arm else 5
        limit=constants['ARM_EFFORT_LIMIT'][idx] if arm else 90
        if name=='arm_f1x':kp,kd,limit=20,.2,3
        ET.SubElement(actuators,'position',name=name,joint=PREFIX+name,kp=str(kp),kv=str(kd),forcerange=f'{-limit} {limit}',ctrlrange=j.get('range'),ctrllimited='true')
    xml=ET.tostring(root,encoding='unicode');(a.output/'scene.xml').write_text(xml);e.close()
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    for name,home in constants['SPOT_DEFAULT_JOINT_POS'].items():
        j=m.joint(PREFIX+name).id;d.qpos[m.jnt_qposadr[j]]=home;d.ctrl[m.actuator(name).id]=home
    d.ctrl[m.actuator('arm_f1x').id]=-1.;d.qpos[m.jnt_qposadr[m.joint(PREFIX+'arm_f1x').id]]=-1.
    mujoco.mj_forward(m,d)
    return m,d,base,fp


def ik(m,d,pos,rot,seed,offset=None):
    qids=np.array([m.jnt_qposadr[m.joint(PREFIX+n).id] for n in ARM]);ids=[m.joint(PREFIX+n).id for n in ARM];w=m.body(PREFIX+'arm_link_wr1').id
    scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos;scratch.mocap_pos[:]=d.mocap_pos;scratch.mocap_quat[:]=d.mocap_quat
    def residual(q):
        scratch.qpos[qids]=q;mujoco.mj_kinematics(m,scratch)
        actual=Rotation.from_matrix(scratch.xmat[w].reshape(3,3))
        if rot is None:
            return np.r_[scratch.xpos[w]+actual.apply(offset)-pos,.001*(q-seed)]
        return np.r_[scratch.xpos[w]-pos,.25*(rot*actual.inv()).as_rotvec()]
    bounds=m.jnt_range[ids].T
    result=least_squares(residual,np.clip(seed,bounds[0]+1e-5,bounds[1]-1e-5),bounds=bounds,max_nfev=200,gtol=1e-9,ftol=1e-9,xtol=1e-9)
    if np.linalg.norm(residual(result.x))>.001:
        for guess in ([0,-1.9,1,0,.9,1.5], [.5,-2,1,-.5,1.1,1.8], [-.5,-2,1,.5,1.1,1.2]):
            candidate=least_squares(residual,guess,bounds=bounds,max_nfev=300)
            if np.linalg.norm(residual(candidate.x))<np.linalg.norm(residual(result.x)):result=candidate
    if np.linalg.norm(residual(result.x))>.001:raise RuntimeError(f'Unreachable wrist pose {pos}: {residual(result.x)} joints={result.x}')
    return result.x


def inside_basket(center, rotation, base):
    """Full ellipsoid containment; allow one micrometre of solver round-off."""
    rel=np.asarray(center)-(np.asarray(base)+CENTER)
    extents=np.sqrt((np.asarray(rotation).reshape(3,3)**2)@np.array([.027,.027,.036])**2)
    return bool(np.all(np.abs(rel[:2])+extents[:2]<=SIZE[:2]/2-WALL+1e-6)
                and rel[2]-extents[2]>=WALL/2-1e-6
                and rel[2]+extents[2]<=SIZE[2]+1e-6)


def main(a):
    if not np.isfinite(a.timestep) or not 0<a.timestep<=.00002:
        raise ValueError('Use a finite timestep in (0, 20 microseconds]')
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    m,d,base,fruit=build(a)
    qids=np.array([m.jnt_qposadr[m.joint(PREFIX+n).id] for n in ARM])
    aids=np.array([m.actuator(n).id for n in ARM]);dofs=np.array([m.jnt_dofadr[m.joint(PREFIX+n).id] for n in ARM])
    w=m.body(PREFIX+'arm_link_wr1').id;fb=m.body('kiwi').id;fg=m.geom('kiwi_collision').id
    jaw=m.actuator('arm_f1x').id;ab=m.equality('abscission').id;assist=m.equality('grip_assist').id
    stem=m.body('stem_3').id;stemgeoms={m.geom(f'stem_collision_{i}').id for i in range(4)}
    fixed={i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name==PREFIX+'arm_link_jaw' and m.geom_contype[i]}
    moving={i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name==PREFIX+'arm_link_fngr' and m.geom_contype[i]}
    basketgeoms={i for i in range(m.ngeom) if m.geom(i).name.startswith(('basket_floor','basket_liner'))}
    armgeoms={i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(PREFIX+'arm_') and m.geom_contype[i]}
    canopygeoms={i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith('seg')}
    floor={i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
    initial=d.qpos[qids].copy();rot=Rotation.from_euler('x',90,degrees=True)
    offset=np.array([.175,0,-.004]);pick=fruit-rot.apply(offset);pre=pick+[0,0,-.12]
    qpre=ik(m,d,pre,rot,initial);qpick=ik(m,d,pick,rot,qpre)
    # Start the stationary fixture at a clear pregrasp pose, before time zero.
    d.qpos[qids]=qpre;d.ctrl[aids]=qpre;initial=qpre.copy();mujoco.mj_forward(m,d)
    (a.output/'workspace.json').write_text(json.dumps(dict(base_m=base.tolist(),fruit_m=fruit.tolist(),pregrasp=qpre.tolist(),grasp=qpick.tolist()),indent=2))
    # Cartesian stages use IK continuation. Only actuator targets change
    # during physics integration; the half-second pregrasp hold settles joints.
    seed=qpre.copy();target=initial.copy();phase='PREGRASP';phase_start=0.;gripped=detached=released=False
    break_info=None;release_time=None;grip_time=None;hold_since=None;passed=False;error=None
    max_obstacle_overlap=0.;max_overlap=0.;max_stem_overlap=0.;peak_jaw=0.;ground=False;rows=[];next_control=next_sample=next_frame=0.
    loads=np.zeros(2);force=np.zeros(6);load=0.;angle=180.;start_pose=None;pivot=None;grip_local=None;peak_slip=0.
    cam=mujoco.MjvCamera();cam.lookat[:]=base+[0,0,.35];cam.distance=2.7;cam.azimuth=-60;cam.elevation=-18
    closecam=mujoco.MjvCamera();closecam.distance=.48;closecam.azimuth=100;closecam.elevation=-15
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    renderer=mujoco.Renderer(m,720,1280) if a.video else None;encoder=None
    if renderer:
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('DejaVuSans.ttf',22)
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'cycle.mp4')],stdin=subprocess.PIPE)
    def smooth(u):
        u=np.clip(u,0,1);return u*u*(3-2*u)
    def enter(name):
        nonlocal phase,phase_start
        phase=name;phase_start=d.time
        print(f'{d.time:.3f}s {phase}',flush=True)
    try:
        while d.time<26:
            t=d.time;elapsed=t-phase_start
            if t>=next_control:
                if phase=='PREGRASP':
                    target=qpre
                    if elapsed>=.5:enter('INSERT')
                elif phase=='INSERT':
                    pos=pre+(pick-pre)*smooth(elapsed/2);seed=ik(m,d,pos,rot,seed);target=seed
                    if elapsed>=2.5:enter('CLOSE')
                elif phase=='CLOSE':
                    d.ctrl[jaw]=-1+smooth(elapsed/2)
                    if elapsed>=2 and min(loads)>.1:
                        mujoco.mj_kinematics(m,d)
                        hs=m.site('grip_frame').id
                        m.site_pos[hs]=d.xmat[w].reshape(3,3).T@(d.xpos[fb]-d.xpos[w])
                        inv=d.xquat[w].copy();inv[1:]*=-1
                        mujoco.mju_mulQuat(m.site_quat[hs],inv,d.xquat[fb]);m.site_sameframe[hs]=0
                        mujoco.mj_kinematics(m,d)
                        assert np.linalg.norm(d.site_xpos[hs]-d.xpos[fb])<1e-9
                        assert np.allclose(d.site_xmat[hs],d.xmat[fb],atol=1e-9)
                        d.eq_active[assist]=True;gripped=True;grip_time=t;grip_local=m.site_pos[hs].copy()
                        start_pose=d.xpos[w].copy();start_rot=Rotation.from_matrix(d.xmat[w].reshape(3,3))
                        pivot=d.xpos[fb]+.036*d.xmat[fb].reshape(3,3)[:,2];enter('ROTATE')
                    elif elapsed>3:raise RuntimeError(f'No bilateral jaw contact: {loads}')
                elif phase=='ROTATE':
                    r=Rotation.from_rotvec([.8*smooth(elapsed/2),0,0]);pos=pivot+r.apply(start_pose-pivot);rpose=r*start_rot
                    seed=ik(m,d,pos,rpose,seed);target=seed
                    if elapsed>=2.2:
                        pull_pos=pos.copy();pull_rot=rpose;enter('PULL DOWN')
                elif phase=='PULL DOWN':
                    pos=pull_pos+[0,0,-.009*min(elapsed,3)]
                    seed=ik(m,d,pos,pull_rot,seed);target=seed
                    if detached:
                        transfer_start=d.xpos[fb].copy()
                        release_center=base+CENTER+[0,0,SIZE[2]+.095]
                        enter('TRANSFER')
                    elif elapsed>3.2:raise RuntimeError('No force-triggered detachment during vertical pull')
                elif phase=='TRANSFER':
                    u=smooth(elapsed/5)
                    center=(1-u)*transfer_start+u*release_center
                    center[1]+=.30*np.sin(np.pi*u)
                    seed=ik(m,d,center,None,seed,grip_local);target=seed
                    if elapsed>=5.5:enter('OPEN')
                elif phase=='OPEN':
                    d.ctrl[jaw]=-smooth(elapsed)
                    if elapsed>=1.2:
                        d.eq_active[assist]=False;released=True;release_time=t;enter('SETTLE')
                next_control+=.01
            # Feed-forward joint bias offsets gravity; effort limits still apply.
            d.ctrl[aids]=target+d.qfrc_bias[dofs]/m.actuator_gainprm[aids,0]
            mujoco.mj_step(m,d)
            loads[:]=0.;basketcontact=False
            for j in range(d.ncon):
                c=d.contact[j];gs=set(c.geom);overlap=max(0.,-float(c.dist))
                if gs&armgeoms and gs&(basketgeoms|canopygeoms):max_obstacle_overlap=max(max_obstacle_overlap,overlap)
                if gs&stemgeoms:max_stem_overlap=max(max_stem_overlap,overlap)
                if fg not in gs:continue
                max_overlap=max(max_overlap,overlap);mujoco.mj_contactForce(m,d,j,force)
                if gs&fixed:loads[0]+=abs(force[0])
                if gs&moving:loads[1]+=abs(force[0])
                if gs&floor and force[0]>.01:ground=True
                if gs&basketgeoms and force[0]>.01:basketcontact=True
            peak_jaw=max(peak_jaw,float(max(loads)))
            if not detached:
                direction=d.xmat[stem].reshape(3,3)[:,2];axis=d.xmat[fb].reshape(3,3)[:,2]
                angle=fruit_stem_angle(-axis,direction)
                eqrows=np.flatnonzero((d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id[:d.nefc]==ab))
                load=max(0.,float(d.efc_force[eqrows]@direction)) if len(eqrows)==3 else 0.
                if load>=detachment_force(angle):
                    if phase!='PULL DOWN':raise RuntimeError(f'Premature detachment in {phase}: {load} N')
                    d.eq_active[ab]=False;detached=True;break_info=dict(time_s=d.time,force_N=load,angle_deg=angle)
                    print(f'DETACHED {break_info}',flush=True)
            if gripped and not released:
                local=d.xmat[w].reshape(3,3).T@(d.xpos[fb]-d.xpos[w]);peak_slip=max(peak_slip,float(np.linalg.norm(local-grip_local)))
            if released:
                inside=inside_basket(d.xpos[fb],d.xmat[fb],base)
                velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,fb,velocity,0)
                stable=inside and basketcontact and np.linalg.norm(velocity[3:])<.05 and np.linalg.norm(velocity[:3])<1.
                hold_since=(d.time if hold_since is None else hold_since) if stable else None
                passed=bool(hold_since is not None and d.time-hold_since>=.5)
            if ground:raise RuntimeError('Fruit hit the ground')
            if max_obstacle_overlap>.001:raise RuntimeError(f'Arm/obstacle overlap exceeded 1 mm: {max_obstacle_overlap}')
            if max_overlap>.001 or max_stem_overlap>.001:raise RuntimeError(f'Contact overlap exceeded 1 mm: fruit={max_overlap}, stem={max_stem_overlap}')
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(warning.number for warning in d.warning):raise RuntimeError('Numerical failure')
            if released and elapsed>8:break
            if t>=next_sample:
                rows.append(dict(time_s=t,phase=phase,fruit_m=d.xpos[fb].tolist(),wrist_m=d.xpos[w].tolist(),wrist_quat=d.xquat[w].tolist(),stem_N=0. if detached else load,jaws_N=loads.tolist(),assist=bool(d.eq_active[assist]),detached=detached));next_sample+=.05
            if renderer and t>=next_frame:
                renderer.update_scene(d,cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE]=0;img=Image.fromarray(renderer.render())
                closecam.lookat[:]=d.xpos[fb];closecam.elevation=-85 if released or phase=='OPEN' else -15
                renderer.update_scene(d,closecam,scene_option=opt)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE]=0
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                inset=Image.fromarray(renderer.render()).resize((480,270));img.paste(inset,(16,430))
                draw=ImageDraw.Draw(img);draw.rectangle((0,0,1280,78),fill=(18,24,32))
                draw.text((16,10),f'ASSISTED PICK + DEPOSIT | FIXED BASE | RIGID FRUIT | {t:.2f}s',font=font,fill='white')
                draw.text((16,42),f'{phase} | ideal grip: {bool(d.eq_active[assist])} | detached: {detached}',font=font,fill='white');encoder.stdin.write(np.asarray(img).tobytes());next_frame+=1/30
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';print(error,flush=True)
    finally:
        if encoder:encoder.stdin.close();encoder.wait();renderer.close()
    if not passed and error is None:error='Fruit did not remain settled in the basket at the end of the observation period'
    report=dict(passed=bool(passed and error is None and not ground and detached and released and not d.eq_active[assist] and not d.eq_active[ab]),error=error,timestep_s=a.timestep,mujoco_version=mujoco.__version__,source_sha256=source_hash,time_s=d.time,detachment=break_info,grip_activation_s=grip_time,release_s=release_time,ground_contact=ground,max_arm_obstacle_overlap_m=max_obstacle_overlap,max_fruit_overlap_m=max_overlap,max_stem_overlap_m=max_stem_overlap,peak_jaw_force_N=peak_jaw,max_assisted_slip_m=peak_slip,final_fruit_m=d.xpos[fb].tolist(),scope='Fixed base, rigid fruit, ideal grip. No grasp, tissue safety, RL or balance validation.',samples=rows)
    np.savez(a.output/'final-state.npz',qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,eq_active=d.eq_active,mocap_pos=d.mocap_pos,mocap_quat=d.mocap_quat)
    (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--relic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timestep',type=float,default=.00002);p.add_argument('--video',action='store_true')
    main(p.parse_args())
