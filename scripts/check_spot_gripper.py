"""Scripted native MuJoCo bench using RELIC's actual Spot gripper meshes.

The wrist is a fixed fixture. Gravity is tared during closure, then a 1 g load
is applied to the free fruit to test retention and release. No grasp constraint.
External assets are referenced, never copied. Tissue response remains a proxy.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from treesim.kiwi_material import XUXIANG, damage_increment
from treesim.native_kiwi import flesh_mass, FLESH_DENSITY_KG_M3, CONTACT_TIME_S


def scene(asset, dt, rigid=False, torque=.3, offset_mm=(0.,0.,0.), tilt_deg=0., count=9):
    from scipy.spatial.transform import Rotation
    position = np.array([.195, -.005, .24]) + np.asarray(offset_mm)/1000
    rotation = Rotation.from_euler('z', tilt_deg, degrees=True) * Rotation.from_euler('x', 90, degrees=True)
    xyzw = rotation.as_quat()
    pose = dict(pos=' '.join(map(str,position)), quat=' '.join(map(str,xyzw[[3,0,1,2]])))
    urdf = ET.parse(asset/'spot_with_arm.urdf').getroot()
    root = ET.fromstring(f'''<mujoco model="Spot jaw contact bench">
      <compiler angle="radian"/>
      <option gravity="0 0 0" timestep="{dt}" integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-9"/>
      <size memory="128M"/>
      <visual><global offwidth="1280" offheight="720"/><headlight ambient=".5 .5 .5"/></visual>
      <default><geom friction=".44 .005 .0001" solref="{CONTACT_TIME_S} 1" solimp=".95 .99 .001"/></default>
      <asset/>
      <worldbody><light pos="0 -.4 1"/>
        <geom name="floor" type="plane" pos="0 0 0" size=".4 .4 .01" rgba=".2 .24 .27 1"/>
        <body name="wrist" pos="0 0 .24" quat=".70710678 .70710678 0 0"/>
      </worldbody>
      <actuator><position name="jaw_motor" joint="jaw" kp="2" kv=".04" forcerange="-{torque} {torque}" ctrllimited="true" ctrlrange="-1.5708 0"/></actuator>
    </mujoco>''')
    assets, world = root.find('asset'), root.find('worldbody')
    wrist = world.find('body')
    for name in ('arm_link_wr1','arm_link_jaw','arm_link_fngr'):
        link = urdf.find(f"link[@name='{name}']")
        body = wrist
        if name == 'arm_link_fngr':
            joint = urdf.find("joint[@name='arm_f1x']")
            body = ET.SubElement(wrist,'body',name='finger',pos=joint.find('origin').get('xyz'))
            limit = joint.find('limit')
            ET.SubElement(body,'joint',name='jaw',type='hinge',axis=joint.find('axis').get('xyz'),range=f"{limit.get('lower')} {limit.get('upper')}",damping='.01',armature='.001')
            inert = link.find('inertial'); inertia = inert.find('inertia')
            ET.SubElement(body,'inertial',pos=inert.find('origin').get('xyz'),mass=inert.find('mass').get('value'),fullinertia=' '.join(inertia.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')))
        for kind in ('visual','collision'):
            for i,geom in enumerate(link.findall(kind)):
                key=f'{name}_{kind}_{i}'
                mesh=geom.find('geometry/mesh')
                ET.SubElement(assets,'mesh',name=key,file=str((asset/mesh.get('filename')).resolve()))
                origin=geom.find('origin')
                if origin is not None and any(float(v) for v in origin.get('rpy','0 0 0').split()):
                    raise ValueError('Rotated URDF geom needs explicit conversion')
                attrs=dict(name=key,type='mesh',mesh=key,pos=origin.get('xyz','0 0 0') if origin is not None else '0 0 0',rgba='.17 .19 .22 1')
                if kind=='visual': attrs.update(contype='0',conaffinity='0',group='2')
                else: attrs.update(group='3',rgba='.3 .32 .35 1')
                ET.SubElement(body,'geom',**attrs)
    if rigid:
        fruit=ET.SubElement(world,'body',name='kiwi',**pose)
        ET.SubElement(fruit,'freejoint')
        ET.SubElement(fruit,'geom',name='kiwi',type='ellipsoid',size='.027 .027 .036',mass=str(flesh_mass(count)),rgba='.45 .29 .11 1')
    else:
        flex=ET.SubElement(world,'flexcomp',name='kiwi',type='ellipsoid',dim='3',count=f'{count} {count} {count}',spacing=' '.join(str(v/(count-1)) for v in (.054,.054,.072)),**pose,mass=str(flesh_mass(count)),radius='.0003',rgba='.45 .29 .11 1')
        ET.SubElement(flex,'elasticity',young=str(XUXIANG['flesh'].young),poisson=str(XUXIANG['flesh'].poisson),damping='.00001')
        ET.SubElement(flex,'contact',selfcollide='none',internal='false',condim='3',friction='.44 .005 .0001',solref=f'{CONTACT_TIME_S} 1',solimp='.95 .99 .001')
    return ET.tostring(root,encoding='unicode')


def run(args):
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    xml=scene(args.relic.resolve()/'source/relic/relic/assets/spot',args.timestep,args.rigid,args.torque,args.offset_mm,args.tilt_deg,args.count)
    (out/'scene.xml').write_text(xml)
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m)
    d.qpos[0]=-1.; d.ctrl[0]=-1.; mujoco.mj_forward(m,d)
    bodies=np.array([m.body('kiwi').id]) if args.rigid else np.unique(m.flex_vertbodyid)
    fruitgeom=m.geom('kiwi').id if args.rigid else -2
    fixed={i for i in range(m.ngeom) if 'arm_link_jaw_collision' in (m.geom(i).name or '')}
    moving={i for i in range(m.ngeom) if 'arm_link_fngr_collision' in (m.geom(i).name or '')}
    def points():
        return d.xpos[bodies] if args.rigid else d.flexvert_xpos
    rest=points().copy(); restcenter=rest.mean(axis=0)
    restwidth=np.ptp(rest[:,1]) if not args.rigid else .072
    centered_rest=rest-restcenter
    if not args.rigid:
        elements=m.flex_elem.reshape(-1,4)
        tetra=rest[elements]
        volumes0=np.linalg.det(tetra[:,1:]-tetra[:,:1])/6
    min_volume_ratio=1.; shape_error=0.; strain=0.; grip_damage=0.; max_grip_strain=0.
    ground_before_release=False; post_release_load=0.; next_shape=0.; next_progress=0.

    renderer=mujoco.Renderer(m,720,1280) if args.video else None
    cam=mujoco.MjvCamera(); cam.lookat[:]=[.17,0,.21]; cam.distance=.42; cam.azimuth=130; cam.elevation=-25
    opts=mujoco.MjvOption(); opts.geomgroup[3]=0
    # Jaw has no visual mesh in the asset, so retain its physical collision geoms.
    for i in fixed: m.geom_group[i]=0
    encoder=None
    if renderer:
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('DejaVuSans.ttf',24)
        args.video.parent.mkdir(parents=True,exist_ok=True)
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','15','-i','-','-r','30','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(args.video)],stdin=subprocess.PIPE)
    hold_samples=bilateral_samples=0
    rows=[]; force=np.zeros(6); damage=0.; peak=np.zeros(2); next_sample=next_frame=0.; held=[]
    try:
        for step in range(round(4/args.timestep)):
            t=d.time
            if t<.2: target=-1.; phase='OPEN - gravity tared'
            elif t<1.2:
                u=(t-.2); target=-1.+u*u*(3-2*u); phase='CLOSE - torque limited'
            elif t<2.8: target=0.; phase='HOLD - gravity on' if t>=1.5 else 'SETTLE'
            else:
                u=min((t-2.8)/.6,1.); target=-u*u*(3-2*u); phase='OPEN AND RELEASE'
            d.ctrl[0]=target
            d.xfrc_applied[bodies,2]=-9.81*m.body_mass[bodies] if t>=1.5 else 0.
            mujoco.mj_step(m,d)
            loads=np.zeros(2); ground=False
            for k in range(d.ncon):
                c=d.contact[k]
                isfruit=fruitgeom in c.geom if args.rigid else 0 in c.flex
                if not isfruit: continue
                mujoco.mj_contactForce(m,d,k,force)
                load=abs(force[0]); geoms=set(c.geom)
                if geoms&fixed: loads[0]+=load
                if geoms&moving: loads[1]+=load
                ground |= m.geom('floor').id in geoms and load>.01
            peak=np.maximum(peak,loads)
            pts=points(); center=pts.mean(axis=0)
            if t>=next_shape and not args.rigid:
                centered=pts-center
                u,_,vt=np.linalg.svd(centered.T@centered_rest)
                aligned=centered@(u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt)
                strain=max(0.,1.-np.ptp(aligned[:,1])/restwidth)
                shape_error=float(np.sqrt(np.mean(np.sum((aligned-centered_rest)**2,axis=1))))
                tetra=pts[elements]
                min_volume_ratio=min(min_volume_ratio,float(np.min(np.linalg.det(tetra[:,1:]-tetra[:,:1])/6/volumes0)))
                if min_volume_ratio<=.1: raise RuntimeError('Collapsed or inverted tissue element')
                next_shape+=.001
            if t<2.8:
                ground_before_release |= ground
                max_grip_strain=max(max_grip_strain,strain)
            if t>3.6: post_release_load=max(post_release_load,float(max(loads)))
            damage=float(damage_increment(strain,0.,args.timestep,damage))
            if t<2.8: grip_damage=damage
            if t>=next_progress:
                print(f'{t:.1f} s: {phase}, forces={loads.round(2)}, compression={strain:.4f}',flush=True)
                next_progress+=.5
            if 1.7<t<2.7:
                held.append(float(np.linalg.norm(center-restcenter)))
                hold_samples += 1
                bilateral_samples += int(min(loads) > .1)
            if t>=next_sample:
                rows.append(dict(time_s=t,phase=phase,jaw_rad=float(d.qpos[0]),torque_Nm=float(d.actuator_force[0]),fixed_force_N=loads[0],moving_force_N=loads[1],center_z_m=center[2],compression_proxy=strain,damage_proxy=damage,shape_rms_error_m=shape_error,ground_contact=ground))
                next_sample+=.01
            if renderer and t>=next_frame:
                renderer.update_scene(d,cam,scene_option=opts)
                img=Image.fromarray(renderer.render()); draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,1280,110),fill=(18,24,32))
                draw.text((20,12),f'SPOT GRIPPER BENCH - scripted | {"RIGID CONTACT" if args.rigid else "DEFORMABLE XUXIANG PROXY"}',font=font,fill='white')
                draw.text((20,46),f'{phase} | {t:.2f} s | jaw loads {loads[0]:.1f} / {loads[1]:.1f} N',font=font,fill='white')
                draw.text((20,78),f'2x slow motion | fixed wrist | torque limit {args.torque:.2f} Nm | no grasp constraint',font=font,fill='white')
                encoder.stdin.write(np.asarray(img).tobytes()); next_frame+=1/30
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(w.number for w in d.warning): raise RuntimeError('Invalid simulation or MuJoCo warning')
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait(): raise RuntimeError('ffmpeg failed')
        if renderer: renderer.close()
    metrics = dict(
        rigid=args.rigid, mass_kg=flesh_mass(args.count), mesh_count=args.count, contact_time_s=CONTACT_TIME_S, flesh_density_kg_m3=FLESH_DENSITY_KG_M3, timestep_s=args.timestep, torque_limit_Nm=args.torque,
        offset_mm=list(args.offset_mm), tilt_deg=args.tilt_deg,
        hold_bilateral_fraction=bilateral_samples/max(hold_samples,1),
        peak_jaw_force_N=peak.tolist(), max_hold_displacement_m=max(held),
        hold_tolerance_m=.02, hold_window_s=[1.7, 2.7],
        final_center_z_m=float(center[2]), damage_proxy=damage,
        grip_damage_proxy=grip_damage, max_grip_compression_proxy=max_grip_strain,
        minimum_tetra_volume_ratio=min_volume_ratio,
        final_shape_rms_error_m=shape_error,
        ground_before_release=bool(ground_before_release),
        post_release_jaw_force_N=post_release_load,
        bilateral_contact=bool(any(r['fixed_force_N']>.1 and r['moving_force_N']>.1
                                   for r in rows if 1.7<r['time_s']<2.7)),
        retained_during_hold=bool(max(held)<.02 and not ground_before_release),
        released_to_floor=bool(any(r['ground_contact'] for r in rows if r['time_s']>3.4)),
        warnings=0,
        scope='Fixed wrist, free fruit, scripted actions; uncalibrated elastic tissue and friction',
    )
    if args.rigid:
        for key in ('damage_proxy', 'grip_damage_proxy', 'max_grip_compression_proxy',
                    'minimum_tetra_volume_ratio', 'final_shape_rms_error_m'):
            metrics[key] = None  # Rigid geometry cannot establish tissue strain/damage.
    metrics['passed_contact_check'] = bool(metrics['bilateral_contact'] and metrics['retained_during_hold']
                                         and metrics['released_to_floor'] and post_release_load<.01)
    with (out/'measurements.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2))
    if args.check and not metrics['passed_contact_check']: raise RuntimeError('Contact/hold/release acceptance failed; inspect metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic',required=True,type=Path); p.add_argument('--output',type=Path,default=Path('output/spot-gripper'))
    p.add_argument('--video',type=Path); p.add_argument('--rigid',action='store_true'); p.add_argument('--check',action='store_true')
    p.add_argument('--offset-mm',type=float,nargs=3,default=(0.,0.,0.))
    p.add_argument('--tilt-deg',type=float,default=0.)
    p.add_argument('--count',type=int,default=9)
    p.add_argument('--timestep',type=float,default=.00002); p.add_argument('--torque',type=float,default=.3)
    a=p.parse_args()
    if a.count < 4: p.error('Mesh count must be >= 4')
    if not np.isfinite([a.timestep,a.torque]).all() or not 0<a.timestep<=.001 or not 0<a.torque<=15.32: p.error('Invalid timestep or jaw torque')
    if not np.isfinite([*a.offset_mm,a.tilt_deg]).all() or max(map(abs,a.offset_mm))>10 or abs(a.tilt_deg)>30:
        p.error('Pose bounds: offsets up to 10 mm, tilt up to 30 degrees')
    run(a)
