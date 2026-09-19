"""Paper-style orientation fixture: ramp axial load at 60, 120 and 180 degrees.

This is a mechanics regression, not a robot policy. The test fixture constrains
rotation and lateral translation, as in the paper's holder; zero gravity tares
fruit weight until release; after release only gravity acts. It uses the SAME KiwiField kernel as the orchard simulation.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import newton
import warp as wp
from treesim.config import TreeConfig
from treesim.kiwi import KiwiField
from treesim.kiwi_material import STEM_LENGTH, detachment_force

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--timestep', type=float, default=.001)
p.add_argument('--video', type=Path)
p.add_argument('--output', type=Path, default=Path('output/detachment-angles.json'))
a = p.parse_args()
steps = round(.02/a.timestep) if a.timestep > 0 else 0
if steps < 2 or steps % 2 or not np.isclose(steps*a.timestep,.02):
    p.error('timestep must divide .02 into a positive even number of steps')
b = newton.ModelBuilder(); b.gravity = 0.
parents, fruits, angles = [], [], [60.,120.,180.]
for index, angle in enumerate(angles):
    x = .25*(index-1)
    anchor = wp.transform(wp.vec3(x,0,.25+STEM_LENGTH),wp.quat_identity())
    parent = b.add_link(xform=anchor, mass=1., inertia=wp.mat33(.001,0,0,0,.001,0,0,0,.001))
    j = b.add_joint_fixed(parent=-1,child=parent,parent_xform=anchor)
    b.add_articulation([j])
    b.add_shape_box(parent,hx=.02,hy=.02,hz=.015,cfg=b.ShapeConfig(density=0.,has_shape_collision=False),color=(.4,.4,.4))
    q = wp.quat_from_axis_angle(wp.vec3(0,1,0),float(np.radians(180-angle)))
    pos = wp.vec3(x,0,.25)-wp.quat_rotate(q,wp.vec3(0,0,.03249))
    pose = wp.transform(pos,q)
    fruit = b.add_link(xform=pose)
    volume = 4*np.pi*.025*.027*.03249/3
    b.add_shape_ellipsoid(fruit,rx=.025,ry=.027,rz=.03249,
        cfg=b.ShapeConfig(density=float(.1/volume)),color=(.48,.31,.12))
    axis = wp.quat_rotate(wp.quat_inverse(q),wp.vec3(0,0,1))
    j = b.add_joint_d6(parent=-1,child=fruit,parent_xform=pose,
        linear_axes=[b.JointDofConfig(axis=axis)])
    b.add_articulation([j])
    parents.append(parent); fruits.append(fruit)
m = b.finalize()
def states():
    pair = m.state(),m.state()
    for s in pair: newton.eval_fk(m,m.joint_q,m.joint_qd,s)
    return pair
cfg = TreeConfig(); cfg.physics.gravity = 0.
tree = SimpleNamespace(model=m,config=cfg,state_pair=states,apple_data=dict(
    apple_body=fruits,parent_body=parents,offset=np.zeros((3,3)),
    hang_drop=np.full(3,STEM_LENGTH+.03249),detach_force=np.full(3,36.5)))
field = KiwiField(tree,a.timestep)
s0,s1 = states()
solver = newton.solvers.SolverMuJoCo(m,disable_contacts=True,solver=1)
ctrl,contacts = m.control(),m.contacts()
ext = wp.zeros(m.body_count,dtype=wp.spatial_vector)
@wp.kernel
def release_load(fruits: wp.array(dtype=int), broken: wp.array(dtype=int),
                 mass: wp.array(dtype=float), forces: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if broken[i] != 0:
        body = fruits[i]
        # Remove the pull on the SAME physics substep that breaks the stem.
        # The fixture tares weight while attached; release restores gravity.
        forces[body] = wp.spatial_vector(wp.vec3(0., 0., -9.81*mass[body]), wp.vec3(0.))

def step():
    global s0,s1
    for _ in range(steps):
        s0.clear_forces(); wp.copy(s0.body_f,ext)
        field.apply(s0)
        wp.launch(release_load, dim=3, inputs=[field.apple_body, field._flag, m.body_mass, s0.body_f])
        solver.step(s0,s1,ctrl,contacts,a.timestep)
        s0,s1=s1,s0
with wp.ScopedCapture() as cap: step()
viewer = encoder = None
if a.video:
    import newton.viewer as V
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    a.video.parent.mkdir(parents=True,exist_ok=True)
    viewer=V.ViewerGL(headless=True); viewer.set_model(m)
    viewer.set_camera(pos=wp.vec3(0.,-1.05,.38),yaw=90.,pitch=-7.)
release_speeds = [None]*3
try:
    for frame in range(260):
        t=frame*.02
        f=np.zeros((m.body_count,6),np.float32)
        for i,body in enumerate(fruits):
            if not field.detached[i]: f[body,2]=-10*t
        ext.assign(f)
        wp.capture_launch(cap.graph)
        field.update(s0)
        poses = s0.body_q.numpy()
        if not np.isfinite(poses).all(): raise RuntimeError('Nonfinite state')
        for i,body in enumerate(fruits):
            if field.detached[i] and release_speeds[i] is None:
                release_speeds[i] = float(np.linalg.norm(s0.body_qd.numpy()[body,:3]))
                assert release_speeds[i] < .25, 'Pull launched fruit after break'

        if viewer and frame%2==0:
            viewer.begin_frame(t); viewer.log_state(s0)
            starts, ends = [], []
            for i,body in enumerate(fruits):
                if not field.detached[i]:
                    starts.append(poses[parents[i],:3])
                    ends.append(poses[body,:3]+np.asarray(wp.quat_rotate(wp.quat(*poses[body,3:]),wp.vec3(0,0,.03249))))
            viewer.log_lines('attached stems', wp.array(starts,dtype=wp.vec3) if starts else None,
                             wp.array(ends,dtype=wp.vec3) if ends else None,
                             colors=(.35,.8,.3), width=.003, hidden=not starts)
            viewer.end_frame()
            pixels=viewer.get_frame().numpy(); h,w=pixels.shape[:2]
            canvas = Image.fromarray(pixels)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0,0,w,135), fill=(20,25,34))
            draw.text((25,12), 'STEM TEST: how much pull breaks each attachment?', font=font, fill='white')
            draw.text((25,47), f'4x slow motion | simulation time {t:.2f} s | pull rises 10 N/s', font=font, fill='white')
            draw.text((25,82), 'Green = stem. Rotation held by test fixture. No robot or gripper.', font=font, fill='white')
            for i,angle in enumerate(angles):
                x = 25+i*(w//3)
                broken = field.detached[i]
                load = field.break_load.numpy()[i] if broken else field._tension.numpy()[i]
                draw.rectangle((x-8,h-132,x+w//3-25,h), fill=(20,25,34))
                draw.text((x,h-122), f'{angle:.0f} degrees', font=font, fill='white')
                draw.text((x,h-87), f'{"BROKE AT" if broken else "PULL"}: {load:.2f} N', font=font, fill=(255,180,75) if broken else (110,240,130))
                draw.text((x,h-52), 'Pull OFF; gravity only' if broken else 'Still attached', font=font, fill='white')
            pixels=np.asarray(canvas)
            if encoder is None:
                encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                    '-s',f'{w}x{h}','-r','6.25','-i','-', '-vf', 'scale=1280:-2', '-r','25',
                    '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(a.video)],stdin=subprocess.PIPE)
            encoder.stdin.write(pixels.tobytes())
finally:
    if encoder:
        encoder.stdin.close()
        if encoder.wait(): raise RuntimeError('Video encoding failed')
results=[]
for i,angle in enumerate(angles):
    measured=float(field.break_load.numpy()[i]); expected=float(detachment_force(angle))
    assert field.detached[i], (angle,'did not detach')
    assert abs(measured-expected) < .5, (angle,measured,expected)
    results.append(dict(fsa_deg=angle,expected_N=expected,measured_break_load_N=measured,release_speed_m_s=release_speeds[i]))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(timestep_s=a.timestep,cases=results,passed=True),indent=2)+'\n')
print(a.output.read_text())
