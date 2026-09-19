"""Stylised showreel clips of the scripted Spot gripper bench.

Scripted motions only: no perception, no learned policy. Rigid surrogate fruit
on native MuJoCo CPU. Reuses scene() from check_spot_gripper and dresses the
XML up (skybox, checker floor, key/fill lights, shadows). Honest on-frame tags.
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from check_spot_gripper import scene
from treesim.native_kiwi import flesh_mass

W, H, FPS = 1280, 720, 30
TAG = 'SCRIPTED | NATIVE MUJOCO CPU | RIGID SURROGATE FRUIT'
POCKET = np.array([.195, -.005, .24])
BANNER = (16, 22, 30)
FONT_PATH = '/usr/share/fonts/liberation/LiberationSans-Bold.ttf'


def dress(root, gravity=None, floor_size='1 1 .01'):
    """Add skybox, floor texture, extra lights and render quality to scene() XML."""
    if gravity is not None:
        root.find('option').set('gravity', gravity)
    root.find('size').set('memory', '256M')
    assets = root.find('asset')
    ET.SubElement(assets, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='.05 .12 .14', rgb2='.004 .006 .01', width='512', height='512')
    ET.SubElement(assets, 'texture', name='floor_tex', type='2d', builtin='checker',
                  rgb1='.09 .11 .13', rgb2='.15 .18 .20', width='512', height='512',
                  mark='edge', markrgb='.3 .42 .42')
    ET.SubElement(assets, 'material', name='floor_mat', texture='floor_tex',
                  reflectance='.2', texrepeat='4 4')
    visual = root.find('visual')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    visual.find('headlight').set('ambient', '.35 .35 .35')
    visual.find('headlight').set('diffuse', '.6 .6 .6')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', name='key', pos='1.2 -1.0 2.4', dir='-.6 .5 -1',
                  directional='true', castshadow='true', diffuse='.9 .9 .85')
    ET.SubElement(world, 'light', name='fill', pos='-1.0 1.0 1.4', dir='.5 -.5 -1',
                  castshadow='false', diffuse='.45 .32 .22')
    floor = world.find("geom[@name='floor']")
    floor.set('material', 'floor_mat')
    floor.set('size', floor_size)
    for kiwi in world.findall("body[@name='kiwi']"):
        for g in kiwi.findall('geom'):
            g.set('rgba', '.48 .34 .17 1')
    return root


def add_kiwi(world, name, pos, quat=(1, 0, 0, 0), mass=None):
    body = ET.SubElement(world, 'body', name=name,
                         pos=' '.join(f'{v:.6f}' for v in pos),
                         quat=' '.join(str(v) for v in quat))
    ET.SubElement(body, 'freejoint')
    ET.SubElement(body, 'geom', name=name, type='ellipsoid', size='.027 .027 .036',
                  mass=str(mass if mass is not None else flesh_mass(9)),
                  rgba='.48 .34 .17 1')
    return body


def remove_kiwi(root):
    world = root.find('worldbody')
    for kiwi in world.findall("body[@name='kiwi']"):
        world.remove(kiwi)


def orbit_cam(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def smoothstep(u):
    u = min(max(u, 0.), 1.)
    return u * u * (3 - 2 * u)


class Encoder:
    def __init__(self, path, fps=None):
        fps = fps or FPS
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '18',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
            stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, img, repeat=1):
        raw = np.asarray(img).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(raw)
            self.frames += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError('ffmpeg encode failed')


def overlay(img, title, line2, font, font2):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 74), fill=BANNER)
    draw.text((16, 8), title, font=font, fill='white')
    draw.text((16, 42), line2, font=font2, fill=(200, 215, 220))
    return img


def jaw_geom_sets(m):
    fixed = {i for i in range(m.ngeom) if 'arm_link_jaw_collision' in (m.geom(i).name or '')}
    moving = {i for i in range(m.ngeom) if 'arm_link_fngr_collision' in (m.geom(i).name or '')}
    return fixed, moving


def jaw_loads(m, d, fruit_geoms, fixed, moving):
    loads = np.zeros(2)
    force = np.zeros(6)
    floor_id = m.geom('floor').id
    fruit_on_floor = False
    for k in range(d.ncon):
        c = d.contact[k]
        gs = set(c.geom)
        if gs & fruit_geoms:
            mujoco.mj_contactForce(m, d, k, force)
            if gs & fixed:
                loads[0] += abs(force[0])
            if gs & moving:
                loads[1] += abs(force[0])
            if floor_id in gs and force[0] > .01:
                fruit_on_floor = True
    return loads, fruit_on_floor


def render_clip(name, xml, sim_s, dt, on_step, camera_fn, outdir, tag=TAG,
                slowmo=1, metrics_extra=None):
    """Generic loop: on_step(m,d,t,ctx)->phase str; camera_fn(t,m,d,ctx)->(lookat,dist,az,el)."""
    from PIL import Image, ImageFont
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    j = m.joint('jaw')
    d.qpos[m.jnt_qposadr[j.id]] = -1.
    d.ctrl[0] = -1.
    mujoco.mj_forward(m, d)
    fixed, moving = jaw_geom_sets(m)
    fruit_geoms = {m.geom(n).id for n in
                   [m.geom(i).name for i in range(m.ngeom)] if n and n.startswith('kiwi')}
    for g in fixed:
        m.geom_group[g] = 0
    opts = mujoco.MjvOption()
    opts.geomgroup[3] = 0
    renderer = mujoco.Renderer(m, H, W)
    font = ImageFont.truetype(FONT_PATH, 24)
    font2 = ImageFont.truetype(FONT_PATH, 20)
    enc = Encoder(outdir / f'{name}.mp4')
    ctx = {}
    peak = np.zeros(2)
    first_floor = None
    next_frame = 0.
    next_progress = 0.
    start_wall = time.time()
    try:
        for _ in range(round(sim_s / dt)):
            t = d.time
            phase = on_step(m, d, t, ctx)
            mujoco.mj_step(m, d)
            loads, on_floor = jaw_loads(m, d, fruit_geoms, fixed, moving)
            peak = np.maximum(peak, loads)
            if on_floor and first_floor is None:
                first_floor = t
            if t >= next_frame:
                lookat, dist, az, el = camera_fn(t, m, d, ctx)
                renderer.update_scene(d, orbit_cam(lookat, dist, az, el), scene_option=opts)
                img = overlay(Image.fromarray(renderer.render()),
                              f'{name} — {tag}', f'{phase} | {t:.2f} s | jaw loads {loads[0]:.1f}/{loads[1]:.1f} N',
                              font, font2)
                enc.write(img, repeat=slowmo)
                next_frame += 1 / FPS
            if t >= next_progress:
                print(f'  {name}: {t:.1f} s {phase}', flush=True)
                next_progress += 1.
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() \
                    or any(w.number for w in d.warning):
                raise RuntimeError(f'{name}: non-finite state or MuJoCo warning at t={t:.3f}')
    finally:
        enc.close()
        renderer.close()
    metrics = dict(clip=name, tag=tag, simulated_s=sim_s, timestep_s=dt, slowmo=slowmo,
                   frames=enc.frames, peak_jaw_force_N=peak.tolist(),
                   first_floor_contact_s=first_floor,
                   wall_s=round(time.time() - start_wall, 1))
    if metrics_extra:
        metrics.update(metrics_extra)
    (outdir / f'{name}.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def base_root(args, torque=.3, gravity=None, floor_size='.4 .4 .01'):
    asset = args.relic.resolve() / 'source/relic/relic/assets/spot'
    root = ET.fromstring(scene(asset, args.timestep, rigid=True, torque=torque))
    return dress(root, gravity=gravity, floor_size=floor_size)


# ---------------- clips ----------------

def clip_orbit_grasp(args, out):
    root = base_root(args, torque=1.)
    xml = ET.tostring(root, encoding='unicode')

    def step(m, d, t, ctx):
        if t < .4:
            target, phase = -1., 'OPEN'
        elif t < 1.2:
            target, phase = -1. + smoothstep((t - .4) / .8), 'CLOSE'
        elif t < 5.5:
            target, phase = 0., 'HOLD' if t >= 1.8 else 'SETTLE'
        else:
            target, phase = -smoothstep((t - 5.5) / .5), 'RELEASE'
        d.ctrl[0] = target
        body = m.body('kiwi').id
        d.xfrc_applied[body, 2] = -9.81 * m.body_mass[body] if t >= 1.8 else 0.
        return phase

    def cam(t, m, d, ctx):
        az = 90. + 360. * t / 7.
        dist = .55 + (.30 - .55) * smoothstep(t / 3) + (.45 - .30) * smoothstep((t - 3.5) / 3.5)
        el = -20. + 10. * math.sin(2 * math.pi * t / 3.5)
        return d.xpos[m_body(m, 'kiwi')], dist, az, el

    return render_clip('orbit_grasp', xml, 7., args.timestep, step, cam, out)


def m_body(m, name):
    return m.body(name).id


def clip_wrist_waltz(args, out):
    root = base_root(args, torque=1.)
    root.find("worldbody/body[@name='wrist']").set('mocap', 'true')
    xml = ET.tostring(root, encoding='unicode')
    base_q = np.array([.70710678, .70710678, 0., 0.])
    base_p = np.array([0., 0., .24])

    def step(m, d, t, ctx):
        d.ctrl[0] = -1. + smoothstep(t / 1.5) if t < 1.5 else 0.
        body = m.body('kiwi').id
        d.xfrc_applied[body, 2] = -9.81 * m.body_mass[body] if t >= 1.8 else 0.
        u = max(0., t - 2.5)
        w = 2 * math.pi / 5.  # gentle figure-8: 5 s period
        off = np.array([.12 * math.sin(w * u), .06 * math.sin(w * u),
                        .12 * math.sin(2 * w * u)])
        roll = -2 * math.pi * min(u / 7.5, 1.)
        qr = np.array([math.cos(roll / 2), 0., 0., math.sin(roll / 2)])  # wxyz, local z
        d.mocap_pos[0] = base_p + off
        d.mocap_quat[0] = quat_mul(base_q, qr)
        return 'WALTZ' if u > 0 else ('CLOSE' if t < 1.5 else 'HOLD')

    def cam(t, m, d, ctx):
        wid = m_body(m, 'wrist')
        wrist = d.xpos[wid]
        fruit = d.xpos[m_body(m, 'kiwi')]
        local = d.xmat[wid].reshape(3, 3).T @ (fruit - wrist)
        if 'ref' not in ctx:
            ctx['ref'] = local.copy()
        ctx['slip'] = max(ctx.get('slip', 0.), float(np.linalg.norm(local - ctx['ref'])))
        sep = float(np.linalg.norm(fruit - wrist))
        lookat = wrist if sep > .15 else wrist + .5 * (fruit - wrist)
        return lookat, .5, 100. + 15. * t, -18.

    # slip recorded via ctx in camera fn (called each rendered frame)
    ctx_holder = {}
    orig_cam = cam

    def cam_wrap(t, m, d, ctx):
        ctx_holder.update(ctx)
        return orig_cam(t, m, d, ctx)

    metrics = render_clip('wrist_waltz', xml, 10., args.timestep, step, cam_wrap, out)
    metrics['max_fruit_slip_wrist_frame_m'] = ctx_holder.get('slip', 0.)
    (out / 'wrist_waltz.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def clip_kiwi_rain(args, out):
    root = base_root(args, torque=1., gravity='0 0 -9.81', floor_size='1 1 .01')
    remove_kiwi(root)
    rng = random.Random(args.seed)
    world = root.find('worldbody')
    xs = [POCKET[0] + d for d in (-.09, 0., .09)]
    ys = [POCKET[1] + d for d in (-.09, 0., .09)]
    n = 0
    for layer in range(4):
        z = .35 + layer * (.6 / 3)
        for x in xs:
            for y in ys:
                q = rng.uniform(0, 2 * math.pi)
                quat = (math.cos(q / 2), 0, 0, math.sin(q / 2))
                add_kiwi(world, f'kiwi_r{n}',
                         (x + rng.uniform(-.02, .02), y + rng.uniform(-.02, .02),
                          z + rng.uniform(-.02, .02)), quat)
                n += 1
    xml = ET.tostring(root, encoding='unicode')

    def step(m, d, t, ctx):
        d.ctrl[0] = -1. if t < 4. else -1. + smoothstep((t - 4.) / .8)
        return 'RAIN' if t < 4. else 'JAW CLOSE'

    def cam(t, m, d, ctx):
        return POCKET + [0, 0, .08], .55, 80. + 8. * t, -8.

    return render_clip('kiwi_rain', xml, 8., args.timestep, step, cam, out)


def clip_zero_g(args, out):
    root = base_root(args, torque=1., gravity='0 0 0')
    remove_kiwi(root)
    root.find("worldbody/body[@name='wrist']").set('mocap', 'true')
    rng = np.random.default_rng(args.seed)
    world = root.find('worldbody')
    # Star kiwi: starts .30 m along +y of pocket, drifts -y at .10 m/s -> pocket at t=3 s.
    add_kiwi(world, 'kiwi_star', POCKET + [0, .30, 0])
    for i in range(9):
        p = POCKET + rng.uniform(-.35, .35, 3)
        p[2] = np.clip(p[2], .08, .8)
        q = rng.uniform(0, 2 * math.pi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        add_kiwi(world, f'kiwi_d{i}', p,
                 (math.cos(q / 2), *(axis * math.sin(q / 2))))
    xml = ET.tostring(root, encoding='unicode')
    base_q = np.array([.70710678, .70710678, 0., 0.])
    base_p = np.array([0., 0., .24])
    seeded = {'done': False}

    def seed_vel(m, d):
        names = [f'kiwi_d{i}' for i in range(9)] + ['kiwi_star']
        for n in names:
            j = m.body_jntadr[m.body(n).id]
            a = m.jnt_dofadr[j]
            if n == 'kiwi_star':
                d.qvel[a:a + 3] = [0, -.10, 0]
                d.qvel[a + 3:a + 6] = [0, 0, .5]
            else:
                v = rng.uniform(-1, 1, 3)
                v *= rng.uniform(.05, .15) / np.linalg.norm(v)
                d.qvel[a:a + 3] = v
                w = rng.uniform(-1, 1, 3)
                w *= rng.uniform(.5, 3.) / np.linalg.norm(w)
                d.qvel[a + 3:a + 6] = w

    def step(m, d, t, ctx):
        if not seeded['done']:
            seed_vel(m, d)
            seeded['done'] = True
        d.ctrl[0] = -1. if t < 3. else min(0., -1. + (t - 3.) / .25)
        u = max(0., t - 4.5)
        spin = math.pi * min(u / 3.5, 1.)
        qs = np.array([math.cos(spin / 2), 0., 0., math.sin(spin / 2)])
        d.mocap_quat[0] = quat_mul(qs, base_q)
        d.mocap_pos[0] = base_p
        return 'DRIFT' if t < 3. else ('SNAP' if t < 4.5 else 'SPIN')

    def cam(t, m, d, ctx):
        return POCKET, .6 - .2 * smoothstep(t / 8.), 110. + 4. * t, -15.

    return render_clip('zero_g_catch', xml, 8., args.timestep, step, cam, out,
                       tag=TAG + ' | ZERO GRAVITY')


def clip_macro_teeth(args, out):
    root = base_root(args, torque=1.)
    xml = ET.tostring(root, encoding='unicode')

    def step(m, d, t, ctx):
        d.ctrl[0] = -1. + smoothstep(t / 3.)
        return 'SLOW CLOSE (2x SLOW MOTION)'

    def cam(t, m, d, ctx):
        center = d.xpos[m_body(m, 'kiwi')]
        finger = d.xpos[m_body(m, 'finger')]
        direction = finger - center
        direction /= np.linalg.norm(direction)
        return center + .027 * direction, .16, 70. + 60. * t / 3., 10.

    return render_clip('macro_teeth', xml, 3., args.timestep, step, cam, out, slowmo=2)


CLIPS = dict(orbit_grasp=clip_orbit_grasp, wrist_waltz=clip_wrist_waltz,
             kiwi_rain=clip_kiwi_rain, zero_g_catch=clip_zero_g,
             macro_teeth=clip_macro_teeth)


# ---------------- post-processing ----------------

def ff(cmd, timeout=None):
    print('  $', ' '.join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'ffmpeg failed: {r.stderr[-400:]}')


def post(args, out):
    cut = out / 'cutouts'
    cut.mkdir(parents=True, exist_ok=True)
    made = []

    src = out / 'orbit_grasp.mp4'
    if src.exists():
        dst = cut / 'orbit_grasp_slowmo.mp4'
        try:
            ff(['ffmpeg', '-y', '-loglevel', 'error', '-ss', '1.0', '-t', '1.2', '-i', src,
                '-vf', 'setpts=2.857*PTS,minterpolate=fps=60:mi_mode=mci,fps=30',
                '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst], timeout=60)
        except (subprocess.TimeoutExpired, RuntimeError):
            ff(['ffmpeg', '-y', '-loglevel', 'error', '-ss', '1.0', '-t', '1.2', '-i', src,
                '-vf', 'setpts=2.857*PTS,fps=30', '-c:v', 'libx264', '-crf', '20',
                '-pix_fmt', 'yuv420p', dst])
        made.append(dst)

    src = out / 'kiwi_rain.mp4'
    if src.exists():
        dst = cut / 'kiwi_rain_reverse.mp4'
        ff(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-vf', 'reverse',
            '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst])
        made.append(dst)

    src = out / 'wrist_waltz.mp4'
    if src.exists():
        dst = cut / 'wrist_waltz_kaleido.mp4'
        f = ('[0:v]crop=1280:610:0:110,split=4[c0][c1][c2][c3];'
             '[c0]hflip[tl];[c1]vflip[tr];[tl][tr]hstack[top];'
             '[c2]vflip[bl];[c3]hflip[br];[bl][br]hstack[bot];'
             '[top][bot]vstack,scale=1280:720')
        ff(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-filter_complex', f,
            '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst])
        made.append(dst)

    src = out / 'zero_g_catch.mp4'
    if src.exists():
        dst = cut / 'zero_g_hue.mp4'
        ff(['ffmpeg', '-y', '-loglevel', 'error', '-i', src,
            '-vf', 'hue=h=t*60,vignette', '-c:v', 'libx264', '-crf', '20',
            '-pix_fmt', 'yuv420p', dst])
        made.append(dst)

    src = out / 'macro_teeth.mp4'
    if src.exists():
        dst = cut / 'macro_edges.mp4'
        ff(['ffmpeg', '-y', '-loglevel', 'error', '-i', src,
            '-vf', 'edgedetect=mode=colormix:high=.1', '-c:v', 'libx264', '-crf', '20',
            '-pix_fmt', 'yuv420p', dst])
        made.append(dst)

    src = out / 'stem_snap.mp4'
    mets = out / 'stem_snap.json'
    if src.exists():
        dst = cut / 'stem_snap_freeze.mp4'
        bt = 3.5
        detached = True
        if mets.exists():
            mj = json.loads(mets.read_text())
            detached = bool(mj.get('detached'))
            bt = mj.get('break_time_s') or bt
        if not detached:
            print('  stem_snap: no detachment — skipping freeze cutout')
        else:
            still = cut / '_freeze_frame.png'
            ff(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{bt:.2f}', '-i', src,
                '-frames:v', '1', still])
            ff(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-i', still,
                '-filter_complex',
                f'[0:v]trim=0:{bt:.2f},setpts=PTS-STARTPTS[main];'
                "[1:v]zoompan=z='min(1.3,1+0.3*on/44)':d=45:"
                'x=iw/2-(iw/zoom/2):y=ih/2-(ih/zoom/2):s=1280x720:fps=30[frz];'
                '[main][frz]concat=n=2:v=1,fps=30',
                '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst])
            made.append(dst)
    return made


def title_card(text, path, dur=1.5):
    esc = text.replace(':', '\\:').replace("'", "\\'")
    ff(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i',
        f'color=c=0x101820:s=1280x720:d={dur}:r=30',
        '-vf', f"drawtext=fontfile={FONT_PATH}:text='{esc}':fontcolor=white:"
               'fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2',
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', path])


def assemble(out):
    cut = out / 'cutouts'
    stem_title = 'STEM SNAP — scripted pull'
    sj = out / 'stem_snap.json'
    if sj.exists() and not json.loads(sj.read_text()).get('detached'):
        stem_title = 'Stem pull — scripted, no detachment reached'
    order = [
        ('KIWI GRIPPER SHOWREEL — scripted, CPU MuJoCo, rigid surrogate fruit', None),
        ('ORBIT GRASP', out / 'orbit_grasp.mp4'),
        ('ORBIT GRASP — slow motion', cut / 'orbit_grasp_slowmo.mp4'),
        ('MACRO TEETH', out / 'macro_teeth.mp4'),
        ('MACRO TEETH — edges', cut / 'macro_edges.mp4'),
        (stem_title, out / 'stem_snap.mp4'),
        ('STEM SNAP — freeze', cut / 'stem_snap_freeze.mp4'),
        ('DEFORMABLE PROXY GRIP — uncalibrated', out / 'deformable_grip.mp4'),
        ('WRIST WALTZ', out / 'wrist_waltz.mp4'),
        ('WRIST WALTZ — kaleidoscope', cut / 'wrist_waltz_kaleido.mp4'),
        ('ZERO-G CATCH', out / 'zero_g_catch.mp4'),
        ('ZERO-G — hue cycle', cut / 'zero_g_hue.mp4'),
        ('KIWI RAIN', out / 'kiwi_rain.mp4'),
        ('KIWI RAIN — reversed', cut / 'kiwi_rain_reverse.mp4'),
        ('CANOPY FLYTHROUGH — Newton CPU', out / 'canopy_flythrough.mp4'),
        ('No perception, no learned policy. Scripted motions only.', None),
    ]
    tmp = out / '_concat'
    tmp.mkdir(exist_ok=True)
    entries = []
    i = 0
    for title, clip in order:
        card = tmp / f'card{i:02d}.mp4'
        title_card(title, card)
        entries.append(card)
        if clip is not None and clip.exists():
            entries.append(clip)
        i += 1
    lst = tmp / 'list.txt'
    lst.write_text(''.join(f"file '{e.resolve()}'\n" for e in entries))
    ff(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
        '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', '30',
        out / 'showreel.mp4'])
    manifest = []
    for e in entries:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'format=duration', '-of', 'csv=p=0', str(e)],
                           capture_output=True, text=True)
        manifest.append(dict(file=str(e.relative_to(out)),
                             duration_s=round(float(r.stdout.strip()), 2),
                             source='title card' if 'card' in e.name else 'rendered clip'))
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', str(out / 'showreel.mp4')],
                       capture_output=True, text=True)
    manifest.append(dict(file='showreel.mp4', duration_s=round(float(r.stdout.strip()), 2),
                         source='assembled'))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    global FPS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('output/showreel'))
    p.add_argument('--clips', nargs='+', default=list(CLIPS), choices=list(CLIPS))
    p.add_argument('--timestep', type=float, default=.0005)
    p.add_argument('--post', action='store_true')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--fps', type=int, default=FPS)
    a = p.parse_args()
    FPS = a.fps
    a.output.mkdir(parents=True, exist_ok=True)
    for name in a.clips:
        print(f'== {name} ==', flush=True)
        CLIPS[name](a, a.output)
    if a.post:
        print('== post ==', flush=True)
        post(a, a.output)
        assemble(a.output)


if __name__ == '__main__':
    main()
