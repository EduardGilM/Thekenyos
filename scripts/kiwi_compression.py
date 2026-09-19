"""Native MuJoCo compression bench with published Xuxiang flesh stiffness.

Opposing kinematic gripper pads squeeze and release one tetrahedral flex.
This isolates fruit contact; damage is a proxy, not a calibrated bruise model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from treesim.kiwi_material import XUXIANG, damage_increment
from treesim.native_kiwi import flesh_mass, FLESH_DENSITY_KG_M3, FLEX_RADIUS_M, CONTACT_TIME_S

import mujoco
import numpy as np


def scene(timestep=0.00001, young=1.57e6, count=7):
    spacing = [0.054 / (count - 1), 0.054 / (count - 1), 0.072 / (count - 1)]
    return f'''<mujoco model="Deformable kiwi compression bench">
      <option gravity="0 0 0" timestep="{timestep}" integrator="implicitfast" solver="Newton" tolerance="1e-9" iterations="100"/>
      <size memory="64M"/>
      <visual><global offwidth="1280" offheight="720"/><headlight ambient=".4 .4 .4"/></visual>
      <default><geom friction=".5 .005 .0001" solref="{CONTACT_TIME_S} 1" solimp=".95 .99 .001"/></default>
      <worldbody>
        <light pos=".1 -.2 .4" diffuse=".8 .8 .8"/>
        <geom name="table" type="plane" size=".3 .3 .01" rgba=".19 .23 .25 1"/>
        <body name="left" mocap="true" pos="-.045 0 .04">
          <geom name="left_pad" type="box" size=".008 .04 .045" rgba=".12 .15 .17 1"/>
          <geom type="box" pos="-.012 0 0" size=".004 .033 .033" rgba=".95 .65 .05 1" contype="0" conaffinity="0"/>
        </body>
        <body name="right" mocap="true" pos=".045 0 .04">
          <geom name="right_pad" type="box" size=".008 .04 .045" rgba=".12 .15 .17 1"/>
          <geom type="box" pos=".012 0 0" size=".004 .033 .033" rgba=".95 .65 .05 1" contype="0" conaffinity="0"/>
        </body>
        <flexcomp name="kiwi" type="ellipsoid" dim="3" count="{count} {count} {count}" spacing="{' '.join(map(str, spacing))}" pos="0 0 .039" mass="{flesh_mass(count)}" radius=".0003" rgba=".42 .25 .10 1">
          <elasticity young="{young}" poisson=".4" damping=".00001"/>
          <contact selfcollide="none" internal="false" condim="3" friction=".5 .005 .0001" solref="{CONTACT_TIME_S} 1" solimp=".95 .99 .001"/>
        </flexcomp>
      </worldbody>
    </mujoco>'''


def gap_at(t, compression=.10):
    closure = .074 - (.054*(1-compression) + 2*FLEX_RADIUS_M)
    # Smooth motion removes velocity jumps from the force measurement.
    if t < 2:
        return .074, 'settle'
    if t < 5:
        s = (t - 2) / 3
        return .074 - closure * (3*s*s - 2*s*s*s), 'compress'
    if t < 6:
        return .074-closure, 'hold'
    if t < 9:
        s = (t - 6) / 3
        return .074-closure + closure * (3*s*s - 2*s*s*s), 'release'
    return .074, 'recover'


def run(output, timestep=.00001, young=1.57e6, count=7, video=None, compression=.10):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xml = scene(timestep, young, count)
    (output / 'kiwi.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    elements = model.flex_elem.reshape(-1, 4)
    def volumes():
        tetra = data.flexvert_xpos[elements]
        return np.linalg.det(tetra[:, 1:] - tetra[:, :1]) / 6
    initial_volumes = volumes().copy()
    rest = data.flexvert_xpos.copy()
    rest -= rest.mean(axis=0)
    pad_ids = [model.geom(name).id for name in ('left_pad', 'right_pad')]
    renderer = mujoco.Renderer(model, height=720, width=1280) if video else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .032]
    camera.distance, camera.azimuth, camera.elevation = .25, 90, -15
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
    encoder = None
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        labels = ["drawtext=text='XUXIANG FLESH - DAMAGE PROXY':x=32:y=32:fontsize=26:fontcolor=white"]
        for label, start, end in [('SETTLE', 0, 2), ('COMPRESS', 2, 5), ('HOLD', 5, 6), ('RELEASE', 6, 9), ('RECOVER', 9, 12)]:
            labels.append(f"drawtext=text='{label}':x=32:y=70:fontsize=22:fontcolor=white:enable='gte(t,{start})*lt(t,{end})'")
        encoder = subprocess.Popen(['ffmpeg' , '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '30', '-i', '-', '-an', '-vf', ','.join(labels), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    rows, next_frame = [], 0.
    force = np.empty(6)
    min_volume_ratio = 1.
    try:
        for step in range(round(11 / timestep)):
            gap, phase = gap_at(data.time, compression)
            data.mocap_pos[0, 0], data.mocap_pos[1, 0] = -gap/2-.008, gap/2+.008
            mujoco.mj_step(model, data)
            if any(w.number for w in data.warning):
                raise RuntimeError(f'MuJoCo numerical warning at step {step}; reduce timestep')
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Non-finite simulation state')
            if step % max(1, round(.01 / timestep)) == 0:
                forces = np.zeros(2)
                for i in range(data.ncon):
                    contact = data.contact[i]
                    for pad, gid in enumerate(pad_ids):
                        if gid in contact.geom:
                            mujoco.mj_contactForce(model, data, i, force)
                            forces[pad] += max(0., force[0])
                vertices = data.flexvert_xpos
                if phase == 'recover':
                    # Remove free rigid rotation before measuring shape recovery.
                    # World AABBs can change even for an undeformed rotating fruit.
                    centered = vertices-vertices.mean(axis=0)
                    u, _, vt = np.linalg.svd(centered.T @ rest)
                    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
                    vertices = centered @ (u @ correction @ vt)
                dimensions = np.ptp(vertices, axis=0)
                ratio = float(np.min(volumes() / initial_volumes))
                min_volume_ratio = min(min_volume_ratio, ratio)
                rows.append([float(data.time), phase, gap, *dimensions.tolist(), *forces.tolist(), ratio])
            if renderer and data.time + 1e-9 >= next_frame:
                renderer.update_scene(data, camera=camera, scene_option=options)
                encoder.stdin.write(renderer.render().tobytes())
                next_frame += 1/30
    finally:
        if renderer:
            renderer.close()
        if encoder:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoding failed')
    names = ['time_s', 'phase', 'pad_gap_m', 'width_m', 'depth_m', 'height_m', 'left_normal_N', 'right_normal_N', 'min_tetra_volume_ratio']
    with (output / 'measurements.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(rows)
    numeric = np.array([[row[i] for i in (0, 2, 3, 4, 5, 6, 7, 8)] for row in rows])
    baseline = numeric[(numeric[:, 0] > 1.5) & (numeric[:, 0] < 2), 2:5].mean(axis=0)
    final = numeric[numeric[:, 0] > 10.5, 2:5].mean(axis=0)
    held = numeric[(numeric[:, 0] > 5.5) & (numeric[:, 0] < 6)]
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    metrics = {
        'scope': 'Homogeneous Xuxiang flesh elasticity with persistent strain damage proxy; no skin/core layers or plastic constitutive response; zero gravity; ideal pads, not Spot jaws. Damping and damage accumulation uncalibrated.',
        'material_source': 'https://doi.org/10.3390/foods13213523',
        'commanded_compression': compression,
        'contact_time_s': CONTACT_TIME_S, 'contact_radius_m': FLEX_RADIUS_M,
        'mujoco_version': mujoco.__version__, 'timestep_s': timestep,
        'wall_time_s': time.perf_counter() - started,
        'young_modulus_Pa': young, 'poisson_ratio': .4, 'mass_kg': flesh_mass(count),
        'flesh_density_kg_m3': FLESH_DENSITY_KG_M3,
        'reference_mesh_volume_m3': float(np.abs(initial_volumes).sum()),
        'vertices': model.nflexvert, 'tetrahedra': len(elements),
        'baseline_dimensions_m': baseline.tolist(), 'recovered_dimensions_m': final.tolist(),
        'compression_fraction': float(1-held[:, 2].mean()/baseline[0]),
        'held_force_per_pad_N': held[:, 5:7].mean(axis=0).tolist(),
        'peak_force_per_pad_N': numeric[:, 5:7].max(axis=0).tolist(),
        'recovery_relative_error': float(np.max(np.abs(final-baseline)/baseline)),
        'released_force_N': float(numeric[numeric[:, 0] > 10.5, 5:7].max()),
        'minimum_signed_tetra_volume_ratio': min_volume_ratio,
        'warnings': warnings,
    }
    strains = np.maximum(1-numeric[:,2]/baseline[0], 0.)
    damage = 0.
    for strain in strains:
        damage = float(damage_increment(strain, 0., .01, damage))
    metrics['irreversible_strain_damage_proxy'] = damage
    metrics['strain_threshold_exceeded'] = bool(np.max(strains) > .05)
    metrics['passed'] = bool(not warnings and min_volume_ratio > .1 and .005 < metrics['compression_fraction'] < .4 and metrics['recovery_relative_error'] < .02 and metrics['released_force_N'] < .001 and min(metrics['held_force_per_pad_N']) > .05)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    if not metrics['passed']:
        raise RuntimeError('Compression/release numerical acceptance checks failed')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/kiwi-compression')
    parser.add_argument('--video')
    parser.add_argument('--timestep', type=float, default=.00001)
    parser.add_argument('--young', type=float, default=XUXIANG['flesh'].young)
    parser.add_argument('--compression', type=float, default=.10, help='Requested whole-fruit strain, e.g. .03 or .10')
    parser.add_argument('--count', type=int, default=7)
    parser.add_argument('--check-timestep', action='store_true', help='Repeat at half timestep; require force agreement within 2 percent and compression within 0.5 percentage points')
    args = parser.parse_args()
    if not 0 < args.compression < .3 or not 0 < args.timestep <= .001 or not np.isfinite(args.young) or args.young <= 0 or args.count < 4:
        parser.error('Require timestep in (0, .001], positive Young modulus, and count >= 4')
    result = run(args.output, args.timestep, args.young, args.count, args.video, args.compression)
    if args.check_timestep:
        reference = run(Path(args.output) / 'half-step', args.timestep / 2, args.young, args.count, compression=args.compression)
        force_error = float(np.max(np.abs(np.array(result['held_force_per_pad_N']) / np.array(reference['held_force_per_pad_N']) - 1)))
        compression_error = abs(result['compression_fraction'] - reference['compression_fraction'])
        comparison = {'force_relative_error': force_error, 'compression_absolute_error': compression_error,
                      'passed': force_error < .02 and compression_error < .005}
        (Path(args.output) / 'timestep-check.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps(comparison, indent=2))
        if not comparison['passed']:
            raise RuntimeError('Timestep sensitivity exceeds numerical acceptance limits')
