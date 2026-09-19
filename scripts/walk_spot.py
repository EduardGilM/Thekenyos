#!/usr/bin/env python
"""Walk RELIC's pretrained Spot under the procedural kiwi pergola."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp
import newton.viewer as V
from treesim.config import TreeConfig, preset
from treesim import builder
from treesim.sim import Sim
from treesim.spot import SpotController


def route_command(time, pose):
    """Track an oval inside the bay; this route is scripted, the gait is learned."""
    if time < 1.0:
        return np.zeros(3)
    phase = 0.32 * (time - 1.0)
    reference = np.array([-.65 * math.cos(phase), .95 * math.sin(phase)])
    tangent = np.array([.65 * .32 * math.sin(phase), .95 * .32 * math.cos(phase)])
    velocity = tangent + .8 * (reference - pose[:2])
    x, y, z, w = pose[3:]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    heading = math.atan2(tangent[1], tangent[0])
    error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([np.clip(c*velocity[0]+s*velocity[1], -.4, .4),
                     np.clip(-s*velocity[0]+c*velocity[1], -.3, .3),
                     np.clip(1.8*error, -.7, .7)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--relic', required=True, type=Path)
    p.add_argument('--frames', type=int, default=1500)
    p.add_argument('--substeps', type=int, default=20, help='Physics substeps per 50 Hz control step')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--video', type=Path)
    p.add_argument('--metrics', type=Path, default=Path('output/spot-metrics.json'))
    p.add_argument('--stand', action='store_true')
    p.add_argument('--spill-test', action='store_true', help='Apply a tipping torque at 2 s to verify spills and penalties')
    p.add_argument('--spill-torque', type=float, default=400., help='World X torque in Nm during the deliberate spill test')
    p.add_argument('--basket', action='store_true')
    p.add_argument('--basket-mass', type=float, default=1.2)
    p.add_argument('--payload', type=float, help='Fruit kg, 0..6; omitted with --basket samples uniformly')
    p.add_argument('--payload-seed', type=int, default=0)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--closeup', action='store_true', help='Follow Spot for basket inspection')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--terrain', action='store_true',
                   help='Use seeded procedural terrain, selected by --terrain-kind.')
    p.add_argument('--terrain-kind', choices=['noise', 'orchard'], default='noise',
                   help='noise = continuous noise; orchard = legacy grassed aisles and slope')
    args = p.parse_args()
    if not np.isfinite(args.spill_torque) or args.frames <= 0 or args.substeps <= 0 or args.substeps % 2 or (args.video and args.no_render):
        p.error('Use positive --frames and enable rendering for --video')
    cfg = TreeConfig.compliant('pergola')
    cfg.seed = args.seed
    cfg.robot.enabled, cfg.robot.kind, cfg.robot.relic_path = True, 'spot', str(args.relic)
    cfg.robot.position, cfg.robot.yaw = (-.65, 0.), math.pi / 2
    if args.payload is not None and not args.basket:
        p.error('--payload requires --basket')
    cfg.robot.basket, cfg.robot.basket_mass = args.basket, args.basket_mass
    cfg.robot.payload_seed = args.payload_seed
    cfg.robot.payload_mass = (args.payload if args.payload is not None else
                              float(np.random.default_rng(args.payload_seed).uniform(0, 6))) if args.basket else 0.
    cfg.fruit.enabled, cfg.fruit.max_count = True, 40
    # Dimensions, mass and stem length come from kiwi_material's Hayward envelope.
    cfg.fruit.joint = 'free'
    cfg.fruit.colors = ((0.39, 0.27, 0.12), (0.48, 0.34, 0.17))
    cfg.foliage.set_density(0.6)
    cfg.foliage.min_order_for_leaves = 2
    cfg.foliage.leaf_length, cfg.foliage.leaf_width = .22, .17
    cfg.physics.terrain = args.terrain
    cfg.physics.terrain_kind = args.terrain_kind
    tree = builder.generate_and_build(cfg)
    sim = Sim(tree, fps=50, substeps=args.substeps, collisions=True)
    from treesim.basket import SpillTracker
    spill = SpillTracker(tree.robot_data['basket']) if args.basket else None
    if args.spill_test and not args.basket:
        p.error('--spill-test requires --basket')
    controller = SpotController(sim)
    viewer = V.ViewerNull() if args.no_render else V.ViewerGL(headless=args.headless)
    sim.set_viewer(viewer)
    if not args.no_render:
        if args.terrain:
            viewer.set_camera(pos=wp.vec3(9.5, -12.0, 6.2), yaw=127.87, pitch=-22.)
        else:
            viewer.set_camera(pos=wp.vec3(3.5, -4.5, 2.1), yaw=127.87, pitch=-12.)
    if tree.terrain_params:
        print('[terrain]', {k: tree.terrain_params[k] for k in
                            ('slope_deg', 'ground_noise_m', 'rut_depth_m',
                             'rut_width_m', 'friction', 'min_z_m', 'max_z_m')
                            if k in tree.terrain_params}, flush=True)
    encoder = None
    positions, tilts = [], []
    shape_body = tree.model.shape_body.numpy()
    first_robot = tree.robot_data['body_start']
    environment_contact_frames = 0
    basket_robot_contact_frames = 0
    basket_min_gap = float('inf')
    max_contacts = 0
    try:
        for frame in range(args.frames):
            pose = sim.state_0.body_q.numpy()[tree.robot_data['chassis']]
            command = np.zeros(3) if args.stand else route_command(sim.sim_time, pose)
            controller.update(command)
            if args.spill_test:
                sim.set_external_force(tree.robot_data['chassis'], torque=(args.spill_torque if 2.0 <= sim.sim_time < 2.4 else 0., 0., 0.))
            sim.step()
            if spill:
                penalty = spill.update(sim.state_0.body_q.numpy(), tree.robot_data['chassis'])
                if penalty:
                    print(f'[spill] new={int(-penalty)} lost={int(spill.spilled.sum())} reward={penalty}', flush=True)
            pose = sim.state_0.body_q.numpy()[tree.robot_data['chassis']]
            positions.append(pose[:3].tolist())
            tilt = float(np.arccos(np.clip(1 - 2*(pose[3]**2 + pose[4]**2), -1, 1)))
            tilts.append(tilt)
            if not np.isfinite(pose).all() or (not args.spill_test and (pose[2] < .25 or tilt > 1.0)):
                raise RuntimeError(f'Spot fell at frame {frame}: pose={pose}, tilt={tilt}')
            count = int(sim.contacts.rigid_contact_count.numpy()[0])
            max_contacts = max(count, max_contacts)
            if count > sim.contact_capacity:
                raise RuntimeError(f'Contact capacity exceeded: {count}')
            sa, sb = (sim.contacts.rigid_contact_shape0.numpy()[:count], sim.contacts.rigid_contact_shape1.numpy()[:count])
            a, b = shape_body[sa], shape_body[sb]
            # Newton includes separated candidate contacts. Count only touching
            # surfaces (1 mm tolerance), using body-frame contact points.
            poses = np.vstack([sim.state_0.body_q.numpy(), [0, 0, 0, 0, 0, 0, 1]])
            points = []
            for ids, array in ((a, sim.contacts.rigid_contact_point0), (b, sim.contacts.rigid_contact_point1)):
                transforms = poses[ids]  # body=-1 correctly selects world identity
                v, q = array.numpy()[:count], transforms[:, 3:]
                points.append(transforms[:, :3]+v+2*np.cross(q[:, :3], np.cross(q[:, :3], v)+q[:, 3:]*v))
            normal = sim.contacts.rigid_contact_normal.numpy()[:count]
            gaps = np.sum((points[1]-points[0])*normal, axis=1)
            gaps -= sim.contacts.rigid_contact_margin0.numpy()[:count]+sim.contacts.rigid_contact_margin1.numpy()[:count]
            touching = (gaps <= .001) & (np.linalg.norm(normal, axis=1) > .5) & (sa >= 0) & (sb >= 0)
            basket = tree.robot_data['basket']
            if basket:
                ba = (sa >= basket['first_shape']) & (sa < basket['shape_end'])
                bb = (sb >= basket['first_shape']) & (sb < basket['shape_end'])
                fruit_bodies = basket['fruit_bodies']
                basket_pairs = (ba & (b >= first_robot) & ~np.isin(b, fruit_bodies)) | (bb & (a >= first_robot) & ~np.isin(a, fruit_bodies))
                if np.any(basket_pairs):
                    basket_min_gap = min(basket_min_gap, float(gaps[basket_pairs].min()))
                basket_robot_contact_frames += int(np.any(basket_pairs & touching))
            environment_contact_frames += int(np.any(touching & (((a >= first_robot) & (b >= 0) & (b < first_robot)) |
                                                      ((b >= first_robot) & (a >= 0) & (a < first_robot)))))
            if frame % 50 == 0:
                print(f'[spot] t={sim.sim_time:.1f}s pos={pose[:3]} tilt={math.degrees(tilt):.1f}', flush=True)
            if not args.no_render:
                if args.closeup:
                    eye = pose[:3] + np.array([1.8, -2.2, .60])
                    look = pose[:3] + np.array([0., 0., .18])
                    d = look-eye
                    viewer.set_camera(pos=wp.vec3(*map(float, eye)), yaw=math.degrees(math.atan2(d[1], d[0])),
                                      pitch=math.degrees(math.atan2(d[2], np.linalg.norm(d[:2]))))
                sim.render()
                if args.video and frame % 2 == 0:
                    pixels = viewer.get_frame().numpy()
                    if encoder is None:
                        args.video.parent.mkdir(parents=True, exist_ok=True)
                        h, w = pixels.shape[:2]
                        encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                            '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', '-framerate', '25',
                            '-i', 'pipe:0', '-an', '-vf', 'scale=1280:-2', '-c:v', 'libx264',
                            '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(args.video)], stdin=subprocess.PIPE)
                    encoder.stdin.write(pixels.tobytes())
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('ffmpeg failed')
    path = np.array(positions)
    if not all(np.isfinite(a.numpy()).all() for a in
               (sim.state_0.body_q, sim.state_0.body_qd, sim.state_0.joint_q, sim.state_0.joint_qd)):
        raise RuntimeError('Non-finite simulation state')
    metrics = dict(frames=len(path), seconds=sim.sim_time, start=path[0].tolist(), end=path[-1].tolist(),
                   path_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
                   height_range_m=[float(path[:,2].min()), float(path[:,2].max())],
                   max_tilt_degrees=math.degrees(max(tilts)),
                   xy_bounds_m=[path[:,:2].min(axis=0).tolist(), path[:,:2].max(axis=0).tolist()],
                   robot_plant_contact_frames=environment_contact_frames, max_contacts=max_contacts,
                   policy='RELIC pretrained ONNX, no retraining', route='scripted oval, learned gait', seed=args.seed,
                   basket=tree.robot_data['basket'], basket_robot_contact_frames=basket_robot_contact_frames,
                   spilled_fruit=int(spill.spilled.sum()) if spill else 0,
                   spill_reward_total=spill.total_penalty if spill else 0., spill_test=args.spill_test, spill_torque_Nm=args.spill_torque if args.spill_test else 0.,
                   detached_canopy_fruit=sim.apples.broken_count if sim.apples else 0,
                   fruit_contact_damage=sim.kiwi_damage.metrics() if sim.kiwi_damage else None,
                   basket_min_sampled_gap_m=basket_min_gap if np.isfinite(basket_min_gap) else None,
                   terrain=tree.terrain_params)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
