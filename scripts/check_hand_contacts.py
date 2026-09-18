"""Native CPU collision coverage for every Spot hand collision mesh.

Place the free fruit in each mesh for a static collision query (never step these
penetrating fixtures). Compare contact discovery with independent intersections.
An optional archived policy replay measures physical penetration and fruit motion;
it does not train and cannot validate tissue or bruising parameters.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from trimesh.triangles import closest_point
from treesim.harvest_env import SpotHarvestEnv


def geometric_overlap(m, d, fruit, geom):
    """Independent ellipsoid/convex-mesh intersection, without CCD warm starts."""
    if np.linalg.norm(d.geom_xpos[geom]-d.geom_xpos[fruit]) > m.geom_rbound[geom]+m.geom_rbound[fruit]:
        return False
    mesh = m.geom_dataid[geom]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    vertices = m.mesh_vert[start:start+count]
    world = vertices @ d.geom_xmat[geom].reshape(3,3).T + d.geom_xpos[geom]
    normalized = ((world-d.geom_xpos[fruit]) @ d.geom_xmat[fruit].reshape(3,3)) / m.geom_size[fruit]
    hull = ConvexHull(normalized)
    if np.max(hull.equations[:,3]) <= 0: return True
    triangles = normalized[hull.simplices]
    nearest = closest_point(triangles, np.zeros((len(triangles),3)))
    return bool(np.min(np.linalg.norm(nearest,axis=1)) < 1.-1e-5)


def coverage(env):
    sim = env.sim
    m, d = sim.solver.mj_model, sim.solver.mj_data
    mapping = sim.solver.mjc_geom_to_newton_shape.numpy()[0]
    bodies = sim.model.shape_body.numpy()
    fruit = next(g for g, s in enumerate(mapping) if s >= 0 and bodies[s] == env.observer.body)
    hand = [g for g, s in enumerate(mapping) if s >= 0 and bodies[s] >= 0 and
            sim.model.body_label[bodies[s]].rsplit('/', 1)[-1] in
            ('arm_link_wr1', 'arm_link_jaw', 'arm_link_fngr')]
    assert len(hand) == 8, 'Pinned Spot asset must provide all eight hand collision meshes'
    adr = m.jnt_qposadr[m.body_jntadr[m.geom_bodyid[fruit]]]
    qpos = d.qpos.copy()
    mujoco.mj_forward(m, d)
    centers = d.geom_xpos[hand].copy()
    rows = []
    for geom, center in zip(hand, centers):
        d.qpos[:] = qpos
        d.qpos[adr:adr+3] = center + [.005, .003, .001]
        d.qpos[adr+3:adr+7] = [1., 0., 0., 0.]
        mujoco.mj_forward(m, d)
        expected = {g for g in hand if geometric_overlap(m, d, fruit, g)}
        found = {int(g) for c in d.contact if fruit in c.geom for g in c.geom if g != fruit}
        rows.append(dict(probe_geom=int(geom), overlapping_geoms=sorted(expected),
                         missed_geoms=sorted(expected-found)))
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relic', required=True, type=Path)
    parser.add_argument('--physics-hz', type=int, default=1000, choices=(1000, 2000))
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/hand-contacts.json'))
    a = parser.parse_args()
    env = SpotHarvestEnv(a.relic, device='cpu', physics_hz=a.physics_hz)
    env.reset(seed=1000)
    m = env.sim.solver.mj_model
    corrected = coverage(env)
    assert all(x['probe_geom'] in x['overlapping_geoms'] and not x['missed_geoms']
               for x in corrected), corrected
    # Preserve evidence that the pinned conversion's midphase misses contacts.
    m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    legacy = coverage(env)
    result = dict(passed=True, physics_hz=a.physics_hz, corrected=corrected,
                  legacy_midphase=legacy, tissue_calibrated=False)
    env.close()
    if a.policy:
        from stable_baselines3 import PPO
        from treesim.reach_grasp_env import ReachGraspEnv
        policy = PPO.load(a.policy, device='cpu')
        replays = {}
        for bypass in (False, True):
            env = ReachGraspEnv(a.relic, device='cpu', guidance_weight=0.)
            env.physics_hz = a.physics_hz
            obs, _ = env.reset(seed=1000)
            sim = env.sim
            if not bypass:
                sim.solver.mj_model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            m, d = sim.solver.mj_model, sim.solver.mj_data
            mapping = sim.solver.mjc_geom_to_newton_shape.numpy()[0]
            bodies = sim.model.shape_body.numpy()
            fruit = next(g for g,s in enumerate(mapping) if s >= 0 and bodies[s] == env.observer.body)
            hand = [g for g,s in enumerate(mapping) if s >= 0 and bodies[s] in
                    (env.observer.tcp_body, *env.observer.pads)]
            if not bypass:
                m.geom_solref[[fruit, *hand]] = [.02, 1.]
                m.geom_solimp[[fruit, *hand]] = [.9, .95, .001, .5, 2.]
            initial = sim.state_0.body_q.numpy()[env.observer.body,:3].copy()
            deepest = 0.; motion = 0.; missed = 0; worst = {}; missed_details = []
            original = sim._cpu_contacts
            # Query a separate data object at the completed step pose. Native
            # step contacts otherwise describe the start of that step.
            query = mujoco.MjData(m)
            def sample():
                nonlocal deepest, motion, missed, worst
                original()
                query.qpos[:] = d.qpos
                mujoco.mj_forward(m, query)
                present = {int(g) for c in query.contact if fruit in c.geom for g in c.geom if g != fruit}
                for g in hand:
                    distance = mujoco.mj_geomDistance(m,query,fruit,g,.1,None)
                    if not geometric_overlap(m,query,fruit,g):
                        continue
                    if -distance > deepest:
                        deepest = -distance
                        worst = dict(time_s=float(d.time), geom=int(g))
                    if g not in present:
                        missed += 1
                        missed_details.append(dict(time_s=float(d.time), geom=int(g), depth_mm=-distance*1000))
                position = sim.state_0.body_q.numpy()[env.observer.body,:3]
                motion = max(motion,float(np.linalg.norm(position-initial)))
            sim._cpu_contacts = sample
            for _ in range(200):
                obs, _, done, _, info = env.step(policy.predict(obs, deterministic=True)[0])
                if done: break
            replays['corrected' if bypass else 'legacy'] = dict(
                maximum_penetration_mm=deepest*1000, maximum_fruit_motion_mm=motion*1000,
                missed_overlapping_pairs=missed, worst=worst, missed_details=missed_details, outcome=info['outcome'], time_s=sim.sim_time)
            env.close()
        assert replays['corrected']['missed_overlapping_pairs'] == 0, replays
        assert replays['corrected']['maximum_penetration_mm'] < 1., replays
        result['archived_policy_replay'] = replays
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
