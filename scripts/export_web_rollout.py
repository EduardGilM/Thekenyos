"""Export the recorded demo rollout as body poses + mesh data for a web replay."""
import json, sys, numpy as np, mujoco
scene, replay, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = mujoco.MjModel.from_xml_path(scene + '/scene.xml'); d = mujoco.MjData(m)
man = json.load(open(scene + '/manifest.json'))
q = np.load(replay + '/states.npz')['qpos']; rep = json.load(open(replay + '/report.json'))
chassis = m.body(man['robot']['chassis']).id
root = m.body_rootid[chassis]
fruit_bodies = [m.body(f['body']).id for f in man['fruits']]
# bodies to animate: robot tree + fruit trees (fruit body + its subtree) + any body that is a stem/anchor
def subtree(b): return [i for i in range(m.nbody) if m.body_rootid[i]==m.body_rootid[b] and (i==b or _desc(i,b))]
def _desc(i,b):
    while i>0:
        i=m.body_parentid[i]
        if i==b: return True
    return False
bodies = set(i for i in range(m.nbody) if m.body_rootid[i]==root)
for f in man['fruits']:
    fb = m.body(f['body']).id; bodies |= set(range(m.nbody)) & set(i for i in range(m.nbody) if m.body_rootid[i]==m.body_rootid[fb])
bodies = sorted(bodies - {0})
geoms = []
meshes = {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if b not in bodies or m.geom_group[g] >= 3 or m.geom_rgba[g,3] <= 0.01: continue
    t = int(m.geom_type[g]); name = m.geom(g).name or ''
    entry = dict(body=bodies.index(b), type=t, size=m.geom_size[g].tolist(), pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist(), rgba=m.geom_rgba[g].tolist(), name=name, group=int(m.geom_group[g]))
    if m.geom_matid[g] >= 0: entry['rgba'] = m.mat_rgba[m.geom_matid[g]].tolist() if m.geom_rgba[g].tolist()==[0.5,0.5,0.5,1] else entry['rgba']
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); entry['mesh'] = mid
        if mid not in meshes:
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            meshes[mid] = dict(name=m.mesh(mid).name, verts=m.mesh_vert[v0:v0+nv].astype(np.float32), faces=m.mesh_face[f0:f0+nf].astype(np.uint32), normals=m.mesh_normal[m.mesh_normaladr[mid]:m.mesh_normaladr[mid]+m.mesh_normalnum[mid]].astype(np.float32) if hasattr(m,'mesh_normaladr') else None)
    geoms.append(entry)
print('bodies', len(bodies), 'geoms', len(geoms), 'meshes', len(meshes), 'frames', len(q))
poses = np.zeros((len(q), len(bodies), 7), np.float32)
for i, qq in enumerate(q):
    d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
    poses[i,:,:3] = d.xpos[bodies]; poses[i,:,3:] = d.xquat[bodies]
import os; os.makedirs(out, exist_ok=True)
poses.tofile(out + '/poses.f32')
blob = bytearray(); mesh_meta = []
for mid, me in meshes.items():
    mesh_meta.append(dict(id=mid, name=me['name'], voff=len(blob), nv=int(len(me['verts'])), foff=None, nf=int(len(me['faces']))))
    blob += me['verts'].tobytes(); mesh_meta[-1]['foff'] = len(blob); blob += me['faces'].tobytes()
open(out + '/meshes.bin','wb').write(blob)
events = [e for e in rep['events'] if e['event'] in ('harvest','walk','stand','retry')]
json.dump(dict(frames=len(q), fps=len(q)/rep['simulated_seconds'], seconds=rep['simulated_seconds'], bodies=[m.body(b).name for b in bodies], chassis=bodies.index(chassis), geoms=geoms, meshes=mesh_meta, events=events, harvested=rep['harvested'], fruits=rep['fruits'],
               start_xy=poses[0, bodies.index(chassis), :2].tolist(), fruit_bodies=[bodies.index(f) for f in fruit_bodies]), open(out + '/meta.json','w'))
print('geom types', sorted(set(g['type'] for g in geoms)), 'mesh bytes', len(blob), 'pose bytes', poses.nbytes)
for g in geoms[:60]: print(g['name'], g['type'], g['group'], m.body(bodies[g['body']]).name, g['rgba'])
