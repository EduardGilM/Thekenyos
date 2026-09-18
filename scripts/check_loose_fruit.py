"""Check that a stationary physical basket retains freely simulated fruit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--timestep", type=float, default=.001)
args = p.parse_args()
if not 0 < args.timestep <= .001:
    p.error("timestep must be in (0, .001]")
substeps = int(round(.02/args.timestep))
if substeps % 2 or not np.isclose(substeps*args.timestep, .02):
    p.error("timestep must divide .02 into an even number of steps")
import warp as wp
import newton
from treesim.basket import add_basket, SpillTracker

b = newton.ModelBuilder()
body = b.add_link(mass=2., label='fixed_mount')
j = b.add_joint_fixed(parent=-1, child=body)
b.add_articulation([j])
data = add_basket(b, body, 1.2, 6., 0)
b.add_ground_plane(height=-.5)
m = b.finalize()
s0, s1 = m.state(), m.state()
newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
c, ctrl = m.contacts(), m.control()
solver = newton.solvers.SolverMuJoCo(m, use_mujoco_contacts=True, nconmax=8192, njmax=32768, solver=1)
tracker = SpillTracker(data)
initial = s0.body_q.numpy()[data['fruit_bodies'], :3].copy()
def step():
    global s0,s1
    for _ in range(substeps):
        s0.clear_forces()
        m.collide(s0, c)
        solver.step(s0, s1, ctrl, c, args.timestep)
        s0,s1=s1,s0
with wp.ScopedCapture() as capture:
    step()
for i in range(250):
    wp.capture_launch(capture.graph)
    tracker.update(s0.body_q.numpy(), body)
final=s0.body_q.numpy()[data['fruit_bodies'],:3]
result={'spilled':int(tracker.spilled.sum()), 'relative_motion_m':float(np.max(np.linalg.norm(final-initial,axis=1))), 'min_z':float(final[:,2].min()), 'finite':bool(np.isfinite(final).all())}
print(json.dumps(result))
assert result['finite'] and result['spilled']==0 and result['relative_motion_m']>.001, result
