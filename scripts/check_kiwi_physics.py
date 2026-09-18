"""GPU regression: attached rest, physical pulling, free fall, contact damage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from treesim.config import TreeConfig, preset
from treesim import builder
from treesim.sim import Sim

cfg = TreeConfig.compliant('pergola')
cfg.seed = 42
cfg.fruit.enabled, cfg.fruit.max_count = True, 1
sim = Sim(builder.generate_and_build(cfg), fps=50, substeps=40, collisions=True)
sim._capture()
body = sim.tree.apple_bodies[0]
for _ in range(100):
    sim.step()
assert sim.apples.broken_count == 0, 'Detached under gravity at rest'
height = float(sim.body_q_np()[body, 2])
sim.set_external_force(body, force=(0,0,-20))
for _ in range(25):
    sim.step()
assert sim.apples.broken_count == 1, 'Stem did not detach from actual load'
sim.clear_external_forces()
for _ in range(175):
    sim.step()
q = sim.body_q_np()
assert np.isfinite(q).all()
assert q[body,2] < height-.5, 'Detached kiwi did not fall'
ground = sim.tree.terrain_height
z_ground = float(ground(q[body,0], q[body,1])) if callable(ground) else 0.
assert z_ground-.02 < q[body,2] < z_ground+.12, f'Kiwi missed orchard ground: z={q[body,2]} ground={z_ground}'
metrics = dict(attached_at_rest=True, detached_after_pull=True, final_height_m=float(q[body,2]),
               ground_height_m=z_ground, orchard_ground=sim.tree.orchard_ground,
               strength_N=float(sim.apples.detach_force[0]), **sim.kiwi_damage.metrics())
assert max(metrics['peak_force_N']) > 0, 'Contact load was not measured'
print(json.dumps(metrics, indent=2))
Path('output').mkdir(exist_ok=True)
Path('output/kiwi-physics-check.json').write_text(json.dumps(metrics, indent=2)+'\n')
