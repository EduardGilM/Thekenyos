"""GPU regression: attached rest, physical pulling, free fall, contact damage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
from treesim.config import TreeConfig, preset
from treesim import builder
from treesim.sim import Sim

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--device", default="cuda")
p.add_argument("--terrain", action="store_true")
p.add_argument("--terrain-amplitude", type=float, default=.24)
p.add_argument("--terrain-wavelength", type=float, default=1.3)
p.add_argument("--terrain-seed", type=int, default=7)
args = p.parse_args()

cfg = TreeConfig.compliant('pergola')
cfg.device = args.device
cfg.seed = 42
cfg.physics.terrain = args.terrain
cfg.physics.terrain_amplitude = args.terrain_amplitude
cfg.physics.terrain_wavelength = args.terrain_wavelength
cfg.physics.terrain_seed = args.terrain_seed
cfg.physics.terrain_extent = 6.
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
ground = sim.tree.terrain_height(*q[body,:2]) if args.terrain else 0.
if args.terrain:
    assert np.max(np.abs(q[body,:2])) < cfg.physics.terrain_extent, 'Kiwi left the terrain patch'
    assert ground > .004, 'Drop check must land on elevated terrain, not a flat pad'
assert -.01 < q[body,2]-ground < .1, f'Kiwi did not settle on terrain: pose={q[body,:3]}, ground={ground}'
metrics = dict(attached_at_rest=True, detached_after_pull=True, final_height_m=float(q[body,2]),
               ground_height_m=ground, terrain=args.terrain, terrain_seed=args.terrain_seed,
               strength_N=float(sim.apples.detach_force[0]), **sim.kiwi_damage.metrics())
assert max(metrics['peak_force_N']) > 0, 'Contact load was not measured'
print(json.dumps(metrics, indent=2))
Path('output').mkdir(exist_ok=True)
path = Path('output/kiwi-terrain-physics-check.json' if args.terrain else 'output/kiwi-physics-check.json')
path.write_text(json.dumps(metrics, indent=2)+'\n')
