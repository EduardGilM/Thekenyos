"""Check the read-only observer against actual Spot/native-MuJoCo state.

Standing and a forced fruit drop are regression inputs, not a picking policy.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from treesim import builder
from treesim.config import TreeConfig
from treesim.sim import Sim
from treesim.spot import SpotController
from treesim.harvest_task import SimHarvestObserver, HarvestOracle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--relic', type=Path, required=True)
a = p.parse_args()
cfg = TreeConfig.compliant('pergola')
cfg.seed = 42
cfg.robot.enabled, cfg.robot.kind, cfg.robot.relic_path = True, 'spot', str(a.relic)
cfg.robot.basket, cfg.robot.payload_mass = True, 0.
cfg.fruit.enabled, cfg.fruit.max_count = True, 1
sim = Sim(builder.generate_and_build(cfg), fps=50, substeps=20, collisions=True)
controller = SpotController(sim)
sim._capture()
labels = {name.rsplit('/',1)[-1]: i for i,name in enumerate(sim.model.body_label)}
observer = SimHarvestObserver(sim, 0, labels['arm_link_fngr'], (0,0,0),
                              (labels['arm_link_fngr'], labels['arm_link_jaw']))
oracle = HarvestOracle()
initial = observer.observe(0.)
oracle.reset(initial)
assert initial.attached and not initial.ground_contact and not initial.in_basket
seen_ground = False
for frame in range(250):
    if frame == 50:
        sim.set_external_force(observer.body, force=(0,0,-50))
    if frame == 75:
        sim.clear_external_forces()
    controller.update(np.zeros(3)); sim.step()
    before = sim.body_q_np().copy()
    obs = observer.observe(0.)  # Work is intentionally not scored by this bridge check.
    np.testing.assert_array_equal(before, sim.body_q_np())
    obs.validate()
    assert max(obs.jaw_forces_N) == 0., 'Unexpected remote jaw contact'
    result = oracle.update(obs)
    seen_ground |= obs.ground_contact
assert not obs.attached and seen_ground
assert not result['success'] and result['outcome'] in ('dropped', 'damage_limit')
metrics = dict(passed=True, observer_is_read_only=True, ground_contact_seen=seen_ground,
               final_observation=asdict(obs), outcome=result['outcome'])
Path('output').mkdir(exist_ok=True)
Path('output/harvest-observer.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
