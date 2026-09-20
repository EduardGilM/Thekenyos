# Orchard walk (web viewer)

First-person walk through the procedural kiwi plantation while five Spot robots replay
recorded harvest rollouts of the RL arm policy. No physics runs in the browser.

Serve the folder statically and open it (module scripts need http, not file://):

```bash
python3 -m http.server 8791 --directory web/orchard-walk
```

Controls: WASD walk, mouse look, shift run, F follow nearest robot, T time speed, H hide UI.

Data pipeline:
- `scripts/export_web_orchard.py OUT rows cols fruit_count canopy_spacing` exports terrain,
  canes, leaves and fruit of the beauty-branch orchard generator (`orchard/`).
- `scripts/export_web_rollout.py SCENE REPLAY OUT` exports body poses and Spot meshes from a
  `demo_orchard_harvest.py` recording (`robot/` for meshes, `rollouts/<name>/` per run).
- `rollouts/index.json` lists the runs the robots cycle through.
