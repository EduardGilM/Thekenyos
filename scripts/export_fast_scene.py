"""Export the bounded rigid-fruit training scene from a base-scene artifact."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treesim.kiwi_rl.fast_scene import assemble_fast_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-scene', type=Path, required=True,
                        help='Directory containing base.xml and manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fruit-count', type=int)
    parser.add_argument('--timestep', type=float, choices=(.002, .005), default=.005,
                        help='Physics step in seconds: .005=200 Hz, .002=500 Hz')
    parser.add_argument('--no-visual-stalk', action='store_true')
    parser.add_argument('--keep-base-arm-pose', action='store_true',
                        help='Diagnostic export without the default fruit-facing starting arm pose')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Preserve existing scene export directories')
    xml, manifest = assemble_fast_scene(args.base_scene, fruit_count=args.fruit_count,
                                        timestep_s=args.timestep,
                                        visual_stalk=not args.no_visual_stalk,
                                        camera_start=not args.keep_base_arm_pose)
    args.output.mkdir(parents=True)
    (args.output / 'scene.xml').write_text(xml)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(schema=manifest['schema'], fruits=len(manifest['fruits']),
                          frequency_hz=manifest['numerical_profile']['frequency_hz'],
                          model_sha256=manifest['model_sha256'])))


if __name__ == '__main__':
    main()
