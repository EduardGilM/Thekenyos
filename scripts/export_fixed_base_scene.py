"""Create the frozen-base manipulation fixture from a verified fast scene."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from treesim.kiwi_rl.fast_scene import fixed_base_scene, load_fast_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    xml, manifest = fixed_base_scene(args.scene)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'scene.xml').write_text(xml)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    load_fast_scene(args.output)
    print(json.dumps(dict(scene=str(args.output), model_sha256=manifest['model_sha256'])))


if __name__ == '__main__':
    main()
