"""Construct canonical Dataset V2 foundation references and derived views."""
import argparse
import json
from pathlib import Path
import shutil

from forgebuilder.foundation import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/dataset_v2/foundation_build.json'))
    parser.add_argument('--output', type=Path, default=Path('data/dataset_v2/foundation_v1'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf8'))
    parent = args.output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < 20 * 1024**3:
        raise RuntimeError('dataset build requires at least 20 GiB free before starting')
    print(json.dumps(build(config, args.output), indent=2))


if __name__ == '__main__':
    main()
