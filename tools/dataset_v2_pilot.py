"""Prepare source-grounded pilot candidates or independently review them locally."""
import argparse
import json
from pathlib import Path

from forgebuilder.pilot import prepare_pilot, review_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','review'])
    parser.add_argument('--foundation', type=Path, default=Path('data/dataset_v2/foundation_v1'))
    parser.add_argument('--pilot', type=Path, default=Path('data/dataset_v2/source_pilot_v1'))
    parser.add_argument('--runtime', type=Path, default=Path('.local/dataset_v2/strata32k'))
    parser.add_argument('--target', type=int, default=5000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    result = prepare_pilot(args.foundation,args.pilot,args.target) if args.action=='prepare' else review_pilot(args.pilot,args.runtime,args.limit)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
