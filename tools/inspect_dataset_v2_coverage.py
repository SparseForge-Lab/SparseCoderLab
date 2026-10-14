"""Save an immutable coverage snapshot while pilot review continues."""
import argparse
import json
from pathlib import Path

from forgebuilder.quality import inspect_coverage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists(): raise FileExistsError('retain prior coverage snapshots')
    pools=[Path('data/dataset_v2/source_pilot_v1'),Path('data/dataset_v2/generated_pilot_v1')]
    terminal=Path('data/dataset_v2/terminal_pilot_v1')
    if (terminal/'preparation.json').exists(): pools.append(terminal)
    report=inspect_coverage(pools,
                            Path('configs/dataset_v2/pilot.json'))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='pilots'},indent=2))


if __name__=='__main__': main()
