"""Inspect trusted local full-state entries without launching training."""
import argparse
import json
from pathlib import Path

from forgebuilder.comparison import inspect_entries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=Path('configs/dataset_v2/comparison_plan.json'))
    parser.add_argument('--report',type=Path,default=Path('results/dataset_v2/entry_preflight_v1.json'))
    args=parser.parse_args()
    if args.report.exists():
        raise FileExistsError('retain prior measurements; select a new report path')
    report=inspect_entries(args.plan)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
