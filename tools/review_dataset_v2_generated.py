"""After generator qualification, independently review and export its new code."""
import json
from pathlib import Path
import time

from forgebuilder.generated import prepare_generated_pilot
from forgebuilder.pilot import review_pilot
from forgebuilder.strata_runtime import health
from tools.export_dataset_v2_pilot import export


def main():
    root=Path('data/dataset_v2')
    source=root/'generator_qualification'
    pilot=root/'generated_pilot_v1'
    state=Path('.local/dataset_v2')
    while not all((source/role/'receipt.json').exists() for role in ('bulk_generator','repository_generator')):
        if (state/'STOP').exists():
            return
        time.sleep(10)
    if not (pilot/'preparation.json').exists():
        print(json.dumps(prepare_generated_pilot(pilot,[source/'bulk_generator',source/'repository_generator'],root/'foundation_v1')),flush=True)
    while not health(8096).get('loaded'):
        if (state/'STOP').exists():
            return
        time.sleep(5)
    print(json.dumps(review_pilot(pilot,state/'strata32k')),flush=True)
    print(json.dumps(export(pilot)),flush=True)


if __name__=='__main__':
    main()
