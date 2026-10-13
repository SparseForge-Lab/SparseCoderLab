"""Finish active foundation preparation, then run the authorized local pilot review."""
import json
from pathlib import Path
import time

from forgebuilder.pilot import prepare_pilot, review_pilot
from forgebuilder.schema import canonical_json
from tools.finalize_dataset_v2_foundation import finalize


def main():
    root=Path('data/dataset_v2')
    state_dir=Path('.local/dataset_v2')
    foundation=root/'foundation_v1'
    pilot=root/'source_pilot_v1'
    runtime=state_dir/'strata32k'
    state_path=state_dir/'campaign_state.json'
    started=time.time()
    def state(stage, **extra):
        value=dict(stage=stage,updated_unix=time.time(),started_unix=started,
                   student_training_authorized=False,**extra)
        tmp=state_path.with_suffix('.tmp')
        tmp.write_bytes(canonical_json(value)+b'\n')
        tmp.replace(state_path)
        print(json.dumps(value),flush=True)
    while not (foundation/'build_receipt.json').exists():
        if (state_dir/'STOP').exists():
            state('stopped_before_pilot')
            return
        state('waiting_for_foundation')
        time.sleep(30)
    state('finalizing_foundation')
    finalize(foundation)
    if not (pilot/'preparation.json').exists():
        state('preparing_source_grounded_pilot')
        prepare_pilot(foundation,pilot,target=5000)
    while not (runtime/'weight_identity.json').exists():
        if (state_dir/'STOP').exists():
            state('stopped_before_teacher_review')
            return
        state('waiting_for_teacher_fingerprint')
        time.sleep(30)
    if (state_dir/'STOP').exists():
        state('stopped_before_teacher_review')
        return
    state('reviewing_source_grounded_pilot')
    result=review_pilot(pilot,runtime)
    state('source_pilot_review_returned',result=result,
          full_dataset_ready=False,
          next_required='Executable generation and real repository patch pools remain pending.')


if __name__=='__main__':
    main()
