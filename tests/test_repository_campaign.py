import copy
import pytest
from tools.repository_training_campaign import check_metadata, rate, INHERITED, START_STEP

CFG={'training':dict(lr=.0003,warmup_steps=500,schedule_steps=18310,min_lr_ratio=.1)}

def metadata(phase):
    from src.config import fingerprint
    return dict(step=START_STEP+phase,inherited_tokens=INHERITED,phase_start_step=START_STEP,
        phase_steps=phase,tokens_seen=INHERITED+phase*8192,new_phase_tokens=phase*8192,
        cursor=phase*8,config_hash=fingerprint(CFG))

@pytest.mark.parametrize('phase',[0,1,500,18310])
def test_matched_phase_counts_and_schedule(phase):
    check_metadata(metadata(phase),CFG,{'last_epoch':phase})
    assert rate(CFG,phase)>0
    if phase==0: assert rate(CFG,phase)==pytest.approx(6e-7)
    if phase==18310: assert rate(CFG,phase)==pytest.approx(3e-5)

@pytest.mark.parametrize('field',['cursor','tokens_seen','new_phase_tokens','phase_steps','inherited_tokens','phase_start_step'])
def test_replayed_or_mislabelled_tokens_rejected(field):
    meta=metadata(1000);meta[field]+=1
    with pytest.raises(ValueError): check_metadata(meta,CFG,{'last_epoch':1000})

def test_scheduler_offset_rejected():
    with pytest.raises(ValueError): check_metadata(metadata(1000),CFG,{'last_epoch':999})
