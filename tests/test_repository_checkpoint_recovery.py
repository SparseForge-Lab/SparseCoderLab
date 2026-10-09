"""Inject a replacement failure and require a recorded, healthy rollback."""
import json
from pathlib import Path
import pytest
import torch
from src.config import fingerprint
from src.utils.hashing import sha256_file
from tools import repository_training_campaign as campaign

def test_checkpoint_replacement_failure_keeps_verified_previous(tmp_path,monkeypatch):
    cfg=dict(training=dict(lr=.0003,warmup_steps=500,schedule_steps=18310,min_lr_ratio=.1),
        data=dict(tokenizer='data/research_v1/tokenizer.json',shards='data/research_v2_real/shards/transition_r2',
            dataset_version='research_v2_repository_transition_2026_10_08_r2'))
    run=tmp_path/'dense75_ref';(run/'checkpoints').mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(cfg))
    def meta(phase):
        return dict(step=12208+phase,inherited_tokens=100007936,phase_start_step=12208,
            phase_steps=phase,tokens_seen=100007936+phase*8192,new_phase_tokens=phase*8192,
            cursor=phase*8,config_hash=fingerprint(cfg))
    def state(phase):
        return dict(model={'weight':torch.ones(2)},optimizer=dict(state={0:{'exp_avg':torch.ones(2)}},
            param_groups=[dict(lr=campaign.rate(cfg,phase)),dict(lr=campaign.rate(cfg,phase))]),
            scheduler=dict(last_epoch=phase),rng=dict(python=None,numpy=None,torch=torch.zeros(1,dtype=torch.uint8),
                cuda=[torch.zeros(1,dtype=torch.uint8)]),metadata=meta(phase))
    initial=run/'checkpoints/initial.pt';torch.save(state(0),initial)
    entry=dict(initial_copies={'dense75_ref':dict(config_hash=fingerprint(cfg),checkpoint_sha256=sha256_file(initial),
        checkpoint_bytes=initial.stat().st_size)},operative_sources_verified={})
    # Identity checks have independent coverage; this test injects write failure.
    monkeypatch.setattr(campaign,'check_identities',lambda identities:None)
    class Scheduler:
        def __init__(self,phase):self.phase=phase
        def state_dict(self):return {'last_epoch':self.phase}
    def injected_save(path,model,optimizer,scheduler,metadata,limit):
        if metadata['phase_steps']==2:
            path.write_bytes(b'corrupted replacement')
            raise OSError('injected replacement failure')
        torch.save(state(metadata['phase_steps']),path)
    monkeypatch.setattr(campaign.engine,'save_training',injected_save)
    def simulated_run(config,run_dir,**kwargs):
        for phase in (1,2):campaign.engine.save_training(run_dir/'checkpoints/last.pt',None,None,Scheduler(phase),meta(phase))
    monkeypatch.setattr(campaign.engine,'run',simulated_run)
    with pytest.raises(OSError,match='injected replacement failure'):
        campaign.run_variant('dense75_ref',tmp_path,entry)
    receipt=json.loads((run/'checkpoint_verification.json').read_text())
    previous=receipt['previous']
    assert previous['phase_steps']==1
    assert sha256_file(Path(previous['path']))==previous['sha256']
    assert sha256_file(run/'checkpoints/last.pt')!=receipt['last']['sha256']
