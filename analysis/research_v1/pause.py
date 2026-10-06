"""CPU-only shutdown checkpoint verification; never resumes or starts training."""
from __future__ import annotations
import datetime,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.training.research import frozen_identity,sha

def main():
    directory=Path('experiments/research_v1/dense')
    summary=json.loads((directory/'summary_50003968.json').read_text(encoding='utf-8'))
    checkpoint=Path(summary['checkpoint']);rolling=directory/'checkpoints/last.pt'
    assert sha(checkpoint)==summary['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    last=torch.load(rolling,map_location='cpu',weights_only=False)
    assert state['metadata']==last['metadata']
    assert state['metadata']['tokens_seen']==50003968 and state['metadata']['step']==6104 and state['metadata']['cursor']==48832
    assert state['scheduler']['last_epoch']==6104 and state['scheduler']==last['scheduler']
    assert all(torch.equal(value,last['model'][key]) for key,value in state['model'].items())
    for key,value in state['optimizer']['state'].items():
        for field,tensor in value.items():
            other=last['optimizer']['state'][key][field]
            assert torch.equal(tensor,other) if torch.is_tensor(tensor) else tensor==other
    assert state['scaler'] is None and set(state['rng'])=={'python','numpy','torch','cuda'}
    expected=frozen_identity(load_config('configs/research_v1/dense.yaml'))
    assert all(state['metadata'][k]==v and summary[k]==v for k,v in expected.items())
    result={'passed':True,'status':'paused_at_user_request','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model':'DenseCompute','tokens':50003968,'step':6104,'cursor':48832,
            'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':summary['checkpoint_sha256'],
            'rolling_checkpoint':str(rolling.resolve()),'rolling_checkpoint_sha256':sha(rolling),
            'model_optimizer_scheduler_saved_and_equal':True,'rng_states_present':True,
            'training_process_exit_code':0,'scope':'CPU verification after the specific dense50M exec session completed with exit0. No further GPU work started.',
            'remaining':'Fresh sparse/Ngram20M; their cumulative50M; all70M/100M; diagnostics/replication decision/final archive. Rerun actual repaired-source resume check and Phase0/readiness gate before starting another GPU stage.'}
    Path('results/research_v1/pause_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
