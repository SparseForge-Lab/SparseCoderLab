"""Reconstruct resume state from trusted local payloads, not checkpoint names."""
from __future__ import annotations
import datetime,json,subprocess
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.engine import make_optimizer
from src.training.data import PackedStream
from src.training.research import RESULTS,ENDPOINTS,frozen_identity,sha
from tools.shard_data import verify_shards

def main():
    configs={tag:load_config(f'configs/research_v1/{tag}.yaml') for tag in ('dense','sparse','memory')}
    base=configs['dense']
    assert all(c['training']==base['training'] and c['data']==base['data'] for c in configs.values())
    assert configs['sparse']['model']['moe_backend']==configs['memory']['model']['moe_backend']=='grouped'
    assert configs['sparse']['model']['top_k']==configs['memory']['model']['top_k']==1
    assert configs['sparse']['model']['experts']==configs['memory']['model']['experts']==12
    assert sha('data/research_v1/shards/manifest.json')==sha(RESULTS/'corpus_manifest.json')
    verify_shards(base)
    report={'passed':True,'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':{},
            'scope':'CPU deserialize/load of every existing primary checkpoint, exact parameter/optimizer shapes, finite state, frozen identities, scheduler and stream cursor. No training or checkpoint mutation.'}
    for tag,cfg in configs.items():
        directory=Path('experiments/research_v1')/tag;identity=frozen_identity(cfg)
        records=[];failures=[]
        model=LanguageModel(cfg);opt,scheduler=make_optimizer(model,cfg)
        for p in sorted((directory/'checkpoints').glob('*.pt')):
            try:
                state=torch.load(p,map_location='cpu',weights_only=False);meta=state['metadata']
                assert all(meta[k]==v for k,v in identity.items()),'Frozen identity mismatch'
                assert meta['tokens_seen']==meta['step']*8192 and meta['cursor']==meta['step']*8
                assert set(state['model'])==set(model.state_dict())
                for key,value in state['model'].items():
                    assert value.shape==model.state_dict()[key].shape and torch.isfinite(value).all(),key
                model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
                assert scheduler.last_epoch==meta['step']
                for parameter,values in opt.state.items():
                    for field in ('exp_avg','exp_avg_sq'):
                        if field in values:assert values[field].shape==parameter.shape and torch.isfinite(values[field]).all()
                assert set(state['rng'])=={'python','numpy','torch','cuda'}
                assert state['rng']['torch'].numel()>0 and state['rng']['cuda']
                stream=PackedStream(Path(cfg['data']['shards']),'train',cfg['training']['context'],42,meta['cursor'])
                assert 0<=meta['cursor']<stream.count
                records.append({'path':str(p.resolve()),'sha256':sha(p),'size_bytes':p.stat().st_size,
                    'tokens':meta['tokens_seen'],'optimizer_step':meta['step'],'cursor':meta['cursor'],'rng_available':True,
                    'scheduler_step':scheduler.last_epoch,'learning_rate':opt.param_groups[0]['lr'],**identity})
            except Exception as error:failures.append({'path':str(p),'error':str(error)})
        complete=[]
        for tokens in ENDPOINTS:
            s=directory/f'summary_{tokens}.json';e=RESULTS/f'{tag}_{tokens}_evaluation.json'
            if not s.exists() or not e.exists():continue
            summary=json.loads(s.read_text(encoding='utf-8'));evaluation=json.loads(e.read_text(encoding='utf-8'))
            checkpoint=next((r for r in records if Path(r['path']).resolve()==Path(summary['checkpoint']).resolve()),None)
            assert checkpoint and checkpoint['tokens']==tokens and checkpoint['sha256']==summary['checkpoint_sha256']
            assert all(summary[k]==v for k,v in identity.items())
            assert evaluation['mixed_predictions']==262144 and len(evaluation['documents'])==743
            assert (RESULTS/f'{tag}_{tokens}_micro_code.json').exists()
            if tag=='memory':assert (RESULTS/f'{tag}_{tokens}_ablation.json').exists()
            complete.append(tokens)
        latest=max(records,key=lambda r:(r['tokens'],Path(r['path']).name=='last.pt')) if records else None
        assert not failures or latest,'No verified checkpoint remains'
        report['models'][tag]={'state':'resumable' if latest else 'fresh_unstarted','consumed_tokens':latest['tokens'] if latest else 0,
            'latest_valid_checkpoint':latest,'fully_evaluated_milestones':complete,'verified_checkpoints':records,'rejected_checkpoints':failures,
            'frozen_identity':identity,'fallback_required':bool(failures)}
        del model,opt,scheduler
    report['planned_order']=['sparse20M','sparse50M','memory20M','memory50M','matched50M_review','dense70M','sparse70M','memory70M','dense100M','sparse100M','memory100M']
    report['interrupted_attempt_excluded']='results/history/Prompt-2/sparse_uncheckpointed_attempt; no resumable weights existed'
    (RESULTS/'resume_state.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'models':{t:{k:v for k,v in r.items() if k in ('state','consumed_tokens','fully_evaluated_milestones')} for t,r in report['models'].items()}},indent=2))

if __name__=='__main__':main()
