"""CPU-only final provenance and cumulative checkpoint audit; never trains."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.model import LanguageModel
from src.training.engine import make_optimizer
from src.training.research import ENDPOINTS,RESULTS,frozen_identity,sha
from tools.shard_data import verify_shards

def main():
    report={'passed':True,'milestones':{},'scope':'CPU checkpoint payload inspection plus exact frozen-input checks. No training or GPU evaluation.'}
    for tag in ('dense','sparse','memory'):
        cfg=load_config(f'configs/research_v1/{tag}.yaml');verify_shards(cfg)
        identity=frozen_identity(cfg);previous=None;model=LanguageModel(cfg).cpu()
        shapes={k:tuple(v.shape) for k,v in model.state_dict().items()}
        for tokens in ENDPOINTS:
            directory=Path('experiments/research_v1')/tag
            summary=json.loads((directory/f'summary_{tokens}.json').read_text())
            checkpoint=Path(summary['checkpoint'])
            assert sha(checkpoint)==summary['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert set(state['model'])==set(shapes)
            assert all(tuple(v.shape)==shapes[k] and bool(torch.isfinite(v).all()) for k,v in state['model'].items())
            model.load_state_dict(state['model'],strict=True)
            opt,scheduler=make_optimizer(model,cfg);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            for param,values in opt.state.items():
                for key,value in values.items():
                    if torch.is_tensor(value):
                        assert bool(torch.isfinite(value).all())
                        if key in ('exp_avg','exp_avg_sq'):assert value.shape==param.shape
            meta=state['metadata'];step=tokens//8192
            assert meta['tokens_seen']==tokens and meta['step']==step and meta['cursor']==step*8
            assert state['scheduler']['last_epoch']==step
            assert all(meta[k]==v and summary[k]==v for k,v in identity.items())
            assert state['scaler'] is None and set(state['rng'])=={'python','numpy','torch','cuda'}
            optimizer_steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
            assert optimizer_steps and max(optimizer_steps)==step and min(optimizer_steps)>0
            if previous:
                assert meta['training_seconds']>previous['training_seconds'] and meta['wall_time']>previous['wall_time']
            evaluation=json.loads((RESULTS/f'{tag}_{tokens}_evaluation.json').read_text())
            assert evaluation['mixed_predictions']==262144
            if tag=='memory':
                ablation=json.loads((RESULTS/f'{tag}_{tokens}_ablation.json').read_text())
                normal={r['sha256']:r['predictions'] for r in evaluation['documents']}
                assert normal=={r['sha256']:r['predictions'] for r in ablation['documents']}
            report['milestones'][f'{tag}/{tokens}']={'step':step,'cursor':meta['cursor'],'tokens':tokens,'checkpoint_sha256':summary['checkpoint_sha256'],
                'optimizer_step_range':[min(optimizer_steps),max(optimizer_steps)],'scheduler_step':state['scheduler']['last_epoch'],
                'model_keys_shapes_finite_verified':True,'optimizer_scheduler_load_verified':True,'optimizer_moments_shapes_finite_verified':True,**identity}
            previous=meta;del state,opt,scheduler
        rolling=directory/'checkpoints/last.pt'
        assert sha(rolling)==summary['checkpoint_sha256'],'Final rolling and milestone checkpoints differ'
        report.setdefault('final_rolling_checkpoints',{})[tag]={'path':str(rolling.resolve()),'sha256':sha(rolling),'matches_100m_milestone':True}
        del model
    report['common_manifest_sha256']=sha('data/research_v1/shards/manifest.json')
    report['common_tokenizer_sha256']=sha('data/research_v1/tokenizer.json')
    report['consumed_unique_prediction_positions_per_model']=ENDPOINTS[-1]
    report['no_training_epoch_repeat']=ENDPOINTS[-1]<json.loads((RESULTS/'corpus_manifest.json').read_text())['tokens']['train']-1024
    report['prompt1_archive_unchanged']=sha(Path.home()/'Data-Zip/Prompt-1/Prompt-1.zip')=='ad0404358128f51a712372260d54206c6f1bec08d346197f41d8f9e4116e1d1e'
    assert report['no_training_epoch_repeat'] and report['prompt1_archive_unchanged']
    (RESULTS/'final_integrity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'milestones':len(report['milestones']),'no_training_epoch_repeat':True,'prompt1_archive_unchanged':True},indent=2))

if __name__=='__main__':main()
