"""Bounded frozen-model features -> checkpointable predictor -> held-out prefetch evaluation."""
from __future__ import annotations
import argparse, hashlib, json, signal, time
from pathlib import Path
import torch
from torch.nn import functional as F
from src.config import load_config,fingerprint
from src.model import LanguageModel
from src.training.data import PackedStream
from src.training.engine import require_cuda,seed_all,amp
from src.training.checkpoint import save_training,resume_training
from src.runtime.routeahead import RouteAhead,prediction_metrics
from src.runtime.cache import IOConfig,simulate
from src.moe.atlas import expert_atlas,pack_experts
from src.moe.trace import global_experts

@torch.no_grad()
def features(model,cfg,split,batches):
    stream=PackedStream(Path(cfg['data']['shards']),split,cfg['training']['context'],cfg['data']['seed'])
    xs=[]; ys=[]; records=[]; layers=cfg['model']['moe_layers']; experts=cfg['model']['experts']; history=cfg['routeahead']['history']
    for batch in range(batches):
        x,_=stream.next(1,'cuda')
        with amp(cfg):
            out=model(x); h=out['hidden'][0]
            if model.mtp: draft=model.mtp.block(0)(out['hidden'],model.embedding(x))[0]
            else: draft=torch.zeros_like(h)
        route=torch.stack([model.routes()[layer]['experts'][0] for layer in layers],dim=1)
        multihot=F.one_hot(route,experts).amax(2).float()
        for t in range(history,len(x[0])-cfg['routeahead']['horizons']):
            # All history states are current/past; future labels never enter features.
            hist=multihot[t-history+1:t+1]
            xs.append(torch.cat([h[t].float(),draft[t].float(),hist.flatten()]).cpu())
            ys.append(torch.stack([multihot[t+horizon] for horizon in range(1,cfg['routeahead']['horizons']+1)]).cpu())
            records.append({'token':len(records),'sequence':batch,'position':t,'layers':{str(layer):route[t,j].tolist() for j,layer in enumerate(layers)}})
    return torch.stack(xs),torch.stack(ys),records

def run(cfg,checkpoint,run_dir,max_wall_minutes,steps,batches,resume=None):
    require_cuda('cuda');seed_all(cfg['training']['seed']);m=cfg['model'];rcfg=cfg['routeahead']
    if not m['moe_layers']:raise ValueError('MoE model required')
    model=LanguageModel(cfg).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
    train_x,train_y,fit_records=features(model,cfg,'train',batches);eval_x,eval_y,records=features(model,cfg,'val',batches)
    predictor=RouteAhead(m['d_model'],m['experts'],len(m['moe_layers']),rcfg).cuda()
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=cfg['training']['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    base_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    meta={'step':0,'cursor':0,'tokens_seen':0,'config_hash':fingerprint(cfg),'base_checkpoint':str(checkpoint.resolve()),'base_sha256':base_hash}
    if resume:
        meta=resume_training(resume,predictor,optimizer,scheduler,'cuda')
        if meta['config_hash']!=fingerprint(cfg) or meta['base_checkpoint']!=str(checkpoint.resolve()) or meta['base_sha256']!=base_hash:raise ValueError('Predictor resume input mismatch')
    stop=[False];previous=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));start=time.perf_counter();run_dir.mkdir(parents=True,exist_ok=True)
    # Deterministic contiguous shuffled-with-RNG batches; RNG state and cursor saved.
    while meta['step']<steps and time.perf_counter()-start<max_wall_minutes*60*.9 and not stop[0]:
        ids=torch.randint(0,len(train_x),(min(128,len(train_x)),));optimizer.zero_grad()
        x=train_x[ids].cuda();labels=train_y[ids].cuda()
        logits=predictor.network(x).view(-1,rcfg['horizons'],len(m['moe_layers']),m['experts']);loss=predictor.loss(logits,labels)
        if not torch.isfinite(loss):raise FloatingPointError('RouteAhead NaN')
        loss.backward();torch.nn.utils.clip_grad_norm_(predictor.parameters(),cfg['training']['grad_clip'],error_if_nonfinite=True);optimizer.step();scheduler.step()
        meta['step']+=1;meta['cursor']+=len(ids);meta['tokens_seen']+=len(ids)
    save_training(run_dir/'checkpoints/last.pt',predictor,optimizer,scheduler,meta,int(cfg['data']['project_gb']*1024**3));signal.signal(signal.SIGINT,previous)
    with torch.no_grad():logits=predictor.network(eval_x.cuda()).view(-1,rcfg['horizons'],len(m['moe_layers']),m['experts']).cpu()
    atlas=expert_atlas(fit_records,m['experts'])
    for layer in m['moe_layers']:
        for expert in range(m['experts']):atlas['load'].setdefault(layer*m['experts']+expert,0)
    for r in records:
        for expert in global_experts(r,m['experts']):atlas['load'].setdefault(expert,0)
    mapping=pack_experts(atlas,2);metrics={};predictions={}
    for horizon in range(rcfg['horizons']):
        for k in (1,rcfg['top_k']):
            indices=logits[:,horizon].topk(k,-1).indices;predicted=[];actual=[]
            for row in range(len(records)):
                p=[layer*m['experts']+int(e) for j,layer in enumerate(m['moe_layers']) for e in indices[row,j]]
                a=[layer*m['experts']+int(e) for j,layer in enumerate(m['moe_layers']) for e in torch.where(eval_y[row,horizon,j]>0)[0]]
                predicted.append(p);actual.append(a)
                if horizon==0 and k==rcfg['top_k']:predictions[row]=list(dict.fromkeys(mapping[e] for e in p))
            metrics[f't+{horizon+1}/top{k}']=prediction_metrics(predicted,actual,mapping)
    r=cfg['runtime'];io=IOConfig(r['expert_bytes'],r['vram_pages']*r['expert_bytes'],r['ram_pages']*r['expert_bytes'],r['ssd_mib_s']*1024**2,r['ssd_latency_ms']/1000,r['ram_mib_s']*1024**2,r['ram_latency_ms']/1000,r['prefetch_window_s'])
    # Separate traces by packed sequence: no fictitious prefetch across reset boundaries.
    cases=[]
    for sequence in range(batches):
        ids=[i for i,record in enumerate(records) if record['sequence']==sequence];subset=[records[i] for i in ids];pred={j:predictions[i] for j,i in enumerate(ids)}
        base=simulate(subset,mapping,m['experts'],io);prefetch=simulate(subset,mapping,m['experts'],io,predictions=pred)
        cases.append({'sequence':sequence,'without_prefetch':base,'with_prefetch':prefetch,'stall_improves':prefetch['predicted_io_stall_s_per_token']<base['predicted_io_stall_s_per_token']})
    report={'metadata':meta,'route_metrics':metrics,'simulations':cases,'feature_sources':['current hidden','shared MTP state' if model.mtp else 'zero draft state (no MTP)','causal recent routing'],
            'scope':'Train/val document split, synthetic smoke base, configured I/O assumptions. No actual routing changes; no architecture success claim.'}
    (run_dir/'summary.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/sparse_mtp.yaml');p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--run-dir',type=Path,default=Path('experiments/routeahead'))
    p.add_argument('--max-wall-minutes',type=float,default=5);p.add_argument('--steps',type=int,default=200);p.add_argument('--feature-batches',type=int,default=2);p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.max_wall_minutes>30:raise ValueError('RouteAhead prototype capped at 30 minutes')
    print(json.dumps(run(load_config(a.config),a.checkpoint,a.run_dir,a.max_wall_minutes,a.steps,a.feature_batches,a.resume),indent=2))
