"""Cumulative, frozen-input research runner built on the existing model/optimizer."""
from __future__ import annotations
import csv,hashlib,json,math,signal,time
from pathlib import Path
import torch
from src.config import fingerprint,load_config
from src.model import LanguageModel
from src.training.engine import amp,make_optimizer,require_cuda,seed_all
from src.training.data import PackedStream
from src.training.checkpoint import save_training,resume_training
from src.eval.research import evaluate,freeze_evaluation
from tools.count_params import count_model

RESULTS=Path('results/research_v1')
ENDPOINTS=[20_004_864,50_003_968,70_000_640,100_007_936]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def training_hash():
    files=sorted(p for root in ('src/model','src/moe','src/memory','src/training','src/eval') for p in Path(root).glob('*.py'))
    files+=[Path('src/config.py'),Path('tools/count_params.py'),Path('verify_install.py')]
    return fingerprint({p.as_posix():sha(p) for p in files})
def frozen_identity(cfg):
    return dict(config_hash=fingerprint(cfg),data_hash=sha(Path(cfg['data']['shards'])/'manifest.json'),
                tokenizer_hash=sha(cfg['data']['tokenizer']),source_hash=training_hash(),evaluation_hash=sha(RESULTS/'evaluation_index.json'))
def checkpoint_identity(meta,cfg):
    expected=frozen_identity(cfg)
    if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError('Frozen source/config/data/tokenizer/evaluation differs from checkpoint')
def route_diagnostics(model):
    report={}
    for layer,r in model.routes().items():
        counts=r['load'].detach().float();total=counts.sum();router=model.blocks[layer].moe.router.weight
        report[layer]=dict(entropy=float(r['entropy']),hard_counts=counts.cpu().tolist(),load_cv=float(counts.std(unbiased=False)/counts.mean().clamp_min(1)),
                           max_expert_share=float(counts.max()/total.clamp_min(1)),unused_experts=int((counts==0).sum()),
                           router_gradient_norm=float(router.grad.norm()) if router.grad is not None else None)
    return report
def run_stage(tag,endpoint):
    cfg=load_config(f'configs/research_v1/{tag}.yaml');t=cfg['training'];require_cuda('cuda');seed_all(t['seed'])
    if t['compile'] or cfg['mtp']['enabled'] or cfg['model']['top_k']!=1:raise RuntimeError('Unfair research options')
    if tag!='dense' and (cfg['model'].get('moe_backend')!='grouped' or cfg['model']['experts']!=12):raise RuntimeError('Primary sparse backend/geometry mismatch')
    from tools.shard_data import verify_shards
    verify_shards(cfg);freeze_evaluation(cfg)
    manifest=json.loads((Path(cfg['data']['shards'])/'manifest.json').read_text());assert manifest['tokens']['train']>=150_000_000
    assert sha(cfg['data']['tokenizer'])==manifest['tokenizer_sha256']
    torch.backends.cuda.matmul.allow_tf32=t['tf32'];torch.backends.cudnn.allow_tf32=t['tf32']
    model=LanguageModel(cfg).cuda();opt,scheduler=make_optimizer(model,cfg)
    directory=Path('experiments/research_v1')/tag;directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'checkpoints/last.pt';milestone=directory/f'checkpoints/tokens_{endpoint}.pt'
    identity=frozen_identity(cfg)
    meta=dict(step=0,tokens_seen=0,cursor=0,training_seconds=0.,wall_time=0.,seed=t['seed'],**identity)
    if checkpoint.exists():
        # Stage checkpoint payload on CPU; load_state_dict moves weights/moments
        # to live CUDA parameters while ordinary Adam step counters stay on CPU.
        meta=resume_training(checkpoint,model,opt,scheduler,'cpu');checkpoint_identity(meta,cfg)
        prior_summary=directory/f"summary_{meta['tokens_seen']}.json"
        if prior_summary.exists():
            prior_meta=json.loads(prior_summary.read_text())
            if prior_meta['step']==meta['step'] and prior_meta['config_hash']==meta['config_hash']:meta['wall_time']=prior_meta['wall_time']
    if meta['tokens_seen']>endpoint:raise RuntimeError('Stage already passed; use immutable milestone evaluation')
    if milestone.exists() and (RESULTS/f'{tag}_{endpoint}_evaluation.json').exists():return
    stream=PackedStream(Path(cfg['data']['shards']),'train',t['context'],42,meta['cursor'])
    (directory/'config.json').write_text(json.dumps(cfg,indent=2));(directory/'provenance.json').write_text(json.dumps(identity,indent=2))
    count=count_model(model);base=meta['wall_time'];start=time.perf_counter();stop=[False];collapse=0
    prior=signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));prior_term=signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    torch.cuda.reset_peak_memory_stats();model.train()
    try:
        with (directory/'metrics.jsonl').open('a',encoding='utf-8') as log:
            while meta['tokens_seen']<endpoint and not stop[0]:
                if time.perf_counter()-start>t['max_wall_minutes']*60:raise RuntimeError('Stage time cap; last safe checkpoint retained')
                begin=time.perf_counter();opt.zero_grad(set_to_none=True);total=balance=0.
                for _ in range(t['accumulation']):
                    x,y=stream.next(t['microbatch'],'cuda')
                    with amp(cfg):out=model(x,y);loss=out['loss']/t['accumulation']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; stop comparison')
                    loss.backward();total+=float(out['lm_loss']);balance+=float(out['aux_loss'])
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],error_if_nonfinite=True))
                router=route_diagnostics(model) if (meta['step']+1)%t['log_steps']==0 else None
                # Collapse checks use each update's final microbatch after warmup.
                severe=any(float(r['load'].max()/r['load'].sum())>.95 and int((r['load']==0).sum())>=10 for r in model.routes().values())
                collapse=collapse+1 if severe and meta['step']>t['warmup_steps'] else 0
                if collapse>=200:raise RuntimeError('Persistent severe router collapse; stop comparison')
                memory=model.memory.diagnostics(x) if model.memory is not None and (meta['step']+1)%200==0 else None
                if memory is not None:
                    memory['gate_histogram10bins']=torch.histc(model.memory.last_gate.float(),bins=10,min=0,max=1).cpu().tolist()
                    memory['update_frequency_scope']='Observed gradient-bearing rows at every200th accumulated update, not exact per-row lifetime update count.'
                opt.step();scheduler.step();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                meta['step']+=1;meta['tokens_seen']+=x.numel()*t['accumulation'];meta['cursor']=stream.cursor
                meta['training_seconds']+=elapsed;meta['wall_time']=base+time.perf_counter()-start
                if router is not None or meta['tokens_seen']==endpoint:
                    log.write(json.dumps(dict(meta,train_loss=total/t['accumulation'],balance_loss=balance/t['accumulation'],grad_norm=norm,
                                              step_tok_s=x.numel()*t['accumulation']/elapsed,vram_peak=torch.cuda.max_memory_allocated(),router=router,ngram=memory))+'\n');log.flush()
                if meta['step']%t['eval_steps']==0 and meta['tokens_seen']<endpoint:
                    quick=evaluate(model,cfg,full=False);log.write(json.dumps(dict(meta,**quick))+'\n');log.flush()
                if meta['step']%t['checkpoint_steps']==0:save_training(checkpoint,model,opt,scheduler,meta,int(cfg['data']['project_gb']*1024**3))
            meta['wall_time']=base+time.perf_counter()-start
            if stop[0]:save_training(checkpoint,model,opt,scheduler,meta);raise RuntimeError('Interrupted safely; resume same checkpoint')
            if meta['tokens_seen']!=endpoint:raise RuntimeError('Exact matched complete-step endpoint mismatch')
            full=evaluate(model,cfg,full=True,routes=tag!='dense')
            full_path=RESULTS/f'{tag}_{endpoint}_evaluation.json';full_path.write_text(json.dumps(full,indent=2))
            from src.eval.micro_code import evaluate_micro
            (RESULTS/f'{tag}_{endpoint}_micro_code.json').write_text(json.dumps(evaluate_micro(model,cfg),indent=2))
            if model.memory is not None:
                model.memory.ablate=True
                try:ablated=evaluate(model,cfg,full=True)
                finally:model.memory.ablate=False
                keys=['val_loss','code_val_loss','general_val_loss','technical_val_loss']
                (RESULTS/f'{tag}_{endpoint}_ablation.json').write_text(json.dumps(dict(normal={k:full.get(k) for k in keys},zero_residual={k:ablated.get(k) for k in keys},
                  delta_ablated_minus_normal={k:ablated[k]-full[k] for k in keys if k in full and k in ablated},documents=ablated['documents'],scope='Same checkpoint, frozen data, evaluation only; no retraining'),indent=2))
            meta['wall_time']=base+time.perf_counter()-start
            save_training(milestone,model,opt,scheduler,meta,int(cfg['data']['project_gb']*1024**3))
            save_training(checkpoint,model,opt,scheduler,meta,int(cfg['data']['project_gb']*1024**3))
            final_wall=base+time.perf_counter()-start
            result=dict(meta,model=cfg['name'],tag=tag,wall_time=final_wall,wall_tok_s=meta['tokens_seen']/final_wall,
                        training_step_tok_s=meta['tokens_seen']/meta['training_seconds'],vram_peak=torch.cuda.max_memory_allocated(),
                        stored_params=count['total'],active_params_est=count['active_estimate'],training_flops_est=6*count['active_estimate']*meta['tokens_seen'],
                        checkpoint=str(milestone),checkpoint_sha256=sha(milestone),
                        **{k:v for k,v in full.items() if k not in ('documents','router','ngram','scope')},
                        evaluation_scope=full.get('scope'),
                        scope='Cumulative fresh seed42 real-data quality run. Wall includes loop/validation/milestone and rolling checkpoint writes, excludes initialization/verification. Estimated6N FLOPs omit quadratic attention/lookup/dispatch; not measured hardware FLOPs.')
            (directory/f'summary_{endpoint}.json').write_text(json.dumps(result,indent=2));print(json.dumps({'completed':tag,'tokens':endpoint,'val_loss':full['val_loss'],'code_val_loss':full.get('code_val_loss'),'step_tok_s':result['training_step_tok_s']},indent=2),flush=True)
    except Exception as error:
        (RESULTS/'training_failure.json').write_text(json.dumps(dict(tag=tag,endpoint=endpoint,metadata=meta,error=str(error)),indent=2));raise
    finally:signal.signal(signal.SIGINT,prior);signal.signal(signal.SIGTERM,prior_term)
