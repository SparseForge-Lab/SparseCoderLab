"""Matched final-gate NLL, generation and restricted WASI scoring.

Uses the existing frozen evaluation index and canonical generation/scoring
functions. Generated source is executed only by the existing WASI runtime.
"""
from __future__ import annotations
import argparse, collections, gc, hashlib, json, math, os, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import psutil
import torch
from tokenizers import Tokenizer
from src.config import fingerprint
from src.eval.research import evaluate, freeze_evaluation
from src.eval.wasi_python import WasiPython
from src.model import LanguageModel
from src.training.checkpoint import resume_training
from src.training.engine import make_optimizer
from src.utils.hashing import sha256_file
from tools.generate_canonical_python import generate_batch, gpu_allowed
from tools.repository_training_campaign import TAGS, TARGET, verify_state, write_json
from tools.score_saved_python import score, summarize, validate_rows

DIRECTORY=Path('results/repository_training_v1')
MANIFEST=DIRECTORY/'matched_gate_checkpoints.json'
BENCHMARK=Path('configs/eval/simple_python_v1.json')
GENERATIONS=Path('data/research_v2_real/evaluation/repository_gate_250M_r1.jsonl')

def validate_evaluation(result):
    for key in ('val_loss','code_val_loss','technical_val_loss','general_val_loss'):
        value=result.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise FloatingPointError('Missing or nonfinite final evaluation loss: '+key)
    # Cover document losses, ablations and nested router/memory diagnostics too.
    # Default JSON encoding would otherwise silently emit NaN or Infinity.
    try:
        json.dumps(result,allow_nan=False)
    except ValueError as error:
        raise FloatingPointError('Nonfinite nested final evaluation measurement') from error

def manifest():
    entries={}
    for tag in TAGS:
        run=Path('experiments/prompt5_v1')/tag
        cfg=json.loads((run/'config.json').read_text())
        summary=json.loads((run/'campaign_summary.json').read_text())
        if summary['tokens_seen']!=TARGET: raise ValueError('All models must reach the matched gate')
        full=verify_state(run/'checkpoints/latest_verified_250M.pt',cfg)
        weights=summary['model_only']
        if sha256_file(Path(weights['path']))!=weights['sha256']: raise ValueError('Final weights changed')
        entries[tag]=dict(total_tokens=TARGET,inherited_tokens=100007936,new_phase_tokens=149995520,
            phase_steps=18310,config=dict(path=(run/'config.json').as_posix(),sha256=sha256_file(run/'config.json'),fingerprint=fingerprint(cfg)),
            full_resume_state=full,model_weights=weights,tokenizer_sha256=sha256_file(Path(cfg['data']['tokenizer'])),
            data_manifest_sha256=sha256_file(Path(cfg['data']['manifest'])))
    result=dict(schema_version=1,matched_gate_checkpoints=entries,
        scope='Final common repository-training gate; canonical 100M starting states remain separate and unchanged.')
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())!=result: raise ValueError('Published gate manifest differs')
    write_json(MANIFEST,result)
    return result

def entry_model(tag, entry):
    gpu_allowed()
    config_path=Path(entry['config']['path'])
    if sha256_file(config_path)!=entry['config']['sha256']: raise ValueError('Final config identity differs')
    cfg=json.loads(config_path.read_text())
    checkpoint=Path(entry['model_weights']['path'])
    if sha256_file(checkpoint)!=entry['model_weights']['sha256']: raise ValueError('Final checkpoint identity differs')
    if sha256_file(Path(cfg['data']['tokenizer']))!=entry['tokenizer_sha256']: raise ValueError('Tokenizer identity differs')
    if sha256_file(Path(cfg['data']['manifest']))!=entry['data_manifest_sha256']: raise ValueError('Corpus manifest identity differs')
    model=LanguageModel(cfg).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model,cfg,checkpoint

def evaluate_gate(entries):
    frozen=json.loads((DIRECTORY/'first_update_verification.json').read_text())
    for name,row in frozen['frozen_document_and_index_references_verified'].items():
        if sha256_file(Path(name))!=row['sha256']: raise ValueError('Frozen corpus artifact changed: '+name)
    for tag,entry in entries.items():
        output=DIRECTORY/(tag+'_evaluation_250M.json')
        if output.exists():
            previous=json.loads(output.read_text())
            if previous['checkpoint_sha256']!=entry['model_weights']['sha256']: raise ValueError('Existing evaluation checkpoint differs')
            continue
        model,cfg,checkpoint=entry_model(tag,entry)
        # Exercise the complete final-state loader, including Adam group identity,
        # scheduler and RNG restoration, without advancing an optimizer update.
        optimizer,scheduler=make_optimizer(model,cfg)
        meta=resume_training(Path(entry['full_resume_state']['path']),model,optimizer,scheduler,'cuda')
        if meta['tokens_seen']!=TARGET or meta['cursor']!=18310*8: raise ValueError('Final resume counters differ')
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if any(not torch.equal(value.cpu(),weights[name]) for name,value in model.state_dict().items()):
            raise ValueError('Final full-state/model-only weights differ')
        del weights,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        index=freeze_evaluation(cfg)
        labels=['mixed']*index['mixed_batches']+[key for key,rows in index['documents'].items() for _ in rows]
        baseline=json.loads(Path('results/research_v2_real/transition_cuda_preflight_r2.json').read_text())['variants'][tag]
        index_path=Path('results/research_v2_real')/f"evaluation_index_{cfg['data']['dataset_version']}.json"
        if sha256_file(index_path)!=baseline['frozen_evaluation_index_sha256']: raise ValueError('Frozen evaluation index changed')
        calls=[0];aux=collections.defaultdict(lambda:[0.,0])
        def observe(module,args,out):
            label=labels[calls[0]];calls[0]+=1
            predictions=int((args[1]!=-100).sum())
            aux[label][0]+=float(out['aux_loss'])*predictions;aux[label][1]+=predictions
        handle=model.register_forward_hook(observe)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
        result=evaluate(model,cfg,index,full=True,routes=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin;handle.remove()
        if calls[0]!=len(labels): raise ValueError('Evaluation auxiliary-loss sample alignment differs')
        if not math.isfinite(result['val_loss']) or any(row['nll'] is not None and not math.isfinite(row['nll']) for row in result['language'].values()):
            raise FloatingPointError('Nonfinite final evaluation')
        for value in result['router'].values():
            value['under_one_percent_experts']=sum(frequency<.01 for frequency in value['frequency'])
        result.update(checkpoint_sha256=entry['model_weights']['sha256'],total_tokens=TARGET,
            frozen_evaluation_index_sha256=sha256_file(index_path),full_resume_loader_verified=True,
            full_state_matches_model_only=True,evaluation_seconds=elapsed,
            evaluated_predictions=result['mixed_predictions']+sum(result['category_predictions'].values()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_rss_bytes=psutil.Process().memory_info().rss,
            auxiliary_loss_by_stratum={key:dict(mean=value/count,predictions=count) for key,(value,count) in aux.items()},
            evaluator_sha256=sha256_file(Path(__file__)))
        result['predictions_per_second']=result['evaluated_predictions']/elapsed
        if model.memory is not None:
            model.eval();model.memory.ablate=True;begin=time.perf_counter()
            result['residual_off']=evaluate(model,cfg,index,full=True,routes=False)
            result['residual_off_seconds']=time.perf_counter()-begin
            model.memory.ablate=False
        if sha256_file(checkpoint)!=entry['model_weights']['sha256']: raise ValueError('Checkpoint changed during evaluation')
        validate_evaluation(result)
        write_json(output,result)
        print(json.dumps(dict(stage='evaluation_complete',variant=tag,mixed_nll=result['val_loss'],seconds=elapsed)),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()

def generate(entries):
    benchmark=json.loads(BENCHMARK.read_text())
    tokenizer=Tokenizer.from_file('data/research_v1/tokenizer.json')
    GENERATIONS.parent.mkdir(parents=True,exist_ok=True)
    existing=[json.loads(line) for line in GENERATIONS.read_text(encoding='utf8').splitlines()] if GENERATIONS.exists() else []
    seen={(row['variant'],row['temperature'],row['prompt_number']) for row in existing}
    if len(seen)!=len(existing): raise ValueError('Duplicate existing generation slots')
    for row in existing:
        if row['checkpoint_sha256']!=entries[row['variant']]['model_weights']['sha256']: raise ValueError('Partial generation uses different checkpoint')
    receipt_path=DIRECTORY/'generation_efficiency_250M.json'
    times=json.loads(receipt_path.read_text())['groups'] if receipt_path.exists() else []
    with GENERATIONS.open('a',encoding='utf8',newline='\n') as handle:
        for tag,entry in entries.items():
            if all((tag,temp,task['prompt_number']) in seen for temp in (0.,.2,.5,.8,1.,1.2,1.5) for task in benchmark['tasks']): continue
            model,cfg,checkpoint=entry_model(tag,entry)
            torch.cuda.reset_peak_memory_stats()
            for temperature in (0.,.2,.5,.8,1.,1.2,1.5):
                gpu_allowed();begin=time.perf_counter();generated_tokens=completed=0
                for offset in range(0,len(benchmark['tasks']),4):
                    tasks=benchmark['tasks'][offset:offset+4]
                    pending=[task for task in tasks if (tag,temperature,task['prompt_number']) not in seen]
                    if not pending: continue
                    # Original grouping retained even when recovering a partial group.
                    gpu_allowed();outputs=generate_batch(model,tokenizer,[task['prompt'] for task in tasks],temperature)
                    for task,(text,tokens) in zip(tasks,outputs):
                        slot=(tag,temperature,task['prompt_number'])
                        if slot in seen: continue
                        row=dict(variant=tag,temperature=temperature,prompt_number=task['prompt_number'],input=task['prompt'],
                            raw_output=text,checkpoint=checkpoint.as_posix(),checkpoint_sha256=entry['model_weights']['sha256'],
                            training_tokens=TARGET,generation_seed=42,max_new_tokens=128,top_p=.95,top_k=40,repetition_penalty=1.,
                            stop_tokens=['<eos>','<doc>'],device='cuda',bf16=True,tf32=True,generation_batch_size=4,
                            generated_tokens=tokens,sampler_sha256=sha256_file(Path('tools/generate_canonical_python.py')),
                            orchestration_sha256=sha256_file(Path(__file__)))
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();seen.add(slot)
                        completed+=1;generated_tokens+=tokens
                if completed:
                    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                    times.append(dict(variant=tag,temperature=temperature,completions=completed,generated_tokens=generated_tokens,
                        seconds=elapsed,generated_tokens_per_second=generated_tokens/elapsed,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        cpu_rss_bytes=psutil.Process().memory_info().rss))
                    write_json(receipt_path,dict(groups=times,completed_slots=len(seen),generation_source_sha256=sha256_file(GENERATIONS),
                        baseline_sampler_sha256=sha256_file(Path('tools/generate_canonical_python.py')),
                        scope='Same sampler, batch4, seven temperatures, per-prompt seed42 and decoding settings as canonical100M baseline. Recovery preserves original prompt grouping.'))
                    print(json.dumps(dict(stage='generation_group',variant=tag,temperature=temperature,completions=len(seen))),flush=True)
            if sha256_file(checkpoint)!=entry['model_weights']['sha256']: raise ValueError('Checkpoint changed during generation')
            del model;gc.collect();torch.cuda.empty_cache()
    if len(seen)!=700: raise ValueError('Incomplete matched generation matrix')
    write_json(receipt_path,dict(groups=times,completed_slots=len(seen),generation_source_sha256=sha256_file(GENERATIONS),
        baseline_sampler_sha256=sha256_file(Path('tools/generate_canonical_python.py')),
        scope='Same sampler, batch4, seven temperatures, per-prompt seed42 and decoding settings as canonical100M baseline. Recovery preserves original prompt grouping.'))

def validate_generation(rows, entries, baseline):
    benchmark=json.loads(BENCHMARK.read_text())
    bound={'canonical_transition_checkpoints':{tag:dict(entry,canonical_transition_tokens=entry['total_tokens']) for tag,entry in entries.items()}}
    policy=validate_rows(rows,benchmark['tasks'],bound)
    if policy!=baseline['decoding_policy'] or sha256_file(BENCHMARK)!=baseline['benchmark_sha256']:
        raise ValueError('Matched baseline task or decoding identity changed')
    for row in rows:
        if row.get('sampler_sha256')!=sha256_file(Path('tools/generate_canonical_python.py')) or not row.get('bf16') or not row.get('tf32') or row.get('generation_batch_size')!=4:
            raise ValueError('Matched sampling implementation/precision/grouping differs')
    return policy

def score_gate(entries):
    output=DIRECTORY/'functional_250M_r1.json'
    if output.exists(): raise FileExistsError('Preserve existing functional report')
    raw=GENERATIONS.read_bytes();rows=[json.loads(line) for line in raw.decode('utf8').splitlines()]
    benchmark=json.loads(BENCHMARK.read_text())
    baseline=json.loads(Path('results/research_v2_real/functional_canonical_gpu_r1.json').read_text())
    policy=validate_generation(rows,entries,baseline)
    runtime=WasiPython()
    if runtime.runtime_identity!=baseline['runtime']: raise ValueError('Restricted runtime identity changed')
    lookup={task['prompt_number']:task for task in benchmark['tasks']};results=[];begin=time.perf_counter()
    for i,row in enumerate(rows):
        task=lookup[row['prompt_number']]
        result=score(runtime,row['input']+row['raw_output'],task)
        result.update(variant=row['variant'],training_tokens=TARGET,temperature=row['temperature'],task_id=task['id'],
            completion_sha256=hashlib.sha256(row['raw_output'].encode()).hexdigest())
        results.append(result)
        if (i+1)%175==0: print(json.dumps(dict(stage='restricted_scoring',completed=i+1,total=700)),flush=True)
    if GENERATIONS.read_bytes()!=raw: raise ValueError('Generation source moved during scoring')
    report=dict(schema_version=1,benchmark_id=benchmark['benchmark_id'],benchmark_sha256=sha256_file(BENCHMARK),
        source_file=GENERATIONS.name,source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),
        manifest_sha256=sha256_file(MANIFEST),decoding_policy=policy,runtime=runtime.runtime_identity,
        scope=benchmark['scope'],scorer_sha256=sha256_file(Path('tools/score_saved_python.py')),
        orchestration_sha256=sha256_file(Path(__file__)),sandbox_source_sha256=sha256_file(Path('src/eval/wasi_python.py')),
        generation_checkpoint_hashes_verified=True,baseline_evaluation_identity_matched=True,
        checkpoint_references={tag:entry['model_weights'] for tag,entry in entries.items()},
        summaries=summarize(results),completions=results,scoring_seconds=time.perf_counter()-begin,
        limitations=baseline['limitations'])
    write_json(output,report)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['manifest','evaluate','generate','score']);args=parser.parse_args()
    if args.mode=='score':
        entries=json.loads(MANIFEST.read_text())['matched_gate_checkpoints'];score_gate(entries);return
    gpu_allowed();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if args.mode=='manifest': manifest();return
    entries=json.loads(MANIFEST.read_text())['matched_gate_checkpoints']
    if args.mode=='evaluate':
        torch.use_deterministic_algorithms(True);evaluate_gate(entries)
    else: generate(entries)

if __name__=='__main__':main()
