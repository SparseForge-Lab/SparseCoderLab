"""Build local release/card evidence from completed primary milestones only."""
from __future__ import annotations
import csv,datetime,json,subprocess
from pathlib import Path
import torch
from src.config import load_config
from src.training.research import ENDPOINTS,RESULTS,sha

FIELDS='date prompt phase experiment model milestone_tokens actual_tokens optimizer_step seed stored_params active_params train_loss val_loss code_loss general_loss technical_loss bits_per_token step_tok_s wall_tok_s elapsed_seconds peak_vram router_entropy router_load_cv max_expert_share unused_experts ngram_gate_mean ngram_ablation_val_delta ngram_ablation_code_delta ngram_ablation_general_delta ngram_ablation_technical_delta learning_rate git_commit source_hash config_hash data_manifest_hash tokenizer_hash checkpoint_hash status notes'.split()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for row in rows:writer.writerow({k:'null' if row.get(k) is None else row[k] for k in FIELDS})

def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    configs={t:load_config(f'configs/research_v1/{t}.yaml') for t in ('dense','sparse','memory')}
    counts=json.loads((RESULTS/'parameter_counts.json').read_text(encoding='utf-8'))
    ledger_path=Path('results/research_history.csv');existing=[]
    if ledger_path.exists():
        with ledger_path.open(newline='',encoding='utf-8') as f:
            existing=[{k:None if v=='null' else v for k,v in r.items()} for r in csv.DictReader(f)]
    indexed={(r['experiment'],str(r['actual_tokens']),str(r['seed'])):r for r in existing}
    releases={};card_runs=[]
    for tokens,nominal in zip(ENDPOINTS,(20000000,50000000,70000000,100000000)):
        records=[];metrics={};references={}
        for tag in ('dense','sparse','memory'):
            directory=Path('experiments/research_v1')/tag;s=directory/f'summary_{tokens}.json';e=RESULTS/f'{tag}_{tokens}_evaluation.json'
            if not s.exists() or not e.exists():continue
            summary=json.loads(s.read_text(encoding='utf-8'));evaluation=json.loads(e.read_text(encoding='utf-8'))
            checkpoint=Path(summary['checkpoint']);assert sha(checkpoint)==summary['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['metadata']['tokens_seen']==tokens and state['metadata']['step']==tokens//8192
            logs=[];samples=[]
            for line in (directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r.get('tokens_seen',tokens+1)<=tokens:samples.append(r)
                if r.get('tokens_seen')==tokens and 'train_loss' in r:logs.append(r)
            assert logs,'A milestone training metric is missing';log=logs[-1]
            routes=list(evaluation['router'].values());n=sum(r['tokens'] for r in routes)
            route_mean=lambda key:sum(r[key]*r['tokens'] for r in routes)/n if n else None
            gates=list(evaluation['ngram'].values());gate_tokens=sum(r['tokens'] for r in gates)
            ablation_path=RESULTS/f'{tag}_{tokens}_ablation.json';ablation=json.loads(ablation_path.read_text()) if ablation_path.exists() else None
            row={k:None for k in FIELDS}
            row.update(date=datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime,datetime.timezone.utc).isoformat(),prompt='Prompt-2',phase='research_v1_seed42',
                experiment=f'research_v1/{tag}',model=summary['model'],milestone_tokens=nominal,actual_tokens=tokens,optimizer_step=state['metadata']['step'],seed=summary['seed'],
                stored_params=summary['stored_params'],active_params=summary['active_params_est'],train_loss=log['train_loss'],val_loss=summary['val_loss'],code_loss=summary['code_val_loss'],
                general_loss=summary['general_val_loss'],technical_loss=summary['technical_val_loss'],bits_per_token=summary['bits_per_token'],step_tok_s=summary['training_step_tok_s'],wall_tok_s=summary['wall_tok_s'],
                elapsed_seconds=summary['wall_time'],peak_vram=summary['vram_peak'],router_entropy=route_mean('soft_entropy'),router_load_cv=route_mean('load_cv'),
                max_expert_share=max((r['max_share'] for r in routes),default=None),unused_experts=max((r['unused_experts'] for r in routes),default=None),
                ngram_gate_mean=sum(r['gate_mean']*r['tokens'] for r in gates)/gate_tokens if gate_tokens else None,
                learning_rate=state['optimizer']['param_groups'][0]['lr'],git_commit=commit,source_hash=summary['source_hash'],config_hash=summary['config_hash'],data_manifest_hash=summary['data_hash'],
                tokenizer_hash=summary['tokenizer_hash'],checkpoint_hash=summary['checkpoint_sha256'],status='completed',
                notes='Measured single-seed research result. Code is fixed language-balanced document diagnostic; insufficient Rust/SQL standalone groups flagged. '+('Dense cumulative wall is a lower bound after reporting recovery. ' if tag=='dense' else '')+'Active params/FLOPs are calculated estimates; source hash binds dirty source, git commit alone does not.')
            operating_receipt=RESULTS/'operating_conditions.json'
            if operating_receipt.exists():
                operating=json.loads(operating_receipt.read_text(encoding='utf-8'))
                resumed=datetime.datetime.fromisoformat(operating['continuation_started_utc']).timestamp()
                if checkpoint.stat().st_mtime>=resumed:
                    row['notes']+=' Runtime includes a user-authorized continuation with Roblox kept open; it is not a controlled speed comparison. See Documentation43 and operating_conditions.json.'
            if ablation:
                for field,key in [('val','val_loss'),('code','code_val_loss'),('general','general_val_loss'),('technical','technical_val_loss')]:row[f'ngram_ablation_{field}_delta']=ablation['delta_ablated_minus_normal'][key]
            key=(row['experiment'],str(tokens),str(row['seed']))
            if key in indexed:
                old=indexed[key]
                assert old['checkpoint_hash']==row['checkpoint_hash'],'Refusing to replace an earlier milestone with different checkpoint evidence'
                row=old
            else:indexed[key]=row
            records.append(row)
            references[tag]={'canonical_path':str(checkpoint.resolve()),'sha256':summary['checkpoint_sha256'],'size_bytes':checkpoint.stat().st_size,
                'architecture':configs[tag]['model'],'ngram':configs[tag]['memory'],'tokens':tokens,'summary_path':str(s.resolve()),'evaluation_path':str(e.resolve())}
            last_sample=lambda key:next(({'tokens':r['tokens_seen'],'step':r['step'],'value':r[key],
                'scope':'Closest preceding logged training sample; not the exact milestone update.'} for r in reversed(samples) if r.get(key) is not None),None)
            micro_path=RESULTS/f'{tag}_{tokens}_micro_code.json'
            micro=json.loads(micro_path.read_text(encoding='utf-8')) if micro_path.exists() else None
            metrics[tag]={'measured':evaluation,'micro_code':micro,'same_checkpoint_memory_ablation':ablation,
                'train_loss':log['train_loss'],'balance_loss':log.get('balance_loss'),'gradient_norm':log.get('grad_norm'),
                'training_seconds':state['metadata']['training_seconds'],'learning_rate':state['optimizer']['param_groups'][0]['lr'],
                'timing_scope':summary['scope'],'router_gradient_sample':last_sample('router'),'ngram_gradient_sample':last_sample('ngram')}
            card_runs.append({'tag':tag,'nominal_tokens':nominal,'actual_tokens':tokens,'measured_metrics':row,'checkpoint':references[tag],
                'micro_code':micro,'same_checkpoint_memory_ablation':ablation,
                'diagnostic_scopes':'Mixed is fixed packed validation; code/general/technical are fixed document samples. Train loss is exact endpoint update, not a smoothed mean. Router entropy/load CV are token-weighted category/layer means; max share/unused experts are maxima across category/layer samples. Gradients are preceding sampled training updates.'})
            del state
        release=Path('results/releases')/f'{nominal//1000000}M';release.mkdir(parents=True,exist_ok=True)
        snapshot={'status':'matched_complete' if len(records)==3 else 'partial' if records else 'planned','nominal_tokens':nominal,'actual_tokens':tokens,'models':references,
                  'scope':'Local release evidence, not a published model release. Checkpoints referenced, not copied.'}
        write_json(release/'summary.json',snapshot);write_json(release/'metrics.json',metrics);write_csv(release/'comparison.csv',records)
        table=['| Model | Actual tokens | Mixed NLL | Code NLL | General NLL | Step tok/s |','|---|---:|---:|---:|---:|---:|']
        for row in records:table.append(f"| {row['model']} | {row['actual_tokens']} | {float(row['val_loss']):.6f} | {float(row['code_loss']):.6f} | {float(row['general_loss']):.6f} | {float(row['step_tok_s']):.0f} |")
        (release/'README.md').write_text(f"# {nominal//1000000}M research snapshot\n\nStatus: {snapshot['status']}. One seed, fixed research tokenizer/corpus. Partial snapshots do not establish matched architecture conclusions.\n\n"+'\n'.join(table)+'\n\nCanonical checkpoint paths/SHA256/size/architecture are in summary.json; full evaluation and bounded routes in metrics.json. Rust/SQL standalone claims are insufficient. Dense wall is a documented lower bound. No SOTA, useful coding-agent or large-model-transfer claim.\n',encoding='utf-8')
        releases[f'{nominal//1000000}M']=snapshot['status']
    write_csv(ledger_path,list(indexed.values()))
    # Append new quality rows while preserving historical leaderboard bytes.
    leaderboard=Path('results/leaderboard.csv')
    if leaderboard.exists():
        with leaderboard.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f);columns=reader.fieldnames;old=list(reader)
        keys={(r['model'],r['tokens'],r['seed']) for r in old}
        with leaderboard.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columns)
            for record in card_runs:
                r=record['measured_metrics'];key=(r['model'],str(r['actual_tokens']),str(r['seed']))
                if key in keys:continue
                values={'model':r['model'],'seed':r['seed'],'stored_params':r['stored_params'],'active_params_est':r['active_params'],
                    'tokens':r['actual_tokens'],'training_flops_est':6*int(r['active_params'])*int(r['actual_tokens']),
                    'wall_time':r['elapsed_seconds'],'tok_s':r['wall_tok_s'],'val_loss':r['val_loss'],'code_val_loss':r['code_loss'],
                    'general_val_loss':r['general_loss'],'router_entropy':r['router_entropy'],'vram_peak':r['peak_vram']}
                writer.writerow({k:values.get(k) if values.get(k) is not None else '' for k in columns});keys.add(key)
    manifest=json.loads((RESULTS/'corpus_manifest.json').read_text(encoding='utf-8'))
    evidence={'status':'primary_complete' if len(card_runs)==12 else 'in_progress','measured':{'completed_runs':card_runs,'hardware':'NVIDIA RTX5070,12GiB; native CUDA BF16; see environment verification/profile evidence',
        'tokenizer_sha256':sha('data/research_v1/tokenizer.json'),'corpus_manifest_sha256':sha(RESULTS/'corpus_manifest.json'),'dataset':manifest},
        'calculated':{'architectures':{t:{'backbone':configs[t]['model'],'ngram':configs[t]['memory'],'context':configs[t]['training']['context'],'training_policy':configs[t]['training'],
        'stored_parameters':counts[t]['total'],'neural_stored_parameters':counts[t]['total']-counts[t]['memory_tables'],'memory_table_parameters':counts[t]['memory_tables'],'active_estimate':counts[t]['active_estimate']} for t in configs},
        'scope':'~25M stage is shorthand: dense16.57M/sparse21.56M/memory25.77M stored. Active parameter and6N FLOP arithmetic is not measured hardware compute.'},
        'planned':{'current_prompt':'Complete all primary100M curves; classify GOOD/MIXED/BAD and stop','future_unstarted':'~75M neural/stored stage with ~10M–25M memory candidates, possible100M/250M/500M/1B token gates; needs a subsequent explicit prompt'},
        'unknown':['training-seed variance until replication','useful coding-agent performance','60–75B scaling transfer','universal corpus contamination exclusion','independent legal verification of every source file'],
        'intended_use':'Architecture research and reproducibility, not production coding assistance',
        'unsupported_claims':['SOTA','superior to named production models','production ready','proved future60–75B performance'],
        'licensing_scope':'Code repository annotations filtered by permissive whitelist; FineWeb ODC-By collection/CommonCrawl terms do not certify per-page ownership. See Documentation32 and pinned cards.',
        'operating_conditions':{'primary_speed_study':'Prompt-1 controlled benchmark','quality_run_runtime':'Cumulative observations; later continuation with Roblox kept open by user request, plus CPU Dense playground. Not controlled speed evidence.','receipt':'results/research_v1/operating_conditions.json','documentation':'Documentation/43_training_resume_with_roblox.md'},
        'release_snapshots':releases,'source_repository':'https://github.com/SparseForge-Lab/SparseCoderLab',
        'source_license':'Apache-2.0; dataset/dependency licenses and checkpoint redistribution rights are separate',
        'publication_status':'Local checkpoint evidence; no Hugging Face model/checkpoint publication. Source publication uses the separately reviewed GitHub repository.'}
    write_json(Path('results/model_card_evidence.json'),evidence)
    timeline=[{'stage':'Phase0','scope':'Engineering validation; results/overfit.json and profile files'},
        {'stage':'Phase1A','scope':'Three5,005,312-token synthetic runs; engineering corpus, not real architecture-quality evidence'},
        {'stage':'Prompt-1','scope':'Controlled grouped MoE runtime study; dense25.27k/sparse18.36k step tok/s,1.38x sparse cost; not quality evidence'}]
    timeline.extend({'stage':f'Prompt-2/{milestone}','status':status,'evidence':f'results/releases/{milestone}/summary.json'} for milestone,status in releases.items())
    write_json(Path('results/research_timeline.json'),timeline)
    print(json.dumps({'ledger_rows':len(indexed),'releases':releases,'publications_performed':False}))

if __name__=='__main__':main()
