"""All milestones, paired held-out deltas and diagnostic trend aggregation."""
from __future__ import annotations
import collections,csv,json,math,statistics
from pathlib import Path
from src.training.research import ENDPOINTS,RESULTS

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
def paired(a,b):
    first={r['sha256']:r for r in a['documents']};rows=[]
    for second in b['documents']:
        if second['sha256'] not in first:raise RuntimeError('Different held-out document identities')
        one=first[second['sha256']];assert one['predictions']==second['predictions']
        rows.append(dict(stratum=second['stratum'],delta=second['nll']-one['nll'],weight=second['predictions']))
    result={}
    for category in ('all','code','general','technical'):
        selected=[r for r in rows if category=='all' or r['stratum'].startswith(category)]
        if not selected:continue
        total=sum(r['weight'] for r in selected);mean=sum(r['delta']*r['weight'] for r in selected)/total
        variance=sum(r['weight']*(r['delta']-mean)**2 for r in selected)/total
        effective=total**2/sum(r['weight']**2 for r in selected);se=math.sqrt(variance/max(effective-1,1))
        result[category]=dict(weighted_delta=mean,approx_95_interval=[mean-1.96*se,mean+1.96*se],documents=len(selected),effective_documents=effective,
                              caveat='Descriptive paired document sampling interval, not seed uncertainty; repository correlation can make it optimistic.')
    return result
def main():
    table=[];curves=[];deltas=[];router={};memory={};ablation={};intervals={}
    for endpoint in ENDPOINTS:
        evaluations={}
        for tag in ('dense','sparse','memory'):
            p=Path(f'experiments/research_v1/{tag}/summary_{endpoint}.json')
            if not p.exists():continue
            r=json.loads(p.read_text());e=json.loads((RESULTS/f'{tag}_{endpoint}_evaluation.json').read_text());evaluations[tag]=e
            table.append({**{key:r.get(key) for key in ('model','tag','seed','tokens_seen','val_loss','bits_per_token','code_val_loss','general_val_loss','technical_val_loss','wall_tok_s','training_step_tok_s','wall_time','vram_peak','stored_params','active_params_est','training_flops_est','checkpoint','checkpoint_sha256','config_hash','data_hash','tokenizer_hash','source_hash')},
                          'wall_time_is_lower_bound':tag=='dense' and (RESULTS/'recovery_audit.json').exists()})
            curves.append(dict(model=tag,tokens=endpoint,step=r['step'],train_loss=None,val_loss=r['val_loss'],bits_per_token=r['bits_per_token'],
                               code_val_loss=r.get('code_val_loss'),general_val_loss=r.get('general_val_loss'),technical_val_loss=r.get('technical_val_loss'),
                               step_tok_s=r['training_step_tok_s'],grad_norm=None,wall_time=r['wall_time']))
            router[f'{tag}/{endpoint}']=e['router'];memory[f'{tag}/{endpoint}']=e['ngram']
            p=RESULTS/f'{tag}_{endpoint}_ablation.json'
            if p.exists():ablation[str(endpoint)]=json.loads(p.read_text())
        if len(evaluations)==3:
            a,b,c=[evaluations[t] for t in ('dense','sparse','memory')]
            deltas.append(dict(tokens=endpoint,sparse_minus_dense=b['val_loss']-a['val_loss'],ngram_minus_dense=c['val_loss']-a['val_loss'],ngram_minus_sparse=c['val_loss']-b['val_loss'],
                               sparse_code_minus_dense=b['code_val_loss']-a['code_val_loss'],ngram_code_minus_dense=c['code_val_loss']-a['code_val_loss'],ngram_code_minus_sparse=c['code_val_loss']-b['code_val_loss'],
                               sparse_general_minus_dense=b['general_val_loss']-a['general_val_loss'],ngram_general_minus_dense=c['general_val_loss']-a['general_val_loss'],ngram_general_minus_sparse=c['general_val_loss']-b['general_val_loss'],
                               sparse_technical_minus_dense=b['technical_val_loss']-a['technical_val_loss'],ngram_technical_minus_dense=c['technical_val_loss']-a['technical_val_loss'],ngram_technical_minus_sparse=c['technical_val_loss']-b['technical_val_loss']))
            intervals[str(endpoint)]={'sparse_minus_dense':paired(a,b),'ngram_minus_sparse':paired(b,c)}
    for tag in ('dense','sparse','memory'):
        p=Path(f'experiments/research_v1/{tag}/metrics.jsonl')
        if not p.exists():continue
        for line in p.read_text().splitlines():
            r=json.loads(line);curves.append(dict(model=tag,tokens=r['tokens_seen'],step=r['step'],train_loss=r.get('train_loss'),val_loss=r.get('val_loss'),bits_per_token=r.get('bits_per_token'),
              code_val_loss=None,general_val_loss=None,technical_val_loss=None,
              step_tok_s=r.get('step_tok_s'),grad_norm=r.get('grad_norm'),wall_time=r['wall_time']))
    write_csv(RESULTS/'training_comparison.csv',table);write_csv(RESULTS/'training_curves.csv',curves);write_csv(RESULTS/'milestone_deltas.csv',deltas)
    for name,value in [('router_diagnostics.json',router),('ngram_diagnostics.json',memory),('ngram_ablations.json',ablation),('paired_document_intervals.json',intervals)]:
        (RESULTS/name).write_text(json.dumps(value,indent=2))
    decision=dict(status='primary_complete' if len(table)==12 else 'in_progress',milestones_completed=len(deltas),quality_decision='pending human-readable analysis of all curves',
                  seed_replication='Pending100M ambiguity/trend review; not automatically doubling all models',large_model_transfer='unproven;100M micro-scale results do not establish60–75B performance')
    (RESULTS/'final_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
