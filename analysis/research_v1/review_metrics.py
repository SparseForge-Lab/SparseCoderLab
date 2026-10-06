"""CPU-derived matched milestone evidence, never infers unexecuted endpoints."""
from __future__ import annotations
import json,math
from pathlib import Path

R=Path('results/research_v1')
ENDPOINTS=(20004864,50003968,70000640,100007936)
TAGS=('dense','sparse','memory')
KEYS=('val_loss','code_val_loss','general_val_loss','technical_val_loss')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def gap_shape(values):
    if len(values)<2:return 'insufficient matched checkpoints'
    diff=[b-a for a,b in zip(values,values[1:])]
    if min(values)<0<max(values):return 'changes sign'
    if all(x<0 for x in diff):return 'monotonically decreasing signed gap'
    if all(x>0 for x in diff):return 'monotonically increasing signed gap'
    if all(abs(x)<.001 for x in diff):return 'approximately stable (<0.001 nats changes)'
    return 'non-monotonic signed gap'
def main():
    routes=read(R/'routing_summary.json') if (R/'routing_summary.json').exists() else {}
    intervals=read(R/'clustered_document_intervals.json') if (R/'clustered_document_intervals.json').exists() else {}
    output={'matched':{},'trends':{},'scope':'Calculated from completed immutable seed42 checkpoint summaries and deterministic held-out evaluations. Confidence intervals are document-cluster sampling only, not training-seed variance.'}
    for tokens in ENDPOINTS:
        paths={tag:Path(f'experiments/research_v1/{tag}/summary_{tokens}.json') for tag in TAGS}
        if not all(p.exists() for p in paths.values()):continue
        summaries={tag:read(p) for tag,p in paths.items()}
        micro={tag:read(R/f'{tag}_{tokens}_micro_code.json')['pass_count'] for tag in TAGS}
        ablation=read(R/f'memory_{tokens}_ablation.json')['delta_ablated_minus_normal']
        deltas={}
        for name,a,b in [('sparse_minus_dense','dense','sparse'),('ngram_minus_dense','dense','memory'),('ngram_minus_sparse','sparse','memory')]:
            deltas[name]={key:summaries[b][key]-summaries[a][key] for key in KEYS}
            deltas[name]['mixed_perplexity_ratio']=math.exp(deltas[name]['val_loss'])
        output['matched'][str(tokens)]={'losses':{t:{k:s[k] for k in KEYS} for t,s in summaries.items()},'deltas':deltas,'ablation':ablation,
            'micro_pass_count_out_of_six':micro,
            'cumulative_training_cost_ratios':{t:summaries['dense']['training_step_tok_s']/summaries[t]['training_step_tok_s'] for t in ('sparse','memory')},
            'allocated_vram_bytes':{t:s['vram_peak'] for t,s in summaries.items()},
            'routing':{t:routes.get(f'{t}/{tokens}') for t in ('sparse','memory')},'paired_document_cluster_intervals':intervals.get(str(tokens)),
            'speed_scope':'Cumulative training-loop time, not randomized repeated runtime benchmark. Prompt-1 remains the primary controlled study. CPU playground may affect host/wall time; the later user-authorized Roblox continuation also has concurrent game/desktop GPU use (Documentation43). Full thermal/power history is unavailable; partial30s samples beginning mid-Sparse20M→50M are retained in hardware_samples.jsonl.'}
    matched=list(output['matched'].values())
    for pair in ('sparse_minus_dense','ngram_minus_dense','ngram_minus_sparse'):
        output['trends'][pair]={key:{'values':[m['deltas'][pair][key] for m in matched],
            'shape':gap_shape([m['deltas'][pair][key] for m in matched])} for key in KEYS}
    output['trends']['ablation']={key:{'values':[m['ablation'][key] for m in matched],'shape':gap_shape([m['ablation'][key] for m in matched])} for key in KEYS}
    output['status']='primary_complete' if len(matched)==4 else 'partial' if matched else 'no_matched_milestone'
    (R/'review_evidence.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'status':output['status'],'matched_tokens':list(output['matched'])}))
if __name__=='__main__':main()
