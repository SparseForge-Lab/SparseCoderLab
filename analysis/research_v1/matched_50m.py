"""Write the ten requested matched50M answers from completed canonical evidence."""
from __future__ import annotations
import json
from pathlib import Path

R=Path('results/research_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def f(x):return f'{x:+.6f}'
def main():
    review=read(R/'review_evidence.json');matched=review['matched']
    assert '20004864' in matched and '50003968' in matched,'All three20M/50M evaluations required'
    assert all(read(Path('results/releases')/milestone/'summary.json')['status']=='matched_complete' for milestone in ('20M','50M')),'Verified release snapshots required'
    first,last=matched['20004864'],matched['50003968']
    a,b=first['deltas'],last['deltas'];ab0,ab1=first['ablation'],last['ablation']
    lines=['# Matched50M review — Prompt-2.5','',
        'All three models completed20,004,864 and50,003,968 cumulative predicted training tokens with frozen common inputs, seed42 and the same100M LR schedule. Negative model deltas favor the first named variant; positive residual-zero ablation deltas mean memory helps. This early review authorizes continuation to the already requested70M/100M comparisons, not a future scale-up.','',
        '| Model | Tokens | Mixed NLL | Code NLL | General NLL | Technical NLL |',
        '|---|---:|---:|---:|---:|---:|']
    for t,m in ((20004864,first),(50003968,last)):
        for tag,v in m['losses'].items():lines.append(f"| {tag} | {t:,} | {v['val_loss']:.6f} | {v['code_val_loss']:.6f} | {v['general_val_loss']:.6f} | {v['technical_val_loss']:.6f} |")
    lines += ['', '## Required review questions','',
        f"1. **Dense versus Sparse at20M:** Sparse-Dense mixed {f(a['sparse_minus_dense']['val_loss'])}; code {f(a['sparse_minus_dense']['code_val_loss'])} nats/token.",
        f"2. **Dense versus Sparse at50M:** Sparse-Dense mixed {f(b['sparse_minus_dense']['val_loss'])}; code {f(b['sparse_minus_dense']['code_val_loss'])}.",
        f"3. **Did Sparse improve relatively?** Signed mixed gap changed {f(b['sparse_minus_dense']['val_loss']-a['sparse_minus_dense']['val_loss'])}; code gap changed {f(b['sparse_minus_dense']['code_val_loss']-a['sparse_minus_dense']['code_val_loss'])}. Negative changes mean relative improvement; two checkpoints cannot establish the final trend.",
        f"4. **Ngram versus Sparse at20M:** mixed {f(a['ngram_minus_sparse']['val_loss'])}; code {f(a['ngram_minus_sparse']['code_val_loss'])}.",
        f"5. **Ngram versus Sparse at50M:** mixed {f(b['ngram_minus_sparse']['val_loss'])}; code {f(b['ngram_minus_sparse']['code_val_loss'])}.",
        f"6. **Is memory becoming more useful?** Same-checkpoint residual-zero mixed penalty {f(ab0['val_loss'])}→{f(ab1['val_loss'])}; code {f(ab0['code_val_loss'])}→{f(ab1['code_val_loss'])}. This measures dependence on learned memory within its model, separately from cross-model superiority.",
        f"7. **Code versus general:** Ngram-Sparse50M code {f(b['ngram_minus_sparse']['code_val_loss'])}, general {f(b['ngram_minus_sparse']['general_val_loss'])}; ablation code {f(ab1['code_val_loss'])}, general {f(ab1['general_val_loss'])}. Different fixed document samples have different baseline difficulty; raw effects are not universal population claims."]
    mi=[];health=[]
    for tag in ('sparse','memory'):
        old,new=first['routing'][tag],last['routing'][tag];assert old and new
        mi.append(tag+': '+', '.join(f"layer{k} {old[k]['category_expert_mutual_information_bits']:.4f}→{new[k]['category_expert_mutual_information_bits']:.4f} bits" for k in ('2','5','8')))
        health.append(tag+': '+', '.join(f"layer{k} CV{v['pooled_load_cv']:.3f}/max share{v['max_share']:.3f}/unused{v['unused_experts']}" for k,v in new.items()))
    lines += ["8. **Routing specialization:** "+'; '.join(mi)+'. These are descriptive category associations, not proof of useful semantic specialization. Tiny Rust/SQL strata remain flagged.',
        "9. **Routing health:** "+'; '.join(health)+'. Checkpoint completion means no predeclared persistent severe collapse/nonfinite condition fired; full route counts/transitions and sampled gradients remain available.',
        "10. **Runtime against Prompt-1:** cumulative50M training cost ratios "+', '.join(f"{tag}/Dense {value:.3f}x" for tag,value in last['cumulative_training_cost_ratios'].items())+'. Prompt-1 controlled sparse/Dense was1.376x and Ngram added3.20% step time. Cumulative quality-run measurements are different scopes and can be affected by CPU playground work; do not overwrite the controlled benchmark.',
        '', '## Sampling, functionality and continuation','',
        'Paired repository/hostname bootstrap intervals are retained in results/research_v1/clustered_document_intervals.json. They quantify fixed document-sample variation, not training-seed variance. Seed1337 is decided only after all primary100M checkpoints.',
        'Six-task functional pass counts at50M: '+', '.join(f'{tag} {count}/6' for tag,count in last['micro_pass_count_out_of_six'].items())+'. These bounded original arithmetic tasks do not establish useful coding-agent performance.',
        'Continue all three to70M, then all three to100M, sequentially. Keep the corpus/tokenizer/model geometry/policy frozen. No early architecture rejection for small gaps; no new features,75M model or Prompt-3.',
        '', 'Canonical sources: review_evidence.json, immutable checkpoint summaries, evaluation/ablation/micro-code files, routing_summary.json and clustered_document_intervals.json.']
    p=Path('Documentation/40_matched_50m_review.md');p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (R/'matched_50m_review.json').write_text(json.dumps({'status':'complete','matched_tokens':[20004864,50003968],
        'documentation':str(p.resolve()),'next_authorized_stage':'all70M then all100M','future_scale_up_started':False},indent=2),encoding='utf-8')
    print(json.dumps({'matched_review_written':str(p),'continue_primary_70m':True}))
if __name__=='__main__':main()
