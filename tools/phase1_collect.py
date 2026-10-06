import csv, json, statistics
from pathlib import Path

rows = []; curves = []; diagnostics = {}
for tag in ('dense', 'sparse', 'memory'):
    root = Path('experiments') / ('phase1a_' + tag)
    summary = json.loads((root / 'summary.json').read_text())
    logs = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    training = [r for r in logs if 'train_loss' in r]
    routes = [v for r in training for v in r['router'].values()]
    gradients = [r['grad_norm'] for r in training]
    row = {k: summary.get(k) for k in ('model','seed','tokens_seen','step','stored_params','active_params_est','training_flops_est','wall_time','tok_s','train_step_tok_s','vram_peak','val_loss','code_val_loss','general_val_loss','bits_per_token','source_hash','data_hash','tokenizer_hash')}
    row.update(router_entropy_mean=statistics.mean(r['entropy'] for r in routes) if routes else None,
               expert_load_cv_mean=statistics.mean(r['imbalance'] for r in routes) if routes else None,
               grad_norm_sample_min=min(gradients), grad_norm_sample_max=max(gradients),
               completed_without_nonfinite=True,
               status='measured', scope='synthetic engineering comparison, one seed')
    rows.append(row)
    for r in logs:
        curves.append({k: r.get(k) for k in ('step','tokens_seen','wall_time','train_loss','val_loss','bits_per_token','grad_norm','tok_s')} | {'model': summary['model']})
    diagnostics[tag] = {'router_sampled_steps': len(training), 'router': {}}
    if routes:
        diagnostics[tag]['router'] = {layer: {'entropy_mean': statistics.mean(r['router'][layer]['entropy'] for r in training), 'cv_mean': statistics.mean(r['router'][layer]['imbalance'] for r in training), 'cv_max': max(r['router'][layer]['imbalance'] for r in training), 'last_sample': training[-1]['router'][layer]} for layer in training[0]['router']}
        for layer, values in diagnostics[tag]['router'].items():
            values['last_20_sample_cv_mean'] = statistics.mean(r['router'][layer]['imbalance'] for r in training[-20:])
            values['last_20_sample_max_expert_share'] = max(max(r['router'][layer]['load'])/sum(r['router'][layer]['load']) for r in training[-20:])
    if tag == 'memory':
        memories = [r['ngram'] for r in training]
        diagnostics[tag]['memory'] = {'first': memories[0], 'last': memories[-1],
            'gate_mean_over_samples': statistics.mean(r['gate_mean_last_microbatch'] for r in memories),
            'logged_steps_with_nonzero_table_gradient': [sum(r['table_gradients'][i]['gradient_norm_after_clip'] > 0 for r in memories) for i in range(4)],
            'scope': f'{len(memories)} sampled steps, last microbatch bucket statistics; not lifetime row counts.'}
        row['ngram_gate_first'] = memories[0]['gate_mean_last_microbatch']
        row['ngram_gate_last'] = memories[-1]['gate_mean_last_microbatch']
        row['ngram_table_gradient_mean_after_clip'] = statistics.mean(b['gradient_norm_after_clip'] for r in memories for b in r['table_gradients'])
        row['ngram_sample_rows_utilized_mean'] = statistics.mean(b['utilized_rows'] for b in memories[-1]['sample_statistics']['banks'])
        row['ngram_sample_collision_fraction_mean'] = statistics.mean(b['collision_fraction'] for b in memories[-1]['sample_statistics']['banks'])
    for key in ('ngram_gate_first','ngram_gate_last','ngram_table_gradient_mean_after_clip','ngram_sample_rows_utilized_mean','ngram_sample_collision_fraction_mean'):
        row.setdefault(key, None)
    row.update(parameter_estimate_status='calculated', flops_estimate_status='calculated', comparison_reproducibility='unknown across seeds')
for name, values in [('phase1a_comparison',rows), ('phase1a_curves',curves)]:
    with Path('results', name + '.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=values[0]); writer.writeheader(); writer.writerows(values)
Path('results/phase1a_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
print(json.dumps(rows,indent=2))
