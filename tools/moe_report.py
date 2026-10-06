"""Aggregate matched runtime evidence without inventing quality measurements."""
from __future__ import annotations
import csv, hashlib, json, platform, statistics
from pathlib import Path
from tools.moe_parity import source_hashes

def load(name): return list(csv.DictReader(Path('results',name).open(newline='',encoding='utf-8')))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def aggregate(rows):
    tokens=sum(int(r['tokens_per_loop']) for r in rows)
    return dict(repetitions=len(rows),training_step_tok_s=tokens/sum(float(r['training_step_seconds']) for r in rows),
                wall_tok_s=tokens/sum(float(r['wall_seconds']) for r in rows),
                training_step_tok_s_range=[min(float(r['training_step_tok_s']) for r in rows),max(float(r['training_step_tok_s']) for r in rows)],
                wall_tok_s_range=[min(float(r['wall_tok_s']) for r in rows),max(float(r['wall_tok_s']) for r in rows)],
                forward_ms=statistics.median(float(r['forward_ms']) for r in rows),
                forward_backward_ms=statistics.median(float(r['forward_backward_ms']) for r in rows),
                peak_allocated_bytes=max(int(r['allocated_peak']) for r in rows),
                peak_reserved_bytes=max(int(r['reserved_peak']) for r in rows),
                stored_params=int(rows[0]['stored_params']),routed_params=int(rows[0]['routed_params']),
                active_params_est=int(rows[0]['active_params_est']))

def main():
    from tools.moe_benchmark import gate
    gate()
    full=load('moe_fullmodel_benchmark.csv'); geometry=load('moe_expert_granularity.csv'); micro=load('moe_microbench.csv')
    models={tag:aggregate([r for r in full if r['tag']==tag]) for tag in ('dense','reference','grouped','memory')}
    shapes={tag:dict(aggregate([r for r in geometry if r['tag']==tag]),
                    experts=int(next(r['experts'] for r in geometry if r['tag']==tag)),
                    expert_ffn=int(next(r['expert_ffn'] for r in geometry if r['tag']==tag))) for tag in ('geometry12','geometry6','geometry4')}
    slowdown={scope:models['dense'][scope]/models['grouped'][scope] for scope in ('training_step_tok_s','wall_tok_s')}
    speedup={scope:models['grouped'][scope]/models['reference'][scope] for scope in ('training_step_tok_s','wall_tok_s')}
    memory_cost={scope:models['grouped'][scope]/models['memory'][scope] for scope in ('training_step_tok_s','wall_tok_s')}
    memory_cost['extra_allocated_bytes']=models['memory']['peak_allocated_bytes']-models['grouped']['peak_allocated_bytes']
    matrix={d:dict(median=statistics.median(v),minimum=min(v),maximum=max(v)) for d in ('balanced','moderate','strong','natural')
            for v in [[float(r['forward_backward_speedup_vs_reference']) for r in micro if r['backend']=='grouped' and r['distribution']==d]]}
    winner=max(shapes,key=lambda t:shapes[t]['training_step_tok_s'])
    worst=max(slowdown.values()); case='A' if worst<=1.5 else 'B' if min(speedup.values())>=1.2 else 'C'
    decision=dict(case=case,correctness_passed=True,models=models,full_model_grouped_speedup_vs_reference=speedup,
                  grouped_cost_ratio_vs_dense=slowdown,ngram_incremental_cost=memory_cost,geometry=shapes,
                  fastest_measured_geometry=winner,
                  fastest_measured_wall_geometry=max(shapes,key=lambda t:shapes[t]['wall_tok_s']),
                  geometry_interpretation='6 wins pure-step by only 0.22% over 4; 4 wins wall. Treat 6/4 as tied at this short-run noise level, both about 9% above 12. No quality winner or automatic architecture change.',
                  layer_forward_backward_speedups=matrix,
                  next_real_data_quality_phase_recommended=case=='A',next_quality_phase_started=False,
                  recommendation='Next prompt may prepare real data and cumulative 20M -> 50M -> 70M -> 100M quality checkpoints.' if case=='A' else 'Resolve remaining execution cost before long quality training.',
                  limitations=['Two short timing repetitions are engineering observations, not confidence intervals.',
                               'Wall loop excludes initialization, evaluation/checkpoint I/O; do not compare this wall scope directly to Phase1A historical wall rates.',
                               'Native BF16 training is non-deterministic even for reference vs reference. Deterministic tested parity passes; no universal bitwise continuation promise.',
                               'Geometry holds routed stored capacity constant; active compute changes. Fresh short loss smoke does not rank quality.',
                               'Microbenchmark memory includes the common checkpoint/input capture fixture. GPU utilization samples are not occupancy.',
                               'Real coding quality, routing specialization/locality, cross-seed reproducibility and useful memory benefit remain unknown.'],
                  archive_prompt='Prompt-1',archive='~/Data-Zip/Prompt-1/Prompt-1.zip')
    Path('results/moe_runtime_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    paths=[p for directory in ('src','tools','tests','configs') for p in Path(directory).rglob('*') if p.suffix in ('.py','.yaml')]
    paths.extend([Path('data/tokenizer.json'),Path('data/manifest.json')]); paths.extend(Path('data/shards').rglob('*.bin'))
    provenance=dict(source_sha256=source_hashes(),inputs_sha256={p.as_posix():sha(p) for p in sorted(paths)},
                    hardware=json.loads(Path('results/gpu_verification.json').read_text()),python=platform.python_version(),
                    benchmark_policy='Sequential RTX5070 GPU jobs, BF16 native SDPA, TF32 enabled, seed42, unchanged synthetic tokenizer/shards. Correctness uses deterministic algorithms and TF32 off. Warm layer 3/repeats10; full 3 warm updates/20 steps per each separate pure and wall loop, two reverse-order repetitions; geometry 10 steps each loop, two reverse-order repetitions.',
                    measured_csv_sha256={p:sha(Path('results',p)) for p in ('moe_microbench.csv','moe_fullmodel_benchmark.csv','moe_expert_granularity.csv')})
    Path('results/moe_benchmark_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(decision,indent=2))

if __name__=='__main__': main()
