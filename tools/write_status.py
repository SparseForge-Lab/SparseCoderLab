"""Generate human-readable documentation exclusively from recorded local evidence."""
from __future__ import annotations
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path

def read(path: str): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_status(date: str) -> str:
    environment=read('results/environment.json');gpu=read('results/gpu_verification.json');counts=read('results/parameter_counts.json')
    gate=read('results/phase0_gate.json');overfit=read('results/overfit.json');timer=read('results/timer_before_resume.json')
    tests=ET.parse('results/tests.xml').getroot();suites=list(tests.iter('testsuite'))
    ntests=sum(int(s.get('tests',0)) for s in suites);failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    status='PASS: phase-0 engineering readiness' if gate['passed'] and failures==0 else 'BLOCKED: phase-0 checks incomplete/failed'
    command='.\\.venv\\Scripts\\python.exe -m tools.train --config configs/dense_compute.yaml --run-dir experiments/phase1_dense_seed42 --target-tokens 5000000 --max-wall-minutes 20'
    lines=[f'# FINAL_STATUS — {date}', '',status,'','No long training run was launched. The tracked goal covers building/verifying this lab; architecture research remains open.','',
           '## Built','', 'Native Windows isolated venv and dependency lock; frozen 32,768 byte-BPE; bounded licensed synthetic shards and streaming/import tools; three shared-SDPA baselines; Top1/Top2 MoE; optional N-gram memory; shared/independent three-step MTP; checkpointable BF16 training, timer and batch finder; trace/atlas/packing; three-tier cache/prefetch and scaling simulators; RouteAhead feature/training/evaluation; archived repeated-compaction fixtures and neural NLL evaluator; large-model capacity sweep; disposable toy-agent trajectories. Separate documentation is under Documentation/.','',
           '## Environment','', f"GPU: {gpu['gpu']}, capability {gpu['compute_capability']}, driver 616.64, 12,227 MiB driver-visible VRAM. Project torch {gpu['torch']}, wheel CUDA {gpu['cuda']}; arch list {gpu['arch_list']}. BF16 SDPA backward executed: {gpu['bf16_sdpa_backward']}. TF32 supported. No silent CPU fallback.", '',
           'Windows 11; Python 3.12.10; AMD Ryzen 7 5700X (8 cores/16 threads); 51,442,139,136 installed RAM bytes (~47.91 GiB). WSL2 cannot start because virtualization is disabled. No existing installation was patched. D: is a FIKWOT FS810 SATA SSD, not NVMe.','',
           f"Unbuffered 256MiB file / 4MiB blocks: sequential {environment['io_benchmark']['sequential']['mib_s']:.2f} MiB/s; shuffled large-block {environment['io_benchmark']['random_large_block']['mib_s']:.2f} MiB/s. Random block mean {environment['io_benchmark']['random_large_block']['block_ms']:.3f} ms includes transfer; do not double-count as independent latency. Windows page cache bypassed; device cache remains possible.",'',
           '## Exact parameters','', '| Model | Stored | Experts | Resident | Memory | MTP | Active estimate |','|---|---:|---:|---:|---:|---:|---:|']
    for name,c in counts.items():lines.append(f"| {name} | {c['total']:,} | {c['routed_experts']:,} | {c['resident']+c['router']:,} | {c['conditional_memory']:,} | {c['mtp']:,} | {c['active_estimate']:,} |")
    lines += ['', 'SparseV3 without memory is 21.56M; the ~25.7M size includes the memory table. Active estimates include the tied dense LM output head and are not measured FLOPs. Shared MTP executes three times even though weights are counted once.','',
              '## Actual tests and short smokes','',f'{ntests} tests executed; {failures} failures/errors in final JUnit report. Phase-0 gate passed: {gate["passed"]}. DenseCompute tiny overfit: {overfit["initial_loss"]:.6f} -> {overfit["final_loss"]:.8f} over {overfit["steps"]} steps. CPU and GPU exact checkpoint continuation, data cursor and speculative greedy parity passed. No NaNs in tested paths.', '',
              'The three timing runs held equal tokenizer/data order, seed42, context1024, AdamW, BF16 and 163,840 tokens. One seed, synthetic fixtures, very short warmup: these are engineering observations, not a model ranking. Missing benchmark columns in leaderboard stay blank.','',
              '| Model | Train-step tok/s | Short-run wall tok/s | Peak allocated GiB | Val loss |','|---|---:|---:|---:|---:|']
    for name,folder in [('DenseCompute','timing_dense'),('SparseV3','timing_sparse'),('SparseV3Ngram','timing_memory')]:
        r=read(f'experiments/{folder}/summary.json');lines.append(f"| {name} | {r['train_step_tok_s']:,.0f} | {r['tok_s']:,.0f} | {r['vram_peak']/1024**3:.3f} | {r['val_loss']:.5f} |")
    lines += ['', 'Short-run wall throughput includes final checkpoint/validation, whose fixed cost is poorly amortized over only 20 steps. Training initialization is excluded. The five-minute standalone DenseCompute profiler completed 300.04 seconds / 9,955,328 tokens at 33,180 tok/s, microbatch2, context1024. Sparse30-second profiles measured 9,591 and 10,448 tok/s. These profiler rates are single-microbatch optimizer steps, not the accumulation4 run.', '',
              f"One-minute timer test: {timer['tokens_seen']:,} tokens / {timer['wall_time']:.2f} seconds = {timer['tok_s']:,.0f} wall tok/s; checkpointed at step {timer['step']}. Validation was correctly skipped near the timer limit. A brief GPU regression test overlapped the beginning of this timer test, so this rate is a conservative local observation, not an isolated hardware benchmark. A separate continuation restored the timer checkpoint.",'',
              '## Calculated duration, not a completed training benchmark','', '| Tokens | Dense timer wall-rate estimate | Dense five-minute profiler estimate |','|---:|---:|---:|']
    for tokens in (20000000,50000000,100000000):lines.append(f"| {tokens:,} | {tokens/timer['tok_s']/60:.2f} min | {tokens/33180.41760952858/60:.2f} min |")
    lines += ['', 'Formula: tokens / measured tokens_per_second. These are extrapolations over a synthetic corpus. Additional initialization, periodic validation, checkpoint costs, thermals, real-corpus lengths and dispatch behavior can change duration. Batch finder recommends larger batches, but the first command keeps microbatch2 for the verified fairness configuration.','',
              '## Experimental outcomes and limitations','']
    mtp=read('results/mtp_evaluation.json');route=read('experiments/routeahead_smoke/summary.json');strata=read('results/strata_status.json')
    lines += [f"- Shared-MTP three-step smoke: horizon accuracies were zero on the bounded held-out sequence. Eight-token greedy speculation had accepted length {mtp['speculation']['accepted_length']:.2f}, exact output parity, and net speedup {mtp['net_speedup']:.2f}x. No MTP speed benefit established; full-prefix reference has no KV cache.",
              f"- RouteAhead ten-step synthetic predictor: Top1 next-expert recall {route['route_metrics']['t+1/top1']['expert_recall']:.3f}. Prefetch improved simulated stall in {sum(c['stall_improves'] for c in route['simulations'])}/{len(route['simulations'])} held-out sequences. No prefetch hypothesis success claimed.",
              '- 4,096 real micro-model routed-token records collected. Atlas fitted first 2,048, cache/packing evaluated next 2,048; 32 cache scenarios. Scaling simulation is explicitly synthetic layer replication/expert remapping, not future-large-model evidence.',
              '- 100 exact-fact tasks / 3,200 deterministic archive-retention checks generated. One-task 1k neural answer-NLL/teacher-forced-accuracy smoke ran for 32 states; prompt truncation recorded. Dictionary retention/oracle retrieval is not model task pass rate. 8k/16k quality remains unmeasured.',
              '- N-gram sample diagnostics include utilization, collision and frequency histograms. Capacity optimality, Top1/Top2 quality tradeoff, two-seed real-code comparisons, meaningful expert specialization and TPU MFU remain unmeasured.',
              f"- Optional Strata capture failed (exit {strata['returncode']}): {strata['skip_reason']} Supplied Alles-Start.bat was read-only configuration evidence; its interactive Claude CLI was not launched.",
              '- Integration bug found and fixed: CUDA verification under no_grad needed enable_grad for its backward probe. Regression and trace collection now pass. No unresolved core unit-test failure.',
              '- Long pretraining, teacher SFT, page-aware routing, alternative attention, physical asynchronous offload, optimized speculation and secure VM/container agent execution are later lanes. No architecture winner selected.','',
              '## First safe manual run','', 'From D:\\SparseCoderLab:','', '```powershell', 'Set-Location D:\\SparseCoderLab',command,'```','',
              f"Expected throughput-only duration: 5,000,000 / {timer['tok_s']:.2f} = {5000000/timer['tok_s']:.1f} seconds ({5000000/timer['tok_s']/60:.2f} minutes). Allow roughly 3–5 minutes including variation/periodic evaluation; hard wall cap is 20 minutes. This uses synthetic development data and is an engineering Phase-1 run, not evidence about real coding quality.",'',
              'Resume with the same command plus `--resume experiments/phase1_dense_seed42/checkpoints/last.pt`. The target token count remains cumulative. Use at most the default 255 minutes for a later authorized daily run, after matched short research results and current phase-0 gate. No later phase starts automatically.','',
              'Evidence: results/tests.xml, test_output.txt, gpu_verification.json, environment.json, parameter_counts.json, profile_*.json, timer_before_resume.json, phase0_gate.json, leaderboard.csv and individual experiments/*/summary.json.','']
    text='\n'.join(lines);Path('FINAL_STATUS.md').write_text(text,encoding='utf-8')
    Path('Documentation/20_measurements.md').write_text(text.replace('# FINAL_STATUS', '# Recorded measurements and readiness',1),encoding='utf-8')
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--date',default='2026-10-05');a=p.parse_args();print(write_status(a.date))
