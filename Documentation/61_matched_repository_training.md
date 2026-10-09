# Matched repository training comparison

The controlled repository-data comparison is complete. All four models start from
the immutable 100,007,936-token states and use the frozen corpus described in the
[preflight record](60_frozen_repository_transition_preflight.md). Each adds
149,995,520 repository tokens in exactly 18,310 updates to reach 250,003,456
total tokens. The final model summaries, gate manifest, matched comparisons,
functional reports and generation records are retained under
`experiments/prompt5_v1/` and `results/repository_training_v1/`.

The [entry verification](../results/repository_training_v1/entry_verification.json)
checks the reviewed public revision, all eight canonical checkpoint hashes,
four separate initial copies, resolved configuration fingerprints, operative
source hashes and physical shards. The preparation run's 118 passing tests and
exact interrupted-resume evidence remain applicable. The first retained Dense
production update matches the preflight weights, Adam state, scheduler and RNG
exactly; [its verification](../results/repository_training_v1/first_update_verification.json)
also checks the frozen document/index identities. This update is counted and
resumed, rather than discarded or repeated under a fresh token counter.

## Training and recovery

The [campaign wrapper](../tools/repository_training_campaign.py) calls the
unchanged validated training engine. Architecture, tokenizer, document isolation,
BF16/TF32, effective batch of 8,192 tokens, optimizer groups and phase-relative LR
schedule remain fixed. Weights, Adam moments/steps and all RNG streams are
inherited; only the new-data cursor and LR phase were reset during preparation.
Resolved configurations live under `experiments/prompt5_v1/`.

GPU training runs sequentially. Every 50 inherited global steps, the logs record
loss, gradient norm, loader wait, optimizer timing, memory, routing and Ngram
diagnostics. Additional telemetry records LR, sampled CPU RSS, auxiliary routing
loss and expert frequencies. Training throughput excludes checkpoint/evaluation
overhead; loop wall-clock throughput includes it. Campaign events and summaries
also record process elapsed time and checkpoint-write/verification costs. Quick four-batch validation during
training is distinct from the full frozen evaluation used for final comparisons.

Atomic full-state checkpoints retain two rolling generations. Each newly written
state is loaded on CPU, checked for finite model/Adam tensors, hashed, and checked
against exact phase/token/cursor/scheduler identities before becoming a verified
resume candidate. A damaged or unverified rolling state cannot silently restart
the phase. The final gate preserves a full resume state and separate model-only
weights for each model. The storage allowance includes initial copies, both
rolling generations, four promoted full states, four model-only states and an
atomic replacement while keeping 20 GiB free. Large payloads remain outside Git.

The wrapper stops on nonfinite loss/gradients/state, identity or counter changes,
insufficient reserve, or allocated VRAM exceeding both the preflight scale and a
conservative additional allowance. Severe routing collapse requires ten
consecutive logged samples with over 98% of assignments concentrated in one
expert and extremely low soft entropy, or at least all but one expert unused.
Single-batch dead/underused counts are diagnostics, not proof of lifetime
starvation. A large quick-validation regression stops the run for diagnosis.
Failures preserve a receipt and the latest verified recovery states.

## Matched evaluation

The [gate evaluator](../tools/evaluate_repository_gate.py) uses the existing
frozen 128-batch mixed sample and hashed document windows for supported language,
code, technical and general NLL. It verifies the full-state loader and equality
with the final model-only weights without advancing training. Routing frequency,
entropy, concentration, unused/underused experts, domain enrichment and auxiliary
loss accompany Ngram gate/bucket diagnostics and residual-off ablations.

Functional generation reuses the existing sampler: 25 tasks, seven temperatures,
per-prompt seed 42, batch 4, BF16/TF32, 128 new tokens, greedy decoding at zero
temperature and top-k 40/top-p 0.95 otherwise. The scorer verifies checkpoint,
prompt, sampler, precision, decoding and restricted-runtime identities against
the canonical baseline. Generated code runs only in the existing WASI guest;
syntax, wrong answers, runtime errors, timeouts and repetition are reported
separately from passing all fixed cases.

All four final checkpoints passed the gate's hash and full-state checks. Mixed
NLL was 2.6803 for Dense, 2.6552 for Sparse, 2.6735 for Sparse +10M Ngram, and
2.6416 for Sparse +25M Ngram. Relative to each model's own inherited baseline,
the mixed-NLL deltas were +0.0078, +0.0037, +0.0169 and +0.0090 nats per
prediction, respectively. Code NLL improved for all four while technical NLL
rose for all four; general NLL improved for all four. Category and language
measurements, routing diagnostics, Ngram statistics, residual-off ablations,
and the full matched comparison are in
[`comparison_250M.json`](../results/repository_training_v1/comparison_250M.json).

The original 25-task greedy functional probe passed 0/25 Dense, 0/25 Sparse,
1/25 Sparse +10M Ngram, and 0/25 Sparse +25M Ngram. A separate 100-prompt greedy
evaluation found 0/100, 0/100, 2/100 and 1/100 correct completions in that same
order. Its four wordings per underlying task are correlated. Truncation
dominated the 100-prompt failures: 86 Dense, 96 Sparse, 75 Sparse +10M Ngram,
and 98 Sparse +25M Ngram outputs reached the 128-token limit. The full results
and per-completion provenance are in
[`unified_python_250M_functional_r3.txt`](../results/unified_python_250M_functional_r3.txt)
and its JSON companion. The 25-task, seven-temperature full-GPU benchmark
reports are available for [Dense](../results/prompt5_dense75_ref_250M_fullgpu_benchmark.txt),
[Sparse](../results/prompt5_sparse75_250M_fullgpu_benchmark.txt),
[Sparse +10M Ngram](../results/prompt5_sparse75_ngram10m_250M_fullgpu_benchmark.txt),
and [Sparse +25M Ngram](../results/prompt5_sparse75_ngram25m_250M_fullgpu_benchmark.txt).

Across the 100-prompt greedy generation, measured throughput ranged from 81.0
to 120.7 generated tokens per second and peak allocated GPU memory from 291 to
483 MB. These measurements cover this decoding workload on the recorded GPU;
they are not general training or inference guarantees. The 175 outputs per
model in the temperature sweep are repeated prompts, not independent tasks or
pass@k samples. Lower NLL or greater parseability alone does not establish
functional improvement or select a winner.

This is one training seed, a narrow general logic/mathematics slice and a small
fixed-case Python probe. Repository-family splits and bounded lexical screens
cannot exclude every semantic or historical overlap. Quality, routing and
efficiency conclusions will be limited to the matched experiment; larger models,
additional corpora and further training are outside its scope.
