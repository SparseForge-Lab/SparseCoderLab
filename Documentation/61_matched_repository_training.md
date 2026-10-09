# Matched repository training comparison

The controlled repository-data comparison is running. All four models start from
the immutable 100,007,936-token states and use the frozen corpus described in the
[preflight record](60_frozen_repository_transition_preflight.md). Each adds
149,995,520 repository tokens in 18,310 updates to reach 250,003,456 total tokens.
Final quality and functional results are pending.

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

The canonical greedy baseline is Dense 2/25 and the other three models 0/25.
The comparison will report all temperatures without treating 175 completions as
175 independent tasks or as pass@k. Lower NLL or greater parseability alone will
not establish functional improvement or select a winner.

This is one training seed, a narrow general logic/mathematics slice and a small
fixed-case Python probe. Repository-family splits and bounded lexical screens
cannot exclude every semantic or historical overlap. Quality, routing and
efficiency conclusions will be limited to the matched experiment; larger models,
additional corpora and further training are outside its scope.
