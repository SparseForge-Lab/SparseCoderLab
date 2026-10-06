# Run artifacts

Each run has config.json, provenance.json, metrics.jsonl, summary.json and checkpoints/last.pt. Checkpoints are trusted local pickle files, ignored by Git. Do not load an untrusted downloaded checkpoint with weights_only=False.

Initial smoke runs deliberately use different short endpoints for integration coverage. Compare the three timing_* runs only at their common 163,840-token endpoint for an engineering observation. They do not establish architecture quality, and the synthetic development corpus is not a real-code benchmark. timer_dense verifies wall-limit stop; resumed rows are cumulative continuation, not independent samples.

Profile checkpoints support pause/resume, but profiler batch-search training is not an architecture-quality comparison. Freeze configs/tokenizer/data and predeclare endpoints for future research. Checkpointed run token limits are cumulative on resume.
