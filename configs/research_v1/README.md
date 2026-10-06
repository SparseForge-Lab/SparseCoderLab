# Primary cumulative real-data configs

Prompt-2 authorizes fresh seed42 DenseCompute, grouped twelve-expert Top1 SparseV3 and unchanged Ngram variant, using the same frozen research tokenizer/corpus/order and matched AdamW schedule. Original synthetic configs remain intact. See Documentation34 for exact common milestone counts and disabled features.

Use python -m tools.research_train only after results/research_v1/readiness_gate.json passes. Runs continue cumulatively through20M/50M/70M/100M; do not restart each stage or use old reference sparse quality execution. No subsequent prompt is authorized automatically.
