# Held-out routing and specialization protocol

## Measured Sparse20M, 2026-10-06

Held-out pooled expert usage is healthy at this checkpoint: layers2/5/8 load CV0.180/0.189/0.154, maximum share0.121/0.113/0.113 and zero unused experts. Category/expert mutual information0.0305/0.0497/0.0624 bits shows some conditional association. Adjacent same-expert fractions0.127/0.149/0.174 exceed independent baselines0.0860/0.0863/0.0853. General prose, Go, Java and C# have supported category biases; this is descriptive routing structure, not proof that the experts improve quality. Sparse is behind Dense in20M NLL. Full counts, transitions, sample sizes, enrichment and caveats are retained in `results/research_v1/sparse_20004864_evaluation.json` and `routing_summary.json`.

## Predeclared protocol

No specialization result is claimed before real training. The twelve-expert grouped Top1 research baseline remains unchanged; no locality/residency constraint is added. Training logs track soft router entropy, hard group counts/CV/max share/unused experts, balance loss and router gradients. These are sampled final microbatches, not an exact lifetime distribution.

At each immutable milestone, frozen real validation documents are grouped by language/code/general/technical/context. Collect token-weighted expert frequencies, soft and hard entropy, load CV, maximum share, unused experts, enrichment relative to the pooled held-out distribution, and within-document adjacent-token transition counts. Keep exact sample sizes and document IDs. Different category frequencies or enrichment may support emerging conditional structure; they are not proof of meaningful semantic specialization, useful locality or large-model transfer.

Persistent severe collapse is predeclared in Documentation34. Balanced uniform routing alone is not evidence of specialization; tiny categories are labelled insufficient by minimum16 documents/8192 predictions. Results/research_v1/router_diagnostics.json will aggregate all executed milestones and retain uncertainty/scope.
