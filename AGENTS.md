# Repository maintenance

Use code, resolved configurations and checked-in results as the source of truth. Read the current branch and latest experiment reports before updating documentation. Keep README, model card and research status consistent, separating completed results from ongoing work.

Commit subjects describe a concrete technical change in a short imperative sentence, for example `add dense benchmark results at 375M tokens` or `fix checkpoint resume handling`. Do not force first-person narration. Commit bodies explain material behavior or validation only when useful.

Public documentation describes methods, behavior and results. Do not include chat requests, numbered task labels, session narration, tool signatures or marketing filler. Technical uses of model, agent, LLM and evaluation prompt are appropriate in this research project. Preserve existing machine-readable identifiers, hashes, historical evidence and file paths unless a separate migration is requested.

Never invent benchmark values, test outcomes, manual work or authorship claims. Report checks only when actually run. Preserve unrelated work, inspect the exact staged diff, and exclude secrets and temporary files. Do not launch training or change evaluation policy for a documentation update. Rewriting published history requires an explicit user request and a verified recovery copy; use an expected-head lease so concurrent updates are not lost.
