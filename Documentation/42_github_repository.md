# Public GitHub repository

Repository: https://github.com/SparseForge-Lab/SparseCoderLab

Project source is published under Apache-2.0 in the root LICENSE. Dataset annotations/collection terms, third-party dependencies and external checkpoint redistribution rights remain separate; no blanket dataset relicensing is implied. Public intent is transparent, reproducible architecture research, with experimental limitations and measured/calculated/planned/unknown evidence preserved.

Git and GitHub CLI are installed, authenticated via the existing system credential store, and configured for HTTPS. The existing correct origin URL and main branch were retained. GitHub initially contained only its README and Apache license. Earlier local engineering history is retained on `local-history/pre-github`, while publication main extends the existing GitHub main; no force push or remote-history rewrite is used.

The public snapshot includes source, tests, configuration, Documentation, README/Model Card, small measured CSV/JSON and reproducibility manifests, plus CPU research figures. It excludes raw/processed datasets, model weight formats, checkpoint trees, caches, environments, secrets, local backups, raw user instructions and selected process/run logs. Ignore rules do not delete canonical data or checkpoints. Personal home-folder prefixes were made portable in notes and metadata; exact originals are retained under ignored `.local/github_preparation/original_paths` with a hash index. The optional Strata launcher now resolves the current user's Desktop and accepts `SPARSECODERLAB_STRATA_LAUNCHER` for another location.

Publication review uses `tools/github_audit.py` to inspect candidate/index/history blobs, including small ZIP contents, for common credential formats, personal home paths and files over20MiB. It reports locations/counts only, never credential values. Pattern scanning is supplemented by reviewing the staged inventory. This is not a guarantee that every conceivable secret format is detectable. Private audit reports remain in `.local/github_preparation`; a value-free publication receipt records the verified scope.

Small experiment configs, provenance and summaries are public alongside result tables and release snapshots. Precise rules exclude checkpoint payloads, raw metrics/logs and caches; the broad experiments/*/ exclusion was removed. GOALS and web remain local directory exclusions at the user's explicit request. Raw request documents and live process coordination remain private; public operating_conditions.json records concurrent runtime scope without process IDs or task instructions. `.gitattributes` preserves byte-exact Python/config/JSON/CSV inputs across Windows/Linux checkouts so Git line-ending conversion cannot silently change frozen source/data/tokenizer/evaluation hashes. Staged blobs are checked against those canonical hashes before the commit. Markdown and shell scripts use portable LF line endings.

Offline Prompt-2 archives remain separate from public Git policy. `tools/archive_prompt.py` retains small canonical research logs, raw requests and historical source snapshots in the private numbered archive, even when excluded from public Git. Large datasets/checkpoints remain path/size/SHA references. Prompt-1 archive bytes and frozen research training math/data/tokenizer/evaluation identities remain unchanged.

During this infrastructure step, the valid Sparse50M training process was allowed to finish normally. No additional training was launched. The user explicitly selected **keep training stopped after the push**; Dense/Sparse50M are retained and the rest of Prompt-2 remains paused, without starting Prompt-3 or75M.

## Permanent publication policy

Classify new files before staging: research source, configs, tests, protocols, measured small CSV/JSON, figures, license/provenance and reproducibility manifests stay public. Raw prompts/task requests, agent workspaces, environments/caches, process logs, machine-specific state, credentials and large data/weights stay local with precise ignore rules. A .gitignore entry does not untrack existing files; remove those from the index while preserving canonical local copies. Scan the exact staged inventory before each publication. Historical research evidence is retained, and normal cleanup does not rewrite Git history.

The latest cleanup occurs during the live Ngram20M→50M continuation. The training process remains untouched; Roblox remains open. Older stop text above describes the completed initial GitHub setup, not the current training state.
