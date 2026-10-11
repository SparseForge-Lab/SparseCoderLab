# Experiment archives and integrity verification

Canonical working files remain in `D:\SparseCoderLab`. Each study has a numbered goal record under `GOALS/Prompt-N.md` and a frozen copy under the current Windows user's home `Data-Zip/Prompt-N/`. These are existing archive identifiers. Historical experiments retain their original labels and file paths.

Allocate an unused archive number after checking existing directories. Record the objective, starting evidence, scope, success and stop criteria, experiments, results, failures, decisions and changed files. Keep source, configurations, tests, documentation and results in their canonical locations.

After the final review, mirror the goal as `GOAL.md`, generate `MANIFEST.json` and refresh `Prompt-N.zip`. Record each included canonical file's SHA-256, project-relative path and byte size. Include changed source, configurations, tests, documentation, benchmark outputs and the final report. Large checkpoints are referenced by path, size, hash and originating experiment without duplicating their payloads. Preserve earlier measurements and failed trials.

Verify ZIP CRC and every included research file's SHA-256 against the manifest and current canonical bytes. Verify the external goal mirror. The manifest and verification receipt are administrative self-references outside their own hash inventory; the ZIP embeds the external manifest. The receipt stores the ZIP hash separately to avoid a circular hash. No canonical file is moved or deleted.

`tools/archive_prompt.py --prompt N` finalizes an established study using `results/history/Prompt-N/starting_manifest.json`. It checks identity before refreshing the archive and rejects unexpected deletions or a different archive identity. It does not allocate a new study number.
