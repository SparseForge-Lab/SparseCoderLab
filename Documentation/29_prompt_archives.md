# Per-prompt goals and frozen archives

Canonical working files remain in `D:\SparseCoderLab`. Each new research prompt has one goal under `GOALS/Prompt-N.md` and a second, frozen copy under the current Windows user's home `Data-Zip/Prompt-N/`. This convention starts with this runtime prompt, Prompt-1; historical experiments are labelled context rather than retroactively renamed.

Before work, inspect existing `Prompt-*` directories, select the next unused integer, and create its goal and directory. Never assume Prompt-1 or overwrite an older goal. Record objective, starting evidence, non-goals, success/stop criteria, experiments, results, failures, decision and changed files. Keep source/config/tests/documentation/results in their canonical locations.

After all work and documentation, mirror the goal as `GOAL.md`, generate `MANIFEST.json`, and refresh `Prompt-N.zip`. Hash each included canonical file with SHA256 and record its original project-relative path and byte size. Include changed source, configs, tests, documentation, raw benchmark outputs, result tables, goal and final report. Record large checkpoint paths, sizes, hashes and originating experiments without duplicating their payloads. Retain earlier measurements and failed trials explicitly.

Open the final ZIP, verify CRC and every research file's SHA256 against both manifest and current canonical bytes; verify the external goal mirror. Record the archive path in final status and research log. The manifest and post-build verification receipt are administrative self-references outside their own hash inventory; the ZIP embeds the exact external manifest. The receipt records the ZIP's SHA256, so it is stored separately to avoid a circular hash. No canonical file is moved or deleted.

`tools/archive_prompt.py --prompt N` implements finalization for a goal already established with `results/history/Prompt-N/starting_manifest.json`. It checks identity before refreshing the current prompt's archive and refuses unexpected deletions or a different archive identity. It is a finalizer, not an automatic allocator of a new prompt number.

Prompt-1 was selected after inspection found no existing numbered archives. Its destination is `~\Data-Zip\Prompt-1\Prompt-1.zip`.
