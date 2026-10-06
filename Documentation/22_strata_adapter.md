# Optional Strata adapter and supplied launcher

User-specified launcher: `~\Desktop\Alles-Start.bat`. It starts `Strata Coder starten.bat`, which selects `D:\Strata\strata-coder-iq1_m.json`, model `qwen3.8-flash-next-coder-iq1_m`, and server port 8095. The wrapper then opens Claude CLI in D:\workspace with skipped permission prompts. The architecture lab only needs expert traces, so the supplied script is used as configuration evidence rather than starting that extra interactive coding session.

Read-only source inspection found the existing native `--dump-routing PATH` feature. Records are little-endian int32 layer, int32 k, k int32 expert IDs, k float32 weights. The multi-token serving path writes layer-major records without position IDs and synthetic weights; token alignment cannot be safely inferred in that mode. The parser therefore preserves dispatch records and only converts exact ordered 0..47 single-token cycles to token traces. Malformed/truncated/ambiguous records fail explicitly.

`tools.strata_trace --capture` reuses the launcher's model/executable references in memory, with a short numeric token prompt, context256, no speculation/prefill and no expert cache, mmap experts, project-local output/logs and a 120-second timeout. It does not edit Strata, model files, profiles, launcher, Python or CUDA. It does not start the HTTP server or Claude. This changes inference policy for diagnostic observability, so it is not a production-launcher benchmark. Default invocation only parses a provided trace.

```powershell
.\.venv\Scripts\python.exe -m tools.strata_trace --capture --timeout-seconds 120 --max-new 8
.\.venv\Scripts\python.exe -m tools.strata_trace --input results/strata_routing.bin
```

Do not run alongside GPU training. `results/strata_status.json` records the actual bounded attempt. If the existing binary/config cannot expose clean traces, retain that failure and skip this optional dataset; never patch/rebuild the existing project as part of this task. Numeric token IDs create a routing fixture, not a representative coding task; later collect reviewed meaningful prompts with its own tokenizer.

Actual initial attempt exited with code 2 in about two seconds: this native IQ pack requires `--spec T>=2` and `--prefill`. Its speculative layer-major dump lacks explicit token/attempt IDs, so the strict single-token adapter cannot establish token positions. The optional real trace dataset was skipped. No Strata installation/config/launcher/model was changed and the interactive launcher was not executed.
