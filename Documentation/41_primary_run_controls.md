# Sequential continuation and operating-condition evidence

`tools/research_schedule.py` is orchestration only; it does not alter model/training/evaluation mathematics or the frozen training hash. Phases run CLI stages sequentially and stop on the first failed child. An actual Windows process query refuses to start while another primary `tools.research_train` process is running. This guard was verified against the live Sparse50M wrapper/worker without launching a duplicate GPU job.

```powershell
.venv\Scripts\python.exe -m tools.research_schedule --phase catchup
```

Catchup verifies/skips completed Sparse20M/50M, runs Ngram20M then50M, and updates CPU-only collection/diagnostics/review metrics/ledger/releases/figures after each stage. It exits before the required matched50M review. `analysis/research_v1/matched_50m.py` can write the ten requested review questions only after all three20M/50M immutable evaluations and verified release snapshots are complete. The agent also reviews those measurements before continuing.

```powershell
.venv\Scripts\python.exe -m analysis.research_v1.matched_50m
.venv\Scripts\python.exe -m tools.research_schedule --phase 70m
.venv\Scripts\python.exe -m tools.research_schedule --phase 100m
```

70M requires all primary50M summaries and the completed matched review;100M additionally requires all70M summaries. The underlying CLI verifies completed checkpoint hashes and frozen identities, or resumes cumulative model/optimizer/scheduler/RNG/cursor. It never reinitializes a later milestone as a fresh shorter schedule. Each phase has a durable `results/research_v1/schedule_state.json`; actual child process/session status determines whether work is running, not that JSON alone. No subsequent prompt or75M architecture starts here.

`analysis/research_v1/hardware_monitor.py` records NVIDIA temperature, power, utilization, used GPU memory and SM-clock snapshots every30 seconds in `results/research_v1/hardware_samples.jsonl`. Coverage starts partway through Sparse20M→50M; earlier Dense/Sparse operating conditions remain unknown. Last logged per-model counters accompany timestamps. These are bounded snapshots, not exact energy, lifetime peaks, GPU-kernel occupancy, or a randomized controlled benchmark. The user-requested CPU playground can affect host/wall throughput. Stop the monitor using its documented `hardware_monitor.stop` marker before the final archive, allowing the process to exit and files to become stable.

Final CPU integrity inspection checks all12 primary checkpoint hashes, architecture keys/shapes, finite weights and optimizer moments, loadable optimizer/scheduler, steps/cursors/RNG/frozen identities, monotonically increasing cumulative time, ablation document identities, final rolling/milestone equality, no training epoch repetition and unchanged Prompt-1 archive. Actual final tests/gates follow training completion; no heavyweight GPU validation overlaps a live training child.
