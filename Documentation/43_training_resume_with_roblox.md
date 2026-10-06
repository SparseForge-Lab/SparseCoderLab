# Training resumed with Roblox open

On 2026-10-06 the user explicitly requested stopping the active GPU process and continuing training while Roblox stays open. Stopped only the identified llama-server.exe process (PID22884). Roblox PID13792 and the CPU Dense playground were retained. GPU memory fell from11730MiB to1633MiB before training.

CPU checkpoint verification passed: Dense and Sparse retain verified20M/50M states; Ngram starts fresh at seed42 with the same frozen corpus, tokenizer, model geometry and100M learning-rate schedule. The catchup phase verifies/skips completed Sparse stages, then runs Ngram20M and50M sequentially, including checkpoint/evaluation evidence and CPU reports. The all-three50M review precedes later70M/100M phases. No Prompt-3 or75M experiment starts.

Roblox remains open by explicit user request. Throughput, wall times and thermal samples from this continuation include concurrent desktop/game activity and cannot be treated as controlled architecture speed comparisons. The historical Prompt-1 controlled benchmark remains separate. Training safety/corruption/nonfinite checks remain enabled. The previous monitor stop marker is preserved as historical evidence before sampling resumes.

## First Ngram recovery checkpoint

At1,000 updates /8,192,000 predicted tokens /cursor8,000, CPU inspection verified strict model keys and shapes, finite weights and Adam moments, optimizer/scheduler loading, scheduler step, RNG presence and all frozen identities. A separately retained byte-identical recovery copy is stored under experiments/research_v1/memory/checkpoints/recovery_tokens_8192000.pt, SHA256 b6919c3c2420c81ae5c983619c7cc9044cc57269516670323328c21ee2c6f266. See results/research_v1/memory_first_recovery_verification.json. This is a periodic recovery point, not a completed20M quality milestone. Training was not interrupted.
