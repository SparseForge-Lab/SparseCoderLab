# Three-step MTP

Design A uses one lightweight block, shared across three autoregressive draft steps. It mixes the current hidden state and previous-token embedding, then applies a small SwiGLU residual. Output weights reuse the base embedding. It is not three full transformer layers. Design B, selectable with `three_layer_mtp`, uses three independent lightweight blocks and remains an optional unmeasured ablation.

For position t, horizon h predicts input token t+h. Horizon 1 predicts the same next-token target as the base head. Teacher forcing provides token t+h-1 to the draft step; it never provides the token being predicted. Loss weights and accuracy/loss metrics are separate for h=1,2,3. Earlier horizons' state is reused for later steps. Prefix lengths shrink to mask unavailable future targets.

`src/eval/speculation.py` proposes three tokens and verifies against causal base logits. It accepts the matching greedy prefix and inserts the base correction at the first mismatch. A parity test compares exact greedy sequences. It measures accepted length and net tokens/sec including draft cost. The reference recomputes full prefixes and has no KV cache; its speed does not predict an optimized serving engine. It implements greedy decoding, not distribution-preserving stochastic speculation.

Base-quality A/B configs disable MTP. `dense_mtp.yaml` and `sparse_mtp.yaml` enable the same mechanism for later fair comparisons. Keep Design B only after net throughput/acceptance justify it. No useful acceptance has been established by random/tiny smoke models.
