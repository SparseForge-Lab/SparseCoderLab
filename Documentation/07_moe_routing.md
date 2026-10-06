# Free expert routing

`FreeMoE` computes a float32 softmax, selects Top1 or Top2 and dispatches to unconstrained expert IDs. No tokens are dropped, no capacity ceiling is applied, and training does not constrain tokens to physical storage pages. Experts execute in an eager loop, which is readable and correct but has host/kernel dispatch overhead.

Top1 keeps the selected probability as the multiplier so language-model gradients reach the router. Top2 renormalizes its two selected probabilities to sum to one. The load-balancing loss is `num_experts * sum(detached_selected_load * mean_gate_probability)`, with config-controlled weight. Balance is measured; it does not prove specialization.

Each forward exposes selected experts, full probabilities, selected weights, entropy, loads and load coefficient of variation. `tools.routing_lab` writes one record per token with per-layer expert IDs, probabilities, entropy and document category/language. Global IDs include the physical layer, avoiding accidental merging of different layers' expert 0.

Traces from smoke checkpoints contain almost-untrained routing and are engineering fixtures. They cannot establish useful locality at scale. Soft locality losses/page routers are intentionally deferred until free-routing quality results exist; no page-aware router was introduced into the initial comparison.
