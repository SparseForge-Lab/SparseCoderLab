# Grouped backend correctness

The original reference backend remains the default; runtime configs explicitly choose `model.moe_backend: grouped`. Both compute the same FP32 router softmax, Top1 selection, selected-probability multiplier and detached-load balancing loss. No overflow, capacity drop, renormalization or N-gram change is introduced. Top2 remains supported by reference only.

`src/moe/grouped.py` sorts expert IDs stably on GPU, computes within-expert positions, packs padded contiguous groups, executes batched forward and input-gradient matrix products, and restores token order. Capacity is the observed maximum group length rounded up to 16. One counts transfer per layer allocates that size. Padding wastes arithmetic under skew; it never truncates tokens. Bias-free zero padding contributes no output for real tokens.

Registered expert weights remain separate `ModuleList` parameters with identical names, shapes and enumeration order. Forward creates temporary differentiable stacks; checkpoints do not store a new representation. Existing Phase1A model and AdamW state load directly. Reference -> grouped -> reference state_dict equality is tested; no converter is required. Existing strict training-run config fingerprints still apply: loading compatible weights/optimizer does not authorize silently resuming a run under a different config.

Stacking gives absent experts zero gradients whereas reference gives None. `GroupedAdamW`, selected automatically by `make_optimizer`, restores None for experts absent across the entire accumulation interval before standard AdamW. This preserves weight decay, momentum and per-parameter step counters, including an expert used in an earlier microbatch but absent in the last. External training loops must use this optimizer factory (or implement equivalent absent-gradient handling); a plain AdamW on stacked gradients changes empty-expert update semantics. Closures are unsupported. This backend targets ordinary first-order LM training; higher-order autograd/compile paths are not validated.

An initially faster fully padded BMM backward failed a stronger trained-checkpoint continuation check. Under deterministic execution, its first-step expert weight-gradient differences were as small as 4.77e-7, but later Top1 routing/Adam updates amplified them: fifth-step maximum logit difference 2.6875 exceeded the predeclared 0.125 limit. It was rejected. The final `_BatchedLinear` keeps batched forward/input gradients and computes each expert's weight reduction on its true, unpadded token count. The rejected source, parity report and timing matrix remain under `results/history/Prompt-1/` and are not final-backend evidence.

Native BF16 execution also failed strict reference-versus-reference repeatability: fifth-step maximum logit difference 1.6875, mean 0.006375 and maximum embedding parameter drift 0.001180. This is baseline numerical nondeterminism, not proof of a grouped defect. Deterministic algorithms plus process-local `CUBLAS_WORKSPACE_CONFIG=:4096:8` made that control identical. The exact underlying nondeterministic kernel was not isolated; do not assert one. These switches are applied only by the parity tool and do not modify global CUDA settings or the original benchmark/training policy.

The final parity gate runs five updates each for small FP32 CPU, RTX5070 BF16 fresh full model, and the existing full Phase1A trained checkpoint with its old AdamW state. Identical weights/data are supplied to both paths. Logits, auxiliary/total loss, per-layer IDs/probabilities/selected weights, every router/resident/expert/other gradient and post-update parameter have reported max/mean absolute and relative-L2 differences in `results/moe_backend_parity.json`. The isolated MoE checks the intermediate output and exact routing tensors. All reported final differences are zero in these deterministic trials. This observation is not a universal bitwise-equality promise for every input/kernel/precision.

Limits were declared before the final trial: FP32 maximum logits/gradients/parameters 2e-5 and mean logits 2e-6; fresh BF16 logits max 0.05/mean 0.003, gradients max 0.003, parameters max 0.0005; trained BF16 logits max 0.125/mean 0.005 with the same gradient/parameter limits. BF16's roughly 0.78% mantissa spacing motivates finite rounding tolerance (0.125 is two ULP in [8,16]); observed equality is stronger than those limits. They were not enlarged to rescue the failed padded backward.

Forty regressions pass. Added coverage includes balanced, empty and collapsed experts; input and parameter gradients; six AdamW updates across changing occupancy; accumulation union; checkpoint roundtrip/order; RTX BF16 and grouped Top2 rejection. Initial collection exposed an existing `src.model`/router circular import; a lazy public `LanguageModel` export resolved it without altering model math. The failure log is retained. The final source-bound parity report is checked by the benchmark launcher, which refuses stale source or a failed gate.

Reproduce sequentially from the project root:

```powershell
.\.venv\Scripts\python.exe -m tools.moe_parity --control --deterministic
.\.venv\Scripts\python.exe -m tools.moe_parity --deterministic
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/moe_regression_tests.xml
```
