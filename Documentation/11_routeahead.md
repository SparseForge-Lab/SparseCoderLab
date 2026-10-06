# RouteAhead

The tiny predictor accepts current hidden state, MTP draft state and recent multi-hot routing history. It predicts independent future expert probabilities per capacity layer for horizons 1,2,3. Multi-label training supports Top2 targets. Prediction never changes real routing or language-model outputs.

The predictor, bounded feature extractor, checkpointable trainer, metrics and simulator interface are implemented in `tools.routeahead_lab`. Hidden/history/MTP features come from a frozen base checkpoint. The predictor fits train-document features and evaluates val-document features; it records its base checkpoint checksum. `--max-wall-minutes`, `--steps` and `--resume` bound/restore optimizer, scheduler, cursor and RNG. Training on representative data is a later experiment; no predictive benefit is assumed. Never report fitting-trace recall as a success.

Report Top1 and Top-k expert recall, precision, page recall, bytes issued/useful/wasted and simulated latency hidden. Compare to no prefetch and simple frequency/history/task-profile prediction under the same bandwidth and byte-capacity budgets. Classification recall is insufficient if wasted transfers or cache pollution increase end-to-end stall. Actual asynchronous prefetch integration remains future runtime work.
