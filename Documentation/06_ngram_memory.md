# Conditional N-gram memory

`src/memory/ngram.py` is independent of the transformer. Config controls enablement, injection layer, bank count, rows, dimensions, hash orders/seeds and gate bias. Default: two bigram banks plus two trigram banks, four times 131,072 rows times 8 dimensions = 4,194,304 table parameters. Projection and scalar learned gate add 10,561 parameters.

Hashing consumes only tokens up to the current position, pads missing prefix tokens with zero, and uses explicitly bounded integer arithmetic modulo 2^31-1. It does not use Python's salted hash. Independent scalar reference and CPU/GPU parity tests verify the exact algorithm. Retrieved vectors are concatenated, projected and multiplied by a sigmoid gate before residual injection.

`NgramMemory.statistics` reports distinct N-grams, occupied rows, collisions, collision fraction, per-bucket distinct-count histogram and lookup-frequency histogram over the supplied sample. Treat those as sample diagnostics, not full-corpus collision estimates. Dense AdamW on the table is the micro-training reference; host residency and asynchronous prefetch are future inference work.

Disable with `memory.enabled: false`. The large candidate sweeps 8B, 12B, 16B, 20B and 25B memory parameters; this tiny table cannot select a winner for that sweep. Large inference should prefer RAM-resident memory and asynchronous host transfer. Random SSD lookup is not an assumed viable strategy.
