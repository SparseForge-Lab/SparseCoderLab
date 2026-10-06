# Expert Atlas and physical packing

The atlas builds expert load counts, task/language usage and an undirected affinity graph from token co-activation and consecutive-token transitions. Expert IDs retain physical layer identity. The deterministic greedy packer starts from a frequent expert and fills a page with high-affinity experts; a sequential-ID packer is the comparison.

Page sizes are 1, 2, 4 and 8 experts. Cache capacity stays fixed in bytes as page size changes, so packed pages do not get extra capacity for free. Packing and frequency/task profiles fit the first trace half; simulation evaluates the held-out second half. Experts first seen in held-out traces get zero-affinity entries rather than fitted future statistics.

Packing changes only storage mappings inside the simulator. It never changes weights, gate decisions, dispatch or model outputs. Graph packing can lose if overfetch exceeds reuse. Compare LRU, LFU, frequency pinning, task-profile prefetch and predictor prefetch with the same bytes and trace. After training use representative tasks, multiple trace folds and seeds before choosing a layout.
