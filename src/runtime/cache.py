from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from src.moe.trace import global_experts

class Cache:
    def __init__(self, capacity_bytes: int, sizes: dict[int, int], policy: str = 'lru', pinned: set[int] | None = None):
        self.capacity = capacity_bytes; self.sizes = sizes; self.policy = policy; self.pinned = pinned or set()
        self.entries: dict[int, int] = {}; self.frequency = Counter(); self.clock = 0; self.evictions = 0
    def has(self, page: int) -> bool: return page in self.entries
    def touch(self, page: int) -> None:
        self.clock += 1; self.frequency[page] += 1
        if page in self.entries: self.entries[page] = self.clock
    def insert(self, page: int) -> None:
        self.touch(page)
        if self.has(page) or self.sizes[page] > self.capacity: return
        while sum(self.sizes[p] for p in self.entries) + self.sizes[page] > self.capacity:
            candidates = [p for p in self.entries if p not in self.pinned]
            if not candidates: return
            victim = min(candidates, key=lambda p: (self.frequency[p], self.entries[p], p)) if self.policy == 'lfu' else min(candidates, key=lambda p: (self.entries[p], p))
            del self.entries[victim]; self.evictions += 1
        self.entries[page] = self.clock

@dataclass
class IOConfig:
    expert_bytes: int
    vram_bytes: int
    ram_bytes: int
    ssd_bytes_s: float
    ssd_latency_s: float
    ram_bytes_s: float
    ram_latency_s: float
    prefetch_window_s: float = 0.0

def simulate(records: list[dict], mapping: dict[int, int], experts_per_layer: int, io: IOConfig,
             policy: str = 'lru', pinned: set[int] | None = None, predictions: dict[int, list[int]] | None = None) -> dict:
    if io.ssd_bytes_s <= 0 or io.ram_bytes_s <= 0: raise ValueError('Bandwidth must be positive')
    sizes = {page: count * io.expert_bytes for page, count in Counter(mapping.values()).items()}
    vram = Cache(io.vram_bytes, sizes, policy, pinned); ram = Cache(io.ram_bytes, sizes, policy)
    accesses = hits = ram_hits = reads = bytes_ssd = bytes_ram = 0; stall = hidden = 0.0
    prefetched: dict[int, tuple[float,bool]] = {}; wasted = useful = 0; prefetched_total = 0; wasted_ssd = wasted_ram = wasted_reads = 0
    def demand(page: int) -> None:
        nonlocal accesses, hits, ram_hits, reads, bytes_ssd, bytes_ram, stall, hidden, useful
        accesses += 1
        if vram.has(page): hits += 1; vram.touch(page)
        else:
            transfer = sizes[page] / io.ram_bytes_s + io.ram_latency_s
            if ram.has(page): ram_hits += 1; ram.touch(page)
            else:
                storage = sizes[page] / io.ssd_bytes_s + io.ssd_latency_s
                reads += 1; bytes_ssd += sizes[page]; ram.insert(page)
                transfer += storage
            if page in prefetched:
                saved = min(transfer, prefetched.pop(page)[0]); transfer -= saved; hidden += saved; useful += sizes[page]
            bytes_ram += sizes[page]; stall += transfer; vram.insert(page)
    for index, record in enumerate(records):
        # Access each expert separately: a tiny cache may thrash even inside one token.
        for expert in global_experts(record, experts_per_layer): demand(mapping[expert])
        # Schedule predictions after observing current token, for future tokens only.
        if predictions:
            budget = io.prefetch_window_s
            for page in dict.fromkeys(predictions.get(index, [])):
                if page not in sizes or vram.has(page) or page in prefetched: continue
                transfer = sizes[page] / io.ram_bytes_s + io.ram_latency_s
                if not ram.has(page): transfer += sizes[page] / io.ssd_bytes_s + io.ssd_latency_s
                saved = min(transfer, budget)
                if saved <= 0: continue
                budget -= saved; prefetched[page] = (saved,not ram.has(page)); prefetched_total += sizes[page]
                # Model prefetch intent; demand completes residual transfer. Does not inject ready pages early.
            # One-token horizon expiration conservatively counts unused reads as waste/I/O.
        if index > 0 and predictions:
            wanted = {mapping[e] for e in global_experts(record, experts_per_layer)}
            for page in list(prefetched):
                if page not in wanted and page not in predictions.get(index, []):
                    wasted += sizes[page]; wasted_ram += sizes[page]
                    if prefetched[page][1]: wasted_ssd += sizes[page]; wasted_reads += 1
                    del prefetched[page]
    for page,(_,needed_ssd) in prefetched.items():
        wasted += sizes[page]; wasted_ram += sizes[page]
        if needed_ssd: wasted_ssd += sizes[page]; wasted_reads += 1
    # Wasted speculative reads consume SSD bandwidth; no free speculative traffic.
    bytes_ssd += wasted_ssd; bytes_ram += wasted_ram; reads += wasted_reads
    stall += wasted_ssd / io.ssd_bytes_s + wasted_reads * io.ssd_latency_s + wasted_ram / io.ram_bytes_s
    n = max(len(records), 1)
    return {'tokens': len(records), 'accesses': accesses, 'vram_hit_rate': hits / max(accesses, 1),
            'ram_hit_rate_on_vram_miss': ram_hits / max(accesses - hits, 1), 'bytes_loaded_per_token': bytes_ssd / n,
            'ram_to_vram_bytes_per_token': bytes_ram / n, 'page_reads_per_token': reads / n,
            'predicted_io_stall_s_per_token': stall / n, 'io_limited_tokens_s': n / stall if stall > 0 else None,
            'wasted_prefetch_bytes': wasted, 'correct_prefetch_bytes': useful, 'issued_prefetch_bytes': prefetched_total,
            'prefetch_latency_hidden_s': hidden, 'cache_churn': vram.evictions + ram.evictions,
            'assumptions': 'Serial demand I/O, fixed bandwidth/latency, shared limited prefetch window; partial transfers and one-token expiration. No measured decode throughput claim.'}
