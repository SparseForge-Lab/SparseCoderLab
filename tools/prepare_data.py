from __future__ import annotations
import argparse, collections, hashlib, json, shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from src.config import load_config, fingerprint
from src.training.data import SPECIAL_TOKENS, bounded_jsonl, development_documents, digest, split_document, tokenizer_report

def prepare(cfg: dict, input_path: Path | None = None) -> dict:
    d = cfg['data']; directory = Path(d['shards']); directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists(): raise RuntimeError('Frozen shards already exist; use a new shard directory/config')
    cache = Path('data/cache'); cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / 'documents.jsonl'; seen = set(); stats = collections.Counter(); bytes_used = 0
    source = bounded_jsonl(input_path, int(d['cache_gb'] * 1024**3)) if input_path else development_documents(d)
    with cache_file.open('w', encoding='utf-8') as out:
        for doc in source:
            sha = digest(doc['text'])
            if sha in seen: continue
            doc.update(sha256=sha, split=split_document(doc['text'], d['seed'], d['eval_fraction']))
            line = json.dumps(doc, ensure_ascii=False) + '\n'; size = len(line.encode('utf-8'))
            if bytes_used + size > d['cache_gb'] * 1024**3: break
            seen.add(sha); bytes_used += size; out.write(line); stats[f"{doc['split']}/{doc['kind']}/{doc['language']}"] += 1
    def documents():
        with cache_file.open('r', encoding='utf-8') as f:
            for line in f: yield json.loads(line)
    tokenizer_path = Path(d['tokenizer'])
    if tokenizer_path.exists(): tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=cfg['model']['vocab_size'], min_frequency=2, special_tokens=SPECIAL_TOKENS,
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tokenizer.train_from_iterator((doc['text'] for doc in documents() if doc['split'] == 'train'), trainer=trainer)
        if tokenizer.get_vocab_size() != cfg['model']['vocab_size']:
            raise RuntimeError(f"Corpus too small for exact {cfg['model']['vocab_size']} vocabulary: got {tokenizer.get_vocab_size()}")
        tokenizer.save(str(tokenizer_path))
    assert tokenizer.get_vocab_size() == cfg['model']['vocab_size'] <= 65536
    counts = {'train': 0, 'val': 0}; by_kind = collections.Counter(); provenance = collections.Counter()
    handles = {split: (directory / f'{split}.bin.tmp').open('wb') for split in counts}
    kinds = {split: (directory / f'{split}_documents.jsonl').open('w', encoding='utf-8') for split in counts}
    try:
        for doc in documents():
            split = doc['split']; tokens = [tokenizer.token_to_id('<bos>')] + tokenizer.encode(doc['text']).ids + [tokenizer.token_to_id('<eos>'), tokenizer.token_to_id('<doc>')]
            if sum(counts.values()) + len(tokens) > d['max_tokens']: break
            if bytes_used + (sum(counts.values()) + len(tokens)) * 2 > d['cache_gb'] * 1024**3: raise RuntimeError('Cache size budget exceeded')
            kinds[split].write(json.dumps({k: doc[k] for k in ('sha256', 'kind', 'language', 'license', 'source')} | {'start': counts[split], 'length': len(tokens)}) + '\n')
            np.array(tokens, dtype=np.uint16).tofile(handles[split]); counts[split] += len(tokens)
            by_kind[f'{split}/{doc["kind"]}'] += len(tokens); provenance[f'{doc["source"]}/{doc["license"]}'] += 1
    finally:
        for f in list(handles.values()) + list(kinds.values()): f.close()
    for split in counts: (directory / f'{split}.bin.tmp').replace(directory / f'{split}.bin')
    report = {'tokens': counts, 'document_stats': dict(stats), 'token_mixture': dict(by_kind), 'sources': dict(provenance),
              'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(), 'seed': d['seed'],
              'data_config': d, 'synthetic_only': input_path is None, 'cache_bytes': bytes_used,
              'shard_sha256': {s: hashlib.sha256((directory / f'{s}.bin').read_bytes()).hexdigest() for s in counts},
              'packing': 'Fixed offsets, EOS/doc separators; causal attention crosses document boundaries within a split.',
              'quality_scope': 'Synthetic dev corpus only: no educational or natural-code quality claims' if input_path is None else 'Explicitly supplied licensed corpus'}
    (directory / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    Path(d['manifest']).write_text(json.dumps(report, indent=2), encoding='utf-8')
    Path('results/tokenizer_quality.json').write_text(json.dumps(tokenizer_report(tokenizer), indent=2), encoding='utf-8')
    from tools.shard_data import shard
    return shard(cfg)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/dense_compute.yaml'); p.add_argument('--input-jsonl', type=Path)
    a = p.parse_args(); print(json.dumps(prepare(load_config(a.config), a.input_jsonl), indent=2))
