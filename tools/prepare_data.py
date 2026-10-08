from __future__ import annotations
import argparse, collections, hashlib, json, shutil
from pathlib import Path
from src.utils.hashing import sha256_file
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from src.config import load_config, fingerprint
from src.training.shards import TokenShardWriter
from tools.token_budget import selected_documents
from src.training.data import SPECIAL_TOKENS, bounded_jsonl, development_documents, digest, split_document, tokenizer_report

PROCESSING_SCRIPT_SHA256 = sha256_file(Path(__file__))
SPLIT_HELPER_SHA256 = sha256_file((Path(__file__).resolve().parents[1] / 'src/training/data.py'))

def prepare(cfg: dict, input_path: Path | None = None) -> dict:
    d = cfg['data']; directory = Path(d['shards']); directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists(): raise RuntimeError('Frozen shards already exist; use a new shard directory/config')
    Path(d['manifest']).parent.mkdir(parents=True, exist_ok=True)
    cache = Path(d.get('cache', 'data/cache')); cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / 'documents.jsonl'; seen = set(); stats = collections.Counter(); bytes_used = 0
    rejected = collections.Counter()
    input_provenance = {'source_revisions': collections.defaultdict(set), 'repositories': collections.defaultdict(set),
                        'languages': collections.Counter(), 'licenses': collections.Counter()}
    source = bounded_jsonl(input_path, int(d['cache_gb'] * 1024**3)) if input_path else development_documents(d)
    with cache_file.open('w', encoding='utf-8') as out:
        for doc in source:
            if doc.get('repository') and not d.get('dataset_version'):
                raise ValueError('Repository corpus requires an explicit dataset_version and new output paths')
            sha = digest(doc['text'])
            duplicate_key = doc.get('normalized_sha256', doc.get('raw_sha256', sha))
            if duplicate_key in seen:
                rejected['exact_or_normalized_duplicate'] += 1
                continue
            split_group = doc.get('repository_family', doc.get('repository'))
            if 'repository_split_assignments' in d:
                assignments = d['repository_split_assignments']
                if split_group not in assignments or assignments[split_group] not in ('train', 'val'):
                    raise ValueError('Missing or invalid reviewed repository-family split assignment')
                split = assignments[split_group]
            else:
                split = split_document(doc['text'], d['seed'], d['eval_fraction'], split_group)
            doc.update(sha256=sha, split=split)
            line = json.dumps(doc, ensure_ascii=False) + '\n'; size = len(line.encode('utf-8'))
            if bytes_used + size > d['cache_gb'] * 1024**3: raise RuntimeError('Input cache budget exceeded; refusing biased prefix selection')
            seen.add(duplicate_key); bytes_used += size; out.write(line); stats[f"{doc['split']}/{doc['kind']}/{doc['language']}"] += 1
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
    packed_stats = collections.Counter()
    writers = {split: TokenShardWriter(directory,split,d.get('shard_tokens',134217728)) for split in counts}
    selection = {}
    kinds = {split: (directory / f'{split}_documents.jsonl').open('w', encoding='utf-8') for split in counts}
    try:
        for doc in selected_documents(cache_file,tokenizer,d['max_tokens'],d['seed'],selection):
            split = doc['split']; tokens = [tokenizer.token_to_id('<bos>')] + tokenizer.encode(doc['text']).ids + [tokenizer.token_to_id('<eos>'), tokenizer.token_to_id('<doc>')]
            if bytes_used + (sum(counts.values()) + len(tokens)) * 2 > d['cache_gb'] * 1024**3: raise RuntimeError('Cache size budget exceeded')
            metadata_keys = ('sha256', 'kind', 'language', 'license', 'source', 'source_url', 'repository',
                             'revision', 'file_path', 'raw_sha256', 'normalized_sha256', 'source_license_sha256', 'role', 'git_blob',
                             'repository_family', 'deduplication', 'fim', 'license_basis', 'license_expression',
                             'member_files', 'sample_sha256', 'license_set', 'format')
            metadata = {k: doc[k] for k in metadata_keys if k in doc}
            kinds[split].write(json.dumps(metadata | {'start': counts[split], 'length': len(tokens)}) + '\n')
            writers[split].add(tokens); counts[split] += len(tokens)
            packed_stats[f"{split}/{doc['kind']}/{doc['language']}"] += 1
            by_kind[f'{split}/{doc["kind"]}'] += len(tokens); provenance[f'{doc["source"]}/{doc["license"]}'] += 1
            input_provenance['languages'][f"{split}/{doc['language']}"] += len(tokens)
            input_provenance['licenses'][doc['license']] += 1
            if doc.get('repository'):
                input_provenance['repositories'][split].add(doc['repository'])
            if doc.get('source') and doc.get('revision'):
                input_provenance['source_revisions'][doc['source']].add(doc['revision'])
    finally:
        for f in kinds.values(): f.close()
        physical = {split: writer.close() for split,writer in writers.items()}
    report = {'dataset_version': d.get('dataset_version', 'synthetic_dev' if input_path is None else 'unversioned-input'),
              'tokens': counts, 'document_stats': dict(packed_stats), 'reservoir_document_stats': dict(stats),
              'rejected': dict(rejected), 'token_mixture': dict(by_kind), 'sources': dict(provenance),
              'source_revisions': {key: sorted(value) for key, value in input_provenance['source_revisions'].items()},
              'repositories_by_split': {key: sorted(value) for key, value in input_provenance['repositories'].items()},
              'language_tokens_by_split': dict(input_provenance['languages']), 'license_document_counts': dict(input_provenance['licenses']),
              'processing_script_sha256': PROCESSING_SCRIPT_SHA256,
              'split_helper_sha256': SPLIT_HELPER_SHA256,
              'input_sha256': sha256_file(input_path) if input_path else None,
              'split_policy': ('Reviewed explicit repository-family assignments; global exact/normalized dedup precedes splitting.' if 'repository_split_assignments' in d else 'Declared repository-family group hash, falling back to repository hash; otherwise legacy document-content hash. Global exact/normalized content deduplication precedes splitting.'),
              'tokenizer_sha256': sha256_file(tokenizer_path), 'seed': d['seed'],
              'data_config': d, 'synthetic_only': input_path is None, 'cache_bytes': bytes_used,
              'physical_shards': physical, 'shard_sha256': {s: 'physical_shards authoritative' for s in counts},
              'token_budget_selection': selection,
              'packing_policy': d.get('packing_policy','document_isolated_v1' if input_provenance['repositories'] else 'legacy_causal'),
              'packing': 'BOS/body/EOS/doc with explicit manifest packing_policy. document_isolated_v1 requires segment masks, boundary target masking and Ngram reset in PackedStream/model.',
              'quality_scope': 'Synthetic dev corpus only: no educational or natural-code quality claims' if input_path is None else 'Explicitly supplied licensed corpus'}
    (directory / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    Path(d['manifest']).write_text(json.dumps(report, indent=2), encoding='utf-8')
    report_path = Path(d.get('tokenizer_report', 'results/tokenizer_quality.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(tokenizer_report(tokenizer), indent=2), encoding='utf-8')
    from tools.shard_data import shard
    return shard(cfg)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/dense_compute.yaml'); p.add_argument('--input-jsonl', type=Path)
    a = p.parse_args(); print(json.dumps(prepare(load_config(a.config), a.input_jsonl), indent=2))
