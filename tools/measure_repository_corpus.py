"""Measure a frozen corpus's mixture, held-out coverage and CPU stream resume."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from src.config import load_config
from src.eval.research import freeze_evaluation
from src.eval.sampling import sample_documents
from src.training.data import PackedStream
from src.utils.hashing import sha256_file
from tools.shard_data import verify_shards

REQUIRED = ('Python', 'JavaScript', 'TypeScript', 'C', 'C++', 'Rust', 'Java',
            'Go', 'C#', 'SQL', 'Bash', 'HTML/CSS')


def measure(cfg, config_path, finalization, overlap_path, output):
    if output.exists():
        raise FileExistsError('Use a new measurement report')
    directory = Path(cfg['data']['shards'])
    final = json.loads(finalization.read_text(encoding='utf8'))
    overlap = json.loads(overlap_path.read_text(encoding='utf8'))
    documents = Path('data/research_v2_real/documents') / (cfg['data']['dataset_version'] + '.jsonl')
    if not final['preprocessing_passed'] or final['documents_sha256'] != sha256_file(documents):
        raise ValueError('Final document identity differs')
    if overlap['documents_sha256'] != final['documents_sha256'] or overlap['matching_documents']:
        raise ValueError('Final overlap screen differs or has unresolved matches')
    verify_shards(cfg)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf8'))
    if sha256_file(directory / 'manifest.json') != final['manifest_sha256']:
        raise ValueError('Final packed manifest differs')
    rows = {}
    languages = {}
    families = {}
    resume = {}
    for split in ('train', 'val'):
        with (directory / f'{split}_documents.jsonl').open(encoding='utf8') as handle:
            rows[split] = [json.loads(line) for line in handle]
        counts, tokens = Counter(), Counter()
        for row in rows[split]:
            if row['kind'] == 'code':
                counts[row['language']] += 1
                tokens[row['language']] += row['length']
        languages[split] = {key: dict(documents=counts[key], tokens=tokens[key]) for key in REQUIRED}
        families[split] = sorted({row['repository_family'] for row in rows[split]})
        stream = PackedStream(directory, split, cfg['training']['context'], cfg['data']['seed'])
        cases = []
        for cursor in (0, stream.count - 3, stream.count + 5):
            stream.cursor = cursor
            stream.next(4, 'cpu')
            resumed = PackedStream(directory, split, cfg['training']['context'], cfg['data']['seed'], stream.cursor)
            x, y = stream.next(4, 'cpu'); rx, ry = resumed.next(4, 'cpu')
            same = (torch.equal(x, rx) and torch.equal(y, ry) and
                    torch.equal(stream.last_segment_ids, resumed.last_segment_ids) and
                    stream.last_prediction_count == resumed.last_prediction_count)
            cases.append(dict(start_cursor=cursor, saved_cursor=stream.cursor-4, next_batch_exact=same))
            resumed.tokens.close()
        resume[split] = cases
        stream.tokens.close()
    selection = sample_documents(rows['val'], 64, cfg['training']['context'], 42,
        lambda row: 'code/' + row['language'] if row['kind'] == 'code' else row['kind'], minimum_length=129)
    held_out = {}
    for key, selected in selection.items():
        predictions = sum(min(cfg['training']['context']+1, row['length']-row['evaluation_offset'])-1 for row in selected)
        held_out[key] = dict(documents=len(selected), predictions=predictions,
            sufficient=len(selected)>=16 and predictions>=8192)
    required_strata = ['code/'+key for key in REQUIRED] + ['general', 'technical']
    index = freeze_evaluation(cfg)
    index_path = Path('results/research_v2_real') / f"evaluation_index_{cfg['data']['dataset_version']}.json"
    train_tokens = manifest['tokens']['train']
    mixture = {key: dict(tokens=value, fraction=value/train_tokens) for key, value in manifest['token_mixture'].items() if key.startswith('train/')}
    family_overlap = sorted(set(families['train']).intersection(families['val']))
    passed = (not family_overlap and
        all(languages['train'][key]['documents'] for key in REQUIRED) and
        all(held_out.get(key, {}).get('sufficient', False) for key in required_strata) and
        all(case['next_batch_exact'] for cases in resume.values() for case in cases))
    receipt = dict(schema_version=1, passed=passed, dataset_version=cfg['data']['dataset_version'],
        config_sha256=sha256_file(config_path), finalization_sha256=sha256_file(finalization),
        overlap_sha256=sha256_file(overlap_path), manifest_sha256=final['manifest_sha256'],
        documents_sha256=final['documents_sha256'], tokenizer_sha256=manifest['tokenizer_sha256'],
        language_code_coverage=languages, held_out=held_out, required_held_out_strata=required_strata,
        repository_families_by_split=families, repository_family_overlap=family_overlap, stream_resume=resume,
        tokens=manifest['tokens'], measured_training_mixture=mixture,
        sampling_recipe='Natural retained token proportions; one shuffled packed-stream order; no oversampling or synthetic mixture weights.',
        proposed_phase_tokens=149_995_520, proposed_phase_epoch_fraction=149_995_520/train_tokens,
        proposed_phase_full_epoch_repeats=149_995_520//train_tokens,
        frozen_evaluation_index_sha256=sha256_file(index_path),
        evaluation_scope=index['scope'], script_sha256=sha256_file(Path(__file__)),
        limitations=['General reasoning is a narrow logic/mathematics subset, not broad knowledge coverage.',
            'Repository-family splits and bounded lexical screens cannot rule out all shared upstream ancestry or semantic contamination.',
            'Mixture counts include repository headers, FIM markers and BOS/EOS/doc tokens.',
            'One hashed document window per sampled document does not measure all repository tasks.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=passed, held_out=held_out, tokens=manifest['tokens'])), flush=True)
    if not passed:
        raise RuntimeError('Corpus coverage/resume gate failed; see the preserved report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    for key in ('finalization', 'overlap', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    measure(load_config(args.config), Path(args.config), args.finalization, args.overlap, args.output)
