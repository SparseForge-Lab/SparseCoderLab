"""Build auditable Dataset V2 foundation views without duplicating source text."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from .schema import canonical_json, sha256_bytes, validate_record
from .strata_runtime import digest


def shingles(text: str, width: int = 8) -> set[bytes]:
    tokens = re.findall(r'[a-z0-9_]+', text.casefold())
    return {hashlib.sha256(' '.join(tokens[i:i+width]).encode()).digest()
            for i in range(max(1, len(tokens)-width+1))} if tokens else set()


class IndexedQuarantine:
    """Reference-overlap screen; held-out text is used only as a negative filter."""
    def __init__(self, entries, threshold=.8):
        self.threshold = threshold
        self.index = defaultdict(list)
        self.references = []
        seen = set()
        for label, text in entries:
            key = sha256_bytes(text.encode())
            if key in seen or not text.strip():
                continue
            seen.add(key)
            values = shingles(text)
            index = len(self.references)
            self.references.append((label, len(values)))
            for value in values:
                self.index[value].append(index)

    def inspect(self, text):
        counts = Counter(i for value in shingles(text) for i in self.index.get(value, ()))
        matches = [dict(reference=self.references[i][0], overlap=count/self.references[i][1])
                   for i, count in sorted(counts.items())
                   if count/self.references[i][1] >= self.threshold]
        return dict(status='flagged' if matches else 'clear_lexical_screen', matches=matches,
                    threshold=self.threshold, universal_contamination_claim=False)


def evaluation_registry(paths):
    entries, receipts = [], []
    fields = {'prompt', 'reference', 'answer', 'input', 'raw_output', 'completion', 'generated_text'}
    def visit(value, label):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in fields and isinstance(item, str):
                    entries.append((label + ':' + key, item))
                elif isinstance(item, (dict, list)):
                    visit(item, label + ':' + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, label + ':' + str(i))
    for path in sorted(set(map(Path, paths))):
        receipts.append(dict(path=path.as_posix(), sha256=digest(path)))
        with path.open(encoding='utf8') as stream:
            if path.suffix == '.jsonl':
                for i, line in enumerate(stream):
                    visit(json.loads(line), path.as_posix() + ':' + str(i+1))
            else:
                visit(json.load(stream), path.as_posix())
    if not entries:
        raise ValueError('no evaluation negatives registered')
    return IndexedQuarantine(entries), receipts


def frozen_assignments(directory: Path):
    assignments, heldout_hashes, receipts = {}, set(), []
    for split in ('train', 'val'):
        path = directory / f'{split}_documents.jsonl'
        receipts.append(dict(path=path.as_posix(), sha256=digest(path)))
        with path.open(encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                key = (row['repository'], row['revision'], row['file_path'])
                target = 'validation' if split == 'val' else 'train'
                if key in assignments and assignments[key] != target:
                    raise ValueError('frozen file appears in both splits')
                assignments[key] = target
                if split == 'val':
                    heldout_hashes.add(row['normalized_sha256'])
    return assignments, heldout_hashes, receipts


def foundation_category(row):
    if row.get('role') in {'test', 'tests', 'test_code'}:
        return 'issues_prs_commits_tests_reviews'
    if row.get('kind') == 'technical':
        return 'technical_documentation_text_to_code'
    if row.get('kind') == 'general':
        return 'algorithms_math_reasoning'
    return 'file_level_code'


def build(config: dict, output: Path):
    from tokenizers import Tokenizer
    from tools.repository_fim import restore, split_header

    if output.exists():
        raise FileExistsError('use a new versioned output; incomplete runs are retained for audit')
    output.mkdir(parents=True)
    started = time.perf_counter()
    source = Path(config['corpus'])
    plan_path = Path(config['source_plan'])
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    sources = {s['id']: s for s in plan['sources']}
    assignments, heldout, split_receipts = frozen_assignments(Path(config['frozen_splits']))
    paths = [p for pattern in config['evaluation_negatives'] for p in Path('.').glob(pattern)]
    screen, registry = evaluation_registry(paths)
    tokenizer_path = Path(config['tokenizer'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    source_hash = digest(source)
    blob = output / 'objects' / source_hash[:2] / source_hash[2:]
    blob.parent.mkdir(parents=True)
    # Copy once: a hard link would allow later edits to mutate the immutable object.
    shutil.copyfile(source, blob)
    if digest(blob) != source_hash:
        raise IOError('canonical source snapshot failed hash verification')
    counts, tokens, languages, rejected = Counter(), Counter(), Counter(), Counter()
    seen, families = set(), defaultdict(set)
    groups = defaultdict(list)
    manifests = {name: (output / f'{name}.jsonl.partial').open('wb')
                 for name in ('foundation', 'views', 'quarantine')}
    def emit(name, row):
        normalized = validate_record(row)
        manifests[name].write(canonical_json(normalized) + b'\n')
        return normalized
    try:
        with blob.open('rb') as stream:
            number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                number += 1
                row = json.loads(raw)
                source_info = sources.get(row['repository'])
                if source_info is None:
                    rejected['not_in_reviewed_source_plan'] += 1
                    continue
                if row['revision'] != source_info['revision']:
                    raise ValueError('source revision is not in the reviewed plan')
                if plan.get('license_allowlist') and row['license'] not in plan['license_allowlist']:
                    rejected['license_not_in_reviewed_allowlist'] += 1
                    continue
                if row['source_license_sha256'] != source_info['root_license_sha256']:
                    raise ValueError('source license evidence changed')
                key = (row['repository'], row['revision'], row['file_path'])
                split = assignments.get(key)
                if split is None:
                    rejected['not_in_frozen_retained_splits'] += 1
                    continue
                planned = 'validation' if source_info['planned_split'] == 'val' else source_info['planned_split']
                if split != planned:
                    raise ValueError('reviewed split differs from the frozen split')
                text = restore(row)
                _, body = split_header(dict(row, text=text))
                normalized_hash = sha256_bytes(body.replace('\r\n','\n').replace('\r','\n').rstrip().encode())
                if normalized_hash != row['normalized_sha256']:
                    raise ValueError('FIM reconstruction/content hash mismatch')
                if split == 'train' and normalized_hash in heldout:
                    rejected['exact_heldout_duplicate'] += 1
                    continue
                identity = (split, normalized_hash)
                if identity in seen:
                    rejected['exact_duplicate'] += 1
                    continue
                seen.add(identity)
                evidence = screen.inspect(body)
                parse_status = 'not_checked_language'
                if row.get('language') == 'Python':
                    try:
                        ast.parse(body)
                        parse_status = 'python_ast_pass'
                    except (SyntaxError, ValueError, RecursionError):
                        parse_status = 'python_ast_fail'
                quarantine = evidence['status'] == 'flagged' or parse_status == 'python_ast_fail'
                category = foundation_category(row)
                count = len(tokenizer.encode(text).ids)
                record = dict(source_id=row['repository'] + ':' + row['file_path'],
                    source_type='pinned_repository_file', content_sha256=sha256_bytes(text.encode()),
                    source_provenance=dict(repository=row['repository'], revision=row['revision'],
                        url=row['source_url'], git_blob=row['git_blob'], raw_sha256=row['raw_sha256'],
                        license_evidence=source_info['license_evidence'], review_scope=source_info['review_scope']),
                    license=row['license'], license_basis=row.get('license_basis'),
                    real_or_generated='real', data_stage='foundation', task_family=category,
                    repository=row['repository'], file_path=row['file_path'],
                    repository_family=row['repository_family'], language=row['language'],
                    natural_language=None, split='quarantine' if quarantine else split,
                    original_split=split, token_count=count, parser_status=parse_status,
                    eval_quarantine_status=evidence['status'], eval_quarantine_evidence=evidence,
                    accepted=not quarantine, execution_verified=False, test_verified=False,
                    content_ref=dict(blob_sha256=source_hash, byte_offset=offset, byte_length=len(raw),
                                     extraction_recipe='json.text_then_repository_fim.restore_v1'),
                    transformation_recipe='canonical_repository_file_v1')
                if quarantine:
                    record['rejection_reason'] = 'evaluation_overlap' if evidence['status'] == 'flagged' else parse_status
                    rejected[record['rejection_reason']] += 1
                    emit('quarantine', record)
                    continue
                record = emit('foundation', record)
                counts[split + '/' + category] += 1
                tokens[split + '/' + category] += count
                languages[split + '/' + row['language']] += count
                families[row['repository_family']].add(split)
                if row.get('fim', {}).get('applied'):
                    view = dict(record, real_or_generated='derived', task_family='fim_verified_transformations',
                        content_sha256=sha256_bytes(row['text'].encode()), token_count=len(tokenizer.encode(row['text']).ids),
                        parent_sample_ids=[record['sample_id']], transformation_recipe=dict(name='original_verified_fim',
                        parameters=row['fim']), content_ref=dict(record['content_ref'], extraction_recipe='json.text_v1'))
                    view.pop('sample_id')
                    emit('views', view)
                    counts[split + '/fim_verified_transformations'] += 1
                    tokens[split + '/fim_verified_transformations'] += view['token_count']
                if count <= 2048:
                    group_key = (row['repository'], split, str(Path(row['file_path']).parent))
                    groups[group_key].append((record, text))
                    if len(groups[group_key]) == 3:
                        group = groups.pop(group_key)
                        parents = [item[0] for item in group]
                        combined = '\n'.join(item[1] for item in group)
                        view = dict(record, source_id=row['repository'] + ':' + group_key[2],
                            real_or_generated='derived', task_family='repository_level_code',
                            content_sha256=sha256_bytes(combined.encode()),
                            parent_sample_ids=[p['sample_id'] for p in parents],
                            token_count=len(tokenizer.encode(combined).ids),
                            transformation_recipe='same_directory_three_file_ordered_concat_v1',
                            content_ref={'parent_sample_ids':[p['sample_id'] for p in parents]})
                        view.pop('sample_id')
                        emit('views', view)
                        counts[split + '/repository_level_code'] += 1
                        tokens[split + '/repository_level_code'] += view['token_count']
                if number % 2000 == 0:
                    print(json.dumps(dict(stage='foundation', source_rows=number, accepted=sum(counts.values()),
                                          rejected=dict(rejected), seconds=round(time.perf_counter()-started,1))), flush=True)
    finally:
        for handle in manifests.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    if any(len(splits) > 1 for splits in families.values()):
        raise ValueError('repository family crosses train/validation')
    outputs = []
    for name in manifests:
        partial = output / f'{name}.jsonl.partial'
        final = output / f'{name}.jsonl'
        partial.rename(final)
        outputs.append(dict(path=final.name, size_bytes=final.stat().st_size, sha256=digest(final)))
    receipt = dict(schema_version=1, status='foundation_views_built_not_training_ready',
        student_training_authorized=False, source_plan_sha256=digest(plan_path),
        canonical_source=dict(original_path=source.as_posix(), sha256=source_hash, size_bytes=blob.stat().st_size),
        tokenizer_sha256=digest(tokenizer_path), frozen_splits=split_receipts,
        evaluation_negative_registry=registry, registered_references=len(screen.references),
        records_by_split_category=dict(counts), token_exposure_by_split_category=dict(tokens),
        language_tokens_by_split=dict(languages), rejected=dict(rejected), manifests=outputs,
        repository_families={k:sorted(v) for k,v in sorted(families.items())},
        unique_source_policy='Reviewed existing corpus reused; views add exposure, not new unique source bytes.',
        lexical_screen_limit='Registered negatives only; no universal contamination guarantee.',
        missing_pools=['verified_posttraining', 'student_on_policy_recovery', 'verified_repository_patches'],
        elapsed_seconds=time.perf_counter()-started, build_config=config)
    (output / 'build_receipt.json').write_bytes(canonical_json(receipt) + b'\n')
    return receipt
