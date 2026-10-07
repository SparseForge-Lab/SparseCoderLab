"""Deterministic code FIM with existing markers and an unchanged vocabulary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from tools.repository_data_smoke import validate_export

MARKERS = ('<fim_prefix>', '<fim_suffix>', '<fim_middle>')
LANGUAGES = {'Python', 'JavaScript', 'TypeScript', 'C', 'C++', 'Rust', 'Java', 'Go', 'C#', 'SQL', 'Bash', 'HTML/CSS'}
FORMAT = 'psm_literal_markers_v1'


def split_header(record: dict) -> tuple[str, str]:
    header = f"<repo>{record['repository']}<file>{record['file_path']}\n"
    if not record['text'].startswith(header):
        raise ValueError('Expected an intact repository/file header')
    return header, record['text'][len(header):]


def transform(record: dict, *, seed: int, ratio: float, min_chars: int = 128) -> dict:
    if not 0 <= ratio <= 1:
        raise ValueError('FIM ratio must be between zero and one')
    if min_chars < 3:
        raise ValueError('FIM requires at least three characters')
    if record.get('fim', {}).get('applied'):
        raise ValueError('Refusing to apply FIM twice')
    header, body = split_header(record)
    out = copy.deepcopy(record)
    eligible = record['kind'] == 'code' and record['language'] in LANGUAGES and len(body) >= min_chars
    collision = any(marker in body for marker in MARKERS)
    eligible = eligible and not collision
    identity = '\0'.join(str(value) for value in (
        seed, record['repository'], record['revision'], record['file_path'], record['raw_sha256'], FORMAT))
    bits = hashlib.sha256(identity.encode('utf-8')).digest()
    selected = int.from_bytes(bits[:8], 'big') / 2**64 < ratio
    info = {'format': FORMAT, 'seed': seed, 'target_ratio': ratio, 'eligible': eligible,
            'applied': eligible and selected, 'marker_policy': 'Existing delimiters encoded by the frozen tokenizer; no vocabulary additions.'}
    if collision:
        info['skip_reason'] = 'marker_collision'
    out['fim'] = info
    if not info['applied']:
        return out
    boundaries = []
    position = 0
    for line in body.splitlines(keepends=True):
        position += len(line)
        if position < len(body):
            boundaries.append(position)
    if len(boundaries) >= 2:
        a = int.from_bytes(bits[8:16], 'big') % (len(boundaries) - 1)
        b = a + 1 + int.from_bytes(bits[16:24], 'big') % (len(boundaries) - a - 1)
        start, end = boundaries[a], boundaries[b]
        info['cut_unit'] = 'line_boundary'
    else:
        start = 1 + int.from_bytes(bits[8:16], 'big') % (len(body) - 2)
        end = start + 1 + int.from_bytes(bits[16:24], 'big') % (len(body) - start - 1)
        info['cut_unit'] = 'unicode_character'
    info.update(prefix_characters=start, middle_characters=end - start, suffix_characters=len(body) - end)
    prefix, middle, suffix = body[:start], body[start:end], body[end:]
    out['text'] = header + MARKERS[0] + prefix + MARKERS[1] + suffix + MARKERS[2] + middle
    return out


def restore(record: dict) -> str:
    if not record.get('fim', {}).get('applied'):
        return record['text']
    header, body = split_header(record)
    if not body.startswith(MARKERS[0]):
        raise ValueError('FIM prefix marker missing')
    prefix, tail = body[len(MARKERS[0]):].split(MARKERS[1], 1)
    suffix, middle = tail.split(MARKERS[2], 1)
    return header + prefix + middle + suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exports', nargs='+', type=Path, default=[
        Path('data/research_v2_real/exports/flask.jsonl'), Path('data/research_v2_real/exports/requests.jsonl')])
    parser.add_argument('--tokenizer', type=Path, default=Path('data/research_v1/tokenizer.json'))
    parser.add_argument('--output', type=Path, default=Path('data/research_v2_real/documents/fim_smoke.jsonl'))
    parser.add_argument('--report', type=Path, default=Path('results/research_v2_real/fim_smoke.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratio', type=float, default=.4)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('FIM fixture already exists; use new versioned output paths')
    tokenizer_bytes = args.tokenizer.read_bytes()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    counts = Counter()
    language_counts = Counter()
    seen = set()
    sources = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        for export in sorted(args.exports):
            source, records = validate_export(export)
            sources.append({'id': source['source_id'], 'revision': source['revision'],
                            'export_sha256': hashlib.sha256(export.read_bytes()).hexdigest()})
            for record in records:
                key = record['normalized_sha256']
                if key in seen:
                    counts['duplicates_before_transform'] += 1
                    continue
                seen.add(key)
                out = transform(record, seed=args.seed, ratio=args.ratio)
                if restore(out) != record['text']:
                    raise ValueError('FIM reconstruction changed original source')
                encoded = tokenizer.encode(out['text']).ids
                if tokenizer.decode(encoded, skip_special_tokens=False) != out['text']:
                    raise ValueError('Frozen tokenizer failed FIM text roundtrip')
                counts['documents'] += 1
                counts['eligible'] += int(out['fim']['eligible'])
                counts['applied'] += int(out['fim']['applied'])
                counts['tokens'] += len(encoded)
                language_counts[record['language']] += int(out['fim']['applied'])
                handle.write(json.dumps(out, ensure_ascii=False) + '\n')
    if tokenizer_bytes != args.tokenizer.read_bytes():
        raise ValueError('Tokenizer changed during FIM preparation')
    report = {
        'schema_version': 1, 'passed': True, 'format': FORMAT, 'seed': args.seed, 'target_ratio': args.ratio,
        'sources': sources, 'counts': dict(counts), 'applied_by_language': dict(language_counts),
        'actual_eligible_fraction': counts['applied'] / max(1, counts['eligible']),
        'marker_token_ids': {marker: tokenizer.encode(marker).ids for marker in MARKERS},
        'marker_policy': 'Existing frozen-tokenizer delimiters; no new tokens.', 'vocabulary_additions': 0,
        'tokenizer_sha256': hashlib.sha256(tokenizer_bytes).hexdigest(),
        'transformed_records_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'preprocessing_seconds': time.perf_counter() - started,
        'exact_reconstruction_and_tokenizer_roundtrips': counts['documents'],
        'scope': 'Two-repository transformation fixture only; no model training and no FIM quality evaluation.',
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'counts': dict(counts), 'actual_eligible_fraction': report['actual_eligible_fraction'],
                      'vocabulary_additions': 0}, indent=2))


if __name__ == '__main__':
    main()
