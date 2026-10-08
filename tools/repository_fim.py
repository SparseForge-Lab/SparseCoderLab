"""Deterministic code FIM with existing markers and an unchanged vocabulary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from src.utils.hashing import sha256_file

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


def transform(record: dict, *, seed: int, ratio: float, min_chars: int = 128,
              tokenizer=None, min_span_tokens: int = 16) -> dict:
    if not 0 <= ratio <= 1:
        raise ValueError('FIM ratio must be between zero and one')
    if min_chars < 3:
        raise ValueError('FIM requires at least three characters')
    if min_span_tokens < 1:
        raise ValueError('FIM span token minimum must be positive')
    if record.get('fim', {}).get('applied'):
        raise ValueError('Refusing to apply FIM twice')
    header, body = split_header(record)
    out = copy.deepcopy(record)
    eligible = record['kind'] == 'code' and record['language'] in LANGUAGES and len(body) >= min_chars
    collision = any(marker in body for marker in MARKERS)
    eligible = eligible and not collision
    format_id = FORMAT if tokenizer is None else 'psm_token_minimums_v2'
    encoded = tokenizer.encode(body) if tokenizer is not None else None
    if encoded is not None and len(encoded.ids) < 3 * min_span_tokens:
        eligible = False
    identity = '\0'.join(str(value) for value in (
        seed, record['repository'], record['revision'], record['file_path'], record['raw_sha256'], format_id))
    bits = hashlib.sha256(identity.encode('utf-8')).digest()
    selected = int.from_bytes(bits[:8], 'big') / 2**64 < ratio
    info = {'format': format_id, 'seed': seed, 'target_ratio': ratio, 'eligible': eligible,
            'applied': eligible and selected, 'marker_policy': 'Existing delimiters encoded by the frozen tokenizer; no vocabulary additions.'}
    if collision:
        info['skip_reason'] = 'marker_collision'
    out['fim'] = info
    if not info['applied']:
        return out
    if encoded is not None:
        info['minimum_span_tokens'] = min_span_tokens
        positions = sorted({start for start, end in encoded.offsets if 0 < start < len(body)})
        # Try a bounded deterministic set, then balanced token offsets. Verify
        # independently encoded spans because BPE merges can cross a cut.
        found = None
        for attempt in range(32):
            bits2 = hashlib.sha256(bits + attempt.to_bytes(2, 'big')).digest()
            if attempt == 31:
                start = encoded.offsets[len(encoded.ids)//3][0]
                end = encoded.offsets[2*len(encoded.ids)//3][0]
            else:
                if len(positions) < 2: break
                a = int.from_bytes(bits2[:8],'big') % (len(positions)-1)
                b = a+1+int.from_bytes(bits2[8:16],'big') % (len(positions)-a-1)
                start,end = positions[a],positions[b]
            lengths = [len(tokenizer.encode(span).ids) for span in (body[:start],body[start:end],body[end:])]
            if min(lengths) >= min_span_tokens:
                found = start,end,lengths; break
        if found is None:
            info.update(applied=False, skip_reason='no_valid_token_spans')
            return out
        start,end,lengths = found
        info.update(cut_unit='token_offset_unicode_character', span_tokens=lengths,
                    prefix_characters=start,middle_characters=end-start,suffix_characters=len(body)-end)
        out['text'] = header+MARKERS[0]+body[:start]+MARKERS[1]+body[end:]+MARKERS[2]+body[start:end]
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
                            'export_sha256': sha256_file(export)})
            for record in records:
                key = record['normalized_sha256']
                if key in seen:
                    counts['duplicates_before_transform'] += 1
                    continue
                seen.add(key)
                out = transform(record, seed=args.seed, ratio=args.ratio, tokenizer=tokenizer)
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
        'schema_version': 1, 'passed': True, 'format': 'psm_token_minimums_v2', 'seed': args.seed, 'target_ratio': args.ratio,
        'sources': sources, 'counts': dict(counts), 'applied_by_language': dict(language_counts),
        'actual_eligible_fraction': counts['applied'] / max(1, counts['eligible']),
        'marker_token_ids': {marker: tokenizer.encode(marker).ids for marker in MARKERS},
        'marker_policy': 'Existing frozen-tokenizer delimiters; no new tokens.', 'vocabulary_additions': 0,
        'tokenizer_sha256': hashlib.sha256(tokenizer_bytes).hexdigest(),
        'transformed_records_sha256': sha256_file(args.output),
        'preprocessing_seconds': time.perf_counter() - started,
        'exact_reconstruction_and_tokenizer_roundtrips': counts['documents'],
        'scope': 'Two-repository transformation fixture only; no model training and no FIM quality evaluation.',
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'counts': dict(counts), 'actual_eligible_fraction': report['actual_eligible_fraction'],
                      'vocabulary_additions': 0}, indent=2))


if __name__ == '__main__':
    main()
