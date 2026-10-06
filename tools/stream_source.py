"""Bounded explicitly approved HF JSONL ingestion; no default network download."""
from __future__ import annotations
import argparse, json, os, urllib.request
from pathlib import Path
from urllib.parse import urlparse
import yaml

def ingest(source: dict, output: Path, budget: int) -> dict:
    if not source.get('enabled') or not source.get('terms_accepted'):
        raise ValueError('Source disabled or access/terms not approved in manifest')
    url = source['url']; parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co': raise ValueError('Only explicit Hugging Face dataset URLs supported; no web scraping')
    if 'the-stack' in url.lower() and not source.get('gated_access_verified'): raise ValueError('The Stack requires verified credentials/access')
    if source['license'] not in ('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','CC-BY-4.0','ISC'):
        raise ValueError('License requires review')
    headers = {'User-Agent': 'SparseCoderLab bounded research ingest'}
    token = os.environ.get('HF_TOKEN')
    if token: headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(url, headers=headers); used = written = docs = 0
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix('.tmp')
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open('w', encoding='utf-8') as out:
            while used < budget:
                line = response.readline(min(1024**2, budget - used)); used += len(line)
                if not line: break
                if not line.endswith(b'\n'): break # Never process a truncated or >1MiB record.
                raw = json.loads(line); text = raw[source.get('text_field', 'text')]
                document = {'text': text, 'source': source['name'], 'license': source['license'], 'kind': source['kind'], 'language': source.get('language','English')}
                # Mixed-license code requires per-record license metadata, not a dataset-level shortcut.
                if source['kind'] == 'code':
                    field = source.get('license_field')
                    if not field or raw.get(field) != source['license']: continue
                encoded = json.dumps(document, ensure_ascii=False) + '\n'; size = len(encoded.encode('utf-8'))
                if written + size > budget: break
                out.write(encoded); written += size; docs += 1
        temporary.replace(output)
    finally: temporary.unlink(missing_ok=True)
    return {'documents': docs, 'network_bytes': used, 'output_bytes': written, 'source': source['name'], 'license': source['license']}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True); p.add_argument('--source', required=True); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-mib', type=int, default=128); a = p.parse_args()
    sources = yaml.safe_load(Path(a.manifest).read_text())['sources']; source = next(s for s in sources if s['name'] == a.source)
    if a.max_mib <= 0 or a.max_mib > 20480: raise ValueError('Download budget must be 1..20480 MiB')
    used = sum(x.stat().st_size for x in Path('data').rglob('*') if x.is_file())
    if used + a.max_mib * 1024**2 > 20*1024**3: raise RuntimeError('20 GiB data budget would be exceeded')
    print(json.dumps(ingest(source, a.output, a.max_mib*1024**2), indent=2))
