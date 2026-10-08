"""Linear token-pattern overlap screen for pinned held-out benchmark registries."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from tools.repository_fim import restore, split_header
from src.utils.hashing import sha256_file

LEXEMES = re.compile(r'\w+|[^\w\s]', re.UNICODE)


class TokenMatcher:
    """Aho-Corasick matching over exact lexemes; no rolling-hash collisions."""
    def __init__(self, patterns):
        self.edges = [{}]; self.fail = [0]; self.outputs = [[]]
        for tokens, metadata in patterns:
            state = 0
            for token in tokens:
                if token not in self.edges[state]:
                    self.edges[state][token] = len(self.edges)
                    self.edges.append({}); self.fail.append(0); self.outputs.append([])
                state = self.edges[state][token]
            self.outputs[state].append(metadata)
        queue = deque(self.edges[0].values())
        while queue:
            state = queue.popleft()
            for token, child in self.edges[state].items():
                queue.append(child); fallback = self.fail[state]
                while fallback and token not in self.edges[fallback]: fallback = self.fail[fallback]
                self.fail[child] = self.edges[fallback].get(token, 0)
                self.outputs[child].extend(self.outputs[self.fail[child]])

    def matches(self, tokens):
        state = 0; hits = set()
        for token in tokens:
            while state and token not in self.edges[state]: state = self.fail[state]
            state = self.edges[state].get(token, 0)
            hits.update(self.outputs[state])
        return sorted(hits)


def patterns(registry):
    result = []; profiles = []
    for row in registry['tasks']:
        reference = LEXEMES.findall(row['reference'].casefold())
        prompt = row['prompt']
        # HumanEval includes import/signature boilerplate; screen its docstring.
        docstring = re.search(r'(?:"""|\x27\x27\x27)(.*?)(?:"""|\x27\x27\x27)', prompt, re.S)
        if docstring: prompt = docstring.group(1)
        # Ignore examples after the prose; exact prompt screen remains bounded.
        prompt = prompt.split('>>>')[0].strip().lstrip('# ')
        prompt_tokens = LEXEMES.findall(prompt.casefold())
        if len(reference) >= 32: result.append((reference, (row['id'], 'reference')))
        if len(prompt_tokens) >= 12: result.append((prompt_tokens, (row['id'], 'prompt')))
        profiles.append(dict(task_id=row['id'], reference_tokens=len(reference), prompt_tokens=len(prompt_tokens),
            reference_screened=len(reference)>=32, prompt_screened=len(prompt_tokens)>=12))
    return result, profiles


def screen(documents, registry_path, output):
    if output.exists(): raise FileExistsError('Use a new overlap report version')
    registry = json.loads(registry_path.read_text(encoding='utf8'))
    entries, profiles = patterns(registry); matcher = TokenMatcher(entries)
    count = 0; hits = []; digest = hashlib.sha256()
    files=sorted(p for p in documents.glob('*.jsonl') if not p.name.endswith('.quarantine.jsonl')) if documents.is_dir() else [documents]
    if not files:raise ValueError('No source exports to screen')
    with ExitStack() as stack:
        handles=[stack.enter_context(path.open('rb')) for path in files]
        for line in (line for handle in handles for line in handle):
            digest.update(line)
            if not line.strip(): continue
            record = json.loads(line); count += 1
            _, body = split_header(dict(record, text=restore(record)))
            matches = matcher.matches(match.group(0).casefold() for match in LEXEMES.finditer(body))
            if matches: hits.append({key:record[key] for key in ('repository','revision','file_path','raw_sha256')} |
                dict(matches=[dict(task_id=task, kind=kind) for task,kind in matches]))
            if count % 1000 == 0: print(json.dumps(dict(stage='contamination', scanned=count, matching_documents=len(hits))), flush=True)
    receipt = dict(schema_version=1, status='REVIEW_REQUIRED' if hits else 'NO_MATCH_WITH_LIMITS',
        input_scope='concatenated_original_exports' if documents.is_dir() else 'versioned_documents',
        input_files=[path.name for path in files],
        registry_sha256=sha256_file(registry_path), sources=registry['sources'],
        documents_sha256=digest.hexdigest(), scanner_sha256=sha256_file(Path(__file__)), records_scanned=count,
        matching_documents=len(hits), hits=hits, reference_profiles=profiles,
        policy='Exact casefolded lexical token sequences, whitespace insensitive; reference minimum32, prompt minimum12. Original source restored before matching.',
        limitations=['No semantic, paraphrase, fuzzy or pretraining-history contamination guarantee.',
            'Short benchmark snippets and prompts omitted; ubiquitous functions can produce legitimate overlap.',
            'Any hit requires provenance review and exclusion in a new corpus version; this scanner never rewrites data.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(records_scanned=count, matching_documents=len(hits), status=receipt['status'])),flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--documents',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); screen(args.documents,args.registry,args.output)
