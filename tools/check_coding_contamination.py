"""Flag benchmark-text overlap in versioned repository records for review.

Elementary functions are common code; a match is evidence to review, not proof
of benchmark leakage. This tool never silently removes or rewrites a corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tools.repository_fim import restore, split_header

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


def fingerprints(benchmark):
    result = []
    for task in benchmark["tasks"]:
        phrase = " ".join(task["prompt"].splitlines()[0].lstrip("# ").casefold().split())
        code = "\n".join(task["reference"].splitlines()[1:])
        tokens = TOKEN.findall(code)
        result.append({"task_id": task["id"], "prompt_phrase": phrase,
                       "reference_tokens": tokens if len(tokens) >= 32 else [],
                       "reference_token_count": len(tokens),
                       "policy": "Prompt phrase and >=32-token exact reference; shorter ubiquitous snippets omitted."})
    return result


def inspect_record(record, registry):
    restored = dict(record, text=restore(record))
    _, body = split_header(restored)
    normalized = " ".join(body.casefold().split())
    tokens = TOKEN.findall(body)
    matches = []
    for entry in registry:
        if entry["prompt_phrase"] in normalized:
            matches.append({"task_id": entry["task_id"], "kind": "prompt_phrase"})
        reference = entry["reference_tokens"]
        if reference:
            size = len(reference)
            if any(tokens[i:i+size] == reference for i in range(len(tokens)-size+1)):
                matches.append({"task_id": entry["task_id"], "kind": "exact_reference_tokens"})
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--benchmark", default="configs/eval/simple_python_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Choose a new overlap-report version")
    benchmark_bytes = Path(args.benchmark).read_bytes()
    registry = fingerprints(json.loads(benchmark_bytes))
    digest = hashlib.sha256()
    count = 0
    hits = []
    with Path(args.documents).open("rb") as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            matches = inspect_record(record, registry)
            if matches:
                hits.append({"repository": record["repository"], "revision": record["revision"],
                             "file_path": record["file_path"], "raw_sha256": record["raw_sha256"], "matches": matches})
    report = {"schema_version": 1, "benchmark_id": json.loads(benchmark_bytes)["benchmark_id"],
              "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
              "documents_file": Path(args.documents).name, "documents_sha256": digest.hexdigest(),
              "scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "records_scanned": count, "source_restoration_before_matching": True,
              "reference_profiles": [{k:v for k,v in p.items() if k != "reference_tokens"} for p in registry],
              "matching_documents": len(hits), "hits": hits,
              "status": "REVIEW_REQUIRED" if hits else "NO_MATCH_WITH_LIMITS",
              "limitations": ["Only this 25-task fixture is covered; other evaluation families remain unreviewed.",
                              "No semantic, paraphrase, external benchmark or training-history contamination proof.",
                              "Elementary functions and comments are ubiquitous; matches require human provenance review.",
                              "Frozen corpus is unchanged. Any exclusion must create a new dataset version."]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    temporary.replace(destination)
    print(json.dumps({k:report[k] for k in ("records_scanned", "matching_documents", "status")}))


if __name__ == "__main__":
    main()
