"""Export an atomic snapshot of objectively verified, critic-accepted candidates."""
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path

from forgebuilder.pilot import verify
from forgebuilder.pipeline import CriticResult, VerificationResult, accept_posttraining_candidate
from forgebuilder.schema import canonical_json, sha256_bytes, validate_record


def export(pilot):
    raw=(pilot/'reviews.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        raw=raw[:raw.rfind(b'\n')+1]
    reviews={r['sample_id']:r for r in map(json.loads,raw.splitlines())}
    candidates=[json.loads(line) for line in (pilot/'candidates.jsonl').read_bytes().splitlines()]
    accepted=[]
    for row in candidates:
        review=reviews.get(row['sample_id'])
        if not review or not review['accepted']:
            continue
        if not accept_posttraining_candidate(verification=VerificationResult(**review['objective_verification']),
                                             critic=CriticResult(**review['critic'])):
            raise ValueError('accepted review does not satisfy the acceptance contract')
        if verify(row).status!='PASS' or sha256_bytes(row['candidate'].encode())!=row['content_sha256']:
            raise ValueError('candidate changed after objective verification')
        row.update(accepted=True,acceptance_status='accepted',critic_verdict=review['critic']['verdict'],
                   critic=review['critic'],objective_verification=review['objective_verification'])
        accepted.append(row)
    pairs=Counter(row['pair_id'] for row in accepted if 'pair_id' in row)
    accepted=[row for row in accepted if 'pair_id' not in row or pairs[row['pair_id']]==2]
    accepted.sort(key=lambda row:row['sample_id'])
    payload=b''.join(canonical_json(validate_record(row))+b'\n' for row in accepted)
    path=pilot/'accepted_snapshot.jsonl'
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp,path)
    receipt=dict(status='partial_reviewed_snapshot_not_frozen',accepted_records=len(accepted),
        reviewed_records=len(reviews),candidate_records=len(candidates),
        task_families=dict(Counter(row['task_family'] for row in accepted)),
        accepted_snapshot_sha256=sha256_bytes(payload),reviewed_prefix_sha256=sha256_bytes(raw),
        reviewed_prefix_bytes=len(raw),student_training_authorized=False,
        coverage='Source-grounded extraction/reasoning and constructed recovery; no executable SWE correctness claim.')
    (pilot/'accepted_snapshot_receipt.json').write_bytes(canonical_json(receipt)+b'\n')
    return receipt


if __name__=='__main__':
    print(json.dumps(export(Path('data/dataset_v2/source_pilot_v1')),indent=2))
