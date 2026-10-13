"""Check offline target-mask serialization against saved accepted pilot records."""
from collections import Counter
import json
from pathlib import Path

from tokenizers import Tokenizer

from forgebuilder.schema import canonical_json, sha256_bytes
from forgebuilder.serialization import serialize_posttraining
from forgebuilder.strata_runtime import digest


def main():
    output=Path('results/dataset_v2/serialization_validation_v1.json')
    private=Path('.local/dataset_v2/serialization_validation_v1.jsonl')
    if output.exists() or private.exists(): raise FileExistsError('retain existing validation snapshots')
    foundation=json.loads(Path('data/dataset_v2/foundation_v1/build_receipt.json').read_bytes())
    tokenizer_path=Path(foundation['build_config']['tokenizer'])
    if digest(tokenizer_path)!=foundation['tokenizer_sha256']:
        raise ValueError('frozen tokenizer identity changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    sources=[]; counts=Counter(); lengths=[]; targets=0; bad_tokens=0
    private.parent.mkdir(parents=True,exist_ok=True)
    with private.open('xb') as stream:
        for pool in ('source_pilot_v1','generated_pilot_v1'):
            path=Path('data/dataset_v2')/pool/'accepted_snapshot.jsonl'
            payload=path.read_bytes()
            sources.append({'path':path.as_posix(),'sha256':sha256_bytes(payload)})
            for line in payload.splitlines():
                record=json.loads(line)
                if not record.get('license'):
                    counts['withheld_unknown_license']+=1
                    continue
                serialized=serialize_posttraining(record,tokenizer)
                encoded=tokenizer.encode(serialized['text'],add_special_tokens=False)
                assert serialized['input_ids'][1:-1]==encoded.ids
                assert len(serialized['input_ids'])==len(serialized['target_weights'])
                for (start,end),weight in zip(encoded.offsets,serialized['target_weights'][1:-1]):
                    overlaps=[s for s in serialized['spans'] if start<s['end'] and end>s['start']]
                    if any(s['weight']==0 for s in overlaps):
                        assert weight==0,'context token leaked into training targets'
                    if any(s['kind'] in ('WRONG_HYPOTHESIS','FAILURE_OUTPUT') for s in overlaps):
                        assert weight==0
                        bad_tokens+=1
                stream.write(canonical_json(serialized)+b'\n')
                counts['serialized']+=1
                counts['recovery_records']+=int('trajectory' in record)
                counts['exceed_student_context']+=int(serialized['token_count']>1024)
                lengths.append(serialized['token_count']); targets+=serialized['target_token_count']
    report=dict(schema_version=1,status='offline_serialization_validated_not_training_ready',
        tokenizer_sha256=foundation['tokenizer_sha256'],sources=sources,counts=dict(counts),
        serialized_tokens=sum(lengths),target_tokens=targets,masked_known_bad_tokens=bad_tokens,
        minimum_record_tokens=min(lengths),maximum_record_tokens=max(lengths),
        serialized_payload_sha256=digest(private),serialized_payload_bytes=private.stat().st_size,
        target_mask_leaks=0,student_context=1024,training_launched=False,gpu_used=False,
        limits=['Saved accepted snapshots only; independent objective checks are not rerun here.',
                'Unknown output licenses are withheld, not counted as functional failures.',
                'Known license metadata is not a new rights-clearance assertion.',
                'Windowing, batching and weighted-loss integration remain unimplemented and unfrozen.'])
    output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
