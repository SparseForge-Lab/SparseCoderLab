import json

from forgebuilder.schema import canonical_json
from forgebuilder.strata_runtime import digest
from tools.finalize_dataset_v2_foundation import finalize, sort_manifest


def test_finalizer_recovers_after_sort_before_receipt_write(tmp_path):
    path=tmp_path/'foundation.jsonl'
    path.write_bytes(b''.join(canonical_json({'sample_id':key})+b'\n' for key in ['c','a','b']))
    receipt={'manifests':[{'path':path.name,'sha256':digest(path),'size_bytes':path.stat().st_size}]}
    (tmp_path/'build_receipt.json').write_text(json.dumps(receipt))
    sort_manifest(path)  # Simulate interruption before the receipt is replaced.
    final=finalize(tmp_path)
    assert [json.loads(line)['sample_id'] for line in path.read_text().splitlines()]==['a','b','c']
    assert final['manifest_order']=='stable_sample_id'
    assert final['manifests'][0]['sha256']==digest(path)
    assert finalize(tmp_path)==final
