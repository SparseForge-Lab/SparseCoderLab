"""Sort foundation manifests with bounded memory and refresh their hash receipts."""
import argparse
import heapq
import json
import os
from pathlib import Path
import tempfile

from forgebuilder.schema import canonical_json
from forgebuilder.strata_runtime import digest


def sort_manifest(path):
    with tempfile.TemporaryDirectory(prefix='.sort-',dir=path.parent) as directory:
        runs, rows = [], []
        def save():
            rows.sort(key=lambda line:json.loads(line)['sample_id'])
            run = Path(directory)/str(len(runs))
            run.write_bytes(b''.join(rows))
            runs.append(run)
            rows.clear()
        with path.open('rb') as source:
            for line in source:
                rows.append(line)
                if len(rows)==5000:
                    save()
        if rows:
            save()
        handles = [run.open('rb') for run in runs]
        target = path.with_suffix(path.suffix+'.sorted')
        previous = None
        try:
            with target.open('wb') as output:
                for line in heapq.merge(*handles,key=lambda line:json.loads(line)['sample_id']):
                    identity = json.loads(line)['sample_id']
                    if identity==previous:
                        raise ValueError('duplicate sample ID in manifest')
                    previous=identity
                    output.write(line)
                output.flush()
                os.fsync(output.fileno())
        finally:
            for handle in handles:
                handle.close()
        journal = {'original_sha256':digest(path),'sorted_sha256':digest(target)}
        path.with_suffix('.sort-identity.json').write_bytes(canonical_json(journal)+b'\n')
        os.replace(target,path)


def finalize(root):
    receipt_path=root/'build_receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('manifest_order')=='stable_sample_id':
        for row in receipt['manifests']:
            if digest(root/row['path'])!=row['sha256']:
                raise ValueError('sorted manifest changed')
        return receipt
    for row in receipt['manifests']:
        path=root/row['path']
        actual=digest(path)
        if actual!=row['sha256']:
            journal_path=path.with_suffix('.sort-identity.json')
            journal=json.loads(journal_path.read_text(encoding='utf8')) if journal_path.exists() else {}
            if journal.get('original_sha256')!=row['sha256'] or journal.get('sorted_sha256')!=actual:
                raise ValueError('manifest changed before finalization')
        else:
            sort_manifest(path)
        row.update(sha256=digest(path),size_bytes=path.stat().st_size)
    receipt.update(manifest_order='stable_sample_id',finalizer_sha256=digest(Path(__file__)))
    temp=receipt_path.with_suffix('.tmp')
    temp.write_bytes(canonical_json(receipt)+b'\n')
    os.replace(temp,receipt_path)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/dataset_v2/foundation_v1'))
    print(json.dumps(finalize(parser.parse_args().root),indent=2))
