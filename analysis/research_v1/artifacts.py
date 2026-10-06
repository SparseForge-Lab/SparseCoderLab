"""Reference large retained research payloads instead of duplicating them in ZIP."""
from __future__ import annotations
import datetime,hashlib,json
from pathlib import Path

ROOT=Path.cwd()
R=Path('results/research_v1')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def utc(value):return datetime.datetime.fromtimestamp(value,datetime.timezone.utc).isoformat()

def main():
    assert json.loads((R/'final_integrity.json').read_text(encoding='utf-8'))['passed'],'Final integrity audit required'
    assert json.loads((R/'final_decision.json').read_text(encoding='utf-8'))['status']=='complete','Final decision required'
    plan=json.loads((R/'source_plan.json').read_text(encoding='utf-8'))
    raw={Path(r['local_path']).as_posix():r for r in plan['files'] if r.get('verified')}
    sources=sorted({(r['dataset'],r['revision']) for r in raw.values()})
    files=[];data=Path('data/research_v1')
    for directory in ('raw','documents','documents_prefreeze_attempt','shards'):
        files.extend(p for p in (data/directory).rglob('*') if p.is_file())
    files.extend(p for p in data.glob('*sqlite*') if p.is_file())
    files.extend(Path('experiments/research_v1').rglob('*.pt'))
    files.extend(p for p in Path('.local/github_preparation/original_paths').rglob('*') if p.is_file())
    redactions=Path('.local/github_preparation/path_redactions.json')
    if redactions.exists():files.append(redactions)
    artifacts=[]
    for p in sorted(set(files)):
        relative=p.as_posix();s=p.stat();digest=sha(p)
        item={'project_relative_path':relative,'canonical_path':str(p.resolve()),'size_bytes':s.st_size,'sha256':digest,
              'created_utc':utc(getattr(s,'st_birthtime',s.st_ctime)),'modified_utc':utc(s.st_mtime),'archived_payload':False}
        if relative in raw:
            r=raw[relative];assert s.st_size==r['size_bytes'] and digest==r['sha256']
            item.update(source=r['dataset'],revision=r['revision'],source_file=r['remote_file'],originating_experiment='research_v1 source acquisition',download_completed_utc=r['download_completed_utc'])
        elif relative.startswith('experiments/'):
            item.update(source='Fresh research_v1 training; common corpus_manifest/tokenizer hashes are in run provenance',revision='Frozen training_source_hash in run provenance',originating_experiment=str(p.parent.parent))
        elif relative.startswith('.local/'):
            item.update(source='Exact private pre-publication metadata originals; numeric observations unchanged by public home-path aliases',
                        revision='Original/public digests recorded in .local/github_preparation/path_redactions.json',originating_experiment='Prompt-2 GitHub publication privacy preparation')
        else:
            item.update(source=[{'dataset':a,'revision':b} for a,b in sources],originating_experiment='Rejected pre-freeze attempt; never trained' if 'prefreeze_attempt' in relative else 'research_v1 frozen preprocessing/packing')
        artifacts.append(item)
    report={'artifacts':artifacts,'total_referenced_bytes':sum(a['size_bytes'] for a in artifacts),
            'scope':'All retained raw/candidate/failed-attempt/shard/index/dedup and research checkpoint payloads. Canonical files remain in place; archive verifies path/size/SHA and stores references only.'}
    (R/'large_artifacts.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'artifacts':len(artifacts),'referenced_bytes':report['total_referenced_bytes']}))

if __name__=='__main__':main()
