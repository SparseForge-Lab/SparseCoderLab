"""Bounded pinned public dataset downloads; no dataset-provided code execution."""
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from pathlib import Path

ROOT=Path('data/research_v1'); RESULTS=Path('results/research_v1')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SparseCoderLab/Research-v1'}),timeout=60) as r:return r.read()
def plan():
    metadata=json.loads((RESULTS/'source_metadata.json').read_text()); files=[]
    for repo,prefix,limit in [('codeparrot/github-code','data',24),('HuggingFaceFW/fineweb-edu','sample/10BT',1)]:
        revision=metadata[repo]['sha']
        inventory=json.loads(get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{prefix}?limit=100'))
        entries=sorted((x for x in inventory if x['path'].endswith('.parquet')),key=lambda x:x['path'])[:limit]
        for e in entries:
            files.append(dict(dataset=repo,revision=revision,remote_file=e['path'],size_bytes=e['size'],
                              sha256=e.get('lfs',{}).get('oid'),local_path=str(ROOT/'raw'/('code' if repo.startswith('codeparrot') else 'general')/Path(e['path']).name)))
        card=get(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md')
        (RESULTS/('codeparrot_card.md' if repo.startswith('codeparrot') else 'fineweb_edu_card.md')).write_bytes(card)
    result=dict(files=files,download_upper_bound_bytes=sum(x['size_bytes'] for x in files),
                policy='Public ungated pinned files only. Code: per-repository MIT/Apache2/BSD2/BSD3/ISC/CC0/Unlicense whitelist, reject unknown/other. FineWebEdu: ODC-By collection/CommonCrawl terms with per-document URL/crawl identifiers; no claim that collection license owns individual pages. Bounded selection; no remote code execution.')
    (RESULTS/'source_plan.json').write_text(json.dumps(result,indent=2))
    return result
def download(entry):
    p=Path(entry['local_path']);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.stat().st_size!=entry['size_bytes'] or sha(p)!=entry['sha256']:raise RuntimeError('Existing download corrupt')
        return
    if sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file())+entry['size_bytes']>16*1024**3:raise RuntimeError('Research data footprint cap')
    temporary=p.with_suffix('.parquet.tmp')
    for attempt in range(4):
        offset=temporary.stat().st_size if temporary.exists() else 0
        url=f"https://huggingface.co/datasets/{entry['dataset']}/resolve/{entry['revision']}/{entry['remote_file']}?download=true&task_ts={int(time.time())}"
        headers={'User-Agent':'SparseCoderLab/Research-v1'}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as r:
                append=offset>0 and r.status==206
                with temporary.open('ab' if append else 'wb') as f:
                    while chunk:=r.read(4*1024**2):f.write(chunk)
            if temporary.stat().st_size!=entry['size_bytes'] or sha(temporary)!=entry['sha256']:raise RuntimeError('Download size/SHA256 mismatch')
            temporary.replace(p);entry['verified']=True;entry['download_completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            print(json.dumps({'downloaded':str(p),'bytes':p.stat().st_size}),flush=True);return
        except Exception as error:
            print(json.dumps({'attempt':attempt+1,'file':str(p),'error':str(error)}),flush=True)
            if attempt==3:raise
            time.sleep(2)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--code-files',type=int,default=2);parser.add_argument('--general',action='store_true');args=parser.parse_args()
    path=RESULTS/'source_plan.json';p=json.loads(path.read_text()) if path.exists() else plan()
    selected=[f for f in p['files'] if ('codeparrot' in f['dataset'] and int(Path(f['remote_file']).name.split('-')[1])<args.code_files) or ('fineweb' in f['dataset'] and args.general)]
    for f in selected:
        download(f);path.write_text(json.dumps(p,indent=2))
if __name__=='__main__':main()
