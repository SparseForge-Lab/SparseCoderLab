"""Redact personal home prefixes in publication metadata, preserving originals."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from tools.github_audit import HOME,git

def main():
    names=git('ls-files','--cached','--others','--exclude-standard','-z').decode('utf-8').split('\0')
    backups=Path('.local/github_preparation/original_paths')
    output=Path('.local/github_preparation/path_redactions.json')
    index=json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
    protected={'results/research_v1/evaluation_index.json','results/research_v1/corpus_manifest.json','data/research_v1/tokenizer.json','data/research_v1/tokenizer_candidate_32768.json'}
    for name in sorted(set(n for n in names if n)):
        p=Path(name)
        if p.suffix not in ('.md','.json'):continue
        original=p.read_bytes()
        try:body=original.decode('utf-8')
        except UnicodeDecodeError:continue
        if not HOME.search(body):continue
        if name in protected:raise RuntimeError('Frozen metadata would need redaction; stop rather than change identity: '+name)
        backup=backups/p;backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():raise RuntimeError('Original backup already exists: '+name)
        shutil.copy2(p,backup)
        result=HOME.sub('~',body)
        p.write_text(result,encoding='utf-8',newline='')
        index.append({'path':name,'original_sha256':hashlib.sha256(original).hexdigest(),'public_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'original_backup':str(backup),'change':'Personal home prefix replaced by portable tilde; measured values unchanged.'})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(index,indent=2),encoding='utf-8')
    print(json.dumps({'redacted_files':[r['path'] for r in index],'private_originals_retained':True}))
if __name__=='__main__':main()
