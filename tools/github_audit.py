"""Inspect publication candidates without printing credential values."""
from __future__ import annotations
import argparse,hashlib,io,json,re,subprocess,zipfile
from pathlib import Path

PATTERNS={
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY',
    'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
    'openai_style_key':r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b',
    'aws_access_id':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'google_api_key':r'\bAIza[A-Za-z0-9_-]{35}\b',
    'slack_token':r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
    'credential_literal':r'''(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\b["']?\s*[:=]\s*["'][^"'\n]{8,}["']''',
}
HOME=re.compile(r'''(?i)(?<![A-Za-z0-9_./\\])(?:[A-Z]:[/\\]+Users[/\\]+[A-Za-z0-9][A-Za-z0-9_.@ -]*|/(?:home|Users)/[A-Za-z0-9][A-Za-z0-9_.@-]*)''')
def git(*args):return subprocess.check_output(['git',*args])
def inspect(name,data,findings,paths):
    if name.endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for item in z.infolist():
                    if item.file_size<=20*1024**2:inspect(name+'!'+item.filename,z.read(item),findings,paths)
        except zipfile.BadZipFile:findings.append({'file':name,'kind':'invalid_zip','line':None})
        return
    try:body=data.decode('utf-8')
    except UnicodeDecodeError:return
    for kind,pattern in PATTERNS.items():
        for match in re.finditer(pattern,body):findings.append({'file':name,'kind':kind,'line':body.count('\n',0,match.start())+1})
    hits=list(HOME.finditer(body))
    if hits:paths.append({'file':name,'home_path_occurrences':len(hits)})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('worktree','staged','history'),default='worktree');parser.add_argument('--ref',default='--all');a=parser.parse_args()
    findings=[];paths=[];large=[];files=[]
    if a.scope=='history':
        for line in git('rev-list','--objects',a.ref).decode('utf-8').splitlines():
            oid,_,name=line.partition(' ')
            if git('cat-file','-t',oid).strip()!=b'blob':continue
            data=git('cat-file','blob',oid);files.append({'path':name,'bytes':len(data),'object':oid})
            if len(data)>20*1024**2:large.append({'file':name,'bytes':len(data),'object':oid})
            inspect(name or oid,data,findings,paths)
    else:
        names=git('ls-files',*(['-z'] if a.scope=='staged' else ['--cached','--others','--exclude-standard','-z'])).decode('utf-8').split('\0')
        for name in sorted(set(n for n in names if n)):
            data=git('show',':'+name) if a.scope=='staged' else Path(name).read_bytes()
            files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            if len(data)>20*1024**2:large.append({'file':name,'bytes':len(data)})
            inspect(name,data,findings,paths)
    report={'scope':a.scope,'history_ref':a.ref if a.scope=='history' else None,'files':files,'credential_candidates':findings,'personal_path_files':paths,'large_files_over_20mib':large,
        'scope_note':'Pattern inspection, including small ZIP contents. Findings never include secret values. Not a guarantee that every possible credential format is detectable.'}
    suffix='_'+re.sub(r'\W','_',a.ref) if a.scope=='history' and a.ref!='--all' else ''
    output=Path('.local/github_preparation')/f'audit_{a.scope}{suffix}.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'scope':a.scope,'files':len(files),'credential_candidate_locations':findings,'personal_path_files':paths,'large_files':large,'private_report':str(output)}))
if __name__=='__main__':main()
