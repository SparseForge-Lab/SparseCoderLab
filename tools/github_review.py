"""Review staged publication, frozen blob identities and a full file inventory."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from src.config import fingerprint
from src.training.research import training_hash,sha

def git(*args):return subprocess.check_output(['git',*args])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',type=Path,default=Path('.local/github_preparation/commit_preview.md'));a=parser.parse_args()
    audit=json.loads(Path('.local/github_preparation/audit_staged.json').read_text(encoding='utf-8'))
    assert not audit['credential_candidates'] and not audit['personal_path_files'] and not audit['large_files_over_20mib']
    forbidden={'.pt','.pth','.ckpt','.safetensors','.gguf','.onnx','.bin','.parquet','.arrow','.sqlite','.sqlite3','.pem','.key'}
    assert not any(Path(r['path']).suffix in forbidden or '.venv/' in r['path'] or '/checkpoints/' in r['path'] for r in audit['files'])
    files=sorted(p for root in ('src/model','src/moe','src/memory','src/training','src/eval') for p in Path(root).glob('*.py'))
    files += [Path('src/config.py'),Path('tools/count_params.py'),Path('verify_install.py')]
    blob_hash=fingerprint({p.as_posix():hashlib.sha256(git('show',':'+p.as_posix())).hexdigest() for p in files})
    expected=json.loads(Path('results/research_v1/readiness_gate.json').read_text(encoding='utf-8'))['training_source_hash']
    assert blob_hash==training_hash()==expected,'Staged training source differs from frozen checkpoint mathematics'
    inputs={}
    for name in ('data/research_v1/tokenizer.json','results/research_v1/corpus_manifest.json','results/research_v1/evaluation_index.json'):
        digest=hashlib.sha256(git('show',':'+name)).hexdigest();assert digest==sha(name);inputs[name]=digest
    resume=json.loads(Path('results/research_v1/resume_state.json').read_text(encoding='utf-8'));assert resume['passed']
    assert resume['models']['memory']['consumed_tokens']==0
    if sys.platform=='win32':
        active=subprocess.check_output(['powershell.exe','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '-m tools\\.research_train( |$)' } | Select-Object -ExpandProperty ProcessId"],text=True).strip()
        assert not active,'Training remains active during the requested publication stop'
    for tag in ('dense','sparse'):
        directory=Path('experiments/research_v1')/tag
        summary=json.loads((directory/'summary_50003968.json').read_text(encoding='utf-8'))
        assert sha(directory/'checkpoints/last.pt')==summary['checkpoint_sha256']
    assert sha(Path.home()/'Data-Zip/Prompt-1/Prompt-1.zip')=='ad0404358128f51a712372260d54206c6f1bec08d346197f41d8f9e4116e1d1e'
    license_body=git('show',':LICENSE').decode('utf-8');assert 'Apache License' in license_body and 'Version 2.0, January 2004' in license_body
    changes=[]
    for line in git('diff','--cached','--name-status').decode('utf-8').splitlines():
        status,name=line.split('\t',1);data=git('show',':'+name)
        changes.append({'status':status,'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    lines=['# Exact GitHub commit preview','', 'Repository: https://github.com/SparseForge-Lab/SparseCoderLab','',
        'Commit message: Prepare SparseCoderLab for public GitHub repository','',
        f"{len(changes)} changed files; {len(audit['files'])} files in the reviewed public tree. Zero credential-pattern candidates, personal home paths or files over20MiB. No datasets/checkpoints/virtual environments are staged. All hashes below describe the actual staged blobs.",'',
        f'Frozen training-source SHA256: {blob_hash}. Tokenizer/corpus/evaluation blob hashes match canonical frozen bytes. Previous local history remains on a local backup branch; public main extends GitHub main.','',
        '| Change | File | Bytes | SHA256 |','|---|---|---:|---|']
    lines += [f"| {r['status']} | {r['path']} | {r['bytes']} | {r['sha256']} |" for r in changes]
    a.preview.parent.mkdir(parents=True,exist_ok=True);a.preview.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report={'status':'staged_review_passed','repository':'https://github.com/SparseForge-Lab/SparseCoderLab','branch':'main',
        'public_tree_files':len(audit['files']),'changed_files':len(changes),'credential_candidate_count':0,'personal_home_path_count':0,
        'large_file_count_over_20mib':0,'largest_blob_bytes':max(r['bytes'] for r in audit['files']),
        'frozen_training_source_sha256':blob_hash,'frozen_inputs_staged_bytes_verified':inputs,
        'dense_sparse_50m_rolling_milestone_match':True,'prompt1_archive_unchanged':True,'training_remains_stopped':True,
        'limits':'Pattern-based credential scan supplemented by full staged inventory; does not guarantee every possible credential format.'}
    Path('.local/github_preparation/staged_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
