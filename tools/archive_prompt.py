"""Freeze changed canonical files; never relocate the working source."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prompt',type=int,required=True)
    parser.add_argument('--references-json',type=Path)
    parser.add_argument('--progress-snapshot',action='store_true',help='Archive incomplete preparation without declaring the phase complete')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    name=f'Prompt-{args.prompt}'; goal=root/'GOALS'/f'{name}.md'
    baseline=json.loads((root/'results/history'/name/'starting_manifest.json').read_text())
    consumed_exclusions={str(item).replace('\\','/').strip('/').casefold()
                         for item in baseline.get('consumed_file_exclusions', [])}
    destination=Path.home()/'Data-Zip'/name
    if not destination.is_dir() or not goal.is_file(): raise RuntimeError('Select unused Prompt-N and establish its goal first')
    prior=destination/'MANIFEST.json'
    prior_manifest={}
    if prior.exists():
        prior_manifest=json.loads(prior.read_text())
        if prior_manifest['project_root']!=str(root) or prior_manifest['starting_commit']!=baseline['starting_commit'] or prior_manifest['prompt']!=name:
            raise RuntimeError('Refusing to overwrite another archive')
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    # Public Git exclusions do not remove private research evidence from the
    # offline prompt archive. Include retained small notes/results/run logs and
    # historical source snapshots, while model/data payloads remain references.
    offline_extensions={'.md','.txt','.json','.jsonl','.csv','.xml','.log','.zip','.png','.svg','.pdf','.html','.css','.js'}
    for directory in ('Documentation','GOALS','web','results','experiments'):
        paths.extend(p.relative_to(root).as_posix() for p in (root/directory).rglob('*')
                     if p.is_file() and p.suffix in offline_extensions and 'checkpoints' not in p.parts)
    manifest_path=f'results/{name.lower()}_archive_manifest.json'
    verification_path=f'results/{name.lower()}_archive_verification.json'
    volatile_local_outputs = {
        'results/prompt3_cpu_generation_benchmark.jsonl',
        'results/prompt3_cpu_generation_prompts.txt',
        'results/prompt3_dense75_250M_test_benchmark.jsonl',
        'results/prompt3_dense75_250M_test_prompts.txt',
        'results/prompt3_sparse75_last_checkpoint_test.jsonl',
        'results/prompt3_sparse75_last_checkpoint_test_prompts.txt',
        'results/prompt3_sparse75_ngram25m_180M_fullgpu_benchmark.jsonl',
        'results/prompt3_sparse75_ngram25m_180M_fullgpu_prompts.txt',
    } if args.prompt in (3,4) else set()
    files=[]
    for relative in sorted(set(paths)):
        normalized=relative.replace('\\','/').casefold()
        if any(normalized==excluded or normalized.endswith('/'+excluded)
               for excluded in consumed_exclusions):
            continue
        if relative in (manifest_path,verification_path) or relative in volatile_local_outputs: continue
        p=root/relative
        if not p.is_file(): raise RuntimeError(f'Missing canonical file: {relative}')
        digest=sha(p)
        if baseline['tracked_files'].get(relative)!=digest:
            files.append(dict(project_relative_path=relative,size_bytes=p.stat().st_size,sha256=digest,
                              change='modified' if relative in baseline['tracked_files'] else 'created'))
    deleted=[p for p in baseline['tracked_files'] if not (root/p).exists()]
    if deleted: raise RuntimeError(f'Unexpected canonical deletions: {deleted}')
    references=[]
    if args.references_json:
        references=json.loads(args.references_json.read_text())['artifacts']
        for item in references:
            p=root/item['project_relative_path']
            if not p.is_file() or p.stat().st_size!=item['size_bytes'] or sha(p)!=item['sha256']:raise RuntimeError('Large artifact reference mismatch')
    else:
        for experiment in ('phase1a_dense','phase1a_sparse','phase1a_memory'):
            p=root/'experiments'/experiment/'checkpoints/last.pt'
            references.append(dict(project_relative_path=p.relative_to(root).as_posix(),size_bytes=p.stat().st_size,
                                   sha256=sha(p),originating_experiment=experiment,archived=False))
    manifest=dict(prompt=name,project_root=str(root),starting_commit=baseline['starting_commit'],files=files,
                  snapshot_type='progress' if args.progress_snapshot else 'final_archive',
                  published_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  large_checkpoint_references=references,large_artifact_references=references,
                  consumed_file_exclusions=baseline.get('consumed_file_exclusions', []),
                  parent_preparation=prior_manifest.get('parent_preparation'),
                  excluded_self_references=[manifest_path,verification_path],
                  excluded_volatile_local_outputs=sorted(volatile_local_outputs),
                  scope=('Progress snapshot; phase preparation remains incomplete. ' if args.progress_snapshot else 'Final archive. ') + 'Changed/new canonical files compared to the beginning-of-prompt byte hashes. ZIP MANIFEST.json and external MANIFEST.json are identical. Manifest and verification receipt are administrative self-references excluded from their own hash inventory; all research source/evidence/goal/report files are hashed. Existing datasets/checkpoints are not copied.')
    body=json.dumps(manifest,indent=2).encode('utf-8')
    (root/manifest_path).write_bytes(body); prior.write_bytes(body)
    (destination/'GOAL.md').write_bytes(goal.read_bytes())
    archive=destination/f'{name}.zip'; temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for item in files: z.write(root/item['project_relative_path'],item['project_relative_path'])
        z.writestr('MANIFEST.json',body)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC failure')
        if z.read('MANIFEST.json')!=prior.read_bytes(): raise RuntimeError('Manifest mismatch')
        for item in files:
            relative=item['project_relative_path']; p=root/relative
            if hashlib.sha256(z.read(relative)).hexdigest()!=item['sha256'] or sha(p)!=item['sha256']:
                raise RuntimeError(f'ZIP/canonical SHA mismatch: {relative}')
    if (destination/'GOAL.md').read_bytes()!=goal.read_bytes(): raise RuntimeError('Goal mirror mismatch')
    receipt=dict(passed=True,prompt=name,archive='~/Data-Zip/'+name+'/'+archive.name,archive_bytes=archive.stat().st_size,
                 snapshot_type='progress' if args.progress_snapshot else 'final_archive',
                 archive_sha256=sha(archive),canonical_files_verified=len(files),zip_crc_verified=True,
                 manifest_paths_and_sha256_verified=True,canonical_files_retained=True,goal_mirror_verified=True)
    (root/verification_path).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (destination/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
