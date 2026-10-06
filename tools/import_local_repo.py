"""Explicit local repository ingestion with user-declared, allowlisted license."""
import argparse, json, subprocess
from pathlib import Path

EXTENSIONS = {'.py':'Python','.c':'C','.h':'C','.cpp':'C++','.rs':'Rust','.java':'Java','.go':'Go','.js':'JavaScript','.ts':'TypeScript','.html':'HTML/CSS','.css':'HTML/CSS','.sql':'SQL','.sh':'Bash','.json':'JSON/YAML','.yaml':'JSON/YAML'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True); p.add_argument('--license', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--max-mib', type=int, default=64)
    a = p.parse_args(); root = a.repo.resolve()
    if a.license not in ('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','CC0-1.0','ISC'): raise ValueError('Explicit permissive license required')
    if not any((root / name).exists() for name in ('LICENSE','LICENSE.txt','LICENSE.md','COPYING')): raise ValueError('Repository license file missing')
    files = subprocess.check_output(['git','-C',str(root),'ls-files'], text=True).splitlines(); total = 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', encoding='utf-8') as out:
        for name in sorted(files):
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path.suffix not in EXTENSIONS or not path.is_file() or path.stat().st_size > 1024**2: continue
            try: text = path.read_text(encoding='utf-8')
            except UnicodeError: continue
            doc = {'text': '<repo>' + root.name + '<file>' + name + '\n' + text, 'source': str(root), 'license': a.license, 'kind':'code', 'language':EXTENSIONS[path.suffix]}
            line = json.dumps(doc, ensure_ascii=False)+'\n'; size = len(line.encode('utf-8'))
            if total + size > a.max_mib * 1024**2: break
            out.write(line); total += size
    print(f'Wrote {total} bytes. Verify per-file license exceptions before training; repo was read-only.')
