"""Sequential primary cumulative stages; refuses an unverified research gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from src.training.research import ENDPOINTS,run_stage,training_hash,sha,frozen_identity
from src.config import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',type=int,choices=ENDPOINTS);p.add_argument('--model',choices=['dense','sparse','memory']);a=p.parse_args()
    gate=json.loads(Path('results/research_v1/readiness_gate.json').read_text())
    if not gate['passed'] or gate['training_source_hash']!=training_hash():raise RuntimeError('Research correctness/resume gate absent or stale')
    for endpoint in ([a.endpoint] if a.endpoint else ENDPOINTS):
        for tag in ([a.model] if a.model else ['dense','sparse','memory']):
            summary=Path(f'experiments/research_v1/{tag}/summary_{endpoint}.json')
            evaluation=Path(f'results/research_v1/{tag}_{endpoint}_evaluation.json')
            if summary.exists() and evaluation.exists():
                completed=json.loads(summary.read_text())
                expected=frozen_identity(load_config(f'configs/research_v1/{tag}.yaml'))
                if completed['tokens_seen']!=endpoint or any(completed.get(k)!=v for k,v in expected.items()):raise RuntimeError('Completed milestone identity mismatch')
                if sha(completed['checkpoint'])!=completed['checkpoint_sha256']:raise RuntimeError('Completed milestone checkpoint corrupted')
                print(json.dumps({'verified_completed':tag,'tokens':endpoint}),flush=True)
                continue
            run_stage(tag,endpoint)
if __name__=='__main__':main()
