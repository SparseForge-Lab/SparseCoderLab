"""Freeze readiness evidence before launching cumulative real-data GPU work."""
from __future__ import annotations
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from src.config import load_config
from src.training.research import RESULTS,training_hash,frozen_identity
from src.eval.research import freeze_evaluation
from tools.shard_data import verify_shards

def main():
    configs={tag:load_config(f'configs/research_v1/{tag}.yaml') for tag in ('dense','sparse','memory')}
    base=configs['dense']
    assert all(c['training']==base['training'] and c['data']==base['data'] for c in configs.values()),'Unmatched training/data policy'
    assert configs['sparse']['model']['moe_backend']==configs['memory']['model']['moe_backend']=='grouped'
    assert all(not c['mtp']['enabled'] and c['model']['top_k']==1 and c['model']['experts']==12 for c in configs.values())
    assert configs['memory']['memory']['enabled'] and not configs['sparse']['memory']['enabled']
    verify_shards(base);freeze_evaluation(base)
    resume=json.loads((RESULTS/'resume_verification.json').read_text());assert resume['passed'] and resume['source_hash']==training_hash()
    from tools.moe_benchmark import gate
    gate()
    tests=ET.parse(RESULTS/'tests.xml').getroot();suites=[tests] if tests.tag=='testsuite' else list(tests)
    assert sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)==0
    assert sum(int(s.get('tests',0)) for s in suites)>=44
    manifest=json.loads((RESULTS/'corpus_manifest.json').read_text());assert manifest['tokens']['train']>=150_000_000 and manifest['leakage']['cross_split_groups']==0
    from tools.phase0 import source_hash
    phase=json.loads(Path('results/phase0_gate.json').read_text());assert phase['passed'] and phase['source_hash']==source_hash()
    report=dict(passed=True,training_source_hash=training_hash(),full_source_hash=source_hash(),frozen_identity=frozen_identity(base),
                test_count=sum(int(s.get('tests',0)) for s in suites),resume_passed=True,grouped_parity_passed=True,
                frozen_tokens=manifest['tokens'],fairness='Identical training/data policies; dense FFN448 vs sparse resident384 plus current12x160 experts; memory alone enabled in third variant. Same tokenizer/data/order.',
                authorization='User Prompt-2 explicitly requests primary cumulative20M/50M/70M/100M quality runs; no additional phase authorized.')
    (RESULTS/'readiness_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
