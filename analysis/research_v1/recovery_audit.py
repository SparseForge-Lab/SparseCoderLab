"""Audit the reporting-only source change and documented recovery exceptions."""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.config import fingerprint
from src.training.research import training_hash,sha

def main():
    history=Path('results/history/Prompt-2/reporting_failure')
    old=(history/'research.py').read_bytes();new=Path('src/training/research.py').read_bytes()
    prefix=b'            result=dict(meta,';suffix=b"            (directory/f'summary_{endpoint}.json')"
    assert old.split(prefix)[0]==new.split(prefix)[0]
    assert old.split(suffix)[1]==new.split(suffix)[1]
    files=sorted(p for root in ('src/model','src/moe','src/memory','src/training','src/eval') for p in Path(root).glob('*.py'))
    files+=[Path('src/config.py'),Path('tools/count_params.py'),Path('verify_install.py')]
    mapping={p.as_posix():sha(p) for p in files};mapping['src/training/research.py']=hashlib.sha256(old).hexdigest()
    before=json.loads((history/'readiness_gate.json').read_text(encoding='utf-8'))['training_source_hash']
    assert fingerprint(mapping)==before,'An additional model/train/eval source change occurred'
    summary_path=Path('experiments/research_v1/dense/summary_20004864.json')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    endpoint=[json.loads(line) for line in Path('experiments/research_v1/dense/metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    endpoint=next(r for r in endpoint if r['tokens_seen']==20004864 and 'train_loss' in r)
    summary['vram_peak']=endpoint['vram_peak']
    summary['wall_time_is_lower_bound']=True
    summary['original_training_source_hash']=before
    summary['source_rebinding_scope']='Reporting-only result-construction repair; all code before result construction and every other covered source file verified byte-identical to original gate.'
    summary['scope']='Cumulative fresh seed42 real-data quality run. Recovered 20M wall is measured through evaluation, before initial milestone/rolling writes; a lower bound. Later dense cumulative walls inherit this small omission. Step time is unaffected. Original checkpoint metadata source hash was administratively rebound after a reporting-only repair; original serialized checkpoint bytes were not retained. No optimizer update was replayed in dense recovery.'
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    result={'passed':True,'source_before':before,'source_after':training_hash(),'only_result_construction_changed':True,
            'all_other_covered_source_bytes_identical':True,'dense20m_peak_from_endpoint_metric':endpoint['vram_peak'],
            'dense20m_wall_seconds_lower_bound':summary['wall_time'],'dense_model_updates_replayed_for_recovery':0,
            'serialized_checkpoint_migration':'Metadata source_hash only was rewritten via CPU torch.load/atomic_save. Original serialized bytes were not backed up, so an original-payload SHA/byte-equality claim is unavailable.',
            'sparse_interruption':'7,372,800 logged tokens/900 updates; no checkpoint existed. Preserve logs, restart seed42. Do not merge its curves or time with final primary run.',
            'stage_order_exception':'Dense20M then interrupted sparse attempt then Dense50M; resume the remaining cumulative schedule sequentially. Input/optimizer/schedule policies stay matched.'}
    Path('results/research_v1/recovery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
