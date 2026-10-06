"""Read live on-disk evidence without touching the training process or GPU."""
from __future__ import annotations
import json
from pathlib import Path

def main():
    progress={}
    for tag in ('dense','sparse','memory'):
        directory=Path('experiments/research_v1')/tag;p=directory/'metrics.jsonl'
        if not p.exists():progress[tag]={'tokens':0};continue
        rows=[]
        for line in p.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        if not rows:progress[tag]={'tokens':0,'state':'initializing'};continue
        last=rows[-1];train=next((r for r in reversed(rows) if 'train_loss' in r),{})
        progress[tag]={'tokens':last['tokens_seen'],'step':last['step'],'train_loss':round(train.get('train_loss',0),4),
                       'step_tok_s_cumulative':round(last['tokens_seen']/last['training_seconds']),
                       'completed_milestones':len(list(directory.glob('summary_*.json')))}
    failure=Path('results/research_v1/training_failure.json')
    if failure.exists():progress['failure']=json.loads(failure.read_text())
    print(json.dumps(progress))

if __name__=='__main__':main()
