"""Frozen real-document losses, category routes and memory ablation evidence."""
from __future__ import annotations
import collections,json,math
import re
from pathlib import Path
import numpy as np
import torch
from src.training.data import PackedStream
from src.training.engine import amp
from src.eval.sampling import sample_documents
from src.utils.hashing import sha256_file

RESULTS=Path('results/research_v1')
GROUPS={'Python':'Python','JavaScript':'JS/TS','TypeScript':'JS/TS','C':'C/C++','C++':'C/C++','Rust':'Rust','Java':'Java','Go':'Go',
        'C#':'C#','SQL':'SQL','Bash':'Bash','HTML/CSS':'HTML/CSS'}
def freeze_evaluation(cfg):
    path=RESULTS/'evaluation_index.json'
    version=cfg['data'].get('dataset_version')
    if version and version.startswith('research_v2'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',version): raise ValueError('Invalid dataset version')
        path=Path('results/research_v2_real')/f'evaluation_index_{version}.json'
    if path.exists():
        existing=json.loads(path.read_text())
        if version and version.startswith('research_v2'):
            expected=(cfg['training']['context'],cfg['training']['microbatch'],sha256_file(cfg['data']['manifest']))
            if (existing['context'],existing['microbatch'],existing['manifest_sha256'])!=expected:
                raise ValueError('Frozen evaluation identity differs; use a new version')
        return existing
    with (Path(cfg['data']['shards'])/'val_documents.jsonl').open(encoding='utf-8') as handle:
        selection=sample_documents((json.loads(line) for line in handle),64,cfg['training']['context'],42,
                    lambda row:'code/'+GROUPS.get(row['language'],'other') if row['kind']=='code' else row['kind'],minimum_length=129)
    # Category reporting requires enough document and token evidence, not a tiny label.
    result=dict(documents=dict(selection),minimum_documents=16,minimum_predictions=8192,mixed_batches=128,
                context=cfg['training']['context'],microbatch=cfg['training']['microbatch'],seed=42,
                sampling_policy='hash_rank_documents_and_positions_v1',dataset_version=version,
                manifest_sha256=sha256_file(cfg['data']['manifest']),
                scope='Hash-seeded 64 eligible held-out documents/stratum with independently hashed window offsets, causal token-weighted NLL. Mixed uses128 fixed packed batches separately; category means cannot reconstruct mixed. Existing historical indices are preserved.')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2));return result
def route_accumulate(model,category,storage):
    for layer,r in model.routes().items():
        ids=r['experts'][...,0].detach().cpu().numpy();counts=np.bincount(ids.flatten(),minlength=r['load'].numel())
        key=f'{category}/layer{layer}';entry=storage.setdefault(key,{'counts':np.zeros(len(counts),dtype=np.int64),'transitions':np.zeros((len(counts),len(counts)),dtype=np.int64),'soft_entropy_sum':0.,'tokens':0})
        entry['counts']+=counts;entry['tokens']+=ids.size;entry['soft_entropy_sum']+=float(r['entropy'])*ids.size
        for seq in ids:
            if len(seq)>1:np.add.at(entry['transitions'],(seq[:-1],seq[1:]),1)
def finish_routes(storage):
    result={};totals={}
    for key,e in storage.items():
        layer=key.split('/layer')[-1];totals.setdefault(layer,np.zeros(len(e['counts']),dtype=np.int64));totals[layer]+=e['counts']
    for key,e in storage.items():
        counts=e['counts'];p=counts/max(counts.sum(),1);layer=key.split('/layer')[-1];base=totals[layer]/max(totals[layer].sum(),1)
        result[key]=dict(tokens=e['tokens'],counts=counts.tolist(),frequency=p.tolist(),hard_entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum()),
                         soft_entropy=e['soft_entropy_sum']/max(e['tokens'],1),load_cv=float(counts.std()/max(counts.mean(),1)),
                         max_share=float(p.max()),unused_experts=int((counts==0).sum()),enrichment_vs_all_categories=(p/np.maximum(base,1e-12)).tolist(),
                         transitions=e['transitions'].tolist())
    return result
@torch.no_grad()
def evaluate(model,cfg,index=None,*,full=True,routes=False):
    model.eval();index=index or freeze_evaluation(cfg)
    stream=PackedStream(Path(cfg['data']['shards']),'val',cfg['training']['context'],42)
    weighted_loss=prediction_count=0;batches=index['mixed_batches'] if full else 16
    for _ in range(batches):
        x,y=stream.next(cfg['training']['microbatch'],'cuda')
        with amp(cfg):value=float(model(x,y,return_outputs=False,segment_ids=stream.last_segment_ids)['lm_loss'])
        weighted_loss+=value*stream.last_prediction_count;prediction_count+=stream.last_prediction_count
    result={'val_loss':weighted_loss/prediction_count,'mixed_predictions':prediction_count}
    result['bits_per_token']=result['val_loss']/math.log(2)
    if not full:model.train();return result
    storage={};documents=[];category=collections.defaultdict(lambda:[0.,0,0]);language={};memory={}
    for key,rows in index['documents'].items():
        subtotal=predictions=0
        for row in rows:
            offset=row.get('evaluation_offset',0);start=row['start']+offset
            length=min(cfg['training']['context']+1,row['length']-offset);data=torch.tensor(np.array(stream.tokens[start:start+length],dtype=np.int64),device='cuda')[None]
            with amp(cfg):out=model(data[:,:-1],data[:,1:],return_outputs=False)
            n=length-1;loss=float(out['lm_loss']);subtotal+=loss*n;predictions+=n
            documents.append(dict(sha256=row['sha256'],source_content_sha256=row.get('source_content_sha256',row.get('raw_sha256',row['sha256'])),stratum=key,language=row['language'],predictions=n,nll=loss))
            kind='code' if key.startswith('code/') else key;category[kind][0]+=loss*n;category[kind][1]+=n;category[kind][2]+=1
            if routes:route_accumulate(model,key,storage)
            if model.memory is not None and not model.memory.ablate:
                gate=model.memory.last_gate.float();entry=memory.setdefault(key,{'gate_sum':0.,'tokens':0,'gate_min':1.,'gate_max':0.,'gate_histogram10bins':[0]*10,'sample_bucket_statistics':None})
                entry['gate_sum']+=float(gate.sum());entry['tokens']+=gate.numel();entry['gate_min']=min(entry['gate_min'],float(gate.min()));entry['gate_max']=max(entry['gate_max'],float(gate.max()))
                entry['gate_histogram10bins']=[a+int(b) for a,b in zip(entry['gate_histogram10bins'],torch.histc(gate,bins=10,min=0,max=1).cpu().tolist())]
                if entry['sample_bucket_statistics'] is None:entry['sample_bucket_statistics']=model.memory.statistics(data[:,:-1])
        language[key]=dict(nll=subtotal/predictions if predictions else None,predictions=predictions,documents=len(rows),
                           sufficient=len(rows)>=index['minimum_documents'] and predictions>=index['minimum_predictions'])
    for kind,(total,n,docs) in category.items():result[f'{kind}_val_loss']=total/n if n else None
    for e in memory.values():e['gate_mean']=e.pop('gate_sum')/max(e['tokens'],1)
    result.update(language=language,documents=documents,router=finish_routes(storage) if routes else {},ngram=memory,
                  category_predictions={k:v[1] for k,v in category.items()},scope=index['scope'])
    model.train();return result
