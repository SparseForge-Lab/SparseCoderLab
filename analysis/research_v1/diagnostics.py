"""CPU-only conditional-routing and clustered paired evaluation summaries."""
from __future__ import annotations
import collections,json,math
from pathlib import Path
import numpy as np

R=Path('results/research_v1')
ENDPOINTS=[20004864,50003968,70000640,100007936]

def routing(e):
    result={}
    for layer in ('2','5','8'):
        selected={k:v for k,v in e['router'].items() if k.endswith('/layer'+layer)}
        if not selected:continue
        matrix=np.array([v['counts'] for v in selected.values()],dtype=float)
        joint=matrix/matrix.sum();base=joint.sum(axis=0);category=joint.sum(axis=1)
        expected=category[:,None]*base[None,:]
        valid=joint>0;information=float((joint[valid]*np.log2(joint[valid]/expected[valid])).sum())
        counts=matrix.sum(axis=0);p=counts/counts.sum()
        transitions=sum((np.array(v['transitions'],dtype=float) for v in selected.values()),np.zeros((len(p),len(p))))
        top=[]
        for key,v in selected.items():
            category_key=key.rsplit('/layer',1)[0]
            evidence=e['language'][category_key]
            if not evidence['sufficient']:continue
            enrichment=np.array(v['enrichment_vs_all_categories']);best=int(enrichment.argmax())
            top.append({'category':category_key,'expert':best,'enrichment':float(enrichment[best]),'frequency':v['frequency'][best],
                        'documents':evidence['documents'],'predictions':evidence['predictions']})
        result[layer]={'category_expert_mutual_information_bits':information,'pooled_counts':counts.astype(int).tolist(),
            'pooled_hard_entropy_nats':float(-(p[p>0]*np.log(p[p>0])).sum()),'uniform_entropy_nats':math.log(len(p)),
            'pooled_load_cv':float(counts.std()/counts.mean()),'max_share':float(p.max()),'unused_experts':int((counts==0).sum()),
            'adjacent_same_expert_fraction':float(np.trace(transitions)/max(transitions.sum(),1)),
            'independent_same_expert_baseline':float((p*p).sum()),'strongest_supported_category_biases':sorted(top,key=lambda x:x['enrichment'],reverse=True)[:6],
            'scope':'Descriptive fixed-sample conditional structure; mutual information includes small flagged strata and is not a semantic/usefulness proof.'}
    return result

def cluster_interval(a,b,index):
    first={r['sha256']:r for r in a['documents']};group={row['sha256']:row['split_group'] for rows in index['documents'].values() for row in rows}
    result={}
    for category in ('all','code','general','technical'):
        clusters=collections.defaultdict(lambda:[0.,0])
        for r in b['documents']:
            if category!='all' and not r['stratum'].startswith(category):continue
            old=first[r['sha256']];assert old['predictions']==r['predictions'];n=r['predictions']
            clusters[group[r['sha256']]][0]+=(r['nll']-old['nll'])*n;clusters[group[r['sha256']]][1]+=n
        values=np.array(list(clusters.values()));numerator,weight=values[:,0],values[:,1]
        rng=np.random.default_rng(42);samples=rng.integers(0,len(values),size=(2000,len(values)))
        boot=numerator[samples].sum(axis=1)/weight[samples].sum(axis=1)
        result[category]={'weighted_delta':float(numerator.sum()/weight.sum()),'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
                          'clusters':len(values),'bootstrap_replicates':2000,'seed':42,
                          'caveat':'Paired repository/hostname cluster resampling of fixed diagnostic documents. Does not estimate training-seed uncertainty or prove population-level generalization.'}
    return result

def main():
    index=json.loads((R/'evaluation_index.json').read_text());routes={};intervals={};memory={}
    for tokens in ENDPOINTS:
        evaluations={}
        for tag in ('dense','sparse','memory'):
            p=R/f'{tag}_{tokens}_evaluation.json'
            if not p.exists():continue
            e=json.loads(p.read_text());evaluations[tag]=e
            if tag!='dense':routes[f'{tag}/{tokens}']=routing(e)
            if tag=='memory':
                summary={}
                for kind in ('code','general','technical','context'):
                    selected=[v for k,v in e['ngram'].items() if k.startswith(kind)]
                    total=sum(v['tokens'] for v in selected)
                    summary[kind]={'tokens':total,'gate_mean':sum(v['gate_mean']*v['tokens'] for v in selected)/max(total,1),
                                   'gate_histogram10bins':np.array([v['gate_histogram10bins'] for v in selected]).sum(axis=0).astype(int).tolist()}
                memory[str(tokens)]=summary
        if len(evaluations)==3:
            intervals[str(tokens)]={'sparse_minus_dense':cluster_interval(evaluations['dense'],evaluations['sparse'],index),
                                   'ngram_minus_sparse':cluster_interval(evaluations['sparse'],evaluations['memory'],index)}
    for name,value in [('routing_summary.json',routes),('clustered_document_intervals.json',intervals),('ngram_gate_summary.json',memory)]:
        (R/name).write_text(json.dumps(value,indent=2))
    print(json.dumps({'routing_milestones':len(routes),'paired_milestones':len(intervals),'memory_milestones':len(memory)}))

if __name__=='__main__':main()
