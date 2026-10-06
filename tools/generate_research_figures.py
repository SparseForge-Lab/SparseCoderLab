"""Render research figures from canonical measured CSV/JSON (CPU only)."""
from __future__ import annotations
import csv,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.research_v1 import plots

R=Path('results/research_v1');OUT=Path('results/figures')
TAGS=('dense','sparse','memory')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(fig,name):
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True);plots.main()
    for name in ('quality_curves','milestone_gaps','ngram_ablation_curve'):
        for ext in ('png','svg','pdf'):
            p=R/f'{name}.{ext}'
            if p.exists():shutil.copy2(p,OUT/p.name)
    table=list(csv.DictReader((R/'training_comparison.csv').open(encoding='utf-8',newline='')))
    counts=read(R/'parameter_counts.json')
    evidence=read(Path('results/model_card_evidence.json'))
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');ax.axis('off')
    for i,tag in enumerate(TAGS):
        arch=evidence['calculated']['architectures'][tag];m=arch['backbone']
        text=(f"{plots.LABELS[tag]}\n{m['layers']} layers · width {m['d_model']} · context {arch['context']}\n"
              f"{counts[tag]['total']/1e6:.2f}M stored / {counts[tag]['active_estimate']/1e6:.2f}M active (estimate)\n"
              +('Dense feed-forward blocks' if tag=='dense' else 'Grouped Top-1 · 12 experts at layers 2, 5, 8')
              +('\n4 hashed Ngram tables · layer 1 residual' if tag=='memory' else ''))
        ax.text((i+.5)/3,.58,text,ha='center',va='center',transform=ax.transAxes,fontsize=10,
                bbox={'boxstyle':'round,pad=.8','facecolor':plots.COLORS[tag],'alpha':.15})
    ax.text(.5,.14,'Shared tokenizer, real corpus, token order, seed42, batch and 100M LR schedule',ha='center',transform=ax.transAxes)
    ax.set_title('Current measured research architectures · future scale-up remains unstarted');save(fig,'architecture_overview')
    timeline=read(Path('results/research_timeline.json'))
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');ax.axis('off')
    for i,event in enumerate(timeline):
        ax.text(.03,.95-i*.13,f"{event['stage']} — {event.get('status',event.get('scope',''))}",transform=ax.transAxes,fontsize=10)
    ax.set_title('Research timeline · synthetic, runtime and real-data evidence are separate');save(fig,'experiment_timeline')
    for name,key,title,ylabel in [('throughput','training_step_tok_s','Cumulative training throughput (single run)','predicted tokens / training second'),
                                  ('parameter_efficiency','val_loss','Quality against stored parameter budget','Mixed validation NLL (nats/token)')]:
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        for tag in TAGS:
            rs=sorted((r for r in table if r['tag']==tag),key=lambda r:int(r['tokens_seen']))
            if not rs:continue
            if name=='parameter_efficiency':
                for row in rs:
                    ax.scatter(float(row['stored_params'])/1e6,float(row[key]),color=plots.COLORS[tag])
                    ax.annotate(f"{int(row['tokens_seen'])/1e6:.0f}M tokens",(float(row['stored_params'])/1e6,float(row[key])),xytext=(5,3),textcoords='offset points',fontsize=8)
                ax.plot([],[],color=plots.COLORS[tag],marker='o',label=plots.LABELS[tag])
            else:ax.plot([int(r['tokens_seen'])/1e6 for r in rs],[float(r[key]) for r in rs],marker='o',color=plots.COLORS[tag],label=plots.LABELS[tag])
        ax.set(title=title,xlabel='Stored parameters (millions)' if name=='parameter_efficiency' else 'Consumed training tokens (millions)',ylabel=ylabel)
        ax.legend();ax.grid(alpha=.2)
        fig.text(.02,-.01,'Active compute is an estimate; quality training timings are not the controlled Prompt-1 benchmark.' if name=='throughput' else 'Architectures have different stored capacity; this is not a causal parameter-efficiency proof.',fontsize=8)
        save(fig,name)
    path=R/'routing_summary.json'
    if path.exists() and read(path):
        routing=read(path);fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,layer in zip(axes,('2','5','8')):
            for tag in ('sparse','memory'):
                rs=sorted((int(k.split('/')[1]),v[layer]['category_expert_mutual_information_bits']) for k,v in routing.items() if k.startswith(tag+'/'))
                if rs:ax.plot([t/1e6 for t,_ in rs],[v for _,v in rs],marker='o',color=plots.COLORS[tag],label=plots.LABELS[tag])
            ax.set(title=f'Layer {layer}',xlabel='Training tokens (millions)',ylabel='Category / expert MI (bits)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Held-out routing structure · descriptive category association, not proof of useful specialization');save(fig,'router_specialization')
    sources=['results/research_v1/training_comparison.csv','results/research_v1/training_curves.csv','results/research_v1/milestone_deltas.csv','results/research_v1/ngram_ablations.json','results/research_v1/routing_summary.json','results/model_card_evidence.json','results/research_timeline.json']
    (OUT/'README.md').write_text('# Research figures\n\nGenerated by tools/generate_research_figures.py using canonical CSV/JSON only. Partial curves include completed checkpoints only. PNG/SVG/PDF share the same data. No values are interpolated into future milestones.\n\nSources:\n\n'+'\n'.join('- '+p for p in sources)+'\n',encoding='utf-8')
    print(json.dumps({'figures':len(list(OUT.glob('*.png'))),'sources':sources}))
if __name__=='__main__':main()
