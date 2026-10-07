"""CPU-only standalone research figures from measured CSV/JSON evidence."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('results/research_v1')
COLORS={'dense':'#2878b5','sparse':'#e08d22','memory':'#218b65'}
LABELS={'dense':'DenseCompute','sparse':'SparseV3','memory':'SparseV3 + Ngram'}

def rows(path):
    return list(csv.DictReader(path.open(newline='',encoding='utf-8'))) if path.exists() else []

def save(fig,name):
    fig.savefig(R/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(R/f'{name}.svg',bbox_inches='tight')
    fig.savefig(R/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    table=rows(R/'training_comparison.csv');curves=rows(R/'training_curves.csv');deltas=rows(R/'milestone_deltas.csv')
    if not curves:return
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,key,title in zip(axs.flat,['train_loss','val_loss','code_val_loss','general_val_loss','technical_val_loss','bits_per_token'],
                           ['Training loss (15-log running mean)','Mixed validation NLL','Code diagnostic NLL','General diagnostic NLL','Technical diagnostic NLL','Mixed validation bits/token']):
        for tag,color in COLORS.items():
            selected=sorted([r for r in (curves if key=='train_loss' else table) if r.get('model')==tag or r.get('tag')==tag],key=lambda r:int(r.get('tokens') or r.get('tokens_seen')))
            selected=[r for r in selected if r.get(key) not in ('',None)]
            if not selected:continue
            x=[int(r.get('tokens') or r.get('tokens_seen'))/1e6 for r in selected];y=[float(r[key]) for r in selected]
            if key=='train_loss':
                ax.plot(x,y,color=color,alpha=.18,lw=.6)
                y=[sum(y[max(0,i-14):i+1])/len(y[max(0,i-14):i+1]) for i in range(len(y))]
            ax.plot(x,y,color=color,label=LABELS[tag],marker=None if key=='train_loss' else 'o',ms=4)
        ax.set(title=title,xlabel='Consumed training tokens (millions)',ylabel='nats/token' if key!='bits_per_token' else 'bits/token');ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=9)
    fig.suptitle('Frozen research_v1 · seed42 · cumulative checkpoints\nCategory losses use fixed language-balanced document samples; mixed uses packed batches',fontsize=13)
    save(fig,'quality_curves')
    if deltas:
        fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax,pairs,title in [(axs[0],[('sparse_minus_dense','Sparse − Dense'),('ngram_minus_dense','Ngram − Dense'),('ngram_minus_sparse','Ngram − Sparse')],'Mixed validation gaps'),
                               (axs[1],[('sparse_code_minus_dense','Sparse − Dense'),('ngram_code_minus_sparse','Ngram − Sparse')],'Code diagnostic gaps')]:
            for key,label in pairs:ax.plot([int(r['tokens'])/1e6 for r in deltas],[float(r[key]) for r in deltas],marker='o',label=label)
            ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.18);ax.set(title=title,xlabel='Consumed training tokens (millions)',ylabel='NLL difference (nats/token)');ax.legend(fontsize=9)
        fig.suptitle('Negative gaps favor the first named model · single-seed evidence',fontsize=12)
        save(fig,'milestone_gaps')
    path=R/'ngram_ablations.json'
    if path.exists():
        ablation=json.loads(path.read_text())
        if ablation:
            fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
            for key,label in [('val_loss','Mixed'),('code_val_loss','Code'),('general_val_loss','General'),('technical_val_loss','Technical')]:
                points=sorted((int(t),r['delta_ablated_minus_normal'][key]) for t,r in ablation.items())
                ax.plot([t/1e6 for t,_ in points],[v for _,v in points],marker='o',label=label)
            ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.18);ax.legend(loc='lower right');ax.set(title='Same-checkpoint Ngram residual-zero ablation\nPositive delta means memory helps',xlabel='Consumed training tokens (millions)',ylabel='Ablated − normal NLL (nats/token)')
            save(fig,'ngram_ablation_curve')

if __name__=='__main__':main()
