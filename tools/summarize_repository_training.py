"""Build matched aggregate tables and standalone research figures."""
from __future__ import annotations
import collections,csv,json,math
from pathlib import Path
import numpy as np
from tools.repository_training_campaign import TAGS,TARGET,write_json
from src.utils.hashing import sha256_file

DIRECTORY=Path('results/repository_training_v1')
LABELS=dict(zip(TAGS,('Dense','Sparse','Sparse + 10M Ngram','Sparse + 25M Ngram')))
COLORS=dict(zip(TAGS,('#2468a0','#d67526','#27835c','#8155a5')))
KINDS=('mixed','code','technical','general')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def jsonl(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]
def nll(record,kind):return record['val_loss' if kind=='mixed' else kind+'_val_loss']
def weighted_routes(record):
    rows=list(record.get('router',{}).values())
    if not rows:return None
    tokens=sum(row['tokens'] for row in rows)
    combined=collections.defaultdict(lambda:np.zeros(12,dtype=np.int64))
    for key,row in record['router'].items():combined[key.split('/layer')[-1]]+=np.asarray(row['counts'])
    return dict(soft_entropy=sum(row['soft_entropy']*row['tokens'] for row in rows)/tokens,
        hard_entropy=sum(row['hard_entropy']*row['tokens'] for row in rows)/tokens,
        mean_load_cv=sum(row['load_cv']*row['tokens'] for row in rows)/tokens,
        maximum_stratum_expert_share=max(row['max_share'] for row in rows),
        pooled_layers={layer:dict(counts=counts.tolist(),frequency=(counts/counts.sum()).tolist(),
            unused_experts=int((counts==0).sum()),under_one_percent_experts=int((counts/counts.sum()<.01).sum())) for layer,counts in combined.items()},
        scope='Frozen document-window routes; pooled across sampled strata, not the full corpus or lifetime training utilization.')
def gates(record):
    rows=list(record.get('ngram',{}).values())
    return sum(row['gate_mean']*row['tokens'] for row in rows)/sum(row['tokens'] for row in rows) if rows else None

def main():
    manifest=read(DIRECTORY/'matched_gate_checkpoints.json')
    baseline=read('results/research_v2_real/transition_cuda_preflight_r2.json')['variants']
    old_functions=read('results/research_v2_real/functional_canonical_gpu_r1.json')
    functions=read(DIRECTORY/'functional_250M_r1.json')
    if not functions['baseline_evaluation_identity_matched']:raise ValueError('Functional evaluation identity differs')
    old_summary={(row['variant'],row['temperature']):row for row in old_functions['summaries']}
    new_summary={(row['variant'],row['temperature']):row for row in functions['summaries']}
    old_cases={(row['variant'],row['temperature'],row['task_id']):row for row in old_functions['completions']}
    new_cases={(row['variant'],row['temperature'],row['task_id']):row for row in functions['completions']}
    curves=[];models={};efficiency=[]
    input_paths=[DIRECTORY/'matched_gate_checkpoints.json',DIRECTORY/'functional_250M_r1.json',
        Path('results/research_v2_real/transition_cuda_preflight_r2.json'),
        Path('results/research_v2_real/functional_canonical_gpu_r1.json')]
    for tag in TAGS:
        run=Path('experiments/prompt5_v1')/tag;summary=read(run/'campaign_summary.json')
        input_paths.extend([run/'campaign_summary.json',run/'metrics.jsonl',run/'campaign_events.jsonl',
            DIRECTORY/(tag+'_evaluation_250M.json')])
        final=read(DIRECTORY/(tag+'_evaluation_250M.json'));prior=baseline[tag]['baseline_evaluation']
        gate=manifest['matched_gate_checkpoints'][tag]
        if summary['tokens_seen']!=TARGET or final['checkpoint_sha256']!=gate['model_weights']['sha256']:
            raise ValueError('Unmatched final gate/checkpoint')
        if (summary['inherited_tokens'],summary['new_phase_tokens'],summary['phase_steps'],summary['cursor'])!=(100007936,149995520,18310,146480):
            raise ValueError('Final inherited/new/cursor counts differ')
        events=jsonl(run/'campaign_events.jsonl')
        sample_events={row['step']:row for row in events if row['stage']=='training_sample'}
        # Latest sample at each step replaces discarded logs following rollback.
        metrics={row['step']:row for row in jsonl(run/'metrics.jsonl') if 'train_loss' in row}
        for step,row in sorted(metrics.items()):
            event=sample_events[step]
            curves.append(dict(variant=tag,step=step,new_phase_tokens=row['new_phase_tokens'],total_tokens=row['tokens_seen'],
                train_loss=row['train_loss'],grad_norm=row['grad_norm'],lr=event['lr'][0],
                tokens_per_second=row['tok_s'],updates_per_second=row['tok_s']/8192,
                data_wait_seconds=row['data_wait_wall_seconds'],peak_allocated_bytes=row['vram_peak'],
                peak_reserved_bytes=row['vram_peak_reserved'],cpu_rss_bytes=event['cpu_rss_bytes'],aux_loss=event['aux_loss']))
        writes=[row for row in events if row['stage']=='checkpoint_verified']
        attempts=[row for row in events if row['stage'] in ('end','failure')]
        total_attempt_seconds=sum(row.get('process_seconds',row.get('seconds',0)) for row in attempts)
        waits=[row['data_wait_wall_seconds'] for row in metrics.values()]
        greedy=new_summary[(tag,0.)];old_greedy=old_summary[(tag,0.)]
        outcomes=greedy['outcomes']
        functional=dict(baseline_greedy_pass=old_greedy['correct'],final_greedy_pass=greedy['correct'],tasks=25,
            wrong_answer=outcomes.get('wrong',0),syntax_error=outcomes.get('syntax_error',0),runtime_error=outcomes.get('runtime_error',0),
            timeout=outcomes.get('timeout',0),other_outcomes={k:v for k,v in outcomes.items() if k not in ('correct','wrong','syntax_error','runtime_error','timeout')},
            repetition_count=round(greedy['repetition_rate']*25),correct_without_repetition=greedy['correct_without_repetition'],
            temperatures=[dict(temperature=temp,baseline_pass=old_summary[(tag,temp)]['correct'],final_pass=new_summary[(tag,temp)]['correct'],
                final_outcomes=new_summary[(tag,temp)]['outcomes'],final_repetition_rate=new_summary[(tag,temp)]['repetition_rate']) for temp in (0.,.2,.5,.8,1.,1.2,1.5)])
        paired=collections.defaultdict(list)
        for (variant,temp,task),row in new_cases.items():
            if variant!=tag or temp!=0:continue
            before=old_cases[(tag,temp,task)]['functional_correct'];after=row['functional_correct']
            paired['retained_pass' if before and after else 'gained_pass' if after else 'lost_pass' if before else 'remained_fail'].append(task)
        functional['paired_greedy_changes']=dict(paired)
        row=dict(model=LABELS[tag],variant=tag,inherited_tokens=100007936,new_phase_tokens=149995520,total_tokens=TARGET,
            phase_updates=18310,training_seconds=summary['training_seconds'],loop_wall_seconds=summary['wall_time'],
            summed_attempt_process_seconds=total_attempt_seconds,training_tokens_per_second=summary['train_step_tok_s'],
            loop_tokens_per_second=summary['tok_s'],training_updates_per_second=summary['train_updates_per_second'],
            peak_allocated_gib=summary['vram_peak']/1024**3,peak_reserved_gib=summary['vram_peak_reserved']/1024**3,
            cpu_rss_peak_sampled_gib=max((event['cpu_rss_peak_sampled_bytes'] for event in sample_events.values()),default=0)/1024**3,
            verified_checkpoint_writes=len(writes),checkpoint_write_rotation_seconds=sum(event['write_and_rotation_seconds'] for event in writes),
            checkpoint_verification_seconds=sum(event['verification_seconds'] for event in writes),
            sampled_loader_wait_mean_ms=1000*np.mean(waits),sampled_loader_wait_p95_ms=1000*np.quantile(waits,.95),
            sampled_loader_wait_max_ms=1000*max(waits),stored_parameters=summary['stored_params'],active_parameters_estimate=summary['active_params_est'],
            greedy_pass=greedy['correct'],greedy_repetition_count=functional['repetition_count'])
        efficiency.append(row)
        models[tag]=dict(label=LABELS[tag],token_counts=dict(inherited=100007936,new=149995520,total=TARGET),
            nll={kind:dict(baseline=nll(prior,kind),final=nll(final,kind),delta_final_minus_baseline=nll(final,kind)-nll(prior,kind)) for kind in KINDS},
            language={key:dict(baseline=prior['language'][key]['nll'],final=value['nll'],sufficient=value['sufficient'],
                predictions=value['predictions'],documents=value['documents']) for key,value in final['language'].items()},
            functional=functional,efficiency=row,router_baseline=weighted_routes(prior),router_final=weighted_routes(final),
            ngram=dict(baseline_gate_mean=gates(prior),final_gate_mean=gates(final),
                baseline_residual_off_mixed_nll=prior.get('residual_off',{}).get('val_loss'),
                final_residual_off_mixed_nll=final.get('residual_off',{}).get('val_loss'),
                final_residual_benefit_nll=final.get('residual_off',{}).get('val_loss',nll(final,'mixed'))-nll(final,'mixed'),
                residual_off_by_stratum={kind:dict(baseline=nll(prior['residual_off'],kind),
                    final=nll(final['residual_off'],kind),
                    final_minus_enabled=nll(final['residual_off'],kind)-nll(final,kind))
                    for kind in KINDS} if 'residual_off' in final else {}),
            checkpoints=gate)
    for tag in TAGS[2:]:
        models[tag]['ngram']['residual_off_minus_sparse']={kind:
            models[tag]['ngram']['residual_off_by_stratum'][kind]['final']-models[TAGS[1]]['nll'][kind]['final']
            for kind in KINDS}
    pairs=[]
    for left,right in ((TAGS[0],TAGS[1]),(TAGS[1],TAGS[2]),(TAGS[1],TAGS[3]),(TAGS[2],TAGS[3])):
        pairs.append(dict(left=left,right=right,nll_left_minus_right={kind:models[left]['nll'][kind]['final']-models[right]['nll'][kind]['final'] for kind in KINDS},
            greedy_pass_left_minus_right=models[left]['functional']['final_greedy_pass']-models[right]['functional']['final_greedy_pass']))
    result=dict(schema_version=1,models=models,pairwise=pairs,
        input_sha256={path.as_posix():sha256_file(path) for path in input_paths},
        summarizer_sha256=sha256_file(Path(__file__)),
        scope='Matched checkpoint-bound measurements, one training seed. NLL deltas are nats/prediction. Greedy outcomes use25 fixed tasks; repetition overlaps correctness. Seven temperatures are repeated tasks, not pass@k.',
        timing_scope='Training/loop counters follow resumed verified state and exclude discarded uncheckpointed updates. Summed attempt process time includes failed attempts and first-update verification, excluding pre-run identity checks and final promotions. Loader/CPU/router statistics are periodic samples. Checkpoint costs cover verified writes only.',
        limitations=['One training seed; no seed robustness or statistical winner claim.',
            'General data is a narrow logic/mathematics subset.',
            'Frozen sampled windows are not full repository tasks; mixed and category samples differ.',
            'Functional correctness means passing the declared finite cases, not instruction compliance or agent capability.',
            'Repetition is a separate heuristic, including for passing completions.',
            'Residual-off changes inference at a checkpoint trained with memory; it does not retrain a matched memory-free counterfactual.',
            'Active parameter/FLOP counts are estimates, not measured hardware FLOPs.'])
    write_json(DIRECTORY/'comparison_250M.json',result)
    for name,rows in (('training_curves_250M.csv',curves),('efficiency_250M.csv',efficiency)):
        with (DIRECTORY/name).open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(models,curves)

def plot(models,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    directory=Path('results/figures');directory.mkdir(exist_ok=True)
    figure,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for tag in TAGS:
        rows=[row for row in curves if row['variant']==tag];x=np.array([row['new_phase_tokens']/1e6 for row in rows])
        for axis,field,scale in ((axes[0,0],'train_loss',1),(axes[0,1],'grad_norm',1),(axes[1,0],'tokens_per_second',1e-3)):
            y=np.array([row[field]*scale for row in rows]);window=21
            axis.plot(x,y,color=COLORS[tag],alpha=.13,linewidth=.6)
            axis.plot(x[window-1:],np.convolve(y,np.ones(window)/window,'valid'),color=COLORS[tag],label=LABELS[tag])
        axes[1,1].plot(x,[row['peak_allocated_bytes']/1024**3 for row in rows],color=COLORS[tag],label=LABELS[tag])
    for axis,title,ylabel in zip(axes.flat,('Training loss','Gradient norm before clipping','Training throughput','Peak allocated CUDA memory'),
            ('Nats / valid prediction','L2 norm','Thousand tokens / second','GiB')):
        axis.set(title=title,xlabel='New repository-phase tokens (millions)',ylabel=ylabel);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    figure.suptitle('Matched repository training — 21 logged-batch rolling means; memory is cumulative peak')
    for suffix in ('png','pdf','svg'):figure.savefig(directory/('repository_training_curves_250M.'+suffix),dpi=160)
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    positions=np.arange(4);width=.18
    for index,tag in enumerate(TAGS):
        axes[0].bar(positions+(index-1.5)*width,[models[tag]['nll'][kind]['final'] for kind in KINDS],width,color=COLORS[tag],label=LABELS[tag])
    axes[0].set_xticks(positions,KINDS);axes[0].set(ylabel='Frozen held-out NLL (nats / prediction)',title='Matched 250M gate')
    before=[models[tag]['functional']['baseline_greedy_pass'] for tag in TAGS];after=[models[tag]['functional']['final_greedy_pass'] for tag in TAGS]
    axes[1].bar(positions-.17,before,.34,color='#aab2bb',label='100M baseline')
    bars=axes[1].bar(positions+.17,after,.34,color=[COLORS[tag] for tag in TAGS],label='250M gate')
    axes[1].bar_label(bars,padding=3);axes[1].set_xticks(positions,['Dense','Sparse','+10M','+25M'])
    axes[1].set(ylabel='Passing fixed tasks out of 25',title='Greedy functional coding',ylim=(0,25));axes[1].legend()
    axes[0].legend(fontsize=8);figure.suptitle('One seed and a bounded Python probe; NLL alone does not establish capability')
    for suffix in ('png','pdf','svg'):figure.savefig(directory/('repository_matched_quality_250M.'+suffix),dpi=160)
    plt.close(figure)

if __name__=='__main__':main()
