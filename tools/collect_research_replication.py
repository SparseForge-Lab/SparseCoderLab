"""CPU-verify immutable seed1337 milestones and publish separate small evidence."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch

from analysis.research_v1.diagnostics import cluster_interval, routing
from src.config import fingerprint, load_config
from src.model import LanguageModel
from src.training.engine import make_optimizer
from src.training.research import ENDPOINTS, frozen_identity, sha, training_hash
from tools.prepare_research_replication import ROOT, WORKSPACE, PUBLIC

KEYS = ('val_loss','code_val_loss','general_val_loss','technical_val_loss')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(tag, tokens, prepared, index):
    directory = WORKSPACE / f'experiments/research_v1/{tag}'
    summary_path = directory / f'summary_{tokens}.json'
    summary = read(summary_path)
    cfg = load_config(WORKSPACE / f'configs/research_v1/{tag}.yaml')
    assert cfg['training']['seed'] == 1337
    assert fingerprint(cfg) == prepared['configs'][tag]['seed1337_config_hash']
    expected = frozen_identity(cfg)
    checkpoint = directory / f'checkpoints/tokens_{tokens}.pt'
    assert sha(checkpoint) == summary['checkpoint_sha256']
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta = state['metadata']
    for source in (summary,meta):
        assert source['seed'] == 1337 and source['tokens_seen'] == tokens
        assert source['step'] == tokens//8192 and source['cursor'] == tokens//8192*8
        assert all(source[k] == v for k,v in expected.items())
    model = LanguageModel(cfg).cpu()
    shapes = model.state_dict()
    assert set(shapes) == set(state['model'])
    assert all(v.shape==shapes[k].shape and bool(torch.isfinite(v).all()) for k,v in state['model'].items())
    model.load_state_dict(state['model'],strict=True)
    opt,scheduler = make_optimizer(model,cfg)
    opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    assert scheduler.last_epoch == meta['step']
    optimizer_steps = []
    for parameter,values in opt.state.items():
        for key,value in values.items():
            if torch.is_tensor(value):
                assert bool(torch.isfinite(value).all())
                if key in ('exp_avg','exp_avg_sq'):
                    assert value.shape == parameter.shape
        if 'step' in values:
            optimizer_steps.append(int(values['step']))
    assert optimizer_steps and max(optimizer_steps)==meta['step'] and min(optimizer_steps)>0
    assert set(state['rng']) == {'python','numpy','torch','cuda'} and state['rng']['cuda']
    assert state['rng']['torch'].numel() > 0 and state['scaler'] is None
    assert meta['tokens_seen'] < read(ROOT/'results/research_v1/corpus_manifest.json')['tokens']['train']-1024
    evaluation_path = WORKSPACE / f'results/research_v1/{tag}_{tokens}_evaluation.json'
    evaluation = read(evaluation_path)
    assert evaluation['mixed_predictions']==262144 and len(evaluation['documents'])==743
    expected_documents = {d['sha256']:(key,min(index['context']+1,d['length'])-1,d['source_content_sha256'])
                          for key,docs in index['documents'].items() for d in docs}
    actual = {d['sha256']:(d['stratum'],d['predictions'],d['source_content_sha256']) for d in evaluation['documents']}
    assert len(actual)==743 and actual==expected_documents
    assert all(math.isfinite(d['nll']) for d in evaluation['documents'])
    for key in KEYS:
        assert math.isfinite(summary[key]) and summary[key]==evaluation[key]
    micro_path = WORKSPACE/f'results/research_v1/{tag}_{tokens}_micro_code.json'
    micro = read(micro_path)
    assert micro['tasks_count']==6 and 0<=micro['pass_count']<=6
    primary_micro = read(ROOT/f'results/research_v1/{tag}_{tokens}_micro_code.json')
    assert [(t['task_id'],t['prompt_sha256'],t['tests']) for t in micro['tasks']] == [(t['task_id'],t['prompt_sha256'],t['tests']) for t in primary_micro['tasks']]
    files = [summary_path,evaluation_path,micro_path]
    ablation = None
    if tag=='memory':
        ablation_path = WORKSPACE/f'results/research_v1/memory_{tokens}_ablation.json'
        ablation = read(ablation_path)
        assert {d['sha256']:d['predictions'] for d in ablation['documents']} == {d['sha256']:d['predictions'] for d in evaluation['documents']}
        for key in KEYS:
            assert ablation['normal'][key] == evaluation[key]
            assert math.isclose(ablation['delta_ablated_minus_normal'][key],ablation['zero_residual'][key]-evaluation[key],rel_tol=0,abs_tol=1e-12)
        files.append(ablation_path)
    destination = PUBLIC/tag
    destination.mkdir(exist_ok=True)
    for source in files:
        target = destination/source.name
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
    reference = {'project_relative_path':checkpoint.relative_to(ROOT).as_posix(),'sha256':sha(checkpoint),
                 'size_bytes':checkpoint.stat().st_size,'tokens':tokens,'seed':1337}
    integrity = {'passed':True,'seed':1337,'tokens':tokens,'optimizer_step_range':[min(optimizer_steps),max(optimizer_steps)],
                 'checkpoint':reference,'frozen_identity':expected,'no_epoch_repeat':True,
                 'scope':'CPU strict model/Adam/scheduler loading, finite weights/moments and shapes, counters/cursor/saved RNG and exact frozen document identities. No new GPU resume equivalence test.'}
    (destination/f'integrity_{tokens}.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    del model,opt,scheduler,state
    return {'summary':summary,'evaluation':evaluation,'integrity':integrity,'micro_pass_count':micro['pass_count'],
            'routing':routing(evaluation),'ablation':ablation['delta_ablated_minus_normal'] if ablation else None,
            'raw_summary_path_base':'work/replication_seed1337',
            'raw_summary_path_scope':'Inherited checkpoint field is relative to the isolated workspace, never the canonical seed42 output directory.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true')
    args = parser.parse_args()
    assert Path.cwd().resolve()==ROOT
    torch.set_num_threads(2)
    prepared = read(PUBLIC/'preparation.json')
    decision = read(ROOT/'results/research_v1/replication_decision.json')
    assert decision['seed']==1337 and decision['models']==prepared['decision']['models']
    assert prepared['training_source_hash']==training_hash()
    index = read(ROOT/'results/research_v1/evaluation_index.json')
    assert sha(WORKSPACE/'results/research_v1/evaluation_index.json')==sha(ROOT/'results/research_v1/evaluation_index.json')
    completed = {};comparisons = {}
    endpoints = [n for n in ENDPOINTS if n<=decision['endpoint']]
    for tokens in endpoints:
        pair = {}
        for tag in decision['models']:
            if not (WORKSPACE/f'experiments/research_v1/{tag}/summary_{tokens}.json').exists():
                continue
            verified = verify(tag,tokens,prepared,index)
            completed[f'{tag}/{tokens}'] = {k:v for k,v in verified.items() if k!='evaluation'}
            pair[tag] = verified
        if len(pair)==2:
            a,b = decision['models']
            delta = {k:pair[b]['summary'][k]-pair[a]['summary'][k] for k in KEYS}
            primary = {tag:read(ROOT/f'experiments/research_v1/{tag}/summary_{tokens}.json') for tag in (a,b)}
            comparisons[str(tokens)] = {'pair':f'{b}_minus_{a}','seed1337_deltas':delta,
                'seed42_deltas':{k:primary[b][k]-primary[a][k] for k in KEYS},
                'seed1337_cluster_diagnostic_intervals':cluster_interval(pair[a]['evaluation'],pair[b]['evaluation'],index)}
    complete = len(completed)==len(endpoints)*2
    if args.final:
        assert complete,'Selected pair is incomplete'
        for tag in decision['models']:
            checkpoint = WORKSPACE/f'experiments/research_v1/{tag}/checkpoints/last.pt'
            assert sha(checkpoint)==completed[f'{tag}/{decision["endpoint"]}']['integrity']['checkpoint']['sha256']
        for tag in decision['models']:
            counters = [completed[f'{tag}/{n}']['summary'] for n in endpoints]
            assert all(b['training_seconds']>a['training_seconds'] and b['wall_time']>a['wall_time'] for a,b in zip(counters,counters[1:]))
    report = {'passed':bool(args.final and complete),'status':'verified_complete' if args.final and complete else 'partial_verification',
              'seed':1337,'decision':decision,'completed':completed,'comparisons':comparisons,
              'scope':'Separate seed1337 selected-pair replication; primary seed42 evidence untouched. Frozen data/evaluation order42. Two seeds do not give a precise population confidence interval. Inherited raw runner scope says seed42; actual config/checkpoint seed1337 controls classification.'}
    (PUBLIC/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'verified_milestones':len(completed),'matched_milestones':len(comparisons)}))


if __name__=='__main__':
    main()
