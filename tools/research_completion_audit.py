"""Read-only Prompt-2 completion inventory; never starts training or infers quality."""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, math, zipfile
from pathlib import Path

TOKENS = (20004864, 50003968, 70000640, 100007936)
TAGS = ('dense', 'sparse', 'memory')
R = Path('results/research_v1')
from src.config import fingerprint, load_config

def current_training_hash():
    paths = sorted(p for root in ('src/model','src/moe','src/memory','src/training','src/eval') for p in Path(root).glob('*.py'))
    paths += [Path('src/config.py'),Path('tools/count_params.py'),Path('verify_install.py')]
    return fingerprint({p.as_posix():sha(p) for p in paths})

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def inspect(final=False):
    checks = []
    def check(name, action):
        try:
            detail = action()
            checks.append({'requirement': name, 'passed': True, 'evidence': detail})
        except (OSError, ValueError, KeyError, AssertionError, zipfile.BadZipFile) as e:
            checks.append({'requirement': name, 'passed': False, 'reason': str(e) or 'Required invariant is false'})

    def corpus():
        m = read(R/'corpus_manifest.json')
        assert m['tokens']['train'] >= 150_000_000
        assert m['leakage']['cross_split_groups'] == 0
        assert sha(R/'corpus_manifest.json') == sha('data/research_v1/shards/manifest.json')
        assert sha('data/research_v1/tokenizer.json') == m['tokenizer_sha256']
        for name in ('source_plan.json','corpus_statistics.json','tokenizer_evaluation.json','tokenizer_decision.json'):
            assert read(R/name)
        return {'train_tokens': m['tokens']['train'], 'manifest_sha256': sha(R/'corpus_manifest.json')}
    check('Frozen corpus, tokenizer and acquisition/statistics evidence', corpus)

    def milestone(tag, tokens):
        s = read(f'experiments/research_v1/{tag}/summary_{tokens}.json')
        assert s['source_hash'] == current_training_hash()
        assert s['config_hash'] == fingerprint(load_config(f'configs/research_v1/{tag}.yaml'))
        assert s['data_hash'] == sha('data/research_v1/shards/manifest.json')
        assert s['tokenizer_hash'] == sha('data/research_v1/tokenizer.json')
        assert s['evaluation_hash'] == sha(R/'evaluation_index.json')
        assert s['tokens_seen'] == tokens and s['seed'] == 42
        assert s['step'] == tokens//8192 and s['cursor'] == s['step']*8
        assert sha(s['checkpoint']) == s['checkpoint_sha256']
        for key in ('val_loss','code_val_loss','general_val_loss','technical_val_loss','bits_per_token',
                    'training_step_tok_s','wall_tok_s','wall_time','vram_peak','stored_params','active_params_est',
                    'config_hash','source_hash','data_hash','tokenizer_hash','evaluation_hash'):
            assert s[key] is not None, key
        for key in ('val_loss','code_val_loss','general_val_loss','technical_val_loss','bits_per_token',
                    'training_step_tok_s','wall_tok_s','wall_time'):
            assert math.isfinite(s[key]), key
        e = read(R/f'{tag}_{tokens}_evaluation.json')
        assert e['mixed_predictions'] == 262144 and len(e['documents']) == 743
        assert 0 <= read(R/f'{tag}_{tokens}_micro_code.json')['pass_count'] <= 6
        if tag != 'dense':
            assert read(R/'routing_summary.json')[f'{tag}/{tokens}']
        if tag == 'memory':
            a = read(R/f'memory_{tokens}_ablation.json')
            assert {d['sha256']:d['predictions'] for d in e['documents']} == {d['sha256']:d['predictions'] for d in a['documents']}
            assert set(a['delta_ablated_minus_normal']) == {'val_loss','code_val_loss','general_val_loss','technical_val_loss'}
        return {'summary': f'experiments/research_v1/{tag}/summary_{tokens}.json', 'checkpoint_sha256': s['checkpoint_sha256']}
    for tokens in TOKENS:
        for tag in TAGS: check(f'{tag}: exact {tokens} checkpoint, evaluation and diagnostics', lambda t=tag,n=tokens: milestone(t,n))

    def release(nominal):
        p = Path(f'results/releases/{nominal}M')
        s = read(p/'summary.json')
        assert s['status'] == 'matched_complete' and set(s['models']) == set(TAGS)
        assert read(p/'metrics.json')
        assert (p/'comparison.csv').stat().st_size and (p/'README.md').stat().st_size
        return str(p)
    for nominal in (20,50,70,100): check(f'{nominal}M complete release evidence', lambda n=nominal: release(n))

    def ledger():
        with Path('results/research_history.csv').open(newline='',encoding='utf-8') as f:
            rows = [row for row in csv.DictReader(f) if row['phase']=='research_v1_seed42' and row['seed']=='42']
        expected = {(f'research_v1/{t}',str(n)) for t in TAGS for n in TOKENS}
        assert {(row['experiment'],row['actual_tokens']) for row in rows} == expected and len(rows)==12
        for row in rows:
            tag = row['experiment'].rsplit('/',1)[1]
            s = read(f"experiments/research_v1/{tag}/summary_{row['actual_tokens']}.json")
            assert row['checkpoint_hash'] == s['checkpoint_sha256']
            for key in ('train_loss','learning_rate','val_loss','code_loss','general_loss','technical_loss'):
                assert row[key] not in ('','null') and math.isfinite(float(row[key])), key
        return {'verified_seed42_records':len(rows)}
    check('Permanent ledger: all 12 exact checkpoint identities and measured loss/LR values', ledger)

    def final_integrity():
        x = read(R/'final_integrity.json')
        assert x['passed'] and len(x['milestones']) == 12
        assert x['no_training_epoch_repeat'] and x['prompt1_archive_unchanged']
        for tag in TAGS:
            s = read(f'experiments/research_v1/{tag}/summary_{TOKENS[-1]}.json')
            assert sha(f'experiments/research_v1/{tag}/checkpoints/last.pt') == s['checkpoint_sha256']
        return str(R/'final_integrity.json')
    check('Final CPU tensor/optimizer/cumulative provenance integrity and rolling checkpoints', final_integrity)

    def review():
        d = read(R/'final_decision.json')
        assert d['status'] == 'complete'
        assert d['classification'] in ('GOOD','MIXED','BAD')
        assert d['human_review_confirmed'] is True
        for group, count in (('original_questions',12),('continuation_questions',14)):
            answers = d[group]
            assert set(answers) == {str(i) for i in range(1,count+1)}
            assert all(isinstance(a,str) and a.strip() for a in answers.values())
        decision = d['replication_decision']
        assert decision['rationale'].strip()
        if decision['required']:
            evidence = read(decision['evidence_path'])
            assert evidence['passed'] and evidence['seed'] == 1337
        assert Path(d['review_document']).is_file()
        return {'classification':d['classification'],'review_document':d['review_document']}
    check('All 12 original and 14 continuation answers, classification and conditional replication decision', review)

    def published_evidence():
        card = read('results/model_card_evidence.json')
        assert card['status'] == 'primary_complete' and len(card['measured']['completed_runs']) == 12
        assert all(card[k] for k in ('calculated','planned','unknown','unsupported_claims','operating_conditions'))
        for name in ('MODEL_CARD.md','README.md','FINAL_STATUS.md','RESEARCH_LOG.md','GOALS/Prompt-2.md'):
            assert Path(name).stat().st_size
        assert read(R/'review_evidence.json')['status'] == 'primary_complete'
        assert read(R/'matched_50m_review.json')['status'] == 'complete'
        for name in ('architecture_overview','experiment_timeline','quality_curves','milestone_gaps',
                     'ngram_ablation_curve','throughput','router_specialization','parameter_efficiency'):
            for ext in ('png','svg','pdf'):assert Path(f'results/figures/{name}.{ext}').stat().st_size
        return 'Card, matched review, curve/diagnostic evidence and documentation present; human interpretation is separately required.'
    check('Complete research card, results, figures and documentation', published_evidence)

    def archive():
        receipt = read('results/prompt-2_archive_verification.json')
        assert receipt['passed']
        p = Path(receipt['archive']); assert sha(p) == receipt['archive_sha256']
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            m = json.loads(z.read('MANIFEST.json'))
            for item in m['files']:
                name = item['project_relative_path']
                assert sha(name) == item['sha256']
                assert hashlib.sha256(z.read(name)).hexdigest() == item['sha256']
            for item in m['large_artifact_references']:
                source = Path(item['project_relative_path'])
                assert source.stat().st_size == item['size_bytes'] and sha(source) == item['sha256']
        assert (p.parent/'GOAL.md').read_bytes() == Path('GOALS/Prompt-2.md').read_bytes()
        return {'archive':str(p),'sha256':receipt['archive_sha256']}
    if final: check('Current complete second-copy ZIP, CRC, manifest, artifact references and goal mirror', archive)
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed':all(c['passed'] for c in checks),
            'scope':'Mechanical evidence audit. Existing integrity receipts are corroborating evidence; inspect their coverage and human review before marking the goal complete.',
            'archive_checked':final,'checks':checks}

def main():
    p = argparse.ArgumentParser();p.add_argument('--final',action='store_true')
    p.add_argument('--output',type=Path,default=Path('.local/prompt2_completion_audit.json'));a=p.parse_args()
    result=inspect(a.final);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'passed_checks':sum(c['passed'] for c in result['checks']),
                      'total_checks':len(result['checks']),'remaining':[c['requirement'] for c in result['checks'] if not c['passed']]}))
    if a.final and not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
