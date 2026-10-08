"""Create a new immutable corpus after benchmark and low-information filtering."""
from __future__ import annotations
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from src.config import load_config
from src.utils.hashing import sha256_file
from tools.prepare_data import prepare
from tools.repository_fim import restore,split_header
from tools.shard_data import verify_shards


def quality_reason(body):
    if len(''.join(body.split()))<64:return 'low_information_under64_nonwhitespace_chars'
    lines=[line.strip() for line in body.splitlines() if line.strip()]
    if len(lines)>=32 and max(Counter(lines).values())/len(lines)>.8:
        return 'repetitive_over80percent_identical_nonempty_lines'
    if len(body)>=1024 and sum(char.isalnum() for char in body)/len(body)<.02:
        return 'low_information_under2percent_alphanumeric'
    return None


def finalize(cfg,documents,upstream_report,overlap_report,report):
    output=Path('data/research_v2_real/documents')/(cfg['data']['dataset_version']+'.jsonl')
    if any(p.exists() for p in (output,report,Path(cfg['data']['manifest']))):
        raise FileExistsError('Never overwrite a corpus; choose new versions')
    upstream=json.loads(upstream_report.read_text(encoding='utf8'))
    if not upstream['preprocessing_passed']:raise ValueError('Upstream pipeline failed')
    overlap=json.loads(overlap_report.read_text(encoding='utf8'))
    if overlap['documents_sha256']!=sha256_file(documents):raise ValueError('Overlap screen input differs')
    # Conservatively hold whole matching source files. This does not claim each
    # elementary-code match proves leakage; all removed identities are recorded.
    exclusions={(r['repository'],r['revision'],r['file_path'],r['raw_sha256']) for r in overlap['hits']}
    rejected=[];counts=Counter();fim=Counter()
    output.parent.mkdir(parents=True,exist_ok=True)
    with documents.open(encoding='utf8') as source,output.open('x',encoding='utf8',newline='\n') as handle:
        for line in source:
            row=json.loads(line)
            key=tuple(row[k] for k in ('repository','revision','file_path','raw_sha256'))
            _,body=split_header(dict(row,text=restore(row)))
            reason='benchmark_exact_lexical_overlap_held' if key in exclusions else quality_reason(body)
            if reason:
                counts[reason]+=1
                rejected.append({k:row[k] for k in ('repository','revision','file_path','raw_sha256')}|dict(reason=reason))
                continue
            fim['documents']+=1;fim['eligible']+=int(row['fim']['eligible']);fim['applied']+=int(row['fim']['applied'])
            handle.write(line)
    manifest=prepare(cfg,output);verify_shards(cfg)
    stats=defaultdict(Counter)
    for split in ('train','val'):
        with (Path(cfg['data']['shards'])/f'{split}_documents.jsonl').open(encoding='utf8') as handle:
            for line in handle:
                row=json.loads(line);stats[row['repository']]['documents']+=1;stats[row['repository']]['tokens']+=row['length']
    receipt=dict(schema_version=1,preprocessing_passed=True,dataset_version=cfg['data']['dataset_version'],
        upstream_dataset_version=upstream['dataset_version'],upstream_pipeline_sha256=sha256_file(upstream_report),
        upstream_documents_sha256=sha256_file(documents),source_plan_sha256=upstream['source_plan_sha256'],
        global_deduplication=upstream['dedup'],quality_policy='minimum64 nonwhitespace characters; max80percent repeated nonempty lines when32+lines; minimum2percent alphanumeric when1024+characters',
        exclusions=rejected,exclusion_counts=dict(counts),benchmark_screen_sha256=sha256_file(overlap_report),
        benchmark_policy='Hold complete files with exact screened benchmark sequences, including legitimate common-code coincidences; no semantic/history guarantee',
        fim=dict(fim),tokens=manifest['tokens'],token_mixture=manifest['token_mixture'],
        language_tokens_by_split=manifest['language_tokens_by_split'],repository_packed_counts={k:dict(v) for k,v in stats.items()},
        repositories_by_split=manifest['repositories_by_split'],tokenizer_sha256=manifest['tokenizer_sha256'],
        documents_sha256=sha256_file(output),manifest_sha256=sha256_file(Path(cfg['data']['manifest'])),
        script_sha256=sha256_file(Path(__file__)),scope='Final candidate preprocessing; a second overlap screen and model/storage preflight are required before training.')
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(tokens=manifest['tokens'],rejected=dict(counts),fim=dict(fim))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    for name in ('documents','upstream-report','overlap-report','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();finalize(load_config(args.config),args.documents,args.upstream_report,args.overlap_report,args.report)
