"""Measure held-out real text, then make one bounded train-only BPE decision."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, statistics
from pathlib import Path
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
from src.training.data import SPECIAL_TOKENS,LANGUAGE_SAMPLES

ROOT=Path('data/research_v1');RESULTS=Path('results/research_v1')
FIM=['<fim_prefix>','<fim_suffix>','<fim_middle>']
def documents(split):
    for file in sorted((ROOT/'documents').glob(f'{split}_*.jsonl')):
        with file.open(encoding='utf-8') as f:
            for line in f:yield json.loads(line)
def sample():
    rows=collections.defaultdict(list)
    for d in documents('val'):
        language='English technical' if d['kind']=='technical' else d['language']
        if len(rows[language])<30:rows[language].append({'text':d['text'][:16000],'sha256':d['sha256'],'kind':d['kind']})
        if d['kind']=='technical':
            for lang,body in re.findall(r'```(json|yaml|yml)\s*\n(.*?)```',d['text'],re.S|re.I):
                key='JSON' if lang.lower()=='json' else 'YAML'
                if len(rows[key])<30:rows[key].append({'text':body[:16000],'sha256':d['sha256'],'kind':'real_document_fenced_code'})
    return dict(rows)
def evaluate(tokenizer,samples):
    result={}
    for language,rows in samples.items():
        chars=bytes_=lines=tokens=0;fragments=[];long=[];ok=True
        for row in rows:
            text=row['text'];ids=tokenizer.encode(text).ids
            ok &= tokenizer.decode(ids,skip_special_tokens=False)==text
            chars+=len(text);bytes_+=len(text.encode('utf-8'));lines+=max(len(text.splitlines()),1);tokens+=len(ids)
            for ident in re.findall(r'\b[A-Za-z_][A-Za-z_0-9]{3,}\b',text)[:100]:
                n=len(tokenizer.encode(ident).ids);fragments.append(n)
                if len(ident)>=20:long.append(n)
        result[language]=dict(documents=len(rows),tokens=tokens,characters=chars,bytes=bytes_,tokens_per_character=tokens/max(chars,1),
                              tokens_per_byte=tokens/max(bytes_,1),tokens_per_line=tokens/max(lines,1),
                              mean_identifier_fragments=statistics.mean(fragments) if fragments else None,
                              long_identifier_fragments=statistics.mean(long) if long else None,roundtrip=bool(ok),document_hashes=[r['sha256'] for r in rows])
    probes=['    \t  \n\n','λ café 日本語 😀\n','<<= >>= -> :: !== ** // += && ||','requests.exceptions.ConnectionError numpy.linalg.solve async_trait serde_json tokio::spawn',
            'very_long_identifier_describing_repository_architecture_configuration_value']
    result['behavior_probes']=[dict(text=p,ids=tokenizer.encode(p).ids,roundtrip=tokenizer.decode(tokenizer.encode(p).ids,skip_special_tokens=False)==p) for p in probes]
    result['special_tokens']={s:tokenizer.token_to_id(s) for s in SPECIAL_TOKENS+FIM}
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');a=parser.parse_args()
    samples=sample();old=Tokenizer.from_file('data/tokenizer.json');report={'old':evaluate(old,samples),'sample_scope':'Up to30 deterministic held-out real documents/language; bounded first16000 characters. Identifier regex is an approximate lexical probe. Whitespace/Unicode/operators/library names use additional explicit probes.'}
    (RESULTS/'tokenizer_evaluation.json').write_text(json.dumps(report,indent=2))
    if not a.candidate:print(json.dumps(report,indent=2));return
    path=ROOT/'tokenizer_candidate_32768.json'
    if not path.exists():
        tok=Tokenizer(models.BPE(unk_token=None,byte_fallback=True));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
        trainer=trainers.BpeTrainer(vocab_size=32768,min_frequency=3,special_tokens=SPECIAL_TOKENS+FIM,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
        used=collections.Counter();hashes=[]
        def iterator():
            # Category/language stratification caps domination and bounds fitting work.
            for d in documents('train'):
                key=d['kind']+'/'+d['language'];limit=8_000_000 if d['kind']=='general' else 6_000_000 if d['kind']=='technical' else 4_000_000
                if used[key]>=limit:continue
                used[key]+=len(d['text']);hashes.append(d['sha256']);yield d['text']
        tok.train_from_iterator(iterator(),trainer=trainer)
        if tok.get_vocab_size()!=32768:raise RuntimeError('Unexpected research vocabulary size')
        tok.save(str(path));(RESULTS/'tokenizer_training_metadata.json').write_text(json.dumps({'split':'train only','characters_by_stratum':dict(used),'documents':len(hashes),'training_document_hash_sha256':hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),'training_document_hashes':hashes},indent=2))
    candidate=Tokenizer.from_file(str(path));report['candidate_32768']=evaluate(candidate,samples)
    keys=[k for k in samples if k not in ('English','English technical','Markdown')]
    ratios=[report['candidate_32768'][k]['tokens_per_character']/report['old'][k]['tokens_per_character'] for k in keys]
    improvement=1-statistics.median(ratios)
    choose=improvement>=.15 and all(v['roundtrip'] for k,v in report['candidate_32768'].items() if isinstance(v,dict) and 'roundtrip' in v)
    chosen=path if choose else Path('data/tokenizer.json');frozen=ROOT/'tokenizer.json'
    if frozen.exists() and frozen.read_bytes()!=chosen.read_bytes():raise RuntimeError('Research tokenizer already frozen differently')
    frozen.write_bytes(chosen.read_bytes())
    decision=dict(chosen='train-only real-data32768 BPE' if choose else 'retained synthetic32768 BPE',vocab_size=32768,
                  median_code_token_reduction=improvement,criterion='At least15% median real-code token reduction with exact roundtrip; same32768 embedding cost. Larger48k/64k not needed for this micro-model.',
                  tokenizer_path=str(frozen),sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),fim_markers_supported=all(candidate.token_to_id(s) is not None for s in FIM) if choose else False,
                  loss_comparison='Only within this frozen tokenizer. Do not compare absolute NLL to old synthetic-tokenizer results.')
    (RESULTS/'tokenizer_evaluation.json').write_text(json.dumps(report,indent=2));(RESULTS/'tokenizer_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
