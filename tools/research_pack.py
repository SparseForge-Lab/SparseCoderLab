"""Freeze one real-data token stream, with global uniqueness and group audit."""
from __future__ import annotations
import collections, hashlib, json, random
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tools.research_tokenizer import ROOT,RESULTS,documents

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
class Writer:
    def __init__(self,root,split):self.root=root;self.split=split;self.files=[];self.total=0;self.current=0;self.handle=None
    def add(self,tokens):
        data=np.asarray(tokens,dtype=np.uint16);self.total+=len(data)
        while len(data):
            if self.handle is None:
                path=self.root/f'{self.split}-{len(self.files):05d}.bin';self.files.append(path);self.handle=path.open('wb');self.current=0
            n=min(len(data),8_388_608-self.current);data[:n].tofile(self.handle);data=data[n:];self.current+=n
            if self.current==8_388_608:self.handle.close();self.handle=None
    def close(self):
        if self.handle:self.handle.close()
        return [dict(path=p.name,tokens=p.stat().st_size//2,sha256=sha(p)) for p in self.files]
def main():
    directory=ROOT/'shards';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise RuntimeError('Frozen real corpus exists; no overwrite')
    tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer.json'));assert tokenizer.get_vocab_size()==32768
    # ~210M intended train tokens; category underfill is reported rather than repeated.
    quotas={'code':132_000_000,'technical':36_000_000,'general':30_000_000,'context':12_000_000}
    accepted={s:[] for s in ('train','val')};counts=collections.Counter();seen=set();normalized=set();groups=collections.defaultdict(set);reject=collections.Counter()
    for split in ('train','val'):
        for doc in documents(split):
            if doc['source_content_sha256'] in seen or doc['normalized_sha256'] in normalized:raise RuntimeError('Duplicate admitted reservoir content')
            seen.add(doc['source_content_sha256']);normalized.add(doc['normalized_sha256'])
            if groups[doc['split_group']] and split not in groups[doc['split_group']]:raise RuntimeError('Repository/domain split leakage')
            groups[doc['split_group']].add(split)
            key=f'{split}/{doc["kind"]}';limit=quotas[doc['kind']]*(1 if split=='train' else .04)
            if counts[key]>=limit:reject['category_quota']+=1;continue
            # Filter before encoding to bound fitting and avoid accidental language domination.
            if doc['kind']=='code' and doc['language']=='Python' and counts[f'{split}/code_language/Python']>=quotas['code']*.20*(1 if split=='train' else .04):reject['python_cap']+=1;continue
            ids=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(doc['text']).ids+[tokenizer.token_to_id('<eos>'),tokenizer.token_to_id('<doc>')]
            counts[key]+=len(ids);counts[f'{split}/language/{doc["language"]}']+=len(ids);counts[f'{split}/license/{doc["license"]}']+=len(ids)
            if doc['kind']=='code':counts[f'{split}/code_language/{doc["language"]}']+=len(ids)
            meta={k:v for k,v in doc.items() if k!='text'};meta['length']=len(ids)
            accepted[split].append((meta,ids))
    train=sum(counts[f'train/{k}'] for k in quotas)
    if train<150_000_000:raise RuntimeError(f'Insufficient unique train tokens: {train}; acquire more verified source, do not repeat')
    physical={};doc_counts={}
    for split,rows in accepted.items():
        random.Random(42).shuffle(rows);writer=Writer(directory,split)
        with (directory/f'{split}_documents.jsonl').open('w',encoding='utf-8') as f:
            for doc,ids in rows:
                doc['start']=writer.total;writer.add(ids);f.write(json.dumps(doc,ensure_ascii=False)+'\n')
        physical[split]=writer.close();doc_counts[split]=len(rows)
    # No row bodies in the manifest; raw/doc/index/shards are referenced large payloads.
    sources=json.loads((RESULTS/'source_plan.json').read_text())
    sources['files']=[entry for entry in sources['files'] if entry.get('verified')]
    sources['admitted_download_bytes']=sum(entry['size_bytes'] for entry in sources['files'])
    sources['scope']='These verified downloaded files only; unused candidates remain in the separate source_plan.json inventory.'
    manifest=dict(tokens={s:sum(x['tokens'] for x in physical[s]) for s in physical},physical_shards=physical,
                  shard_sha256={s:'physical_shards authoritative' for s in physical},tokenizer_sha256=sha(ROOT/'tokenizer.json'),
                  sources=sources,token_statistics=dict(counts),documents=doc_counts,
                  preprocessing={'min_bytes':128,'max_bytes':200000,'max_line_chars':3000,'dedup':'global exact source-content and CRLF-normalized/trailing-newline hash; original body retained',
                                 'split_seed':42,'eval_fraction':.03,'code_split_group':'casefolded repository','web_split_group':'hostname','shuffle_seed':42,'train_token_quotas':quotas,
                                 'python_code_cap':.20,'reservoir_state_sha256':sha(RESULTS/'ingestion_state.json'),'packing':'BOS/body/EOS/doc; whole documents deterministically shuffled before contiguous packing; causal attention can cross doc boundaries within same split'},
                  leakage={'duplicate_hashes':0,'cross_split_groups':0,'groups':len(groups)},synthetic_only=False,
                  training_order='Existing PackedStream seed42 permutation of context1024 fixed offsets; >=150M train predictions, 100M consumes less than one epoch.',
                  source_hashes={p.as_posix():sha(p) for p in (Path('tools/research_corpus.py'),Path('tools/research_tokenizer.py'),Path('tools/research_pack.py'))})
    body=json.dumps(manifest,indent=2);(directory/'manifest.json').write_text(body);(RESULTS/'corpus_manifest.json').write_text(body)
    statistics=dict(tokens=manifest['tokens'],documents=doc_counts,token_statistics=dict(counts),ingestion=json.loads((RESULTS/'ingestion_state.json').read_text()),
                    selection_rejections=dict(reject),footprint_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),leakage=manifest['leakage'])
    (RESULTS/'corpus_statistics.json').write_text(json.dumps(statistics,indent=2));print(json.dumps({'tokens':manifest['tokens'],'documents':doc_counts,'footprint_bytes':statistics['footprint_bytes']},indent=2))
if __name__=='__main__':main()
