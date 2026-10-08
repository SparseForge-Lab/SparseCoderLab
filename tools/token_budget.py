"""Disk-indexed, hash-seeded proportional token-budget selection."""
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from collections import Counter
from pathlib import Path


def selected_documents(cache_path, tokenizer, max_tokens, seed, report):
    if max_tokens < 1: raise ValueError('Token budget must be positive')
    with tempfile.TemporaryDirectory(prefix='token-budget-', dir=Path(cache_path).parent) as tmp:
        with closing(sqlite3.connect(str(Path(tmp)/'index.sqlite'))) as db:
            db.execute('CREATE TABLE docs (id INTEGER PRIMARY KEY, offset INTEGER, bytes INTEGER, tokens INTEGER, stratum TEXT, rank TEXT, selected INTEGER DEFAULT 0)')
            totals=Counter(); count=0
            with Path(cache_path).open('rb') as handle:
                while True:
                    offset=handle.tell(); line=handle.readline()
                    if not line:break
                    doc=json.loads(line)
                    length=len(tokenizer.encode(doc['text']).ids)+3
                    stratum=json.dumps([doc['split'],doc['kind'],doc['language'],doc.get('repository_family',doc.get('repository',doc['source']))],ensure_ascii=False)
                    rank=hashlib.sha256(f'{seed}\0{doc["sha256"]}\0{stratum}'.encode()).hexdigest()
                    db.execute('INSERT INTO docs VALUES (?,?,?,?,?,?,0)',(count,offset,len(line),length,stratum,rank))
                    totals[stratum]+=length; count+=1
            total=sum(totals.values()); used=Counter()
            if total<=max_tokens:
                db.execute('UPDATE docs SET selected=1'); used.update(totals)
            else:
                quotas={s:max_tokens*t//total for s,t in totals.items()}
                db.execute('CREATE INDEX ranks ON docs(stratum,rank)')
                for stratum in sorted(totals):
                    for index,length in db.execute('SELECT id,tokens FROM docs WHERE stratum=? ORDER BY rank,id',(stratum,)):
                        if used[stratum]+length<=quotas[stratum]:
                            db.execute('UPDATE docs SET selected=1 WHERE id=?',(index,)); used[stratum]+=length
                remaining=max_tokens-sum(used.values())
                for index,length,stratum in db.execute('SELECT id,tokens,stratum FROM docs WHERE selected=0 ORDER BY rank,id'):
                    if length<=remaining:
                        db.execute('UPDATE docs SET selected=1 WHERE id=?',(index,)); used[stratum]+=length; remaining-=length
            report.update(policy='proportional_token_strata_hash_rank_v1' if total>max_tokens else 'all_fit_original_order',
                          seed=seed,available_tokens=total,selected_tokens=sum(used.values()),
                          available_tokens_by_stratum=dict(totals),selected_tokens_by_stratum=dict(used),
                          excluded_documents=count-db.execute('SELECT count(*) FROM docs WHERE selected=1').fetchone()[0])
            with Path(cache_path).open('rb') as handle:
                for offset,size in db.execute('SELECT offset,bytes FROM docs WHERE selected=1 ORDER BY id'):
                    handle.seek(offset); yield json.loads(handle.read(size))
