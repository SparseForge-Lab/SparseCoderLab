import hashlib
import json
import random
from pathlib import Path
import numpy as np

from src.eval.sampling import sample_documents
from src.training.shards import TokenShardWriter
from src.training.data import ShardedTokens
from tools.token_budget import selected_documents


def test_evaluation_samples_documents_and_positions_independently_of_input_order():
    rows=[dict(sha256=str(i),start=i*2000,length=2000,kind='code',language='Python') for i in range(200)]
    expected=sample_documents(rows,16,64,42,lambda r:r['kind'])
    random.Random(7).shuffle(rows)
    assert expected==sample_documents(rows,16,64,42,lambda r:r['kind'])
    offsets=[r['evaluation_offset'] for r in expected['code']]
    assert len(set(offsets))>1 and max(offsets)>0 and all(0<=x<=1935 for x in offsets)
    assert {r['sha256'] for r in expected['code']} != {str(i) for i in range(16)}


def test_streaming_physical_shards_preserve_tokens_and_hashes(tmp_path):
    writer=TokenShardWriter(tmp_path,'train',shard_tokens=7)
    writer.add(range(3)); writer.add(range(3,18)); writer.add([])
    physical=writer.close(); assert [r['tokens'] for r in physical]==[7,7,4]
    assert writer.total==18 and not (tmp_path/'train.bin').exists()
    for row in physical:
        assert hashlib.sha256((tmp_path/row['path']).read_bytes()).hexdigest()==row['sha256']
    stream=ShardedTokens([tmp_path/r['path'] for r in physical],max_open=1)
    assert stream[:].tolist()==list(range(18)); stream.close()


def test_token_budget_sampling_reaches_later_repositories_and_is_deterministic(tmp_path):
    class Tokenizer:
        def encode(self,text):
            class Encoded: ids=list(range(len(text)))
            return Encoded()
    docs=[dict(text='abcde',sha256=str(i),split='train',kind='code',language='Python' if i<12 else 'Rust',
               repository='repo/early' if i<12 else 'repo/later',source='fixture') for i in range(24)]
    cache=tmp_path/'docs.jsonl'; cache.write_text(''.join(json.dumps(d)+'\n' for d in docs))
    report={}; chosen=list(selected_documents(cache,Tokenizer(),64,42,report))
    again={}; assert chosen==list(selected_documents(cache,Tokenizer(),64,42,again)) and report==again
    assert len(chosen)==8 and {d['repository'] for d in chosen}=={'repo/early','repo/later'}
    assert report['selected_tokens']==64 and report['excluded_documents']==16
    all_report={}; assert list(selected_documents(cache,Tokenizer(),9999,42,all_report))==docs


def test_monolith_converter_streams_and_matches_boundary_slices(tmp_path,monkeypatch):
    from tools.shard_data import shard, verify_shards
    monkeypatch.chdir(tmp_path)
    directory=tmp_path/'shards';directory.mkdir()
    manifest={'tokens':{'train':23,'val':13},'shard_sha256':{}}
    for split,length in manifest['tokens'].items():
        raw=np.arange(length,dtype=np.uint16).tobytes();(directory/f'{split}.bin').write_bytes(raw)
        manifest['shard_sha256'][split]=hashlib.sha256(raw).hexdigest()
    (directory/'manifest.json').write_text(json.dumps(manifest))
    config={'data':{'shards':str(directory),'manifest':str(tmp_path/'manifest.json'),'shard_tokens':7}}
    result=shard(config);verify_shards(config)
    tokens=ShardedTokens([directory/r['path'] for r in result['physical_shards']['train']],max_open=1)
    assert tokens[5:19].tolist()==list(range(5,19));tokens.close()
