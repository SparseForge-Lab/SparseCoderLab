import json
from pathlib import Path
import pytest
import torch
from src.context.archive import Archive
from src.context.tasks import make_task, retention_fixture
from src.config import load_config
from src.moe.trace import write_trace, read_trace
from src.moe.atlas import expert_atlas, pack_experts
from src.runtime.cache import IOConfig, simulate
from src.runtime.routeahead import RouteAhead, prediction_metrics
from src.runtime.agent import RepoSandbox, TaskSpec, ToolCall
from tools.project_large_model import project

def test_compaction_pointer_exact_retrieval(tmp_path):
    archive = Archive(tmp_path); pointer = archive.put(123, 'x = 17\n')
    assert pointer == '<mem:000123>' and archive.get(pointer) == 'x = 17\n'
    with pytest.raises(ValueError): archive.get('../../secret')
    with pytest.raises(ValueError): archive.put(123, 'different')
    cfg = load_config('configs/test.yaml'); task = make_task(4, cfg, archive)
    lossy = retention_fixture(task, 4, 3, archive, False); recovered = retention_fixture(task, 4, 3, archive, True)
    assert lossy[-1]['exact_field_retention'] < 1 and all(r['exact_field_retention'] == 1 for r in recovered)

def test_trace_atlas_serialization(tmp_path):
    records = [{'token': i, 'layers': {'2': [i % 2], '5': [1]}, 'language': 'Python'} for i in range(8)]
    path = tmp_path / 'trace.jsonl'; assert write_trace(path, records) == 8
    assert list(read_trace(path)) == records
    atlas = expert_atlas(records, 12); mapping = pack_experts(atlas, 2)
    assert set(mapping) == {24, 25, 61} and len(set(mapping.values())) == 2

def test_cache_hand_calculated():
    records = [{'token': i, 'layers': {'0': [e]}} for i, e in enumerate([0,1,0,2,0])]
    io = IOConfig(100, 200, 300, 1000, .01, 10000, .001)
    result = simulate(records, {0:0,1:1,2:2}, 3, io)
    assert result['vram_hit_rate'] == pytest.approx(2/5)
    assert result['bytes_loaded_per_token'] == 60 and result['page_reads_per_token'] == pytest.approx(.6)
    assert result['cache_churn'] == 1
    assert result['predicted_io_stall_s_per_token'] == pytest.approx(3*(.1+.01+.01+.001)/5)

def test_cache_ram_and_overfetch():
    records = [{'token': i, 'layers': {'0': [e]}} for i, e in enumerate([0,1,0])]
    io = IOConfig(100, 100, 200, 1000, 0, 10000, 0)
    r = simulate(records, {0:0,1:1}, 2, io)
    assert r['bytes_loaded_per_token'] == pytest.approx(200/3) and r['ram_hit_rate_on_vram_miss'] == pytest.approx(1/3)
    r2 = simulate(records, {0:0,1:0}, 2, io)
    assert r2['vram_hit_rate'] == 0 # 200-byte page cannot fit in 100-byte VRAM

def test_prefetch_traffic_and_usefulness():
    records=[{'token':i,'layers':{'0':[e]}} for i,e in enumerate([0,1,0])]
    io=IOConfig(100,100,200,1000,0,10000,0,.05);mapping={0:0,1:1,2:2}
    baseline=simulate(records,mapping,3,io)
    useful=simulate(records,mapping,3,io,predictions={0:[1],1:[0]})
    assert useful['correct_prefetch_bytes']==200 and useful['wasted_prefetch_bytes']==0
    assert useful['predicted_io_stall_s_per_token']<baseline['predicted_io_stall_s_per_token']
    wasted=simulate(records,mapping,3,io,predictions={0:[2]})
    assert wasted['wasted_prefetch_bytes']==100 and wasted['bytes_loaded_per_token']>baseline['bytes_loaded_per_token']

def test_routeahead_shapes_and_metrics():
    cfg = {'hidden': 8,'horizons':3,'history':4}; predictor = RouteAhead(16,4,2,cfg)
    out = predictor(torch.randn(5,16), torch.randn(5,16), torch.zeros(5,4,2,4)); assert out.shape == (5,3,2,4)
    predictor.loss(out, torch.zeros_like(out)).backward(); assert predictor.network[0].weight.grad.abs().sum() > 0
    metrics = prediction_metrics([[0,1],[2]], [[1],[3]], {0:0,1:0,2:1,3:1})
    assert metrics['expert_recall'] == .5 and metrics['precision'] == pytest.approx(1/3) and metrics['page_recall'] == 1

def test_projection_independent_arithmetic():
    c = {'d_model':32,'layers':3,'capacity_layers':1,'experts':4,'expert_ffn':16,'top_k':1,'resident_ffn':48,'q_heads':2,'kv_heads':1,
         'head_dim':16,'vocab_size':512,'ngram_params':1000,'precision':'int4','page_experts':2,'gpu_gib':12,'ram_gib':48,
         'activation_reserve_gib':2,'ssd_mib_s':400,'cache_hit_rate':.75}
    result = project(c); expected_expert = 3*32*16
    assert result['routed_expert_params'] == 4*expected_expert
    assert result['expert_page_bytes'] == expected_expert and result['cached_bytes_per_token'] == expected_expert*.25
    assert result['weight_bytes']['bf16'] == 4 * result['weight_bytes']['int4']

def test_agent_path_and_loop(tmp_path):
    cfg = load_config('configs/test.yaml')['agent']; source = tmp_path/'source'; source.mkdir(); (source/'file.py').write_text('x=1\n')
    task = TaskSpec('safe', 'edit', ['file.py'], ['unused']); sandbox = RepoSandbox(source, task, cfg)
    with pytest.raises(ValueError): sandbox.call(ToolCall('inspect', {'file':'../../outside'}))
    sandbox.call(ToolCall('inspect', {'file':'file.py'})); sandbox.call(ToolCall('inspect', {'file':'file.py'}))
    assert sandbox.events[-1]['loop_signal']
    with pytest.raises(ValueError): sandbox.call(ToolCall('shell', {'command':['powershell','-Command','anything']}))

def test_agent_non_python_progress_hash(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'config.json').write_text('{"timeout":1}')
    sandbox=RepoSandbox(source,TaskSpec('state','fix',['config.json'],['unused']),load_config('configs/test.yaml')['agent'])
    before=sandbox.state_hash();sandbox.call(ToolCall('edit',{'file':'config.json','text':'{"timeout":2}'}))
    assert sandbox.state_hash()!=before

def test_strata_parser_preserves_alignment(tmp_path):
    import struct
    from src.runtime.strata_adapter import parse_binary_trace, token_records
    path = tmp_path/'native.bin'; path.write_bytes(b''.join(struct.pack('<iiif', layer,1,2,.5) for layer in [0,1,0,1]))
    dispatches = list(parse_binary_trace(path,2,4)); rows = token_records(dispatches,2)
    assert len(rows) == 2 and rows[0]['layers'] == {'0':[2],'1':[2]}
    with pytest.raises(ValueError): token_records([dispatches[0],dispatches[0]],2)
    path.write_bytes(b'bad')
    with pytest.raises(ValueError): list(parse_binary_trace(path,2,4))

def test_scaled_trace_is_explicit_synthetic():
    from src.runtime.scaling import scale_trace
    records=[{'token':i,'layers':{'2':[0,1]}} for i in range(3)]
    result=scale_trace(records,2,4,8,1)
    assert result[2]['layers']['3']==[4,5] and 'Synthetic' in result[0]['source']
