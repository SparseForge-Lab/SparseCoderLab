import copy,json
from pathlib import Path
import pytest
from tools.evaluate_repository_gate import validate_generation,validate_evaluation

def fixture_rows():
    benchmark=json.loads(Path('configs/eval/simple_python_v1.json').read_text())
    baseline=json.loads(Path('results/research_v2_real/functional_canonical_gpu_r1.json').read_text())
    entries={tag:dict(total_tokens=250003456,model_weights=dict(sha256='a'*64)) for tag in ('dense75_ref','sparse75','sparse75_ngram10m','sparse75_ngram25m')}
    from src.utils.hashing import sha256_file
    rows=[]
    for tag in entries:
        for temp,policy in baseline['decoding_policy'].items():
            for task in benchmark['tasks']:
                rows.append(dict(policy,variant=tag,temperature=float(temp),prompt_number=task['prompt_number'],input=task['prompt'],
                    checkpoint=f'experiments/prompt5_v1/{tag}/checkpoints/model_250M.pt',checkpoint_sha256='a'*64,
                    training_tokens=250003456,sampler_sha256=sha256_file(Path('tools/generate_canonical_python.py')),
                    bf16=True,tf32=True,generation_batch_size=4))
    return rows,entries,baseline

def test_same_frozen_decoding_identity_accepts_new_bound_gate():
    rows,entries,baseline=fixture_rows()
    assert validate_generation(rows,entries,baseline)==baseline['decoding_policy']

@pytest.mark.parametrize('field,value',[
    ('training_tokens',100007936),('checkpoint_sha256','b'*64),('sampler_sha256','b'*64),
    ('generation_batch_size',25),('bf16',False),('generation_seed',43),('input','different task')])
def test_unmatched_stage_sampler_precision_or_task_is_rejected(field,value):
    rows,entries,baseline=fixture_rows();rows[0][field]=value
    with pytest.raises(ValueError):validate_generation(rows,entries,baseline)

def test_missing_task_slot_is_rejected():
    rows,entries,baseline=fixture_rows()
    with pytest.raises(ValueError):validate_generation(rows[:-1],entries,baseline)

def evaluation_fixture():
    baseline=json.loads(Path('results/research_v2_real/transition_cuda_preflight_r2.json').read_text())
    return baseline['variants']['sparse75_ngram10m']['baseline_evaluation']

def test_frozen_baseline_including_ablation_passes_finite_measurement_gate():
    validate_evaluation(evaluation_fixture())

@pytest.mark.parametrize('location',['category','document','ablation','router'])
def test_nonfinite_nested_measurement_cannot_be_published(location):
    result=evaluation_fixture()
    if location=='category':result['technical_val_loss']=float('inf')
    elif location=='document':result['documents'][0]['nll']=float('nan')
    elif location=='ablation':result['residual_off']['val_loss']=float('-inf')
    else:next(iter(result['router'].values()))['soft_entropy']=float('nan')
    with pytest.raises(FloatingPointError):validate_evaluation(result)
