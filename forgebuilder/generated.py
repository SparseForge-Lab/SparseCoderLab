"""Ingest generated qualification code without confusing tests with acceptance."""
from dataclasses import asdict
import json
from pathlib import Path

from .foundation import evaluation_registry
from .pipeline import VerificationResult
from .schema import canonical_json, sha256_bytes, validate_record
from .strata_runtime import digest


def verify_generated(record):
    from src.eval.wasi_python import WasiPython
    task=record['task']
    config=Path(task['registry_path'])
    if digest(config)!=task['registry_sha256']:
        raise ValueError('generated task registry changed')
    spec=next(t for t in json.loads(config.read_text(encoding='utf8'))['tasks'] if t['id']==task['task_id'])
    runtime=WasiPython()
    execution=runtime.execute(record['candidate'],spec['function'],[c['args'] for c in spec['cases']])
    observations=execution.get('results',[])
    complete=(execution.get('kind')=='completed' and len(observations)==len(spec['cases'])
              and not execution.get('module_stdout') and not execution.get('module_stderr'))
    passed=sum(canonical_json(o.get('value'))==canonical_json(c['expected'])
               and not o.get('stdout') and not o.get('stderr')
               for o,c in zip(observations,spec['cases'])) if complete else 0
    return VerificationResult(status='PASS' if complete and passed==len(spec['cases']) else 'FAIL',
        verifier_id='pinned_cpython_wasi',verifier_version='project_generated_function_v1',
        test_hash=sha256_bytes(canonical_json(spec)),
        environment_hash=sha256_bytes(canonical_json(runtime.runtime_identity)),
        passed=passed,failed=len(spec['cases'])-passed,
        timed_out=execution.get('kind')=='timeout',
        evidence={'execution':execution,'scope':'Finite new project-specific cases; no universal functional correctness claim.'})


def prepare_generated_pilot(output, qualifications, foundation):
    from tokenizers import Tokenizer
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    registry=Path('configs/dataset_v2/generator_qualification.json')
    specs={t['id']:t for t in json.loads(registry.read_text(encoding='utf8'))['tasks']}
    foundation_receipt=json.loads((Path(foundation)/'build_receipt.json').read_text(encoding='utf8'))
    negatives,negative_receipts=evaluation_registry([r['path'] for r in foundation_receipt['evaluation_negative_registry']])
    tokenizer=Tokenizer.from_file(foundation_receipt['build_config']['tokenizer'])
    rows=[]
    for directory in map(Path,qualifications):
        receipt=json.loads((directory/'receipt.json').read_text(encoding='utf8'))
        model=receipt['model']
        for proof in receipt['results']:
            spec=specs[proof['task_id']]
            path=directory/(proof['task_id']+'.candidate.py')
            code=path.read_text(encoding='utf8')
            if sha256_bytes(code.encode())!=proof['candidate_sha256']:
                raise ValueError('generated candidate differs from its saved qualification')
            screen=negatives.inspect(spec['instruction']+'\n'+code)
            if screen['status']=='flagged':
                continue
            row=validate_record(dict(source_id=model['repository']+':'+proof['task_id'],
                source_type='local_teacher_generation',license=None,
                source_provenance={'task_registry_sha256':digest(registry),'task_id':proof['task_id'],
                    'model_repository':model['repository'],'model_revision':model['revision'],
                    'model_weights_sha256':model['sha256'],'model_license':model['license'],
                    'output_license':'unknown; model license is not asserted as an output license',
                    'runtime_executable_sha256':receipt['runtime_executable_sha256'],
                    'raw_response_sha256':digest(directory/(proof['task_id']+'.response.json'))},
                content_sha256=sha256_bytes(code.encode()),task_family='code_from_scratch',
                real_or_generated='generated',data_stage='posttraining',split='train',
                language='Python',natural_language=['en'],candidate=code,
                generator_model=model['repository'],generator_version=model['revision'],
                transformation_recipe=proof['transformation_recipe'],token_count=len(tokenizer.encode(code).ids),
                task={'kind':'new_project_function','task_id':spec['id'],'instruction':spec['instruction'],
                      'registry_path':registry.as_posix(),'registry_sha256':digest(registry),
                      'evidence':{'specification':spec['instruction'],'cases':spec['cases'],
                                  'scope':'New project-specific function contract, not a frozen evaluation task.'}},
                execution_verified=bool(proof['objective_pass']),test_verified=bool(proof['objective_pass']),
                accepted=None,acceptance_status='pending_independent_critic',critic_verdict='NOT_RUN',
                eval_quarantine_status=screen['status'],eval_quarantine_evidence=screen,
                finish_reason=proof['finish_reason'],cap_reached=proof['cap_reached']))
            verification=verify_generated(row)
            row['objective_verification']=asdict(verification)
            row['qualification_objective_pass']=bool(proof['objective_pass'])
            row['execution_verified']=verification.status=='PASS'
            row['test_verified']=verification.status=='PASS'
            rows.append(row)
    rows.sort(key=lambda row:row['sample_id'])
    if len({row['sample_id'] for row in rows})!=len(rows):
        raise ValueError('duplicate generated candidate IDs')
    payload=b''.join(canonical_json(row)+b'\n' for row in rows)
    output.mkdir(parents=True)
    (output/'candidates.jsonl').write_bytes(payload)
    preparation={'status':'generated_candidates_pending_critic','candidate_count':len(rows),
        'candidate_sha256':sha256_bytes(payload),'evaluation_negative_registry':negative_receipts,
        'student_training_authorized':False,'coverage_limit':'Small function qualification pool; not the full 5K pilot or real SWE patch pool.'}
    (output/'preparation.json').write_bytes(canonical_json(preparation)+b'\n')
    return preparation
