"""Read-only terminal navigation tasks backed by actual bounded rg execution."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .pipeline import VerificationResult
from .schema import canonical_json,sha256_bytes,validate_record
from .strata_runtime import digest
from .trajectory import normalize_trajectory
from .foundation import evaluation_registry


def locate_definitions(excerpt):
    if not isinstance(excerpt,str) or len(excerpt.encode())>20000:
        raise ValueError('terminal excerpt exceeds the bounded input contract')
    nodes=[n for n in ast.parse(excerpt).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if not nodes: raise ValueError('no top-level definitions')
    executable=shutil.which('rg')
    if not executable: raise RuntimeError('trusted rg executable is unavailable')
    pattern=r'^(async )?def ('+'|'.join(re.escape(n.name) for n in nodes)+r')\s*\('
    argv=['rg','--line-number','--no-heading','--color','never','--',pattern,'source.py']
    staging=Path('.local/dataset_v2/terminal_staging').resolve()
    staging.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inspection_',dir=staging) as directory:
        folder=Path(directory).resolve()
        if folder.parent!=staging: raise ValueError('temporary terminal directory escaped staging')
        (folder/'source.py').write_bytes(excerpt.encode())
        result=subprocess.run([executable,*argv[1:]],cwd=folder,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=3,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if len(result.stdout)+len(result.stderr)>16384:
        raise ValueError('terminal output exceeds the capture contract')
    stdout=result.stdout.decode('utf8')
    observed=[]
    for line in stdout.splitlines():
        match=re.match(r'^(\d+):(?:async )?def ([^\s(]+)\s*\(',line)
        if not match: raise ValueError('unexpected terminal output format')
        observed.append({'name':match[2],'excerpt_line':int(match[1])})
    expected=[{'name':n.name,'excerpt_line':n.lineno} for n in nodes]
    passed=result.returncode==0 and not result.stderr and observed==expected
    return dict(argv=argv,stdout=stdout,stderr=result.stderr.decode('utf8'),
                exit_code=result.returncode,definitions=observed,objective_pass=passed,
                excerpt_sha256=sha256_bytes(excerpt.encode()),
                runtime={'executable_sha256':digest(Path(executable)),'timeout_seconds':3,
                         'source_byte_limit':20000,'output_byte_limit':16384,
                         'shell':False,'candidate_code_executed':False})


def verify_terminal(record):
    task=record['task']; proof=locate_definitions(task['evidence']['excerpt'])
    saved=task['evidence']['terminal_execution']
    if proof['runtime']!=saved['runtime'] or proof['excerpt_sha256']!=saved['excerpt_sha256']:
        raise ValueError('terminal runtime or source identity changed')
    if proof!=saved:
        raise ValueError('recorded terminal output differs from actual rerun')
    expected={'definitions':proof['definitions']}
    passed=proof['objective_pass'] and json.loads(record['candidate'])==expected
    return VerificationResult(status='PASS' if passed else 'FAIL',verifier_id='bounded_rg_and_ast',
        verifier_version='excerpt_definition_locations_v1',passed=int(passed),failed=int(not passed),
        test_hash=sha256_bytes(canonical_json({'excerpt_sha256':proof['excerpt_sha256'],'argv':proof['argv']})),
        environment_hash=sha256_bytes(canonical_json(proof['runtime'])),
        evidence={'terminal_execution':proof,'scope':'Actual read-only terminal search over a reviewed excerpt. Relative excerpt lines only; no code execution, repository edit or SWE patch correctness claim.'})


def prepare_terminal_pilot(source_pilot,output,tokenizer):
    source_pilot,output=Path(source_pilot),Path(output)
    if output.exists(): raise FileExistsError('retain existing terminal pools')
    payload=(source_pilot/'candidates.jsonl').read_bytes()
    preparation=json.loads((source_pilot/'preparation.json').read_bytes())
    if sha256_bytes(payload)!=preparation['candidate_sha256']:
        raise ValueError('source pilot changed')
    records=[]
    for line in payload.splitlines():
        source=json.loads(line)
        if source['task_family']!='evidence_grounded_reasoning': continue
        excerpt=source['task']['evidence']['excerpt']
        proof=locate_definitions(excerpt)
        if not proof['objective_pass']: raise ValueError('rg output disagrees with AST-derived source lines')
        candidate=canonical_json({'definitions':proof['definitions']}).decode()
        instruction='Use the recorded read-only terminal search over the supplied excerpt. Return JSON with definitions, listing each top-level function name and excerpt_line in source order. These are excerpt-relative lines, not full-file locations.'
        row=validate_record(dict(source_id=source['source_id'],source_type='verified_terminal_navigation',
            source_provenance=source['source_provenance'],license=source['license'],
            parent_sample_ids=[source['sample_id']],repository=source['repository'],
            content_sha256=sha256_bytes(candidate.encode()),real_or_generated='derived',
            data_stage='posttraining',split='train',language='Python',natural_language=['en'],
            task_family='terminal_use',candidate=candidate,token_count=len(tokenizer.encode(candidate).ids),
            generator_model='deterministic_terminal_inspector',generator_version='bounded_rg_definition_locations_v1',
            transformation_recipe={'name':'bounded_rg_definition_locations_v1'},
            task={'kind':'rg_definition_locations','instruction':instruction,
                  'evidence':dict(source['task']['evidence'],terminal_execution=proof)},
            execution_verified=False,test_verified=False,terminal_execution_verified=True,
            eval_quarantine_status=source['eval_quarantine_status'],
            acceptance_status='pending_independent_critic',accepted=None,critic_verdict='NOT_RUN',
            trajectory=normalize_trajectory([
                {'type':'TOOL_CALL','content':canonical_json({'argv':proof['argv']}).decode(),
                 'tool_name':'bounded_terminal_search','tool_args':proof['argv']},
                {'type':'TOOL_RESULT','content':proof['stdout'],'loss_mask':0},
                {'type':'FINAL','content':candidate}])))
        records.append(row)
    records.sort(key=lambda r:r['sample_id'])
    if len({r['sample_id'] for r in records})!=len(records): raise ValueError('duplicate terminal task IDs')
    result=b''.join(canonical_json(r)+b'\n' for r in records)
    output.mkdir(parents=True)
    (output/'candidates.jsonl').write_bytes(result)
    receipt=dict(schema_version=1,status='actual_terminal_candidates_pending_independent_critic',
        candidate_count=len(records),candidate_sha256=sha256_bytes(result),
        source_candidates_sha256=sha256_bytes(payload),task_families={'terminal_use':len(records)},
        coverage='Read-only source navigation through actual rg subprocesses; no generated code was executed and no real SWE patch was verified.',
        student_training_authorized=False)
    (output/'preparation.json').write_bytes(canonical_json(receipt)+b'\n')
    return receipt


def finalize_terminal_quarantine(source_pilot,output):
    """Screen new instructions/answers before any independent review starts."""
    output=Path(output)
    if (output/'reviews.jsonl').exists(): raise ValueError('cannot change a reviewed pool')
    source=json.loads((Path(source_pilot)/'preparation.json').read_bytes())
    preparation=json.loads((output/'preparation.json').read_bytes())
    if 'evaluation_negative_registry' in preparation: raise ValueError('terminal pool already screened')
    path=output/'candidates.jsonl'; raw=path.read_bytes()
    if sha256_bytes(raw)!=preparation['candidate_sha256']: raise ValueError('terminal pool changed')
    quarantine,receipts=evaluation_registry([r['path'] for r in source['evaluation_negative_registry']])
    if receipts!=source['evaluation_negative_registry']: raise ValueError('evaluation negative identity changed')
    accepted=[]; rejected=[]
    for line in raw.splitlines():
        row=json.loads(line); task=row['task']
        screen=quarantine.inspect(task['instruction']+'\n'+task['evidence']['excerpt']+'\n'+
                                  task['evidence']['terminal_execution']['stdout']+'\n'+row['candidate'])
        row['eval_quarantine_status']=screen['status']; row['eval_quarantine_evidence']=screen
        (rejected if screen['status']=='flagged' else accepted).append(row)
    (output/'candidates.pre_quarantine.jsonl').write_bytes(raw)
    screened=b''.join(canonical_json(validate_record(r))+b'\n' for r in accepted)
    temp=output/'candidates.jsonl.tmp'; temp.write_bytes(screened); temp.replace(path)
    (output/'quarantine.jsonl').write_bytes(b''.join(canonical_json(r)+b'\n' for r in rejected))
    preparation.update(candidate_count=len(accepted),candidate_sha256=sha256_bytes(screened),
        task_families={'terminal_use':len(accepted)},evaluation_negative_registry=receipts,
        quarantined_records=len(rejected),pre_quarantine_candidate_sha256=sha256_bytes(raw))
    (output/'preparation.json').write_bytes(canonical_json(preparation)+b'\n')
    return preparation
