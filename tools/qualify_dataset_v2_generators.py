"""Qualify downloaded generators sequentially, preserving and resuming critic work."""
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from forgebuilder.foundation import evaluation_registry
from forgebuilder.schema import canonical_json, sha256_bytes
from forgebuilder.strata_runtime import digest
from src.eval.wasi_python import WasiPython

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.local/dataset_v2'
PILOT=ROOT/'data/dataset_v2/source_pilot_v1'


def request_json(url,payload=None,timeout=10):
    req=Request(url,data=canonical_json(payload) if payload is not None else None,
                headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as response:
        return json.load(response)


def stopped():
    return (STATE/'STOP').exists()


def review_workers_running():
    command=("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
             "-and $_.CommandLine -match '-m tools\\.dataset_v2_pilot review' } "
             "| Select-Object -ExpandProperty ProcessId")
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,
        text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return bool(result.stdout.strip())


def qualify(model):
    destination=ROOT/'data/dataset_v2/generator_qualification'/model['role']
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'receipt.json').exists():
        return
    path=Path(model['directory'])/model['filename']
    while not path.exists():
        if stopped():
            return
        time.sleep(10)
    if path.stat().st_size!=model['size_bytes'] or digest(path)!=model['sha256']:
        raise ValueError('generator does not match pinned publisher identity')
    task_path=ROOT/'configs/dataset_v2/generator_qualification.json'
    tasks=json.loads(task_path.read_text(encoding='utf8'))['tasks']
    foundation=json.loads((ROOT/'data/dataset_v2/foundation_v1/build_receipt.json').read_text(encoding='utf8'))
    screen,_=evaluation_registry([r['path'] for r in foundation['evaluation_negative_registry']])
    for task in tasks:
        if screen.inspect(task['instruction'])['status']=='flagged':
            raise ValueError('qualification task overlaps registered evaluation negatives')
    if stopped():
        return
    # This marker belongs to this worker; never remove a user-created STOP.
    marker='generator_qualification:'+model['role']
    stop_path=PILOT/'STOP'
    if stop_path.exists():
        raise RuntimeError('pilot already stopped; preserve existing stop instruction')
    stop_path.write_text(marker,encoding='utf8')
    server=None
    try:
        deadline=time.time()+1800
        while review_workers_running() or request_json('http://127.0.0.1:8096/status').get('busy'):
            if stopped() or time.time()>deadline:
                return
            time.sleep(1)
        request_json('http://127.0.0.1:8096/v1/unload',{},timeout=120)
        executable=Path('D:/llama.cpp/build/bin/llama-server.exe')
        command=[str(executable),'-m',str(path),'-c','32768','-np','1','-ngl','99',
            '-fa','on','-ctk','q8_0','-ctv','q8_0','--host','127.0.0.1','--port','8097',
            '--no-webui','--reasoning-effort','medium','--reasoning-format','auto','-n','16384']
        if model['role']=='repository_generator':
            command[command.index('-ngl')+1]='auto'
            command += ['--fit','on','--fit-target','1024']
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',8097))
        with (destination/'server.log').open('ab') as log:
            server=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        (destination/'process.json').write_bytes(canonical_json({'pid':server.pid,'command':command})+b'\n')
        deadline=time.time()+600
        while True:
            if server.poll() is not None:
                raise RuntimeError('generator server exited; inspect its saved log')
            if stopped() or time.time()>deadline:
                raise RuntimeError('generator startup stopped or timed out')
            try:
                if request_json('http://127.0.0.1:8097/health').get('status')=='ok':
                    break
            except OSError:
                pass
            time.sleep(1)
        runtime=WasiPython()
        results=[]
        for task in tasks:
            if stopped():
                break
            payload={'messages':[{'role':'user','content':task['instruction']}],
                'max_tokens':16384,'temperature':.6,'top_p':.95,'top_k':20,'stream':False,
                'reasoning_effort':'medium'}
            started=time.time()
            raw=request_json('http://127.0.0.1:8097/v1/chat/completions',payload,timeout=1800)
            (destination/(task['id']+'.response.json')).write_bytes(canonical_json(raw)+b'\n')
            choice=raw['choices'][0]
            code=choice['message'].get('content') or ''
            recipe='identity'
            if code.strip().startswith('```python') and code.strip().endswith('```'):
                code=code.strip()[9:-3].strip()+'\n'
                recipe='strip_single_python_fence_v1'
            (destination/(task['id']+'.candidate.py')).write_text(code,encoding='utf8')
            try:
                ast.parse(code)
                execution=runtime.execute(code,task['function'],[c['args'] for c in task['cases']])
                observed=execution.get('results',[])
                passed=(execution.get('kind')=='completed' and len(observed)==len(task['cases'])
                    and not execution.get('module_stdout') and not execution.get('module_stderr')
                    and all(canonical_json(o.get('value'))==canonical_json(c['expected']) and not o.get('stdout') and not o.get('stderr')
                            for o,c in zip(observed,task['cases'])))
            except (SyntaxError,ValueError) as exc:
                execution={'kind':'syntax_error','error':str(exc)}
                passed=False
            result={'task_id':task['id'],'objective_pass':passed,'execution':execution,
                'candidate_sha256':sha256_bytes(code.encode()),'transformation_recipe':recipe,
                'finish_reason':choice.get('finish_reason'),'cap_reached':choice.get('finish_reason')=='length',
                'cap_policy':'Run available code regardless of cap; cap alone does not mean functional failure.',
                'usage':raw.get('usage'),'timings':raw.get('timings'),'wall_seconds':time.time()-started,
                'accepted_for_training':False,'critic_status':'NOT_RUN'}
            (destination/(task['id']+'.verification.json')).write_bytes(canonical_json(result)+b'\n')
            results.append(result)
        receipt={'status':'qualification_completed' if len(results)==len(tasks) else 'qualification_interrupted',
            'model':model,'runtime_executable_sha256':digest(executable),'command':command,
            'tasks_sha256':digest(task_path),'wasi_runtime':runtime.runtime_identity,'results':results,
            'scope':'Three new project-specific functional qualification tasks; no general benchmark claim.',
            'student_training_authorized':False}
        (destination/'receipt.json').write_bytes(canonical_json(receipt)+b'\n')
        print(json.dumps({'model':model['repository'],'qualified_cases':sum(r['objective_pass'] for r in results),
                          'tasks':len(results)}),flush=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=30)
        own_marker=stop_path.exists() and stop_path.read_text(encoding='utf8')==marker
        if own_marker:
            stop_path.unlink()
        if own_marker and not stopped():
            request_json('http://127.0.0.1:8096/v1/load',{},timeout=300)
            with (STATE/'critic_after_qualification.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-X','utf8','-m','tools.dataset_v2_pilot','review'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)


if __name__=='__main__':
    models=json.loads((ROOT/'configs/dataset_v2/local_models.json').read_text(encoding='utf8'))['models']
    for model in models:
        if stopped():
            break
        qualify(model)
