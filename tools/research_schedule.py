"""Sequential Prompt-2.5 phases with explicit matched-review barriers."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

R=Path('results/research_v1')
def run(module,*args,log=None):
    command=[sys.executable,'-u','-m',module,*args]
    if log:
        with log.open('a',encoding='utf-8') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(command,check=True)
def cpu_reports():
    for module in ('tools.research_collect','analysis.research_v1.diagnostics','analysis.research_v1.review_metrics','tools.research_publish_evidence','tools.generate_research_figures'):run(module)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=('catchup','70m','100m'));args=parser.parse_args()
    # Windows wrappers and their actual Python children both appear. Never launch
    # this orchestration alongside an existing primary training process.
    if sys.platform=='win32':
        query="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '-m tools\\.research_train( |$)' } | Select-Object -ExpandProperty ProcessId"
        active=subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True).strip()
        if active:raise RuntimeError('A primary training process is already running: '+active.replace('\n',', '))
    if args.phase=='catchup':stages=[('sparse',20004864),('sparse',50003968),('memory',20004864),('memory',50003968)]
    else:
        required=50003968 if args.phase=='70m' else 70000640
        for tag in ('dense','sparse','memory'):
            assert Path(f'experiments/research_v1/{tag}/summary_{required}.json').exists(),f'{tag} has not reached the prior matched checkpoint'
        gate=json.loads((R/'matched_50m_review.json').read_text(encoding='utf-8'))
        assert gate['status']=='complete','Required matched50M review is missing'
        stages=[(tag,70000640 if args.phase=='70m' else 100007936) for tag in ('dense','sparse','memory')]
    state={'phase':args.phase,'status':'running','completed':[]}
    def save():
        p=R/'schedule_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2),encoding='utf-8');t.replace(p)
    save()
    try:
        for tag,tokens in stages:
            state['current']={'model':tag,'endpoint':tokens};save()
            print(json.dumps({'stage':state['current']}),flush=True)
            run('tools.research_train','--model',tag,'--endpoint',str(tokens),log=R/f'{tag}_{tokens}_schedule_output.txt')
            cpu_reports();state['completed'].append(state['current']);save()
        state['status']='phase_complete';state['current']=None;save()
        print(json.dumps(state),flush=True)
    except Exception as error:
        state['status']='failed';state['error']=str(error);save();raise
if __name__=='__main__':main()
