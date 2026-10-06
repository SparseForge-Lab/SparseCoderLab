"""Bounded GPU thermal/power samples during the remaining primary run."""
from __future__ import annotations
import datetime,json,os,subprocess,time
from pathlib import Path

R=Path('results/research_v1');STOP=R/'hardware_monitor.stop'
def main():
    if STOP.exists():raise RuntimeError('Monitor stop marker exists; do not restart silently')
    output=R/'hardware_samples.jsonl'
    print(json.dumps({'monitor_pid':os.getpid(),'interval_seconds':30,'stop_marker':str(STOP.resolve())}),flush=True)
    with output.open('a',encoding='utf-8') as stream:
        while not STOP.exists():
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Single GPU samples every30s beginning mid-Sparse20M→50M. Not a controlled benchmark, lifetime maximum, or exact energy measurement.'}
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10,check=True)
                values=result.stdout.strip().splitlines()[0].split(',')
                for key,value in zip(('temperature_c','power_w','gpu_utilization_percent','gpu_memory_mib','sm_clock_mhz'),values):
                    try:row[key]=float(value.strip())
                    except ValueError:row[key]=None
            except (subprocess.SubprocessError,OSError) as error:row['sampling_error']=str(error)
            progress={}
            for tag in ('dense','sparse','memory'):
                path=Path('experiments/research_v1')/tag/'metrics.jsonl'
                if not path.exists():continue
                with path.open('rb') as f:
                    f.seek(max(0,path.stat().st_size-65536));lines=f.read().splitlines()
                for line in reversed(lines):
                    try:last=json.loads(line)
                    except (json.JSONDecodeError,UnicodeDecodeError):continue
                    if 'tokens_seen' in last:
                        progress[tag]={'tokens':last['tokens_seen'],'step':last['step']};break
            row['last_logged_progress']=progress;stream.write(json.dumps(row)+'\n');stream.flush()
            for _ in range(30):
                if STOP.exists():break
                time.sleep(1)
    print(json.dumps({'monitor_stopped':True,'samples':str(output.resolve())}),flush=True)
if __name__=='__main__':main()
