from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
from src.runtime.strata_adapter import parse_binary_trace, token_records
from src.moe.trace import write_trace

LAUNCHER = Path(os.environ.get('SPARSECODERLAB_STRATA_LAUNCHER', str(Path.home() / 'Desktop' / 'Alles-Start.bat')))
CONFIG = Path('D:/Strata/strata-coder-iq1_m.json')

def command_from_launcher(output: Path, tokens: str, max_new: int) -> tuple[list[str], dict]:
    launcher = LAUNCHER.read_text(encoding='utf-8')
    if '8095' not in launcher or 'Strata Coder starten.bat' not in launcher: raise ValueError('Launcher configuration changed; review it')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8')); args = list(cfg['args'])
    # One-shot trace: no speculative/batched order, bounded context/cache; original config immutable.
    paired = {'--spec','--mtp','--spec-min-p','--max-context','--kv-resident','--prefill','--expert-cache','--vram-reserve-mib'}
    flags = {'--vision'}; cleaned = []; i = 0
    while i < len(args):
        if args[i] in paired: i += 2
        elif args[i] in flags: i += 1
        else: cleaned.append(args[i]); i += 1
    # pack/native/model references stay read-only; no shared arena path or profile writes.
    command = [cfg['exe'], *cleaned, '--max-context','256','--expert-cache','0','--mmap-experts','--tokens',tokens,
               '--max-new',str(max_new),'--dump-routing',str(output.resolve())]
    return command, cfg

def capture(output: Path, timeout: float, tokens: str, max_new: int) -> dict:
    if not output.resolve().is_relative_to(Path.cwd().resolve()): raise ValueError('Trace output must stay inside project')
    command, cfg = command_from_launcher(output, tokens, max_new); env = os.environ.copy()
    env['PATH'] = os.pathsep.join(cfg.get('lib_dirs', [])) + os.pathsep + env.get('PATH','')
    output.parent.mkdir(parents=True, exist_ok=True); log = Path('results/strata_capture.log'); start = time.perf_counter()
    with log.open('w', encoding='utf-8') as f:
        process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=f, stderr=subprocess.STDOUT)
        try: returncode = process.wait(timeout=timeout); timed_out = False
        except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=10); returncode = process.returncode; timed_out = True
    status = {'launcher': str(LAUNCHER), 'config_read_only': str(CONFIG), 'command': command, 'returncode': returncode,
              'timed_out': timed_out, 'seconds': time.perf_counter()-start, 'raw_trace': str(output),
              'original_launcher_executed': False, 'reason': 'Only native trace engine used; launcher would also start an interactive Claude CLI.',
              'configuration_changes_in_memory': 'No cache/spec/MTP/prefill, context256, mmap experts, short numeric token prompt. Trace topology retained; not production launcher performance.'}
    Path('results/strata_status.json').write_text(json.dumps(status, indent=2)); return status

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--capture', action='store_true'); p.add_argument('--input', type=Path, default=Path('results/strata_routing.bin'))
    p.add_argument('--output', type=Path, default=Path('results/strata_routes.jsonl')); p.add_argument('--layers', type=int, default=48); p.add_argument('--experts', type=int, default=256)
    p.add_argument('--timeout-seconds', type=float, default=120); p.add_argument('--tokens', default='1,2,3'); p.add_argument('--max-new', type=int, default=8)
    a = p.parse_args()
    if a.capture:
        if a.timeout_seconds > 300 or a.max_new > 32: raise ValueError('Optional initial capture is capped at 5 minutes and 32 tokens')
        status = capture(a.input, a.timeout_seconds, a.tokens, a.max_new); print(json.dumps(status, indent=2))
        if status['returncode'] != 0: raise SystemExit('Strata capture failed; see results/strata_capture.log. No external files changed.')
    records = list(parse_binary_trace(a.input, a.layers, a.experts))
    dispatch_path = Path('results/strata_dispatches.jsonl'); dispatch_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    rows = token_records(records, a.layers); print(f'Converted {write_trace(a.output, rows)} strict token cycles.')
