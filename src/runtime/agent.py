from __future__ import annotations
import difflib, hashlib, json, shutil, subprocess, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TaskSpec:
    task_id: str
    goal: str
    allowed_files: list[str]
    test_command: list[str]
    license: str = 'CC0-1.0'

@dataclass
class ToolCall:
    action: str
    arguments: dict
    hypothesis: str = ''

class RepoSandbox:
    """Disposable project-local copy with bounded commands and trajectory recording.

    This is an engineering sandbox, NOT an OS security boundary. Only trusted toy
    tasks/commands are allowed. Untrusted shell requires VM/container isolation later.
    """
    def __init__(self, source: Path, task: TaskSpec, cfg: dict):
        self.task = task; self.cfg = cfg; self.root = Path(tempfile.mkdtemp(prefix='agent-', dir='work'))
        self.repo = self.root / 'repo'; shutil.copytree(source, self.repo); self.events = []; self.original = {}
        for file in task.allowed_files: self.original[file] = self.path(file).read_text(encoding='utf-8')
    def path(self, file: str) -> Path:
        path = (self.repo / file).resolve()
        if not path.is_relative_to(self.repo.resolve()): raise ValueError('Path escapes sandbox')
        return path
    def call(self, call: ToolCall) -> dict:
        if len(self.events) >= self.cfg['max_actions']: raise RuntimeError('Action cap reached')
        before = self.state_hash(); action = call.action; args = call.arguments
        if action == 'inspect': result = {'text': self.path(args['file']).read_text(encoding='utf-8')}
        elif action == 'edit':
            if args['file'] not in self.task.allowed_files: raise ValueError('Edit outside task allowlist')
            self.path(args['file']).write_text(args['text'], encoding='utf-8'); result = {'edited': args['file']}
        elif action in ('shell','test'):
            command = self.task.test_command if action == 'test' else args['command']
            if command != self.task.test_command: raise ValueError('Only reviewed task command allowed in V1')
            try:
                run = subprocess.run(command, cwd=self.repo, capture_output=True, text=True, timeout=self.cfg['shell_timeout'], shell=False)
                result = {'returncode': run.returncode, 'output': (run.stdout + run.stderr)[:self.cfg['max_output_chars']]}
            except subprocess.TimeoutExpired: result = {'returncode': -1, 'output': 'timeout'}
        elif action == 'diff':
            result = {'diff': ''.join(''.join(difflib.unified_diff(old.splitlines(True), self.path(file).read_text(encoding='utf-8').splitlines(True), fromfile=file, tofile=file)) for file, old in self.original.items())}
        elif action in ('hypothesize','replan'): result = {'hypothesis': call.hypothesis}
        else: raise ValueError('Unknown action')
        event = {'call': asdict(call), 'result': result, 'state_before': before, 'state_after': self.state_hash()}
        event['loop_signal'] = any(e['call'] == event['call'] and e['result'] == result and e['state_after'] == event['state_after'] for e in self.events)
        self.events.append(event); return result
    def state_hash(self) -> str:
        h = hashlib.sha256()
        for file in sorted(self.task.allowed_files):
            path=self.path(file);h.update(file.encode());h.update(path.read_bytes())
        return h.hexdigest()
    def verify(self) -> bool: return self.call(ToolCall('test', {}))['returncode'] == 0
    def record(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({'task': asdict(self.task), 'events': self.events}, indent=2), encoding='utf-8')
