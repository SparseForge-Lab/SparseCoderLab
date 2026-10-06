import sys
from pathlib import Path
from src.config import load_config
from src.runtime.agent import TaskSpec, ToolCall, RepoSandbox

if __name__ == '__main__':
    task = TaskSpec('toy-add', 'Repair integer addition', ['calculator.py'], [sys.executable, '-B', 'check.py'])
    cfg = load_config('configs/dense_compute.yaml')['agent']; sandbox = RepoSandbox(Path('data/toy_repo'), task, cfg)
    sandbox.call(ToolCall('inspect', {'file': 'calculator.py'})); sandbox.call(ToolCall('hypothesize', {}, 'Operator is incorrect'))
    sandbox.call(ToolCall('edit', {'file': 'calculator.py', 'text': 'def add(a, b):\n    return a * b\n'})); sandbox.call(ToolCall('test', {}))
    sandbox.call(ToolCall('test', {})); sandbox.call(ToolCall('replan', {}, 'Multiplication fails; addition required'))
    sandbox.call(ToolCall('edit', {'file': 'calculator.py', 'text': 'def add(a, b):\n    return a + b\n'}))
    assert sandbox.verify(); sandbox.call(ToolCall('diff', {})); sandbox.record(Path('results/agent_trajectory.json'))
    print('Recorded failed, repeated and repaired trajectories; final verifier passed.')
