import json
from pathlib import Path
from src.config import load_config
from src.context.archive import Archive
from src.context.tasks import make_task, retention_fixture

if __name__ == '__main__':
    cfg = load_config('configs/dense_compute.yaml'); archive = Archive(Path('data/archive')); rows = []; tasks = []
    for index in range(cfg['context']['task_count']):
        task = make_task(index, cfg, archive); tasks.append(task)
        for ratio in cfg['context']['ratios']:
            for retrieve in (False, True): rows.extend(dict(task=index, **r) for r in retention_fixture(task, ratio, cfg['context']['boundaries'], archive, retrieve))
    Path('data/context_tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks), encoding='utf-8')
    Path('results/context_fixtures.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'{len(tasks)} delayed-fact tasks, {len(rows)} retention checks; oracle fixtures only.')
