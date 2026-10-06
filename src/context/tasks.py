from __future__ import annotations
import random
from pathlib import Path
from src.context.archive import Archive, compact_fields

def make_task(index: int, cfg: dict, archive: Archive) -> dict:
    rng = random.Random(cfg['data']['seed'] + index)
    fields = {'identifier': f'repair_worker_{rng.randrange(1000000):06d}', 'filename': f'src/worker_{index}.py',
              'signature': f'def process_{index}(payload: bytes, timeout: int) -> int:',
              'constraint': 'must preserve input ordering and never mutate payload', 'numeric_value': str(rng.randrange(10, 500)),
              'failed_approach': 'changing the return type to str broke caller compatibility',
              'tool_output': f'pytest: case_{index} failed; expected errno=17', 'compiler_error': f'NameError: cache_{index} is undefined'}
    chunks = [f'FACT {k} = {v}\n' for k, v in fields.items()]
    pointers = [archive.put(index * len(fields) + n, text) for n, text in enumerate(chunks)]
    work = 'Unrelated work log: inspect -> edit -> run tests -> reconsider.\n' * 160
    return {'id': index, 'facts': fields, 'pointers': pointers,
            'prompt': chunks[0] + work + ''.join(chunks[1:4]) + work + ''.join(chunks[4:]) + work,
            'question': 'Return all exact FACT values, including the initial identifier and failed approach.'}

def retention_fixture(task: dict, ratio: int, boundaries: int, archive: Archive, retrieve: bool) -> list[dict]:
    """Oracle dictionary retention: infrastructure upper bound, never neural-model quality."""
    fields = dict(task['facts']); results = []
    for boundary in range(boundaries + 1):
        if boundary:
            state = compact_fields(fields, task['pointers'], ratio); fields = state['fields']
        recovered = dict(fields)
        if retrieve:
            for pointer in task['pointers']:
                text = archive.get(pointer).strip(); name, value = text.removeprefix('FACT ').split(' = ', 1); recovered[name] = value
        score = sum(recovered.get(k) == v for k, v in task['facts'].items()) / len(task['facts'])
        results.append({'boundary': boundary, 'ratio': ratio, 'retrieval': retrieve, 'exact_field_retention': score,
                        'lost_fields': [k for k, v in task['facts'].items() if fields.get(k) != v],
                        'scope': 'Deterministic archive/compactor fixture, NOT model task accuracy'})
    return results
