"""Record explicit dataset/checkpoint storage and future retention headroom."""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from src.utils.hashing import sha256_file


def bytes_in(directory):
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file()) if directory.exists() else 0


def measure(run_root, output):
    if output.exists():
        raise FileExistsError('Use a new measured storage report')
    canonical = json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text(encoding='utf8'))
    entries = canonical['canonical_transition_checkpoints'].values()
    full = [Path(row['full_resume_state']['path']).stat().st_size for row in entries]
    weights = sum(Path(row['model_weights']['path']).stat().st_size for row in canonical['canonical_transition_checkpoints'].values())
    data = Path('data/research_v2_real')
    measured = {str(directory): bytes_in(directory) for directory in (
        data/'documents', data/'exports'/'transition_r1', data/'cache'/'transition_r1',
        data/'cache'/'transition_r2', data/'shards'/'transition_r1', data/'shards'/'transition_r2', run_root)}
    active = bytes_in(data/'shards'/'transition_r2') + Path('data/research_v1/tokenizer.json').stat().st_size
    # Retain both initial/smoke/engine states already measured in run_root.
    # Future phase allowance is four rolling + four promoted full states,
    # four promoted model-only states and one largest atomic replacement.
    future = 2*sum(full) + weights + max(full)
    free = shutil.disk_usage(run_root).free
    reserve = 20*1024**3
    receipt = dict(schema_version=1, measured_utc=datetime.now(timezone.utc).isoformat(),
        scoped_existing_bytes=measured, active_training_data_bytes=active,
        active_training_data_budget_bytes=6*1024**3, project_volume_free_bytes=free,
        future_checkpoint_allowance_bytes=future, free_after_future_allowance_bytes=free-future,
        free_space_reserve_bytes=reserve, largest_atomic_checkpoint_bytes=max(full),
        four_canonical_full_state_bytes=sum(full), four_canonical_model_only_bytes=weights,
        active_data_budget_passed=active<=6*1024**3, checkpoint_retention_reserve_passed=free-future>=reserve,
        checkpoint_retention_policy='Keep canonical/history and initial/disposable evidence; reserve four rolling + four promoted full states, four model-only gate weights and one atomic temporary replacement. No automatic deletion.',
        scope='Scoped artifact bytes and actual remaining volume space. This does not scan unrelated project files or claim a total-project inventory.',
        script_sha256=sha256_file(Path(__file__)))
    receipt['passed'] = receipt['active_data_budget_passed'] and receipt['checkpoint_retention_reserve_passed']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(receipt), flush=True)
    if not receipt['passed']:
        raise RuntimeError('Storage gate failed; preserved the measurements')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure(args.run_root, args.output)
