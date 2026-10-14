"""Prepare actual read-only terminal examples without loading a new model."""
import json
from pathlib import Path
from tokenizers import Tokenizer
from forgebuilder.strata_runtime import digest
from forgebuilder.terminal_tasks import prepare_terminal_pilot,finalize_terminal_quarantine


def main():
    foundation=json.loads(Path('data/dataset_v2/foundation_v1/build_receipt.json').read_bytes())
    path=Path(foundation['build_config']['tokenizer'])
    if digest(path)!=foundation['tokenizer_sha256']: raise ValueError('frozen tokenizer changed')
    source=Path('data/dataset_v2/source_pilot_v1'); output=Path('data/dataset_v2/terminal_pilot_v1')
    prepare_terminal_pilot(source,output,Tokenizer.from_file(str(path)))
    print(json.dumps(finalize_terminal_quarantine(source,output),indent=2))


if __name__=='__main__': main()
