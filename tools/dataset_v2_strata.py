"""Start an independent 32K teacher using the installed IQ3_XXS Strata config."""
import argparse
import json
from pathlib import Path

from forgebuilder.strata_runtime import assert_sources_unchanged, fingerprint_weights, health, prepare, start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'start', 'status', 'verify-source', 'fingerprint'])
    parser.add_argument('--source', type=Path, default=Path('D:/strata-abliterated/strata-iq3_xxs.json'))
    parser.add_argument('--runtime', type=Path, default=Path('.local/dataset_v2/strata32k'))
    parser.add_argument('--port', type=int, default=8096)
    parser.add_argument('--vram-reserve-mib', type=int, default=2560,
                        help='Runtime reserve allowing roughly 2 GiB free after extra engine buffers')
    args = parser.parse_args()
    if args.action in ('prepare', 'start'):
        receipt = prepare(args.source, args.runtime, args.port, args.vram_reserve_mib)
        result = start(receipt, args.runtime.resolve()) if args.action == 'start' else receipt
    elif args.action == 'fingerprint':
        result = fingerprint_weights(args.runtime)
    else:
        receipt = json.loads((args.runtime / 'prepared.json').read_text(encoding='utf8'))
        assert_sources_unchanged(receipt)
        result = health(receipt['port']) if args.action == 'status' else {'source_unchanged': True}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
