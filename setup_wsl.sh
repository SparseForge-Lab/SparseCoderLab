#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 tools/env_probe.py --output results/environment_before_wsl.json
python3 -m venv .venv-wsl
PY=.venv-wsl/bin/python
# Windows lock contains platform wheels: resolve and lock Linux separately.
"$PY" -m pip install --no-cache-dir torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
"$PY" -m pip install --no-cache-dir numpy pyyaml tokenizers psutil pytest
"$PY" verify_install.py
"$PY" -m pip freeze > requirements-wsl.lock
