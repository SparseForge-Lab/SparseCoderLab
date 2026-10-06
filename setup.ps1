param([string]$TorchIndex = 'https://download.pytorch.org/whl/cu130')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv')) { py -3.12 -m venv .venv }
if ($LASTEXITCODE) { throw 'venv creation failed' }
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python tools/env_probe.py --output results/environment_before.json
if (Test-Path requirements.lock) {
    & $python -m pip install --no-cache-dir -r requirements.lock --extra-index-url $TorchIndex
} else {
    & $python -m pip install --no-cache-dir torch --index-url $TorchIndex
    if ($LASTEXITCODE) { throw 'torch installation failed' }
    & $python -m pip install --no-cache-dir numpy pyyaml tokenizers psutil pytest
}
if ($LASTEXITCODE) { throw 'dependency installation failed' }
& $python verify_install.py
if ($LASTEXITCODE) { throw 'GPU verification failed' }
& $python -m pip freeze | Set-Content -Encoding utf8 requirements.lock
