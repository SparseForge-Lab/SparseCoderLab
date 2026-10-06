$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python=Join-Path (Get-Location) '.venv\Scripts\python.exe'
function Invoke-Check([string[]]$Arguments) {
    & $python @Arguments
    if ($LASTEXITCODE) { throw ('Check failed: '+($Arguments -join ' ')) }
}
Invoke-Check @('verify_install.py')
if (-not (Test-Path 'data\shards\manifest.json')) { Invoke-Check @('-m','tools.prepare_data') }
Invoke-Check @('-m','pytest','-q')
Invoke-Check @('-m','tools.overfit','--config','configs/dense_compute.yaml')
Invoke-Check @('-m','tools.count_params')
Invoke-Check @('-m','tools.profile','--config','configs/dense_compute.yaml','--seconds','300')
Invoke-Check @('-m','tools.profile','--config','configs/sparse_v3.yaml','--seconds','30')
Invoke-Check @('-m','tools.profile','--config','configs/sparse_memory.yaml','--seconds','30')
Invoke-Check @('-m','tools.phase0')
Write-Output 'Phase 0 complete. No long training run launched.'
