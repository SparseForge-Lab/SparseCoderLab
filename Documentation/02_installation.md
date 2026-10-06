# Installation and isolation

Run PowerShell from `D:\SparseCoderLab`:

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe verify_install.py
.\.venv\Scripts\python.exe -m pip check
```

`requirements.lock` captures the exact installed Windows packages. The official torch CUDA index is `https://download.pytorch.org/whl/cu130`; no global pip install is performed. First-time setup probes before installation, selects the stable official wheel and verifies CUDA runtime >=12.8, SM120 coverage and BF16 execution. Reproducing the lock uses the same torch index.

If WSL2 later works, run `bash setup_wsl.sh` in `/mnt/d/SparseCoderLab`. It creates `.venv-wsl` separately and records `requirements-wsl.lock`. The Windows lock does not promise cross-platform wheel parity. Do not modify the current CUDA toolkit merely to match a torch wheel: the wheel brings its runtime.

No FlashAttention, Triton, JAX, MaxText or model downloads are required. Eager native SDPA is the correctness baseline. Fused AdamW and compile are disabled initially. Training commands reject CPU devices explicitly; unit tests may intentionally run small tensors on CPU.
