# Environment

Native Windows 11 is selected because Ubuntu WSL2 cannot start: virtualization/Virtual Machine Platform is unavailable. No BIOS, Windows feature, driver, CUDA toolkit or existing Python installation was changed.

- GPU: NVIDIA GeForce RTX 5070, compute capability 12.0 (SM120), 12,227 MiB driver-visible VRAM.
- Driver: 616.64. The driver's advertised CUDA 13.4 compatibility is distinct from the wheel's CUDA 13.0 runtime.
- Project interpreter: Python 3.12.10 under `D:\SparseCoderLab\.venv`.
- Project torch: official stable 2.14.1+cu130, architecture list includes `sm_120`.
- BF16 native SDPA forward/backward was actually executed by `verify_install.py`.
- CPU: AMD Ryzen 7 5700X, 8 cores / 16 threads.
- Installed RAM: 51,442,139,136 bytes (~47.91 GiB).
- D: resides on FIKWOT FS810 1TB, SATA SSD. This machine's project drive is not NVMe.

`tools/env_probe.py` runs before heavy installation. Windows storage reads use `FILE_FLAG_NO_BUFFERING` and an aligned VirtualAlloc buffer over a 256 MiB project-local file. It benchmarks sequential and shuffled 4 MiB reads and removes its own file. Windows page cache is bypassed, but device cache remains possible. Per-block time includes transfer; adding it as a separate latency alongside measured bandwidth double-counts transfer.

`results/environment_before.json` contains the initial buffered benchmark, clearly labelled as cached. `results/environment.json` contains the later unbuffered measurement. GPU free memory can vary with the desktop and WDDM process budgeting. Use actual training allocated/reserved peaks plus driver telemetry when choosing headroom.

Primary references: https://pytorch.org/get-started/locally/ and https://discuss.pytorch.org/t/pytorch-support-for-sm-120/222119/2 . Actual local kernel execution is the compatibility evidence.
