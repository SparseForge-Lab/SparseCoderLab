"""Read-only hardware probe, with bounded project-local buffered I/O benchmarks."""
from __future__ import annotations
import argparse, ctypes, json, os, platform, random, shutil, subprocess, sys, time
from pathlib import Path

def command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

def io_benchmark(root: Path, mib: int = 256, block_mib: int = 4) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'io_probe.bin'
    block = os.urandom(block_mib * 1024**2)
    count = mib // block_mib
    try:
        with path.open('wb') as f:
            for _ in range(count): f.write(block)
            f.flush(); os.fsync(f.fileno())
        if os.name == 'nt':
            return windows_unbuffered_read(path, mib, block_mib)
        order = list(range(count)); random.Random(42).shuffle(order)
        rates = {}
        for name, offsets in [('sequential', range(count)), ('random_large_block', order)]:
            start = time.perf_counter()
            with path.open('rb', buffering=0) as f:
                for i in offsets:
                    f.seek(i * len(block)); f.read(len(block))
            elapsed = time.perf_counter() - start
            rates[name] = {'mib_s': mib / elapsed, 'block_ms': elapsed * 1000 / count}
        return {'size_mib': mib, 'block_mib': block_mib, **rates,
                'caveat': 'Buffered, recently written file: OS-cache upper bound, NOT cold SSD bandwidth. Supply measured cold-device values for credible projections.'}
    finally:
        path.unlink(missing_ok=True)

def windows_unbuffered_read(path: Path, mib: int, block_mib: int) -> dict:
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]; kernel.VirtualAlloc.restype = ctypes.c_void_p
    kernel.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD]; kernel.VirtualFree.restype = w.BOOL
    kernel.SetFilePointerEx.argtypes = [w.HANDLE, ctypes.c_longlong, ctypes.c_void_p, w.DWORD]; kernel.SetFilePointerEx.restype = w.BOOL
    kernel.ReadFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]; kernel.ReadFile.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]; kernel.CloseHandle.restype = w.BOOL
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1, None, 3, 0x20000000, None)
    if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    size = block_mib * 1024**2; buffer = kernel.VirtualAlloc(None, size, 0x3000, 0x04)
    if not buffer: kernel.CloseHandle(handle); raise ctypes.WinError(ctypes.get_last_error())
    order = list(range(mib // block_mib)); random.Random(42).shuffle(order); rates = {}
    try:
        for name, offsets in [('sequential', range(mib // block_mib)), ('random_large_block', order)]:
            durations = []; amount = w.DWORD()
            for offset in offsets:
                start = time.perf_counter()
                if not kernel.SetFilePointerEx(handle, offset * size, None, 0): raise ctypes.WinError(ctypes.get_last_error())
                if not kernel.ReadFile(handle, buffer, size, ctypes.byref(amount), None): raise ctypes.WinError(ctypes.get_last_error())
                if amount.value != size: raise RuntimeError('Short unbuffered read')
                durations.append(time.perf_counter() - start)
            rates[name] = {'mib_s': mib / sum(durations), 'block_ms': sum(durations) * 1000 / len(durations),
                           'block_p95_ms': sorted(durations)[int(.95 * (len(durations)-1))] * 1000}
        return {'size_mib': mib, 'block_mib': block_mib, 'mode': 'Windows FILE_FLAG_NO_BUFFERING, aligned VirtualAlloc buffer',
                **rates, 'caveat': 'Bypasses Windows page cache; device cache/firmware may still contribute. 4MiB block times include transfer, do not add them as a separate latency to bandwidth. SATA on this machine, not NVMe.'}
    finally: kernel.VirtualFree(buffer, 0, 0x8000); kernel.CloseHandle(handle)

def probe(bench: bool = True) -> dict:
    r = {'os': platform.platform(), 'wsl': 'microsoft' in platform.release().lower(),
         'python': sys.version, 'executable': sys.executable, 'cpu': platform.processor(),
         'logical_cpus': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(Path(__file__).resolve().parents[1]).free,
         'nvidia_smi': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader'])}
    try:
        import psutil
        memory=psutil.virtual_memory(); r['system_ram']={'total_bytes':memory.total,'available_bytes':memory.available}
    except ImportError: pass
    if os.name == 'nt':
        r['hardware'] = command(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json; Get-PhysicalDisk | Select-Object FriendlyName,MediaType,Size | ConvertTo-Json'])
    else:
        r['ram'] = command(['free', '-b']); r['disks'] = command(['lsblk', '-d', '-o', 'NAME,MODEL,SIZE,ROTA'])
    try:
        import torch
        r.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, arch_list=torch.cuda.get_arch_list(), cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cc = torch.cuda.get_device_capability()
            r.update(gpu=torch.cuda.get_device_name(), compute_capability=cc,
                     bf16=torch.cuda.is_bf16_supported(), tf32=cc[0] >= 8,
                     free_vram_bytes=free, total_vram_bytes=total)
    except ImportError:
        r['torch'] = 'not installed in this interpreter'
    if bench: r['io_benchmark'] = io_benchmark(Path(__file__).resolve().parents[1] / 'work')
    return r

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default='results/environment.json'); p.add_argument('--no-io', action='store_true')
    a = p.parse_args(); result = probe(not a.no_io)
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
