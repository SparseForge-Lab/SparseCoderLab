"""Execute generated Python in CPython/WASI, never in the host interpreter.

Only the pinned runtime's standard library is mounted, read-only. No project
files, writable directories, sockets, host environment or stdin are inherited.
Answers are checked by the host; the guest never receives expected answers.
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import sys
import threading
import time
from pathlib import Path

# These instructions run inside the Wasm guest, not on the host. No test oracle
# or pass/fail decision is placed in the guest process.
GUEST_RUNNER = r'''
import base64, contextlib, io, json, sys
payload = json.loads(base64.b64decode(sys.argv[1]))
source = payload["source"]
encode = json.dumps
write = sys.stdout.write
capture = io.StringIO
redirect = contextlib.redirect_stdout
redirect_error = contextlib.redirect_stderr
results = []
namespace = {"__name__": "candidate"}
module_output, module_error = capture(), capture()
current_output, current_error = module_output, module_error
try:
    compiled = compile(source, "<candidate>", "exec")
except SyntaxError:
    write(encode({"kind": "syntax_error"}) + "\n")
else:
    try:
        with redirect(module_output), redirect_error(module_error):
            exec(compiled, namespace)
        function = namespace[payload["function"]]
        for args in payload["arguments"]:
            output, error_output = capture(), capture()
            current_output, current_error = output, error_output
            with redirect(output), redirect_error(error_output):
                value = function(*args)
            results.append({"value": value, "stdout": output.getvalue(), "stderr": error_output.getvalue()})
        write(encode({"kind": "completed", "results": results,
                      "module_stdout": module_output.getvalue(),
                      "module_stderr": module_error.getvalue()}, allow_nan=False) + "\n")
    except BaseException as error:
        write(encode({"kind": "runtime_error", "exception": type(error).__name__,
                      "module_stdout": module_output.getvalue(),
                      "module_stderr": module_error.getvalue(),
                      "partial_stdout": current_output.getvalue(),
                      "partial_stderr": current_error.getvalue()}) + "\n")
'''


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class WasiPython:
    def __init__(self, root=".local/wasi_eval", config_path="configs/eval/wasi_runtime.json"):
        self.root = Path(root).resolve()
        self.config = json.loads(Path(config_path).read_text(encoding="utf8"))
        receipt = json.loads((self.root / "integrity.json").read_text(encoding="utf8"))
        expected_config = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        if receipt["config_sha256"] != expected_config:
            raise ValueError("Runtime configuration changed; run setup again")
        # Validate native binding binaries and all read-only guest library files
        # before import. A pinned archive alone does not prove extracted identity.
        observed = {
            str(p.relative_to(self.root)).replace("\\", "/"): file_hash(p)
            for folder in (self.root / "cpython", self.root / "dependencies")
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
        }
        if observed != receipt["files"]:
            raise ValueError("Extracted runtime integrity mismatch")
        sys.path.insert(0, str(self.root / "dependencies"))
        w = importlib.import_module("wasmtime")
        if importlib.metadata.version("wasmtime") != self.config["wasmtime_version"]:
            raise ValueError("Wrong Wasmtime version")
        self.w = w
        config = w.Config()
        config.consume_fuel = True
        config.epoch_interruption = True
        config.parallel_compilation = False
        self.engine = w.Engine(config)
        self.module = w.Module.from_file(self.engine, str(self.root / "cpython/python.wasm"))
        if any(i.module != "wasi_snapshot_preview1" for i in self.module.imports):
            raise ValueError("Unexpected guest host imports")
        self.linker = w.Linker(self.engine)
        self.linker.define_wasi()
        self.lock = threading.Lock()
        self.runtime_identity = {
            "cpython_version": self.config["cpython_version"],
            "cpython_archive_sha256": self.config["cpython_archive_sha256"],
            "wasmtime_version": self.config["wasmtime_version"],
            "wasmtime_wheel_sha256": self.config["windows_wheel_sha256"],
            "module_sha256": observed["cpython/python.wasm"],
            "config_sha256": file_hash(config_path),
            "guest_runner_sha256": hashlib.sha256(GUEST_RUNNER.encode()).hexdigest(),
            "guest_memory_bytes": self.config["guest_memory_bytes"],
            "fuel": self.config["fuel"], "deadline_seconds": self.config["deadline_seconds"],
            "output_bytes": self.config["output_bytes"], "gpu_used": False,
        }

    def execute(self, source, function, arguments):
        if len(source.encode("utf8")) > self.config["source_bytes"]:
            return {"kind": "source_limit", "wall_seconds": 0.0, "fuel_consumed": 0}
        payload = base64.b64encode(json.dumps({
            "source": source, "function": function, "arguments": arguments,
        }).encode()).decode()
        with self.lock:
            return self._execute(payload)

    def _execute(self, payload):
        w = self.w
        output, errors = bytearray(), bytearray()
        exceeded = False

        def consume(target, data):
            nonlocal exceeded
            if len(output) + len(errors) + len(data) > self.config["output_bytes"]:
                exceeded = True
                self.engine.increment_epoch()
                return -1
            target.extend(data)
            return len(data)

        store = w.Store(self.engine)
        store.set_limits(memory_size=self.config["guest_memory_bytes"], instances=1, memories=1, tables=1)
        store.set_fuel(self.config["fuel"])
        store.set_epoch_deadline(1)
        wasi = w.WasiConfig()
        wasi.argv = ["python", "-B", "-S", "-P", "-c", GUEST_RUNNER, payload]
        wasi.env = [("PYTHONHOME", "/runtime"), ("PYTHONHASHSEED", "0")]
        wasi.preopen_dir(str(self.root / "cpython/lib"), "/runtime/lib", fs_mutable=False)
        wasi.stdout_custom = lambda data: consume(output, data)
        wasi.stderr_custom = lambda data: consume(errors, data)
        store.set_wasi(wasi)
        timer = threading.Timer(self.config["deadline_seconds"], self.engine.increment_epoch)
        timer.daemon = True
        started = time.perf_counter()
        timer.start()
        result = {"kind": "sandbox_error"}
        try:
            instance = self.linker.instantiate(store, self.module)
            instance.exports(store)["_start"](store)
            result = json.loads(output.decode("utf8"))
            if not isinstance(result, dict) or result.get("kind") not in ("completed", "syntax_error", "runtime_error"):
                result = {"kind": "invalid_output"}
        except w.Trap as error:
            if error.trap_code in (w.TrapCode.OUT_OF_FUEL, w.TrapCode.INTERRUPT):
                result = {"kind": "timeout"}
            else:
                result = {"kind": "runtime_error", "exception": "WasmTrap"}
        except w.ExitTrap as error:
            result = {"kind": "runtime_error", "exception": "GuestExit", "exit_code": error.code}
        except (ValueError, UnicodeError):
            result = {"kind": "invalid_output"}
        finally:
            timer.cancel()
            # Join prevents an old timer firing during the next fresh store.
            timer.join()
            result["fuel_consumed"] = self.config["fuel"] - store.get_fuel()
            store.close()
        if exceeded:
            result = {"kind": "output_limit", "fuel_consumed": result["fuel_consumed"]}
        result["wall_seconds"] = time.perf_counter() - started
        result["stderr"] = errors.decode("utf8", errors="replace")
        return result
