"""Install hash-pinned CPU evaluation artifacts into a private directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, url_key, sha_key, folder in (
        ("cpython.zip", "cpython_url", "cpython_archive_sha256", "cpython"),
        ("wasmtime.whl", "windows_wheel_url", "windows_wheel_sha256", "dependencies"),
    ):
        archive = root / name
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            with urllib.request.urlopen(config[url_key], timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if digest(temporary) != config[sha_key]:
                raise ValueError("Runtime download hash mismatch")
            temporary.replace(archive)
        if digest(archive) != config[sha_key]:
            raise ValueError("Runtime archive hash mismatch")
        target = root / folder
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename:
                    raise ValueError("Unsafe archive member")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Runtime archive symlink")
            if zipped.testzip() is not None:
                raise ValueError("Runtime archive CRC failure")
            target.mkdir(exist_ok=True)
            expected_files = set()
            for entry in zipped.infolist():
                if entry.is_dir():
                    continue
                destination = target / entry.filename
                expected_files.add(destination.relative_to(target).as_posix())
                raw = zipped.read(entry)
                if destination.exists():
                    if digest(destination) != hashlib.sha256(raw).hexdigest():
                        raise ValueError("Installed runtime differs from pinned archive")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(raw)
            observed_files = {p.relative_to(target).as_posix() for p in target.rglob("*")
                              if p.is_file() and "__pycache__" not in p.parts}
            if observed_files != expected_files:
                raise ValueError("Unexpected installed runtime files")
    receipt = {
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "files": {
            str(p.relative_to(root)).replace("\\", "/"): digest(p)
            for folder in (root / "cpython", root / "dependencies")
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
        },
    }
    (root / "integrity.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".local/wasi_eval")
    parser.add_argument("--config", default="configs/eval/wasi_runtime.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    receipt = install(args.root, config)
    print(json.dumps({"verified_runtime_files": len(receipt["files"]), "gpu_used": False}))


if __name__ == "__main__":
    main()
