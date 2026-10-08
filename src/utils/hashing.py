from pathlib import Path
import hashlib


def sha256_file(path: str | Path) -> str:
    """Hash arbitrary-sized files with a fixed 1 MiB read buffer."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
