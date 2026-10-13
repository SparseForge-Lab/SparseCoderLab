import io
import json
from types import SimpleNamespace

import pytest

from forgebuilder.schema import sha256_bytes
from tools import download_dataset_v2_models as downloads


def config(tmp_path, payload):
    return dict(directory=str(tmp_path),filename='test.gguf',repository='fixture/model',
                revision='a'*40,size_bytes=len(payload),sha256=sha256_bytes(payload))


def test_download_resumes_exact_range_and_verifies_hash(tmp_path,monkeypatch):
    payload=b'GGUF verified fixture weights'
    model=config(tmp_path,payload)
    (tmp_path/'test.gguf.partial').write_bytes(payload[:7])
    monkeypatch.setattr(downloads.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    def open_range(request,timeout):
        assert request.get_header('Range')==f'bytes=7-{len(payload)-1}'
        result=io.BytesIO(payload[7:])
        result.status=206
        result.headers={'Content-Range':f'bytes 7-{len(payload)-1}/{len(payload)}'}
        return result
    monkeypatch.setattr(downloads,'urlopen',open_range)
    receipt=downloads.download(model)
    assert (tmp_path/'test.gguf').read_bytes()==payload
    assert receipt['status']=='verified_download'
    assert json.loads((tmp_path/'download_receipt.json').read_text())['sha256']==model['sha256']


def test_wrong_existing_model_is_preserved(tmp_path):
    (tmp_path/'test.gguf').write_bytes(b'other')
    with pytest.raises(RuntimeError,match='refusing to replace'):
        downloads.download(config(tmp_path,b'expected'))
    assert (tmp_path/'test.gguf').read_bytes()==b'other'


def test_corrupt_download_never_becomes_final_model(tmp_path,monkeypatch):
    model=config(tmp_path,b'GGUF expected data')
    monkeypatch.setattr(downloads.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    def wrong_range(request,timeout):
        result=io.BytesIO(b'x'*model['size_bytes'])
        result.status=206
        result.headers={'Content-Range':f"bytes 0-{model['size_bytes']-1}/{model['size_bytes']}"}
        return result
    monkeypatch.setattr(downloads,'urlopen',wrong_range)
    with pytest.raises(RuntimeError,match='verification'):
        downloads.download(model)
    assert not (tmp_path/'test.gguf').exists()
