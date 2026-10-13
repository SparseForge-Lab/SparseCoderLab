"""Download pinned generator GGUFs and verify publisher SHA256 identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from forgebuilder.schema import canonical_json
from forgebuilder.strata_runtime import digest


def download(model):
    start=time.time()
    root=Path(model['directory'])
    root.mkdir(parents=True,exist_ok=True)
    target=root/model['filename']
    existing=target.stat().st_size if target.exists() else 0
    if target.exists() and (existing!=model['size_bytes'] or digest(target)!=model['sha256']):
        raise RuntimeError(f'refusing to replace a different existing model: {target}')
    reserve=20*1024**3
    if shutil.disk_usage(root).free < model['size_bytes']-existing+reserve:
        raise RuntimeError(f'not enough space with 20 GiB reserve: {root}')
    partial=target.with_suffix(target.suffix+'.partial')
    if not target.exists():
        attempts=0
        while not partial.exists() or partial.stat().st_size < model['size_bytes']:
            offset=partial.stat().st_size if partial.exists() else 0
            end=min(model['size_bytes']-1,offset+32*1024**2-1)
            url=(f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/"
                 f"{model['filename']}?download=true&offset={offset}")
            try:
                request=Request(url,headers={'Range':f'bytes={offset}-{end}'})
                with urlopen(request,timeout=120) as response:
                    expected=f"bytes {offset}-{end}/{model['size_bytes']}"
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise RuntimeError('server returned an unexpected byte range')
                    remaining=end-offset+1
                    with partial.open('ab') as stream:
                        while remaining:
                            block=response.read(min(4*1024**2,remaining))
                            if not block:
                                raise IOError('download ended before requested byte range')
                            stream.write(block)
                            remaining-=len(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                attempts=0
                print(json.dumps({'model':model['repository'],'downloaded_bytes':end+1,
                    'size_bytes':model['size_bytes'],'elapsed_seconds':time.time()-start}),flush=True)
            except Exception as exc:
                attempts+=1
                print(json.dumps({'model':model['repository'],'retry':attempts,'error':str(exc)}),flush=True)
                if attempts>=5:
                    raise
                time.sleep(min(10,2*attempts))
        if partial.stat().st_size!=model['size_bytes'] or digest(partial)!=model['sha256']:
            raise RuntimeError(f'downloaded model failed size/SHA256 verification: {partial}')
        verified_stat=partial.stat()
        os.replace(partial,target)
        final_stat=target.stat()
        if (final_stat.st_size,final_stat.st_mtime_ns)!=(verified_stat.st_size,verified_stat.st_mtime_ns):
            raise RuntimeError('model changed during atomic finalization')
    path=target
    receipt=dict(model,status='verified_download',path=str(path),elapsed_seconds=time.time()-start,
                 mtime_ns=path.stat().st_mtime_ns,verification=('SHA256 of existing final file' if existing else
                 'SHA256 before atomic finalization; size/mtime after rename'))
    (root/'download_receipt.json').write_bytes(canonical_json(receipt)+b'\n')
    print(json.dumps({'model':model['repository'],'status':'verified_download','path':str(path)}),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/dataset_v2/local_models.json'))
    parser.add_argument('--receipt',type=Path,default=Path('.local/dataset_v2/generator_downloads.json'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf8'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(download,config['models']))
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_bytes(canonical_json({'status':'verified_downloads','models':rows})+b'\n')


if __name__=='__main__':
    main()
