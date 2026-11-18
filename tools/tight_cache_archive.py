"""Losslessly archive an explicit list of inactive derived caches; never raw data.

Every compressed file is fully decompressed and hashed before its original is
removed. Receipts preserve exact bytes, permissions, timestamps and restoration.
Historical consumers must restore their cache before replay; current protected
paths are excluded by the caller's audited manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import zstandard as zstd


def digest(stream):
    h=hashlib.sha256()
    while block:=stream.read(4*2**20):h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as f:return digest(f)


def assert_closed(path):
    identity=(path.stat().st_dev,path.stat().st_ino)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    s=fd.stat()
                    if (s.st_dev,s.st_ino)==identity:raise RuntimeError(f'Open cache: {path}, PID {proc.name}')
                except FileNotFoundError:pass
        except (PermissionError,FileNotFoundError):continue


def archive_file(path,expected):
    path=Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
    assert 'perception' in path.parts and path.suffix=='.pt'
    target=path.with_suffix('.pt.zst');receipt=path.with_suffix('.pt.archive.json')
    assert not target.exists() and not receipt.exists()
    assert_closed(path);before=path.stat();assert sha(path)==expected
    with path.open('rb') as src,target.open('xb') as out:
        zstd.ZstdCompressor(level=3,threads=0).copy_stream(src,out)
        out.flush();os.fsync(out.fileno())
    with target.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as decoded:
        assert digest(decoded)==expected,'Compressed recovery does not match original'
    after=path.stat()
    assert (before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size)
    assert_closed(path)
    row=dict(path=str(path),archive=str(target),sha256=expected,archive_sha256=sha(target),
        original_bytes=before.st_size,compressed_bytes=target.stat().st_size,
        mode=stat.S_IMODE(before.st_mode),mtime_ns=before.st_mtime_ns,utc=time.time(),
        archive_verified=True,original_removed=False)
    receipt.write_text(json.dumps(row,indent=2)+'\n')
    path.unlink()
    row['original_removed']=True;receipt.write_text(json.dumps(row,indent=2)+'\n')
    return row


def restore_file(receipt):
    row=json.loads(Path(receipt).read_text());path=Path(row['path']);source=Path(row['archive'])
    assert not path.exists() and sha(source)==row['archive_sha256']
    temp=path.with_suffix('.pt.restoring');assert not temp.exists()
    try:
        with source.open('rb') as src,temp.open('xb') as out:
            zstd.ZstdDecompressor().copy_stream(src,out);out.flush();os.fsync(out.fileno())
        assert sha(temp)==row['sha256']
        os.chmod(temp,row['mode']);os.utime(temp,ns=(row['mtime_ns'],row['mtime_ns']))
        os.replace(temp,path)
    except BaseException:
        temp.unlink(missing_ok=True);raise
    # Keep archive and its receipt as independent evidence after restoration.
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['archive','restore']);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    if a.mode=='restore':
        for r in manifest['files']:restore_file(Path(r['path']).with_suffix('.pt.archive.json'))
    else:
        protected={Path(x).resolve() for x in manifest['protected_cache_paths']}
        for r in manifest['files']:
            assert Path(r['path']).resolve() not in protected
            row=archive_file(r['path'],r['sha256']);print(json.dumps(row),flush=True)


if __name__=='__main__':main()
