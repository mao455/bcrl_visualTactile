"""One-file read-only I/O probe; never evicts OS cache or changes live workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def counters():
    return {key:int(value) for key,value in (line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())}


def probe(path, mode='sequential', expected_sha256=None):
    path=Path(path).resolve();before_stat=path.stat()
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if before_stat.st_size>1<<30 or available-before_stat.st_size<10<<30:
        raise RuntimeError('Single probe capped at 1 GiB with 10 GiB RAM reserve')
    # Import overhead is intentionally excluded from the data read interval.
    if mode!='sequential':
        import torch
    before=counters();started=time.perf_counter();digest=None;payload_bytes=0
    if mode=='sequential':
        hasher=hashlib.sha256()
        with path.open('rb',buffering=0) as stream:
            if hasattr(os,'posix_fadvise'):os.posix_fadvise(stream.fileno(),0,before_stat.st_size,os.POSIX_FADV_SEQUENTIAL)
            while chunk:=stream.read(8<<20):
                payload_bytes+=len(chunk);hasher.update(chunk)
        digest=hasher.hexdigest()
    elif mode in ('mmap_clone','eager'):
        shard=torch.load(path,map_location='cpu',weights_only=False,mmap=mode=='mmap_clone')
        if mode=='mmap_clone':shard={key:value.clone() for key,value in shard.items()}
        payload_bytes=sum(value.numel()*value.element_size() for value in shard.values())
        del shard
    else:raise ValueError('Unknown single-file read mode')
    elapsed=time.perf_counter()-started;after=counters();after_stat=path.stat()
    if (before_stat.st_dev,before_stat.st_ino,before_stat.st_size,before_stat.st_mtime_ns)!=(after_stat.st_dev,after_stat.st_ino,after_stat.st_size,after_stat.st_mtime_ns):
        raise ValueError('Source file changed during probe')
    if expected_sha256 is not None and digest is not None and digest!=expected_sha256:
        raise ValueError('Registered source checksum differs')
    return dict(path=str(path),mode=mode,file_bytes=before_stat.st_size,payload_bytes=payload_bytes,
        elapsed_seconds=elapsed,throughput_mib_s=payload_bytes/(1<<20)/elapsed,
        process_io_delta={key:after[key]-before[key] for key in ('read_bytes','rchar','syscr')},
        sha256=digest,source_mtime_ns=before_stat.st_mtime_ns,source_inode=before_stat.st_ino,
        memory_available_before=available,utc=time.time(),
        scope='Single read-only existing shard; OS cache retained; mode order and concurrent jobs confound direct speed comparison')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['sequential','mmap_clone','eager'],default='sequential')
    parser.add_argument('--sha256');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous I/O evidence')
    result=probe(args.path,args.mode,args.sha256)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
