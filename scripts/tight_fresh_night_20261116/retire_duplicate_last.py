"""Remove only closed15 last snapshots proven equal to retained selected bests."""
import json
import os
from pathlib import Path
import shutil
import time
import torch
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261115_13')
OUT=ROOT/'exact_duplicate_last_retirement'


def equal(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def run():
    torch.set_num_threads(2)
    assert (ROOT/'monitor_exit.json').exists() and (ROOT/'continuation/owner_finished.json').exists()
    OUT.mkdir(exist_ok=False);rows=[];before=shutil.disk_usage(ROOT).free
    try:
        for role in 'AB':
            d=ROOT/'round_15'/role/'training';last=d/'last.pt';best=d/'best.pt'
            assert last.is_file() and not last.is_symlink() and last.stat().st_nlink==1
            done=json.loads((d/'finished.json').read_text());assert done['complete'] and done['updates']==2460
            a=torch.load(last,map_location='cpu',mmap=True,weights_only=False)
            b=torch.load(best,map_location='cpu',mmap=True,weights_only=False)
            assert a['epoch']==b['epoch']==12 and a['updates']==b['updates']==2460
            assert equal(a,b),'Complete optimizer/model/EMA/RNG/history payload must match exactly'
            del a,b
            assert sha256(best)==done['checkpoint_sha256']
            s=last.stat();rows.append(dict(path=str(last),bytes=s.st_size,allocated_bytes=s.st_blocks*512,
                device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,sha256=sha256(last),
                retained_best=str(best),retained_sha256=done['checkpoint_sha256'],entire_payload_equal=True))
        atomic_json(OUT/'inventory.json',dict(created=time.time(),rows=rows,
            semantics='Only redundant serialization removed; retained best contains identical complete model, EMA, Adam, RNG, history and completed counters. No parameter/optimizer state or raw observation is lost.'))
        inodes={(r['device'],r['inode']) for r in rows}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                for fd in (p/'fd').iterdir():
                    try:s=fd.stat()
                    except (OSError,PermissionError):continue
                    assert (s.st_dev,s.st_ino) not in inodes,'Checkpoint still open: '+p.name
            except (OSError,PermissionError):continue
        removed=[]
        for row in rows:
            p=Path(row['path']);s=p.stat()
            assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(row['device'],row['inode'],row['mtime_ns'],row['bytes'])
            p.unlink();removed.append(row['path'])
            atomic_json(OUT/'progress.json',dict(removed=removed))
        atomic_json(OUT/'finished.json',dict(complete=True,utc=time.time(),removed=removed,
            retired_allocated_bytes=sum(r['allocated_bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(ROOT).free))
    except BaseException as e:
        atomic_json(OUT/'failure.json',dict(error=repr(e),utc=time.time()));raise


if __name__=='__main__':run()
