"""Authorized classified cleanup of obsolete caches and duplicate last weights.

Historical selected bests, raw trajectories, exports, expert and DINO are retained.
"""
import argparse
from pathlib import Path
import os
import shutil
import time
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def run(run_root):
    old=run_root.parent/'tight_execution_learning_20261113_01'
    receipt=run_root/'cleanup_old_20261113.json'
    if receipt.exists():raise FileExistsError('Never overwrite deletion evidence')
    protected={str(p):sha256(p) for p in old.rglob('best.pt') if p.is_file()}
    rows=[]
    for p in sorted(old.rglob('*.pt')):
        if p.is_symlink():continue
        kind='old_last_checkpoint' if p.name=='last.pt' and 'training' in p.parts else 'old_nominal_cache' if p.parent.name=='nominal' else None
        if kind:
            stat=p.stat()
            rows.append(dict(path=str(p),sha256=sha256(p),size=stat.st_size,blocks=stat.st_blocks,
                links=stat.st_nlink,kind=kind,inode=stat.st_ino,device=stat.st_dev))
    plan=dict(authorization='2026-11-14 user permits old student checkpoints and invalid derivative caches cleanup',
        rows=rows,protected_selected_bests=protected,status='classified',utc=time.time(),
        excluded='all raw h5/npz, initial exports, selected best weights, expert and generic visual pretraining; current fresh root')
    atomic_json(receipt,plan);before=shutil.disk_usage(old).free
    removed=[]
    try:
        for row in rows:
            p=Path(row['path'])
            if p.stat().st_size!=row['size'] or sha256(p)!=row['sha256']:raise RuntimeError('Candidate changed before deletion')
            p.unlink();removed.append(row['path'])
        for path,digest in protected.items():
            if sha256(path)!=digest:raise RuntimeError('Protected historical best changed')
    finally:
        plan.update(status='complete' if len(removed)==len(rows) else 'partial',removed=removed,
            removed_logical_bytes=sum(r['size'] for r in rows if r['path'] in removed),
            uniquely_linked_blocks_removed=sum(r['blocks']*512 for r in rows if r['path'] in removed and r['links']==1),
            observed_free_disk_delta_bytes=shutil.disk_usage(old).free-before,
            free_disk_limitation='Concurrent research writes affect observed free-space delta; logical bytes are not guaranteed freed blocks',finished_utc=time.time())
        atomic_json(receipt,plan)
    print({k:plan[k] for k in ['status','removed_logical_bytes','uniquely_linked_blocks_removed','observed_free_disk_delta_bytes']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();run(args.root.resolve())
