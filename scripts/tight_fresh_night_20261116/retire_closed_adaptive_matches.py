"""Retire only two exact duplicate Nov13 adaptive validation-best containers."""
import json,os,time,shutil
from pathlib import Path
import torch
from scripts.tight_fresh_night_20261116.retire_duplicate_last import equal
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
CLOSED=BASE/'tight_execution_learning_20261113_01/adaptive_contact_recovery_01'
OUT=BASE/'tight_fresh_20261116_02/closed_adaptive_duplicate_retirement'

def run():
 torch.set_num_threads(2)
 assert json.loads((CLOSED/'status.json').read_text())['state']=='COMPLETE'
 assert json.loads((CLOSED/'owned_cleanup.json').read_text())['verified']
 OUT.mkdir(exist_ok=False);rows=[];before=shutil.disk_usage(BASE).free
 for role in ('G01','QG01'):
  d=CLOSED/'round_01'/role/'dp/training';duplicate=d/'validation_best.pt';best=d/'best.pt'
  assert not duplicate.is_symlink() and not best.is_symlink()
  s=duplicate.stat();assert s.st_nlink==1
  a=torch.load(duplicate,map_location='cpu',mmap=True,weights_only=False)
  b=torch.load(best,map_location='cpu',mmap=True,weights_only=False)
  assert equal(a,b),'Not a complete-payload duplicate; leave it intact'
  del a,b
  rows.append(dict(duplicate=str(duplicate),retained=str(best),duplicate_sha256=sha256(duplicate),retained_sha256=sha256(best),
   device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size,allocated_bytes=s.st_blocks*512,entire_payload_equal=True))
 atomic_json(OUT/'inventory.json',dict(utc=time.time(),rows=rows,scope=__doc__))
 inodes={(r['device'],r['inode']) for r in rows}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   for fd in (p/'fd').iterdir():
    try:s=fd.stat()
    except OSError:continue
    assert (s.st_dev,s.st_ino) not in inodes,'Duplicate checkpoint still open'
  except OSError:continue
 for row in rows:
  p=Path(row['duplicate']);s=p.stat()
  assert (s.st_ino,s.st_mtime_ns,s.st_size)==(row['inode'],row['mtime_ns'],row['bytes'])
  atomic_json(p.with_name('validation_best_retired.json'),dict(**row,utc=time.time()))
  p.unlink()
 atomic_json(OUT/'finished.json',dict(complete=True,utc=time.time(),rows=rows,free_before=before,
  free_after=shutil.disk_usage(BASE).free,retired_allocated_bytes=sum(r['allocated_bytes'] for r in rows)))
 print(json.dumps(dict(complete=True,retired_gib=sum(r['allocated_bytes'] for r in rows)/2**30)))

if __name__=='__main__':run()
