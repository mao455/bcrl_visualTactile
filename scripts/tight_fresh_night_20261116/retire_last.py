"""Retire only named inactive final optimizer checkpoints; every selected best stays."""
import argparse,json,time,os
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
PATHS=[f'tight_dagger_matched_20261112_01/round_{r:02d}/{g}/dp/training' for r in (1,2) for g in ('QG01','G01')]
PATHS+=['narrow_state_20261105_02/training']
PATHS+=[f'{base}/{r}/training' for base in ('tight_fresh_20261115_04/round_14','tight_fresh_20261114_02/round_05') for r in 'AB']
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
if not a.apply:
 a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for rel in PATHS:
  d=BASE/rel;f=d/'last.pt';best=d/'best.pt';finish=d/'finished.json'
  assert f.is_file() and not f.is_symlink() and best.is_file() and finish.is_file()
  j=json.loads(finish.read_text());assert j.get('complete') is True or j.get('target_reached') is True
  s=f.stat();rows.append(dict(path=str(f),bytes=s.st_size,sha256=sha256(f),inode=s.st_ino,mtime_ns=s.st_mtime_ns,
   best=str(best),best_sha256=sha256(best),finished_sha256=sha256(finish)))
 atomic_json(a.output/'inventory.json',dict(utc=time.time(),rows=rows,total_bytes=sum(r['bytes'] for r in rows),
  scope='Only completed inactive final optimizer checkpoints, all selected best weights/raw/cache/current13 checkpoints retained'))
else:
 assert not (a.output/'applied.json').exists();j=json.loads((a.output/'inventory.json').read_text())
 # Explicitly reject any live command naming an inactive source, excluding this script.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except OSError:continue
  assert not any(str(BASE/x) in cmd for x in PATHS),cmd
  try:
   for fd in (proc/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert target not in {row['path'] for row in j['rows']},target
  except PermissionError:continue
 for row in j['rows']:
  f=Path(row['path']);s=f.stat()
  assert s.st_ino==row['inode'] and s.st_mtime_ns==row['mtime_ns'] and s.st_size==row['bytes']
  assert sha256(row['best'])==row['best_sha256']
 removed=[]
 try:
  for row in j['rows']:Path(row['path']).unlink();removed.append(row['path'])
 finally:atomic_json(a.output/'applied.json',dict(utc=time.time(),removed=removed,complete=len(removed)==len(j['rows']),
  removed_bytes=sum(row['bytes'] for row in j['rows'] if row['path'] in removed)))
print(json.dumps(dict(phase='applied' if a.apply else 'prepared',output=str(a.output))))
