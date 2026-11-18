"""Retire only named reproducible pre-layout visual shards after dependency/FD checks."""
import os,time,shutil
from pathlib import Path
from scripts.tight_recovery_20261118.run import ROOT,OUT,read,write
from scripts.tight_cycle_20261118.cycle import RUNS
from visuotactile.compact import sha256

def main():
 dest=OUT/'capacity_cleanup';dest.mkdir(exist_ok=True);assert not(dest/'applied.json').exists()
 protected=set();indexes=[]
 for r in RUNS.iterdir():
  if r.name.startswith(('tight_layout10_','tight_long_','tight_balance76_','tight_sampling76_','tight_fork_diagnosis_','tight_cycle_','tight_recovery_')):
   for p in r.rglob('cache_index.json'):
    indexes.append(str(p));d=read(p)
    protected.update(str(Path(v['path']).resolve()) for v in d.get('entries',{}).values() if isinstance(v,dict) and 'path' in v)
 roots=[RUNS/'tight_fresh_20261114_06/round_08',RUNS/'tight_fresh_20261114_05/pool07_untrained'];rows=[]
 for r in roots:
  p=r/'A/cache_index.json';data=read(r/'A/state_data/data.json');raw={v['source_sha256']:v['path'] for v in data['entries'].values()}
  for v in read(p)['entries'].values():
   f=Path(v['path'])
   if not f.exists() or f.parent.resolve()!=(r/'A/state_data/perception').resolve():continue
   assert str(f.resolve()) not in protected and not f.is_symlink()
   assert Path(raw[v['source_sha256']]).is_file() and sha256(f)==v['sha256']
   st=f.stat();rows.append(dict(path=str(f),bytes=st.st_size,sha256=v['sha256'],mtime_ns=st.st_mtime_ns,raw=raw[v['source_sha256']],index=str(p)))
   if sum(x['bytes'] for x in rows)>=12*1024**3:break
  if sum(x['bytes'] for x in rows)>=12*1024**3:break
 paths={r['path'] for r in rows};readers=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   for fd in (p/'fd').iterdir():
    try:
     if str(fd.resolve()) in paths:readers.append(str(fd))
    except (OSError,RuntimeError):pass
  except OSError:pass
 assert not readers
 before=shutil.disk_usage(RUNS).free;write(dest/'inventory.json',dict(files=rows,current_indexes=indexes,active_readers=readers,free_before=before,reason='Reproducible frozen DINO features from retired pre10cm roots only, raw/checkpoints/indexes retained; current105/new data and all current-cache dependencies excluded. Not RAM/VRAM cleanup.'))
 for row in rows:
  f=Path(row['path']);assert f.stat().st_mtime_ns==row['mtime_ns'];f.unlink()
 write(dest/'applied.json',dict(utc=time.time(),removed_files=len(rows),logical_bytes=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(RUNS).free,raw_and_all_models_preserved=True));print(read(dest/'applied.json'))
if __name__=='__main__':main()
