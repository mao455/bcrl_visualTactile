"""Authorized retirement of named reproducible old-layout frozen visual shards."""
import os,json,time,shutil,hashlib,sys
from pathlib import Path
B=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
O=Path('research_docs/evidence/tight_cycle_20261118')/('cleanup_capacity' if '--capacity' in sys.argv else 'cleanup');O.mkdir(parents=True,exist_ok=True)
assert not (O/'applied.json').exists(),'Already retired; preserve receipt'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
roots=[B/'tight_fresh_20261114_06/round_09'] if '--capacity' in sys.argv else [B/'tight_fresh_20261114_01'/s for s in ['', 'round_02','round_03','round_04']]
protected=set();indexes=[]
for root in B.iterdir():
 if root.name.startswith(('tight_layout10_', 'tight_long_', 'tight_balance76_', 'tight_sampling76_', 'tight_fork_diagnosis_')):
  for idx in root.rglob('cache_index.json'):
   v=read(idx)
   for e in v.get('entries',{}).values():
    if isinstance(e,dict) and 'path' in e:protected.add(str(Path(e['path']).resolve()))
   indexes.append(str(idx))
rows=[]
for root in roots:
 idx=root/'A/cache_index.json';data=read(root/'A/state_data/data.json');cache=read(idx)
 raw={e['source_sha256']:e['path'] for e in data['entries'].values()}
 for e in cache['entries'].values():
  f=Path(e['path'])
  if f.parent.resolve()!=(root/'A/state_data/perception').resolve():continue
  assert str(f.resolve()) not in protected and not f.is_symlink()
  assert Path(raw[e['source_sha256']]).is_file(),'Reconstruction source missing'
  assert sha(f)==e['sha256']
  st=f.stat();rows.append(dict(path=str(f),sha256=e['sha256'],bytes=st.st_size,mtime_ns=st.st_mtime_ns,raw=raw[e['source_sha256']],index=str(idx)))
paths={r['path'] for r in rows};openfiles=[];processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  processes.append(dict(pid=int(p.name),command=cmd))
  for fd in (p/'fd').iterdir():
   try:
    if str(fd.resolve()) in paths:openfiles.append(str(fd))
   except (OSError,RuntimeError):pass
 except OSError:pass
assert not openfiles
before=shutil.disk_usage(B).free
write(O/'inventory.json',dict(utc=time.time(),files=rows,total_bytes=sum(r['bytes'] for r in rows),protected_cache_paths=sorted(protected),checked_indexes=indexes,open_readers=openfiles,processes=processes,free_before=before,reason='Retired pre-10cm frozen DINO features only; raw, indexes, all checkpoints, current76 and original400 untouched. Reconstruct from retained raw with pinned encoder; not a RAM/VRAM release.'))
for r in rows:
 f=Path(r['path']);assert f.stat().st_mtime_ns==r['mtime_ns'];f.unlink()
for root in roots:write(root/'A/state_data/perception/RETIRED.json',dict(inventory=str(O.resolve()/'inventory.json'),utc=time.time(),raw_and_weights_retained=True))
write(O/'applied.json',dict(complete=True,utc=time.time(),removed_files=len(rows),logical_bytes=sum(r['bytes'] for r in rows),free_before=before,free_after=shutil.disk_usage(B).free))
print(read(O/'applied.json'))
