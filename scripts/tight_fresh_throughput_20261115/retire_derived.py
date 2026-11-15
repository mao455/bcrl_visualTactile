"""Retire only named, closed, reproducible backend input shards; retain raw/weights.

Prepare an immutable inventory, inspect it, then pass --apply explicitly. Current
A11/B09, original fallback05/06, all model weights, raw observations and frozen
visual features are excluded. Old cache indexes remain as historical evidence.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
NAMES=['tight_fresh_20261114_06/round_08','tight_fresh_20261115_02/round_12',
 'tight_fresh_20261115_03/round_13','tight_fresh_20261115_04/round_14']
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
protected=[]
for r in ['tight_fresh_20261115_07','tight_fresh_20261115_01/round_11','tight_fresh_20261114_06/round_09','tight_fresh_20261114_02/round_05','tight_fresh_20261114_05/round_06']:
 for role in ['A','B']:protected.append((BASE/r/role/'backend'/role/'nominal').resolve())
if not a.apply:
 a.output.mkdir(exist_ok=False);rows=[];indexes=[]
 for name in NAMES:
  root=BASE/name
  for role in ['A','B']:
   rear=root/role/'backend'/role;directory=rear/'nominal'
   assert read(rear/'training/finished.json')['target_reached'] is True
   assert (rear/'training/best.pt').is_file() and (root/role/'training/best.pt').is_file()
   assert directory.resolve() not in protected
   index=read(directory/'index.json');indexes.append(dict(path=str(directory/'index.json'),sha256=sha(directory/'index.json')))
   for identity,entry in index['episodes'].items():
    assert Path(identity).name==identity
    f=directory/(identity+'.pt');assert f.is_file() and not f.is_symlink()
    st=f.stat();digest=sha(f);assert digest==entry['sha256']
    rows.append(dict(path=str(f),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest))
 write(a.output/'inventory.json',dict(created=time.time(),kind='recomputable_selected_DP_backend_inputs_only',
  protected=[str(p) for p in protected],indexes=indexes,files=rows,total_bytes=sum(r['bytes'] for r in rows),
  reconstruction='Use retained original frozen source, selected frontend, raw/numeric and frozen visual histories to rerun residual-cache into a new output. Historical indexes/hashes are kept; this is cache eviction, not exact serialized-byte restoration.'))
 print(json.dumps(dict(files=len(rows),gib=sum(r['bytes'] for r in rows)/2**30)))
else:
 inv=read(a.output/'inventory.json');assert not (a.output/'applied.json').exists()
 eligible={(BASE/n/role/'backend'/role/'nominal').resolve() for n in NAMES for role in ['A','B']}
 for i in inv['indexes']:assert sha(Path(i['path']))==i['sha256']
 # No current reader may have a selected shard open, even through a symlink.
 paths={r['path'] for r in inv['files']}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   for fd in (proc/'fd').iterdir():
    try:assert str(fd.resolve()) not in paths,'Selected derived shard has an active reader'
    except (FileNotFoundError,PermissionError):pass
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 for row in inv['files']:
  f=Path(row['path']);assert f.parent.resolve() in eligible and f.parent.resolve() not in protected
  st=f.stat();assert not f.is_symlink() and st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns']
 for index in inv['indexes']:
  write(Path(index['path']).parent/'DERIVED_CACHE_RETIRED.json',dict(inventory=str((a.output/'inventory.json').resolve()),
   inventory_sha256=sha(a.output/'inventory.json'),utc=time.time(),raw_trajectories_retained=True,all_weights_retained=True,
   selected_A11_B09_and_fallback_caches_retained=True))
 removed=[]
 try:
  for row in inv['files']:
   Path(row['path']).unlink();removed.append(row['path'])
 finally:
  write(a.output/'applied.json',dict(complete=len(removed)==len(inv['files']),utc=time.time(),removed=removed,
   removed_bytes=sum(r['bytes'] for r in inv['files'] if r['path'] in removed)))
 print(json.dumps(dict(removed=len(removed),gib=inv['total_bytes']/2**30)))
