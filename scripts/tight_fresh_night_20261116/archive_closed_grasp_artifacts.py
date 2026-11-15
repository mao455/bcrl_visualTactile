"""Bounded retirement of obsolete Adam states and lossless old plan-cache archives.

Raw trajectories, frozen visual features, all model/EMA tensors, retained
A11/B09/B15/05/06 pairs and selected18 checkpoints are protected. Old16/17
optimizer states are not resumable after retirement; their full models remain.
Plan-cache archives restore the exact original bytes and original paths.
"""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261116_03'
OUT=ROOT/'closed_grasp_archives'
read=lambda p:json.loads(p.read_text())

def space(extra=0):
 if shutil.disk_usage(ROOT).free<(16<<30)+extra:raise RuntimeError('Archive temporary space would violate16GiB reserve')

def closed_files_unused(paths):
 names={str(p.resolve()) for p in paths}
 for process in Path('/proc').iterdir():
  if not process.name.isdigit() or int(process.name)==os.getpid():continue
  try:
   for fd in (process/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert target not in names,'File still open: '+target
   maps=(process/'maps').read_text()
   assert not any(name in maps for name in names if name.endswith('best.pt') or name.endswith('last.pt'))
  except (OSError,PermissionError):continue

def restore(manifest):
 m=read(manifest);archive=Path(m['archive']);directory=Path(m['directory'])
 assert sha256(archive)==m['archive_sha256']
 missing=[row for row in m['members'] if not (directory/row['name']).exists()]
 space(sum(row['bytes'] for row in missing))
 with zipfile.ZipFile(archive) as source:
  for row in m['members']:
   assert Path(row['name']).name==row['name']
   path=directory/row['name']
   if path.exists():assert sha256(path)==row['sha256'];continue
   temp=path.with_suffix('.restore.tmp');assert not temp.exists()
   with source.open(row['name']) as a,temp.open('xb') as b:
    shutil.copyfileobj(a,b,1<<20);b.flush();os.fsync(b.fileno())
   assert sha256(temp)==row['sha256'];os.chmod(temp,row['mode'])
   os.utime(temp,ns=(row['mtime_ns'],row['mtime_ns']));os.replace(temp,path)
 atomic_json(manifest.with_suffix('.restored.json'),dict(complete=True,utc=time.time(),restored=len(missing)))

def run():
 import torch
 torch.set_num_threads(2)
 old=BASE/'tight_fresh_20261116_02';state=read(old/'night/state.json')
 assert state['safe_stop'] and [r['index'] for r in state['rounds']]==[17,18]
 assert (old/'monitor_exit.json').exists() and (ROOT/'monitor_exit.json').exists()
 assert read(ROOT/'fixed_fit_owner/finished.json')['software_complete']
 assert not OUT.exists();before=shutil.disk_usage(ROOT).free
 spec=importlib.util.spec_from_file_location('archive_equality',Path(__file__).with_name('retire_duplicate_last.py'))
 eq=importlib.util.module_from_spec(spec);spec.loader.exec_module(eq)
 selected=[BASE/'tight_fresh_20261114_06/round_08'/r/'training/best.pt' for r in 'AB']
 selected += [BASE/root/r/'training'/n for root in ('tight_fresh_20261116_01/round_16','tight_fresh_20261116_02/round_17') for r in 'AB' for n in ('best.pt','last.pt')]
 selected += [old/'round_18/A/training/last.pt']
 cache_roots=[BASE/'tight_fresh_20261115_01/round_11/B/backend/B/nominal',BASE/'tight_fresh_20261116_00/round_15/A/backend/A/nominal']
 cache_roots += [BASE/root/r/'backend'/r/'nominal' for root in ('tight_fresh_20261116_01/round_16','tight_fresh_20261116_02/round_17') for r in 'AB']
 protected=[old/'round_18'/r/'training/best.pt' for r in 'AB']
 protected += [BASE/path/r/'training/best.pt' for path,roles in [('tight_fresh_20261115_01/round_11','A'),('tight_fresh_20261114_06/round_09','B'),('tight_fresh_20261114_02/round_05','AB'),('tight_fresh_20261114_05/round_06','AB')] for r in roles]
 protected.append(BASE/'tight_fresh_20261116_00/round_15/B/training/best.pt')
 # Most recent data/weights remain ordinary, immediately deployable snapshots.
 protected=[p.resolve() for p in protected];bindings={str(p):sha256(p) for p in protected}
 groups={};inventory=[];cache_inventory=[]
 for path in selected:
  assert path.is_file() and not path.is_symlink() and path.resolve() not in protected
  s=path.stat();done=read(path.parent/'finished.json');assert done['complete']
  key=(s.st_dev,s.st_ino);groups.setdefault(key,[]).append(path)
 for paths in groups.values():
  source=paths[0];s=source.stat();assert s.st_nlink==len(paths),'Unknown checkpoint hard-link owner'
  inventory.append(dict(paths=[str(p) for p in paths],sha256=sha256(source),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
 for directory in cache_roots:
  index=read(directory/'index.json');members=[]
  for identity,item in index['episodes'].items():
   assert Path(identity).name==identity
   path=directory/(identity+'.pt');s=path.stat();assert path.is_file() and not path.is_symlink() and s.st_nlink==1
   members.append(dict(name=path.name,sha256=item['sha256'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode&0o777))
  cache_inventory.append(dict(directory=str(directory),index_sha256=sha256(directory/'index.json'),members=members))
 closed_files_unused(selected+[Path(c['directory'])/m['name'] for c in cache_inventory for m in c['members']])
 OUT.mkdir(exist_ok=False)
 atomic_json(OUT/'inventory.json',dict(utc=time.time(),optimizer_groups=inventory,caches=cache_inventory,protected=bindings,scope=__doc__))
 reduced=[];archived=[]
 try:
  for row in inventory:
   source=Path(row['paths'][0]);space(row['bytes']);archive=source.with_name(source.stem+'.weights_only.pt');temp=archive.with_suffix('.tmp')
   assert not archive.exists() and not temp.exists()
   saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False);assert 'optimizer' in saved
   payload={k:v for k,v in saved.items() if k!='optimizer'}
   payload['archival']=dict(original_sha256=row['sha256'],original_paths=row['paths'],optimizer_resume_available=False,
    preserved='Every model/EMA tensor and every nonoptimizer field; new explicit archive identity')
   torch.save(payload,temp)
   with temp.open('rb') as f:os.fsync(f.fileno())
   checked=torch.load(temp,map_location='cpu',mmap=True,weights_only=False)
   assert all(eq.equal(v,checked[k]) for k,v in payload.items())
   del checked,payload,saved;os.replace(temp,archive)
   proof=dict(row,archive=str(archive),archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size,all_nonoptimizer_fields_exact=True)
   for name in row['paths']:
    path=Path(name);s=path.stat();assert (s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(row['device'],row['inode'],row['mtime_ns'],row['bytes'])
    atomic_json(path.with_name(path.stem+'_archival.json'),proof)
   for name in row['paths']:Path(name).unlink()
   reduced.append(proof);atomic_json(OUT/'progress.json',dict(optimizer_groups=reduced,cache_archives=archived))
   print(json.dumps(dict(optimizer_archive=str(archive),free_gib=shutil.disk_usage(ROOT).free/2**30)),flush=True)
  for number,item in enumerate(cache_inventory):
   directory=Path(item['directory']);space(sum(m['bytes'] for m in item['members']))
   archive=OUT/('cache_%02d.zip'%number);temp=archive.with_suffix('.tmp')
   with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as output:
    for row in item['members']:
     path=directory/row['name'];raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
     output.writestr(row['name'],raw)
   with temp.open('rb') as f:os.fsync(f.fileno())
   with zipfile.ZipFile(temp) as check:
    for row in item['members']:assert hashlib.sha256(check.read(row['name'])).hexdigest()==row['sha256']
   os.replace(temp,archive)
   manifest=OUT/('cache_%02d.json'%number);proof=dict(item,archive=str(archive),archive_sha256=sha256(archive),
    archive_bytes=archive.stat().st_size,verified_original_bytes=True,restore_command='python '+str(Path(__file__).resolve())+' restore --manifest '+str(manifest))
   atomic_json(manifest,proof);atomic_json(directory/'ARCHIVED_INPUTS.json',dict(manifest=str(manifest),**{k:proof[k] for k in ('archive','archive_sha256','restore_command')}))
   for row in item['members']:
    path=directory/row['name'];s=path.stat();assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'];path.unlink()
   archived.append(str(manifest));atomic_json(OUT/'progress.json',dict(optimizer_groups=reduced,cache_archives=archived))
   print(json.dumps(dict(cache_archive=str(archive),free_gib=shutil.disk_usage(ROOT).free/2**30)),flush=True)
  assert all(sha256(path)==digest for path,digest in bindings.items())
  atomic_json(OUT/'finished.json',dict(complete=True,utc=time.time(),optimizer_groups=reduced,cache_archives=archived,
   free_before=before,free_after=shutil.disk_usage(ROOT).free,protected_sha256_unchanged=True,
   optimizer_bytes_retired=sum(r['bytes']-r['archive_bytes'] for r in reduced)))
 except BaseException as error:
  atomic_json(OUT/'failure.json',dict(error=repr(error),utc=time.time(),optimizer_groups=reduced,cache_archives=archived));raise

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['run','restore']);p.add_argument('--manifest',type=Path);a=p.parse_args()
 if a.mode=='restore':restore(a.manifest)
 else:run()
