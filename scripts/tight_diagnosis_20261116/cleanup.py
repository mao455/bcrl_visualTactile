"""Explicit manifest-driven retirement; never select raw, visual, selected model tensors or recent diagnostics."""
import json,os,sys,time,shutil,hashlib,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
B=Path('outputs/visuotactile/run').resolve();OUT=B/'tight_diagnostic_20261116_01/cleanup'
PUB=Path('research_docs/evidence/tight_diagnosis_20261116')
read=lambda p:json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def st(p):
 s=p.stat();return dict(bytes=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,links=s.st_nlink)
def protected_roots():
 roots=[B/'tight_fresh_20261115_01/round_11/A',B/'tight_fresh_20261116_00/round_15/B']
 roots += [p for p in B.glob('tight_fresh_20261116_*')]
 return roots
def unused(paths):
 names={str(p) for p in paths};inodes={(p.stat().st_dev,p.stat().st_ino) for p in paths}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:s=fd.stat()
    except OSError:continue
    if (s.st_dev,s.st_ino) in inodes:raise RuntimeError('Open file '+str(fd))
   maps=(proc/'maps').read_text()
   if any(name in maps for name in names):raise RuntimeError('Mapped candidate '+proc.name)
  except (OSError,PermissionError):continue

def plan():
 assert not (OUT/'plan.json').exists()
 protected=protected_roots();protected_files={p.resolve() for root in protected for p in root.rglob('*.pt') if p.is_file()}
 # Current shards may physically live in older roots. Resolve every shared link.
 data=B/'tight_fresh_20261116_04/round_19/A/state_data/data.json';pool=read(data)
 protected_files.update(Path(v['path']).resolve() for v in pool['entries'].values())
 links={p.resolve() for p in B.rglob('*') if p.is_symlink() and p.exists()}
 # Keep all core selected model tensors. Retire only obsolete optimizer state,
 # unselected endpoints, and regenerable learned inputs with surviving provenance.
 rows=[];groups=collections.defaultdict(list)
 for p in B.rglob('*.pt'):
  if p.is_symlink() or p.resolve() in protected_files or any(p.is_relative_to(r) for r in protected):continue
  s=p.stat()
  if p.parent.name=='nominal' and (p.parent/'index.json').exists():
   # Nominal records bind exact DP and parameters in index. Preserve that index.
   index=read(p.parent/'index.json');base=index.get('base_checkpoint')
   if not base:
    cfg=p.parent.parent/'training/configuration.json'
    base=read(cfg).get('base_checkpoint') if cfg.exists() else None
   if base and Path(base).exists() and p.resolve() not in links:rows.append(dict(path=str(p),kind='regenerable_old_dp_inputs',rebuild_index=str(p.parent/'index.json'),base_checkpoint=base,**st(p)))
  elif p.parent.name=='training' and p.name in ('best.pt','last.pt'):
   if (p.parent/'finished.json').exists():groups[(s.st_dev,s.st_ino)].append(p)
  elif p.parent.name=='training' and (p.name.startswith('comparison_') or p.name=='last.weights_only.pt'):
   # Distinct unselected historical states are explicitly retired; selected best retained.
   if p.resolve() not in links and ((p.parent/'best.pt').exists() or (p.parent/'best.weights_only.pt').exists()):rows.append(dict(path=str(p),kind='obsolete_unselected_checkpoint',**st(p)))
 for paths in groups.values():
  if len(paths)!=paths[0].stat().st_nlink or any(p.resolve() in links for p in paths):continue
  # Rewriting weights changes SHA. Explicit archive + provenance, never silent replacement.
  src=next((p for p in paths if p.name=='best.pt'),paths[0]);dst=src.with_name(src.stem+'.weights_only.pt')
  if dst.exists():continue
  rows.append(dict(path=str(src),paths=[str(p) for p in paths],kind='retire_optimizer_only',archive=str(dst),**st(src)))
 paths=[Path(x) for r in rows for x in r.get('paths',[r['path']])];unused(paths)
 protected_models={str(p):sha(p) for p in protected_files if p.name in ('best.pt','last.pt')}
 inventory=collections.Counter();types=collections.Counter();seen=set()
 for p in B.parent.rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  s=p.stat();key=(s.st_dev,s.st_ino)
  if key in seen:continue
  seen.add(key);inventory[str(p.relative_to(B.parent).parts[0])]+=s.st_blocks*512;types[p.suffix]+=s.st_blocks*512
 payload=dict(status='planned',utc=time.time(),free_before=shutil.disk_usage(B).free,rows=rows,
  protected_models=protected_models,pool_manifest=str(data),pool_manifest_sha256=sha(data),pool_entries=pool['entries'],
  disk_allocated_by_top_directory=dict(inventory),disk_allocated_by_extension=dict(types),
  policy='Preserve all raw HDF5/partial/npz, all354 current frozen shards, every selected model/EMA tensor, configs, curves, evaluations. All16-19/sourceA11/B15 and fixed-fit diagnostics untouched. Retire only optimizer history or unselected endpoints and exact-version rebuildable inputs. Historical optimizer resumption intentionally unavailable.')
 write(OUT/'plan.json',payload)
 print(json.dumps(dict(rows=len(rows),logical_selected_gib=sum(r['bytes'] for r in rows)/2**30,kinds=dict(collections.Counter(r['kind'] for r in rows)),free_gib=payload['free_before']/2**30)))

def execute():
 import torch,h5py,importlib.util
 torch.set_num_threads(2);p=read(OUT/'plan.json');assert not (OUT/'finished.json').exists()
 spec=importlib.util.spec_from_file_location('eq',Path('scripts/tight_fresh_night_20261116/retire_duplicate_last.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 paths=[Path(x) for r in p['rows'] for x in r.get('paths',[r['path']])];unused(paths);done=[]
 for r in p['rows']:
  path=Path(r['path']);assert st(path)=={k:r[k] for k in st(path)};oldsha=sha(path)
  if r['kind']=='retire_optimizer_only':
   saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
   if 'optimizer' not in saved:del saved;continue
   payload={k:v for k,v in saved.items() if k!='optimizer'};payload['archival']=dict(original_sha256=oldsha,original_paths=r['paths'],optimizer_resume_available=False)
   dst=Path(r['archive']);tmp=dst.with_suffix('.tmp');assert not tmp.exists() and shutil.disk_usage(B).free>r['bytes']+(12<<30)
   torch.save(payload,tmp);checked=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
   assert all(mod.equal(v,checked[k]) for k,v in payload.items());del checked,payload,saved
   with tmp.open('rb') as f:os.fsync(f.fileno())
   os.replace(tmp,dst);item=dict(r,original_sha256=oldsha,archive_sha256=sha(dst),archive_bytes=dst.stat().st_size,nonoptimizer_fields_exact=True)
   for s in r['paths']:write(Path(s).with_name(Path(s).stem+'_archival.json'),item)
   for s in r['paths']:Path(s).unlink()
  else:
   item=dict(r,sha256=oldsha);path.unlink()
  done.append(item);write(OUT/'progress.json',dict(rows=done,free_current=shutil.disk_usage(B).free))
  print(r['kind'],len(done),round(shutil.disk_usage(B).free/2**30,2),flush=True)
 assert all(sha(Path(k))==v for k,v in p['protected_models'].items())
 # Every current successful trajectory remains readable, including real labels.
 checks=[]
 for identity,entry in p['pool_entries'].items():
  with h5py.File(entry['path'],'r') as f:
   g=f['episodes'][entry['episode']];n=len(g['obs/time_s']);assert n>0
   q=g['transition/target_after_rad'];assert q.shape==(n,29);assert bool(g.attrs['complete'])
   checks.append(dict(identity=identity,path=entry['path'],episode=entry['episode'],frames=n,first_last_target_finite=bool(__import__('numpy').isfinite(q[[0,n-1]]).all())))
 assert all(r['first_last_target_finite'] for r in checks)
 for k in p['protected_models']:
  x=torch.load(k,map_location='cpu',mmap=True,weights_only=False);assert isinstance(x['model'],dict) and x['model'];del x
 result=dict(complete=True,utc=time.time(),free_before=p['free_before'],free_after=shutil.disk_usage(B).free,rows=done,protected_hashes_unchanged=True,current_pool_readable=len(checks),pool_checks=checks)
 result['free_increase_bytes']=result['free_after']-result['free_before'];write(OUT/'finished.json',result)
 write(PUB/'cleanup.json',{k:v for k,v in result.items() if k not in ('pool_checks',)});write(PUB/'cleanup_inventory.json',{k:p[k] for k in ('disk_allocated_by_top_directory','disk_allocated_by_extension','policy','pool_manifest_sha256')})
 print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pool_checks')}))
if __name__=='__main__':plan() if sys.argv[1]=='plan' else execute()
