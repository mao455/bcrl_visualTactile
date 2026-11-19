"""Delete only the reviewed manifest's verified, regenerable cache files.

No recursive delete, no other project, no model/raw file, no followed symlink.
Every operation has a durable intent and outcome. A stopped recovery supervisor
prevents training from starting halfway through asset verification.
"""
import hashlib,json,os,shutil,time
from pathlib import Path
P=Path.cwd();OUT=P/'research_docs/evidence/tight_disk_cleanup_20261119'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*2**20):h.update(b)
 return h.hexdigest()
def closed(p):
 identity=(p.stat().st_dev,p.stat().st_ino)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:
     s=fd.stat()
     if (s.st_dev,s.st_ino)==identity:raise RuntimeError(f'Open file {p}, PID {proc.name}')
    except FileNotFoundError:pass
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
hold=read(OUT/'supervisor_hold.json');pid=hold['pid'];proc=Path('/proc')/str(pid)
assert proc.joinpath('stat').read_text().split()[21]==hold['process_start_ticks']
assert '\nState:\tT ' in proc.joinpath('status').read_text()
assert not proc.joinpath(f'task/{pid}/children').read_text().strip()
manifest=read(OUT/'cleanup_manifest.json');assets=read(OUT/'protected_and_rebuild_hashes.json');assert assets['passed']
protected={str(Path(r['path']).resolve()) for r in assets['rows']}
for r in assets['rows']:
 s=Path(r['path']).stat();assert (s.st_ino,s.st_mtime_ns,s.st_size)==(r['inode'],r['mtime_ns'],r['size'])
for path,digest in manifest['code_sha256'].items():assert sha(Path(path))==digest
before=shutil.disk_usage(P).free;started=time.time();results=[]
with (OUT/'cleanup_operations.jsonl').open('x') as log:
 def record(row):log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
 for r in manifest['rows']:
  p=Path(r['path']);assert r['category']==1 and not p.is_symlink()
  assert str(p.resolve()) not in protected and p.resolve().is_relative_to(P/'outputs')
  assert p.resolve()==p and p.stat().st_dev==P.stat().st_dev
  assert r['type'] in ('retired_rebuildable_frozen_DINO_visual_cache_archive','completed_cache_rebuild_probe_output')
  if r['type'].startswith('retired_'):
   assert p.parent.name=='perception' and p.suffixes[-2:]==['.pt','.zst']
  else:assert p.parent==P/'outputs/disk_audit_20261119/rebuild_probe'
  s=p.stat();assert (s.st_ino,s.st_mtime_ns,s.st_size,s.st_nlink)==(r['inode'],r['mtime_ns'],r['bytes'],1)
  assert sha(p)==r['sha256'];closed(p)
  record(dict(r,action='intent',utc=time.time()))
  p.unlink()
  result=dict(path=str(p),action='deleted_verified_derived_cache',utc=time.time(),allocated_released=r['allocated'],sha256=r['sha256'],absent=not p.exists())
  record(result);results.append(result)
  if r['type'].startswith('retired_'):
   # Keep the original archival receipt unchanged; explicit new status alongside it.
   note=p.with_suffix('.retired.json')
   note.write_text(json.dumps(dict(result,source=r['source'],source_sha256=r['source_sha256'],recipe_row=r['recipe_row'],rebuild_tool=str(OUT/'rebuild_visual_cache.py'),reason=r['reason'],original_archive_restore_unavailable=True),indent=2)+'\n')
summary=dict(utc=time.time(),started=started,files=len(results),free_before_bytes=before,free_after_bytes=shutil.disk_usage(P).free,
 attributable_freed_allocated_bytes=sum(r['allocated_released'] for r in results),other_projects_changed=False,raw_data_deleted=0,checkpoints_deleted=0,current_caches_deleted=0,optimizer_updates=0,manifest_sha256=sha(OUT/'cleanup_manifest.json'))
(OUT/'cleanup_result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
