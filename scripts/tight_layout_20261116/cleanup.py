"""Retire only closed unsuccessful RGB-D captures with preserved numeric evidence.

Never treat student failure as corruption. Complete success and all manifest
sources are protected; representative failures remain per run/reason.
"""
import os,json,time,shutil,hashlib,sys
from pathlib import Path
from collections import Counter
B=Path('outputs/visuotactile/run').resolve()
OUT=B/'tight_layout_20261116_01/cleanup'
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def stat(p):
 s=p.stat();return dict(bytes=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,links=s.st_nlink)
def unused(paths):
 ino={(p.stat().st_dev,p.stat().st_ino) for p in paths}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:s=fd.stat()
    except OSError:continue
    if (s.st_dev,s.st_ino) in ino:raise RuntimeError('Active file: '+str(fd))
   maps=(proc/'maps').read_text()
   if any(str(p) in maps for p in paths):raise RuntimeError('Mapped file: '+proc.name)
  except (OSError,PermissionError):pass
if sys.argv[1]=='plan':
 assert not (OUT/'plan.json').exists()
 protected=set()
 for m in B.rglob('data.json'):
  if 'state_data' not in m.parts:continue
  try:d=read(m)
  except (ValueError,OSError):continue
  protected.update(str(Path(e['path']).resolve()) for e in d.get('entries',{}).values() if 'path' in e)
 rows=[];kept=[];counts=Counter()
 for p in sorted(B.rglob('result.json')):
  try:r=read(p)
  except (ValueError,OSError):continue
  if r.get('software_complete') is not True or r.get('success') is not False or not r.get('legal'):continue
  f=p.parent/'demonstration/data.h5';numeric=p.parent/'numeric.h5'
  if not f.is_file() or not numeric.is_file() or f.is_symlink() or str(f.resolve()) in protected:continue
  reason=','.join(k for k,v in r.get('reasons',{}).items() if v) or 'other'
  key=(p.relative_to(B).parts[0],reason);counts[key]+=1
  if counts[key]<=2:kept.append(str(f));continue
  # Raw numeric/control, outcome, configuration and any video remain available.
  rows.append(dict(path=str(f),kind='closed_task_failure_rgbd_only',reason=reason,result=str(p),result_sha256=sha(p),result_summary=r,numeric=str(numeric),numeric_bytes=numeric.stat().st_size,**stat(f)))
 unused([Path(r['path']) for r in rows]);write(OUT/'plan.json',dict(utc=time.time(),free_before=shutil.disk_usage(B).free,rows=rows,kept_representative=kept,protected_manifest_sources=len(protected),policy=__doc__))
 print(json.dumps(dict(files=len(rows),gib=sum(r['allocated'] for r in rows)/2**30,representatives=len(kept),reasons={str(k):v for k,v in counts.items()})))
else:
 p=read(OUT/'plan.json');assert not (OUT/'finished.json').exists();paths=[Path(r['path']) for r in p['rows']];unused(paths)
 for r in p['rows']:
  f=Path(r['path']);assert stat(f)=={k:r[k] for k in stat(f)};assert sha(Path(r['result']))==r['result_sha256'];assert Path(r['numeric']).stat().st_size==r['numeric_bytes']
 done=[]
 for r in p['rows']:
  f=Path(r['path']);digest=sha(f)
  item=dict(r,sha256=digest,retired_utc=time.time());write(f.parent/'RGBD_RETIRED.json',item);f.unlink();done.append(item)
  with (OUT/'deleted.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n');stream.flush();os.fsync(stream.fileno())
 report=dict(complete=True,rows=done,free_before=p['free_before'],free_after=shutil.disk_usage(B).free,logical_retired_bytes=sum(r['bytes'] for r in done),protected_manifest_sources=p['protected_manifest_sources'],representatives_retained=len(p['kept_representative']))
 report['free_increase_bytes']=report['free_after']-report['free_before'];write(OUT/'finished.json',report)
 write(Path('research_docs/evidence/tight_layout_20261116/cleanup.json'),{k:v for k,v in report.items() if k!='rows'})
 print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
