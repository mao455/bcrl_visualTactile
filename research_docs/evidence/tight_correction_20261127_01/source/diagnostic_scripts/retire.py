"""Apply only previously identity-audited retirements after all scientific consumers exit."""
from pathlib import Path
import json,hashlib,subprocess,csv,time,os
P=Path.cwd();D=P/'outputs/diagnostics/tight_correction_20261127_01';R=P/'outputs/visuotactile/run/tight_correction_20261127_01'
assert (D/'science_complete.json').exists(),'Do not alter hashed source while any scientific worker runs'
assert not (D/'retired_paths.csv').exists(),'Retirement has already been applied; inspect original ledger'
for owner in R.rglob('project_job_started.json'):
 exitp=owner.parent/'project_exit.json'
 assert exitp.exists() and json.loads(exitp.read_text()).get('children_reaped'),str(owner)
items=json.loads((D/'retirement_candidate_audit.json').read_text());assert len(items)==30
start=time.time();fs_before=os.statvfs(P);rows=[]
for r in items:
 p=P/r['path'];row=dict(path=r['path'],git_blob=r['git_blob'],sha256=r['sha256'],logical_bytes=r.get('bytes',r.get('logical_bytes')),reason=r.get('reason',r.get('action')),canonical=r.get('canonical','git:60e7b1f:'+r['path']),birth_time='unknown',mtime='unknown',inode='unknown',links='unknown',allocated_bytes='unknown',deleted_this_apply=False)
 if p.exists():
  assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
  s=p.stat();birth=subprocess.check_output(['stat','-c','%w','--',str(p)],text=True).strip()
  row.update(birth_time=birth if birth!='-' else 'unknown',mtime=s.st_mtime,inode=s.st_ino,links=s.st_nlink,allocated_bytes=s.st_blocks*512,deleted_this_apply=True)
  assert r['path'].startswith('tools/'),'Only five deferred tools may still exist'
  p.unlink()
 else:assert r['path'].startswith('research_docs/evidence/'),'Unexpected missing canonical tool'
 assert not p.exists();rows.append(row)
assert sum(r['deleted_this_apply'] for r in rows)==5
with (D/'retired_paths.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fs_after=os.statvfs(P)
(D/'retirement_completed.json').write_text(json.dumps(dict(utc=start,paths=30,deferred_tools_deleted_now=5,duplicate_snapshots_previously_deleted_this_task=25,logical_bytes=sum(r['logical_bytes'] for r in rows),known_unlinked_file_allocated_bytes=sum(r['allocated_bytes'] for r in rows if isinstance(r['allocated_bytes'],int)),filesystem_available_change_bytes=(fs_after.f_bavail-fs_before.f_bavail)*fs_after.f_frsize,filesystem_change_attribution='Concurrent system activity; not a causal release measurement',previous_release_gib_not_counted=[10.468,407.722],unknown_prior_snapshot_file_metadata=True),indent=2)+'\n')
print('Retired all30 exact registered paths;5 deferred tools unlinked after consumers closed')
