from pathlib import Path
import datetime,json,time,shutil
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';read=lambda p:json.loads(p.read_text())
def stamp(row):
 if isinstance(row,dict):
  for key in ['utc','utc_epoch','ended_utc','finished_utc','started_utc']:
   if isinstance(row.get(key),(int,float)):return float(row[key])
 return None
owners=[];intervals=[];samples=0;minimum_ram=None;maximum_gpu_used=None;minimum_gpu_free=None;maximum_gap=0
for p in sorted(R.rglob('project_job_started.json')):
 owner=p.parent;start=read(p);end_path=next((owner/name for name in ['project_complete.json','project_failure.json','project_exit.json'] if (owner/name).exists()),None);end=read(end_path) if end_path else None;exit_path=owner/'project_exit.json';exit_receipt=read(exit_path) if exit_path.exists() else None
 a,b=stamp(start),stamp(end)
 if a is not None and b is not None and b>=a:intervals.append((a,b))
 owners.append(dict(owner=str(owner.relative_to(P)),started=start,end=end,end_source=str(end_path) if end_path else None,exit=exit_receipt,children_reaped=bool(exit_receipt and exit_receipt.get('children_reaped'))))
 telemetry=owner/'resources/resources.jsonl';last=None
 if telemetry.exists():
  with telemetry.open() as f:
   for line in f:
    try:v=json.loads(line)
    except json.JSONDecodeError:continue
    samples+=1
    for key in ['ram_available_bytes','gpu_used_mib','gpu_free_mib']:
     if not isinstance(v.get(key),(float,int)):continue
     if key=='ram_available_bytes':minimum_ram=v[key] if minimum_ram is None else min(minimum_ram,v[key])
     elif key=='gpu_used_mib':maximum_gpu_used=v[key] if maximum_gpu_used is None else max(maximum_gpu_used,v[key])
     else:minimum_gpu_free=v[key] if minimum_gpu_free is None else min(minimum_gpu_free,v[key])
    if last is not None:maximum_gap=max(maximum_gap,v['utc_epoch']-last)
    last=v['utc_epoch']
merged=[]
for a,b in sorted(intervals):
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
seen=set();logical=allocated=files=0
for p in R.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 s=p.stat();logical+=s.st_size;files+=1;key=(s.st_dev,s.st_ino)
 if key not in seen:seen.add(key);allocated+=s.st_blocks*512
now=time.time();reg=read(R/'registration.json');space=shutil.disk_usage(R)
summary=dict(utc=now,start_utc=reg['utc'],wall_seconds=now-reg['utc'],owners=owners,certified_intervals=merged,certified_union_seconds=sum(b-a for a,b in merged),sum_owner_seconds=sum(b-a for a,b in intervals),resource_samples=samples,minimum_sampled_ram_available_gib=minimum_ram/2**30 if minimum_ram else None,maximum_sampled_gpu_used_mib=maximum_gpu_used,minimum_sampled_gpu_free_mib=minimum_gpu_free,maximum_within_owner_sample_gap_seconds=maximum_gap,not_gpu_hours=True,external_processes_modified=0,zero_interference_claim=False,footprint=dict(files=files,unique_inodes=len(seen),logical_bytes=logical,allocated_unique_inode_bytes=allocated,includes_reused_old_features=True,not_claimed_as_new_growth=True,deleted_this_round_bytes=0,available_bytes=space.free),probe_coverage='Temporary synthetic/actual-data probes have separate timing receipts; owner-monitor union excludes gaps between stages')
atomic_json(D/'runtime_accounting.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['owners','certified_intervals']}),flush=True)
