"""Final identities, causal capture denominators, actual owner times and resources."""
from pathlib import Path
import json,csv,time,hashlib,datetime
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';D=P/'outputs/diagnostics/tight_correction_20261127_01';read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
models=[]
for name,pair in read(R/'registration.json')['pairs'].items():
 for key in ('frontend','backend'):
  p=Path(pair[key]);actual=sha(p);assert actual==pair[key+'_sha256'];models.append(dict(policy=name,component=key,path=str(p),sha256=actual,bytes=p.stat().st_size,preserved=True))
for condition in ('old_only','with_corrections'):
 root=R/('short_'+condition);assert read(root/'short_complete.json')
 for role in 'AB':
  for component,p in [('frontend',root/role/'training/best.pt'),('backend',root/role/'backend'/role/'training/best.pt')]:models.append(dict(policy=role+'_'+condition,component=component,path=str(p),sha256=sha(p),bytes=p.stat().st_size,preserved=True))
atomic_json(D/'final_model_identities.json',dict(rows=models,original_four_preserved=True,second_repository_writes=0,terminal64_used_for_selection=False))
rows=[]
for p in sorted((R/'current_takeovers').rglob('targeted_stage_audit.json')):
 directory=p.parent;case=next(c for c in read(directory.parent/'plan.json')['cases'] if c['name']==directory.name);res=read(directory/'result.json');a=read(p);audit=read(directory/'capture_audit.json');causal=a['causal_trigger']['receipt'];switch=a['causal_trigger'].get('actual_switch_tick');valid=audit.get('valid_expert_ticks',0)
 rows.append(dict(case=directory.name,role=directory.name[0],family=case['spec']['family'],stage=directory.parent.name.split('_')[0],software_complete=res['software_complete'],strict_success=res['success'],actual_handoff=switch is not None,actual_switch_tick=switch,first_eligible_completed_tick=causal.get('first_eligible_completed_tick'),ever_entered_15mm=causal.get('ever_entered_15mm'),held12_at_trigger=(causal.get('at_trigger') or {}).get('held12'),near_root_at_trigger_m=(causal.get('at_trigger') or {}).get('root_m'),valid_expert_ticks=valid,admitted=valid>0 and res['success'],expert_recovery=switch is not None and res['success'],source=str(directory)))
assert len(rows)==8
with (D/'current_capture_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for role in 'AB':
 rr=[r for r in rows if r['role']==role];summary[role]=dict(opportunities=len(rr),actual_handoffs=sum(r['actual_handoff'] for r in rr),complete_expert_recoveries=sum(r['expert_recovery'] for r in rr),admitted_trajectories=sum(r['admitted'] for r in rr),admitted_families=len({r['family'] for r in rr if r['admitted']}),legal_ticks=sum(r['valid_expert_ticks'] for r in rr if r['admitted']),never_entry_alignment_handoffs=sum(r['stage']=='alignment' and r['actual_handoff'] for r in rr))
atomic_json(D/'current_capture_summary.json',summary)
owners=[];intervals=[];peaks=[]
for p in sorted(R.rglob('project_job_started.json')):
 owner=p.parent;b=read(p);endp=owner/'project_complete.json' if (owner/'project_complete.json').exists() else owner/'project_failure.json';end=read(endp) if endp.exists() else None
 owners.append(dict(owner=str(owner.relative_to(P)),started=b,ended=end))
 # Original metadata keys are retained; aggregate only numeric real UTC endpoints.
 def stamp(row):
  if not isinstance(row,dict):return None
  for key in ('utc','utc_epoch','ended_utc','finished_utc','started_utc'):
   if isinstance(row.get(key),(int,float)):return float(row[key])
  return None
 start=stamp(b);finish=stamp(end)
 if start is not None and finish is not None and finish>=start:intervals.append((start,finish))
 for resource in (owner/'resources/resources.jsonl',):
  if resource.exists():
   for line in resource.read_text().splitlines():
    try:peaks.append(json.loads(line))
    except json.JSONDecodeError:pass
merged=[]
for a,b in sorted(intervals):
 if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
 else:merged.append([a,b])
atomic_json(D/'runtime_accounting.json',dict(owners=owners,certified_intervals=intervals,certified_union_seconds=sum(b-a for a,b in merged),wall_from_registration_seconds=time.time()-read(R/'registration.json')['utc'],end_utc=time.time(),resource_samples=len(peaks),resource_schema_example=peaks[-1] if peaks else None,minimum_sampled_ram_available_gib=min((x['ram_available_bytes']/2**30 for x in peaks if isinstance(x.get('ram_available_bytes'),(float,int))),default=None),maximum_sampled_gpu_used_mib=max((x['gpu_used_mib'] for x in peaks if isinstance(x.get('gpu_used_mib'),(float,int))),default=None),minimum_sampled_gpu_free_mib=min((x['gpu_free_mib'] for x in peaks if isinstance(x.get('gpu_free_mib'),(float,int))),default=None),not_gpu_hours=True,external_processes_untouched=True,zero_interference_claim=False))
print(summary)
