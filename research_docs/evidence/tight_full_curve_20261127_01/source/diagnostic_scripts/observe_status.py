from pathlib import Path
import json,time
D=Path('outputs/diagnostics/tight_full_curve_20261127_01');R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01');read=lambda p:json.loads(p.read_text())
status={'utc':round(time.time()),'rear':{},'evaluation':{}}
for c in ['old_only','plus_corrected24']:
 for role in 'AB':
  for ep in [30,60]:
   directory=R/c/f'rear{ep}'/role/'backend'/role/'training'
   fixed=directory/'fixed_final_selection.json';p=directory/'progress.json'
   if fixed.exists():continue
   if p.exists():
    v=read(p);status['rear'][c+'_'+role+'_'+str(ep)]={'epoch':v['epoch'],'update':v['update']}
for owner in list(R.glob('evaluation_*'))+list(R.glob('final16_*')):
 if any((owner/n).exists() for n in ['project_complete.json','project_failure.json','project_exit.json']):continue
 p=owner/'rollouts/batch_progress.json'
 if p.exists():
  v=read(p);status['evaluation'][owner.name]={'completed':v['closed_results'],'remaining':v['remaining_tasks'],'tick':v['tick']}
status['cache']={}
for c in ['old_only','plus_corrected24']:
 p=R/c/'rear60/B/backend/cache_progress.json'
 if p.exists() and not (R/c/'rear60/B/backend/B/training/progress.json').exists():
  v=read(p);status['cache'][c+'_B60']={k:v[k] for k in ['episodes','total','tick','phase'] if k in v}
for name in ['continuation15_failure.json','postprocess15_failure.json','audits_complete.json']:
 if (D/name).exists():status[name]=read(D/name)
print(json.dumps(status))
