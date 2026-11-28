from pathlib import Path
import json
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01');D=Path('outputs/diagnostics/tight_full_curve_20261127_01')
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
status={}
for c in ['old_only','plus_corrected24']:
 for r in 'AB':
  for label,p in [('DP',R/c/r/'training/validation.json'),('rear30',R/c/'rear30'/r/'backend'/r/'training/validation.json'),('rear60',R/c/'rear60'/r/'backend'/r/'training/validation.json')]:
   v=read(p)
   if isinstance(v,list):v=v[-1] if v else {}
   if v:status[c+' '+r+' '+label]={k:v[k] for k in ['epoch','updates','update','loss','validation_score'] if k in v}
for o in sorted(list(R.glob('evaluation_*'))+list(R.glob('final16_*'))):
 if (o/'project_complete.json').exists():status[o.name]='complete'
 elif (o/'project_job_started.json').exists():
  v=read(o/'rollouts/batch_progress.json');status[o.name]={k:v.get(k) for k in ['closed_results','remaining_tasks','tick']}
p=R/'plus_corrected24/A/cache_progress.json'
if p.exists() and not (R/'visual_cache_complete.json').exists():
 v=read(p);status['visual_cache']={k:v.get(k) for k in ['episodes','total_episodes','frames','total_frames','eta_seconds']}
failures=[]
for p in list(R.rglob('project_failure.json'))+list(D.glob('*failure.json')):
 failures.append(dict(path=str(p),error=read(p).get('error')))
status['historical_failure_count']=len(failures);status['latest_failure']=failures[-1] if failures else None;status['round_complete']=(R/'curve_complete.json').exists()
print(json.dumps(status,ensure_ascii=False))
