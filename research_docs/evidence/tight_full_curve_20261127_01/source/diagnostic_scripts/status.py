from pathlib import Path
import json,time
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01');D=Path('outputs/diagnostics/tight_full_curve_20261127_01');read=lambda p:json.loads(p.read_text())
for d in sorted(R.glob('experts_*')):
 rows=read(d/'results.json') if (d/'results.json').exists() else []
 print(d.name,dict(attempts=len(rows),accepted=sum(r['accepted'] for r in rows),collisions=sum(r['result'].get('not_started',False) for r in rows),complete=(d/'project_complete.json').exists()))
for d in sorted(R.glob('evaluation_*')):
 res=list(d.glob('rollouts/*/result.json'));print(d.name,dict(completed=len(res),success=sum(read(p).get('success',False) for p in res),owner_complete=(d/'project_complete.json').exists()))
for c in ['old_only','plus_corrected24']:
 for r in 'AB':
  p=R/c/r/'training/validation.json'
  if p.exists():
   row=read(p);print(c,r,'DP',row.get('epoch'),row.get('updates'),row.get('loss'))
 for p in sorted((R/c).glob('rear*/? /backend/?/training/validation.json')):print(p)
for p in list(R.rglob('project_failure.json'))+list(D.glob('*failure.json')):
 print('FAILURE',p,read(p))
print('utc',time.time())
