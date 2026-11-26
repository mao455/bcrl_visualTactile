from pathlib import Path
import json,csv,sys
import h5py,numpy as np
sys.path.insert(0,str(Path('outputs/diagnostics/tight_machine_review_20261126_01').resolve()))
from audit_existing import replay
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01')
def summarize(cohort):
 rows=[]
 waves=sorted((R/cohort).glob('wave*'))
 if cohort=='development_before':
  waves=[R/cohort/'wave00']+[R/'development_retry01'/f'wave{i:02d}' for i in range(1,5)]+sorted((R/'development_retry03').glob('wave*'))
 for w in waves:
  plan=json.loads((w/'plan.json').read_text())
  for case in plan['cases']:
   d=w/case['name'];rp=d/'result.json';result=json.loads(rp.read_text()) if rp.exists() else {}
   row=dict(wave=w.name,role=case['role'],family=case['spec']['family'],seed=case['spec']['seed'],complete=result.get('software_complete',False),success=result.get('success',False),steps=result.get('steps'),reasons='|'.join(k for k,v in result.get('reasons',{}).items() if v),source=str(rp))
   if result.get('software_complete') and (d/'numeric.h5').exists():
    row.update(replay(d/'numeric.h5'));row.pop('eligible_completed_ticks',None)
    with h5py.File(d/'numeric.h5') as f:
     t=f['episode/transition'];key=t['native_keypoints_max_dist'][:];before=t['diagnostics_before/successes'][:];after=t['diagnostics_after/successes'][:]
     def first(x):
      a=np.flatnonzero(x);return int(a[0]) if len(a) else None
     row.update(first_gate_action_tick=first((key<=np.float32(.015))&(before==0)),first_subgoal_action_tick=first(after>0))
    receipt=json.loads((d/'observe_only_trigger.json').read_text()) if (d/'observe_only_trigger.json').exists() else {}
    row['live_first_eligible_completed_tick']=receipt.get('first_eligible_completed_tick')
    row['live_matches_replay']=row['live_first_eligible_completed_tick']==row['never_alignment_first']
   rows.append(row)
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (D/(cohort+'_cases.csv')).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
 atomic_json(D/(cohort+'_cases.json'),rows)
 summary={role:dict(registered=sum(r['role']==role for r in rows),complete=sum(r['complete'] for r in rows if r['role']==role),success=sum(r['success'] for r in rows if r['role']==role),held12=sum(r.get('held12',False) for r in rows if r['role']==role),near_held=sum(r.get('near_held',False) for r in rows if r['role']==role),gate=sum(r.get('ever15mm',False) for r in rows if r['role']==role),subgoal=sum(r.get('native_first_subgoal',False) for r in rows if r['role']==role),never_stall=sum(r.get('never_alignment_first') is not None for r in rows if r['role']==role)) for role in dict.fromkeys(r['role'] for r in rows)}
 atomic_json(D/(cohort+'_summary.json'),summary);print(json.dumps(summary))
if __name__=='__main__':summarize(sys.argv[1] if len(sys.argv)>1 else 'development_before')
