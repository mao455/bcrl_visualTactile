"""Summarize actual within-target training/validation trends, never rank A/B nominal losses."""
from pathlib import Path
import csv,json,time
from visuotactile.training_resources import atomic_json
P=Path.cwd();D=P/'outputs/diagnostics/tight_full_curve_20261127_01';R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01'
assert (R/'curve_complete.json').exists()
allrows=[];trends=[]
for condition in ['old_only','plus_corrected24']:
 for role in 'AB':
  for component,root in [('DP',R/condition/role/'training'),('Transformer30',R/condition/'rear30'/role/'backend'/role/'training'),('Transformer60',R/condition/'rear60'/role/'backend'/role/'training')]:
   p=root/'family_phase_curves.csv'
   assert p.exists(),str(p)
   rows=list(csv.DictReader(p.open()))
   for r in rows:
    r=dict(r,condition=condition,source=str(p.relative_to(P)),curve_component=component)
    for k in ['epoch','samples']:r[k]=int(r[k])
    for k in ['arm_rmse_rad','hand_rmse_rad','normalized_equal_mse']:r[k]=float(r[k])
    allrows.append(r)
   keys={tuple(r[k] for k in ['split','family','phase']) for r in rows}
   for key in sorted(keys):
    series=sorted([r for r in rows if tuple(r[k] for k in ['split','family','phase'])==key],key=lambda r:int(r['epoch']))
    first,last=series[0],series[-1];window=series[-10:]
    trends.append(dict(condition=condition,role=role,component=component,split=key[0],family=key[1],phase=key[2],first_epoch=int(first['epoch']),last_epoch=int(last['epoch']),first_samples=int(first['samples']),last_samples=int(last['samples']),first_arm_rmse_rad=float(first['arm_rmse_rad']),last_arm_rmse_rad=float(last['arm_rmse_rad']),first_hand_rmse_rad=float(first['hand_rmse_rad']),last_hand_rmse_rad=float(last['hand_rmse_rad']),last10_hand_rmse_change_rad=float(window[-1]['hand_rmse_rad'])-float(window[0]['hand_rmse_rad']),epochs_in_last_window=len(window),scope='Within own target space and exact phase/family only; loss is not autonomous success'))
for name,rows in [('fitting_phase_family_curves.csv',allrows),('fitting_phase_family_trends.csv',trends)]:
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (D/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
atomic_json(D/'fitting_audit.json',dict(utc=time.time(),curves=len(allrows),trends=len(trends),aggregate=[r for r in trends if r['family']=='__all__' and r['phase']=='__all__'],comparison='No cross-target A-versus-B nominal ranking',coverage='Actual optimizer-ledger audit is separate from fitting and family-disjoint autonomous outcomes'))
print('Actual phase/family fitting trends exported')
