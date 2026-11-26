"""Descriptive per-family first observable divergence, no exact-state causal claim."""
from pathlib import Path
import json,csv
import h5py,numpy as np
D=Path('outputs/diagnostics/tight_correction_20261127_01')
table=D/'all_completed_development_cases.csv'
if table.exists():rows=list(csv.DictReader(table.open()))
else:
 rows=json.loads((D/'development_before_cases.json').read_text())
 for role in 'AB':
  directory=Path('outputs/visuotactile/run/tight_correction_20261127_01')/('development_after_'+role)/'rollouts'
  plan=json.loads((directory/'plan.json').read_text())
  for case in plan['cases']:
   result=directory/case['name']/'result.json';r=json.loads(result.read_text());assert r['software_complete']
   rows.append(dict(role=case['role'],seed=case['spec']['seed'],success=r['success'],source=str(result)))
index={(r['role'],int(r['seed'])):r for r in rows}
def read(r):
 p=Path(r['source']).parent
 with h5py.File(p/'numeric.h5') as f:
  t=f['episode/transition'];a=t['diagnostics_after'];b=t['diagnostics_before'];v={k:a[k][:] for k in ['joint_position_rad','object_pose_w','palm_body_pose_w','base_pose_w']};v['object_initial']=b['object_pose_w'][0,:3]-b['base_pose_w'][0,:3]
  for k in ['object_pose_w','palm_body_pose_w']:v[k]=v[k][:,:3]-v['base_pose_w'][:,:3]
  v['object_palm_xyz_w']=v['object_pose_w']-v['palm_body_pose_w']
 with np.load(p/'trace.npz') as f:
  for key in ['nominal','residual_proposed_rad','execution_preprojection_target_rad','execution_projection_rad']:
   if key in f:v[key]=f[key].copy()
 with h5py.File(p/'numeric.h5') as f:
  t=f['episode/transition'];v['issued']=t['position_requested_rad'][:];v['applied']=t['control_trace/applied_position_target_rad'][:]
 return v
cache={}
def values(k):
 if k not in cache:cache[k]=read(index[k])
 return cache[k]
def first(d,tolerance,consecutive=1):
 valid=np.max(np.abs(d),axis=-1)>tolerance
 if consecutive>1:valid=np.convolve(valid,np.ones(consecutive,dtype=int),'valid')>=consecutive
 ii=np.flatnonzero(valid);return int(ii[0]) if len(ii) else None
out=[]
for role in 'AB':
 for seed in range(28911000,28911008):
  for candidate,reference in [(role+'_with_corrections',role+'_new'),(role+'_old_only',role+'_new'),(role+'_with_corrections',role+'_old_only')]:
   a=values((candidate,seed));b=values((reference,seed));n=min(len(a['issued']),len(b['issued']));item=dict(seed=seed,candidate=candidate,reference=reference,compared_ticks=n,initial_object_xyz_difference_m=float(np.max(np.abs(a['object_initial']-b['object_initial']))),candidate_success=index[candidate,seed]['success'],reference_success=index[reference,seed]['success'])
   for k in ['nominal','residual_proposed_rad','execution_preprojection_target_rad','execution_projection_rad','issued','applied','joint_position_rad','object_pose_w','palm_body_pose_w','object_palm_xyz_w']:
    if k not in a or k not in b:continue
    tolerance=1e-3 if k in ['object_pose_w','palm_body_pose_w','object_palm_xyz_w'] else 1e-6
    item[k+'_first_difference_tick']=first(a[k][:n]-b[k][:n],tolerance,3 if tolerance==1e-3 else 1)
    item[k+'_tick0_max_abs_difference']=float(np.max(np.abs(a[k][0]-b[k][0])))
   out.append(item)
(D/'first_divergence.json').write_text(json.dumps(dict(rows=out,action_tick_index='0-based; geometry is after executed action',thresholds=dict(commands_rad=1e-6,position_m=1e-3,position_confirmation_ticks=3),interpretation='First observed difference is descriptive, not harmful-error ground truth. Different trained parameters and potentially different hidden/reset streams; review initial observables before any causal interpretation.'),indent=2)+'\n')
with (D/'first_divergence.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('First divergences exported',len(out))
