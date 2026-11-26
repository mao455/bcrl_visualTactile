from pathlib import Path
import json,csv,gzip
import h5py,numpy as np
import torch
old=Path('outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/A/state_data/numeric.pt');raw=torch.load(old,map_location='cpu',weights_only=False,mmap=True);limits=next(iter(raw['records'].values()))['joint_limits'].numpy()
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01')
def first(mask):
 js=np.flatnonzero(mask);return int(js[0]) if len(js) else None
def rmse(a):return float(np.sqrt(np.mean(a*a)))
summary=[];rows=[]
for seed in (28911001,28911002):
 paths=list((R/f'development_retry01/wave{seed-28911000:02d}').glob('*/numeric.h5'))+[R/f'grasp_expert_controls/expert_{seed}/numeric.h5']
 for path in paths:
  with h5py.File(path) as f:
   tr=f['episode/transition'];load=lambda k:tr[k][:]
   obj=load('diagnostics_after/object_pose_w');palm=load('diagnostics_after/palm_body_pose_w');base=load('diagnostics_after/base_pose_w')
   force=load('object_contact_force_w');contacts=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
   q=load('diagnostics_after/joint_position_rad');requested=load('position_requested_rad') if 'position_requested_rad' in tr else load('target_after_rad');applied=load('control_trace/applied_position_target_rad') if 'control_trace/applied_position_target_rad' in tr else load('actuator_substep_target_rad')[:,-1];target=load('target_after_rad')
   n=len(q);root=np.linalg.norm(obj[:,:3]-load('diagnostics_before/goal_pose_w')[:,:3],axis=-1)
   v=obj[:,:3]-palm[:,:3];inv=palm[:,3:].copy();inv[:,1:]*=-1;cross=2*np.cross(inv[:,1:],v);relative=v+inv[:,:1]*cross+np.cross(inv[:,1:],cross)
   settled=float(np.median(obj[24:48,2]));lift=obj[:,2]-settled
   two=contacts>=2;held=np.convolve(two&(lift>=.02),np.ones(12,dtype=int),mode='full')[:n]>=12
   tracepath=path.parent/'trace.npz';trace=np.load(tracepath) if tracepath.exists() else None
   role=path.parent.name.split('_paired')[0] if 'paired' in path.parent.name else 'expert'
   item=dict(seed=seed,role=role,source=str(path),steps=n,first_any_contact_tick=first(contacts>0),first_two_fingers_tick=first(two),first_lift20mm_tick=first((lift>=.02)&(np.arange(n)>=48)),first_held12_tick=first(held),first_gate_tick=first((load('native_keypoints_max_dist')<=np.float32(.015))&(load('diagnostics_before/successes')==0)),windows=[])
   for a,b in [(0,48),(48,96),(96,150),(150,min(240,n))]:
    if a>=n or b<=a:continue
    item['windows'].append(dict(start=a,stop=b,minimum_fingers=int(contacts[a:b].min()),two_finger_ticks=int(two[a:b].sum()),maximum_elevation_mm=float(lift[a:b].max()*1000),minimum_root_mm=float(root[a:b].min()*1000),arm_applied_tracking_rmse_rad=rmse((q-applied)[a:b,:7]),hand_applied_tracking_rmse_rad=rmse((q-applied)[a:b,7:])))
   if trace is not None and 'nominal' in trace:
    item['joint_clipped_nominal_plus_rear_to_execution_preprojection_max_rad']=float(np.max(np.abs(np.clip(trace['nominal']+trace['residual_proposed_rad'],limits[:,0],limits[:,1])-trace['execution_preprojection_target_rad'])))
    item['nominal_plus_applied_rear_to_issued_max_rad']=float(np.max(np.abs(trace['nominal']+trace['residual_applied_rad']-requested)))
    item['projection_to_issued_max_rad']=float(np.max(np.abs(trace['execution_preprojection_target_rad']+trace['execution_projection_rad']-requested)))
   for t in range(n):
    row=dict(seed=seed,role=role,action_tick=t,contact_fingers=int(contacts[t]),elevation_mm=float(lift[t]*1000),root_error_mm=float(root[t]*1000))
    for prefix,value in [('object_base',obj[t,:3]-base[t,:3]),('palm_base',palm[t,:3]-base[t,:3]),('object_palm',relative[t]),('requested',requested[t]),('applied',applied[t]),('reached',q[t])]:
     row.update({f'{prefix}_{j}':float(x) for j,x in enumerate(value)})
    if trace is not None and 'nominal' in trace:
     for prefix in ['nominal','residual_applied_rad','execution_preprojection_target_rad','execution_projection_rad']:
      row.update({f'{prefix}_{j}':float(x) for j,x in enumerate(trace[prefix][t])})
    rows.append(row)
   summary.append(item)
keys=list(dict.fromkeys(k for r in rows for k in r))
with gzip.open(D/'grasp_chain_ticks.csv.gz','wt') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
atomic_json(D/'grasp_chain_summary.json',dict(rows=summary,tick_index='0-based action followed by actual transition',same_state_counterfactual=False,expert_reset_stream='Original serial expert; actual initial difference must be checked, no hidden-state equality assumption'))
print(json.dumps(summary))
