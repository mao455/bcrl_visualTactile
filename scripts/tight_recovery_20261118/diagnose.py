"""Offline observable regression audit; censor at real student prefix."""
import json,csv
from pathlib import Path
import numpy as np,h5py
from scripts.tight_cycle_20261118.cycle import ROOT as OLD,read,write,TREE
from visuotactile.deviation_takeover import profile
from visuotactile.takeover_trigger import make_trigger,observe_transition
OUT=TREE/'research_docs/evidence/tight_recovery_20261118'
def replay(t,n):
 tr=make_trigger(profile(),10**9)
 for i in range(n):
  if observe_transition(tr,i,t,index=i) is not None:break
 return tr.receipt()
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[]
 for wave in ['collection','capability']:
  for c in read(OLD/wave/'plan.json')['cases']:
   d=OLD/wave/c['name'];r=read(d/'result.json')
   if not r['legal']:continue
   with h5py.File(d/'numeric.h5') as f:
    t=f['episode/transition'];n=len(t['target_after_rad']);switch=None;first=None;valid=[]
    if 'dagger' in t:
     switchs=np.flatnonzero(t['dagger/control_source'][:].reshape(-1)==1);switch=int(switchs[0]) if len(switchs) else None
     valid=np.flatnonzero(t['dagger/expert_label_valid'][:].reshape(-1));first=int(valid[0]) if len(valid) else None
    prefix=switch if switch is not None else n;receipt=replay(t,prefix);onset=receipt['first_regression_transition_tick'];at=prefix-1
    pose=t['diagnostics_after/object_pose_w'][at];palm=t['diagnostics_after/palm_body_pose_w'][at];goal=t['diagnostics_before/goal_pose_w'][at]
    row=dict(wave=wave,case=c['name'],role=c['role'],family=c['spec']['family'],seed=c['spec']['seed'],original_trigger=c.get('dagger',{}).get('trigger'),success=r['success'],student_prefix=prefix,observable_regression_tick=onset,regression=receipt,takeover_tick=switch,first_legal_tick=first,label_lag_from_screen=None if onset is None or first is None else first-onset,supervised_ticks=len(valid),entry_object_pose=pose.tolist(),entry_object_palm_delta=(pose[:3]-palm[:3]).tolist(),entry_goal_distance=float(np.linalg.norm(pose[:3]-goal[:3])),entry_velocity=t['diagnostics_after/object_velocity_w_m_s_rad_s'][at].tolist(),entry_fingers=int((np.linalg.norm(t['object_contact_force_w'][at],axis=-1)>.1).sum(-1).min()))
    if first is not None:
     end=min(first+24,n-1);q=t['diagnostics_after/joint_position_rad'][:];cmd=t['target_after_rad'][:]
     row['expert_first24']=dict(command_change_rms_rad=float(np.sqrt(np.mean((cmd[end]-cmd[first])**2))),reached_change_rms_rad=float(np.sqrt(np.mean((q[end]-q[first])**2))),object_delta_m=(t['diagnostics_after/object_pose_w'][end,:3]-t['diagnostics_after/object_pose_w'][first,:3]).tolist(),new_screen_in_first24=replay(t,first+1)['first_regression_transition_tick'])
    rows.append(row)
 write(OUT/'old_regression_audit.json',dict(rows=rows,scope='First sustained observable contact/height/progress regression within actual student prefix. No counterfactual error time for censored normal prefixes. Stall can be deliberate alignment; trigger is a testable screen, not causal proof. No new simulation.'))
 for wave in ['collection','capability']:
  rs=[r for r in rows if r['wave']==wave];print(wave,len(rs),'screen',sum(r['observable_regression_tick'] is not None for r in rs))
  for r in rs:print(r['role'],r['seed'],r['success'],r['takeover_tick'],r['observable_regression_tick'],r['regression']['at_trigger'])
 # Original naturally successful full demonstrations estimate screen ambiguity.
 data=read(OLD/'learning/A/state_data/data.json');ref=[]
 for i in data['normalization_train_ids']:
  e=data['entries'][i]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];rec=replay(t,len(t['target_after_rad']))
   ref.append(dict(identity=i,receipt=rec))
 write(OUT/'expert_screen_ambiguity.json',ref);print('original30 screened',sum(r['receipt']['trigger_completed_tick'] is not None for r in ref))
if __name__=='__main__':main()
