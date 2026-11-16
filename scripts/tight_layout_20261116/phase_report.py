"""Closed full-policy stage outcomes, real labels, throughput and failure geometry."""
import argparse,json,time
from pathlib import Path
import h5py
import numpy as np
from visuotactile.phase_takeover import profile,STAGES
from visuotactile.takeover_trigger import make_trigger,observe_transition
from visuotactile.grasp_curriculum import holding_metrics
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

def summarize(root,output):
 read=lambda p:json.loads(p.read_text())
 plan=read(root/'plan.json');complete=read(root/'result.json')
 assert complete['software_complete'] and complete['worker_error'] is None
 paired=read(root/'paired_initial_check.json');assert paired['passed']
 rows=[]
 for case in plan['cases']:
  folder=root/case['name'];r=read(folder/'result.json')
  row=dict(case=case['name'],role=case['role'],seed=case['spec']['seed'],legal=r['legal'],
   source_family=case['spec'].get('source_identity'),requested_stage=case.get('dagger',{}).get('trigger',{}).get('stage','early' if 'dagger' in case else 'autonomous'),
   success=r['success'],steps=r.get('steps'),reasons=r.get('reasons'),software_complete=r['software_complete'],
   native_subgoals=r.get('native_subgoals'),native_alignment=r.get('native_alignment'),native_lift=r.get('native_lift'))
  if r['legal']:
   with h5py.File(folder/'numeric.h5') as f:
    t=f['episode/transition'];n=len(t['target_after_rad']);prefix=n
    if 'dagger' in t:
     d=t['dagger'];control=d['control_source'][:].reshape(-1);mask=d['expert_label_valid'][:].reshape(-1).astype(bool)
     assert not mask[control==0].any() and not (mask&~d['switch_clean'][:].reshape(-1).astype(bool)).any()
     assert np.array_equal(d['q_reached'][:][mask],t['diagnostics_after/joint_position_rad'][:][mask])
     assert np.array_equal(d['u_expert'][:][mask],t['target_after_rad'][:][mask])
     switch=np.flatnonzero(control==1);prefix=int(switch[0]) if len(switch) else n
     row.update(takeover_tick=prefix if len(switch) else None,takeover_phase=prefix%12 if len(switch) else None,
      valid_expert_ticks=int(mask.sum()),admitted=bool(r.get('accepted_for_training')),expert_rescue=bool(r['dagger']['expert_takeover_success']),
      actual_trigger=r['dagger']['trigger_receipt'],full_history_ticks=n,student_prefix_ticks=prefix,masked_expert_ticks=int(((control==1)&~mask).sum()))
    stages={}
    for stage in STAGES:
     trigger=make_trigger(profile(stage),10**9)
     for tick in range(prefix):
      if observe_transition(trigger,tick,t,index=tick) is not None:break
     stages[stage]=trigger.receipt()
    pose=t['diagnostics_after/object_pose_w'][:prefix];force=t['object_contact_force_w'][:prefix]
    goal=t['diagnostics_before/goal_pose_w'][:prefix]
    root_distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1)
    angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
    fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
    loss=np.flatnonzero((fingers[1:]==0)&(fingers[:-1]>0))+1
    row.update(student_stages=stages,hold60=holding_metrics(pose,force),
     hold120=holding_metrics(pose,force,window=120),student_no_contact_onsets=loss.tolist(),
     student_near_goal=dict(min_root_distance_m=float(root_distance.min()),final_root_distance_m=float(root_distance[-1]),
      minimum_keypoint_distance_m=float(t['native_keypoints_max_dist'][:prefix].min()),final_orientation_error_rad=float(angle[-1]),
      interpretation='Active native goal; root proximity alone does not imply alignment. All values restricted to actual student prefix.'))
   row['numeric_sha256']=sha256(folder/'numeric.h5')
  rows.append(row)
 counts={}
 for role in sorted(plan['pairs']):
  selected=[r for r in rows if r['role']==role];legal=[r for r in selected if r['legal']]
  counts[role]=dict(opportunities=len(selected),legal=len(legal),successes=sum(r['success'] for r in legal),
   expert_rescues=sum(r.get('expert_rescue',False) for r in legal),admitted=sum(r.get('admitted',False) for r in legal),
   admitted_expert_ticks=sum(r.get('valid_expert_ticks',0) for r in legal if r.get('admitted')),
   stage_reached={s:sum(r['student_stages'][s]['trigger_completed_tick'] is not None for r in legal) for s in STAGES})
 stages={}
 for stage in sorted({r['requested_stage'] for r in rows}):
  group=[r for r in rows if r['requested_stage']==stage];legal=[r for r in group if r['legal']]
  taken=[r for r in legal if r.get('takeover_tick') is not None];admitted=[r for r in taken if r.get('admitted')]
  stages[stage]=dict(opportunities=len(group),legal=len(legal),takeovers=len(taken),
   not_triggered=sum(r.get('takeover_tick') is None for r in legal),rescued=sum(r.get('expert_rescue',False) for r in taken),
   admitted_expert_ticks=sum(r['valid_expert_ticks'] for r in admitted),
   distinct_source_families=len({r['source_family'] for r in admitted}),
   takeover_phases=sorted({r['takeover_phase'] for r in taken}),
   interpretation='Rescue denominator is actual takeovers; untaken cases remain in opportunity denominator. Family count is not independent correction-state count.')
 waves=read(root/'wave_profiles.json') if (root/'wave_profiles.json').exists() else None
 result=dict(utc=time.time(),root=str(root),counts=counts,requested_stages=stages,rows=rows,complete=complete,paired=paired,
  wave_profiles=waves,plan_sha256=sha256(root/'plan.json'),software_interruptions=sum(not r['software_complete'] for r in rows),
  valid_expert_ticks_per_wall_hour=sum(v['admitted_expert_ticks'] for v in counts.values())/complete['elapsed_seconds']*3600,
  interpretation='Collection rescue success is not autonomous success; stage-reaching is censored by actual takeover, never inferred beyond student prefix.')
 atomic_json(output,result);print(json.dumps(counts))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();summarize(a.root,a.output)
