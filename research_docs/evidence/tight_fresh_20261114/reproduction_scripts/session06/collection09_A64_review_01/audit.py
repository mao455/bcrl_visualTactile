"""One-shot closed/indexed A64 census, held schedules numeric only; no active HDF."""
from pathlib import Path
from collections import Counter
import json,time,hashlib
import h5py,numpy as np
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_audit import audit_numeric
from tools.tight_fresh_takeover_pose_audit import window_components
from tools.tight_collection_summary import held_recheck
from visuotactile.compact import sha256
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/new_family_09')
OUT=Path(__file__).with_name('closed_A64.json');assert not OUT.exists();started=time.time()
read=lambda p:json.loads(p.read_text())
plan=read(ROOT/'plan.json');planned={c['name']:c for c in plan['cases'] if c['role']=='A'}
raw=(ROOT/'results.json').read_bytes();indexed={r['case']:r['result'] for r in json.loads(raw) if r['role']=='A'}
assert set(indexed)==set(planned) and len(planned)==64
entries=read(ROOT/'correction_entries.json');rows=[];held=[]
for name,case in planned.items():
 result=indexed[name];directory=ROOT/name
 assert result['software_complete'] is True
 row=dict(case=name,family=case['spec']['family'],schedule=case['expansion_schedule'],legal=bool(result.get('legal')),
   accepted=bool(result.get('accepted_for_training')),success=bool(result.get('success')),steps=result.get('steps',0),
   expert_controls=result.get('dagger',{}).get('expert_ticks',0),legal_expert_labels=result.get('dagger',{}).get('expert_valid_ticks',0),
   expert_recovery_success=bool(result.get('dagger',{}).get('expert_takeover_success')),native_reasons=result.get('reasons'))
 if row['accepted']:
  entry=entries[name];assert row['legal'] and row['success'] and row['expert_recovery_success']
  assert entry['episode']==name and entry['family']==case['spec']['family']
  assert entry['source_sha256']==result['dataset_capture']['sha256']
  assert all(entry[k]==plan['pairs']['A'][k] for k in ('frontend_sha256','backend_sha256'))
 rows.append(row)
 if row['schedule'] not in ('held60','held120'):continue
 rec=dict(row,classification=held_recheck(case,result))
 if not row['legal']:held.append(rec);continue
 capture=read(directory/'capture_complete.json')
 assert capture['status']=='source_verified' and capture['dataset_capture']['verified_source_fields']==185
 assert capture['dataset_capture']==result['dataset_capture']
 replay=audit_trigger_case(directory,result);labels,phases=audit_numeric(directory/'numeric.h5',result)
 states=[]
 with h5py.File(directory/'numeric.h5','r') as f:
  t=f['episode/transition'];a=t['diagnostics_after'];pose=a['object_pose_w'][:];palm=a['palm_body_pose_w'][:];force=t['object_contact_force_w'][:]
  assert len(pose)==result['steps'] and (t['dagger/control_source'][:48]==0).all()
  height=float(np.median(pose[24:48,2]));first=replay['receipt']['first_contact_height_window_end_completed_tick'];switch=labels['switch_action_index']
  for tick in sorted({x for x in (first,switch,labels['student_history_ticks']) if x is not None}):
   state=window_components(pose,palm,force,height,tick);i=tick-1;before=t['diagnostics_before'];goal=before['goal_pose_w'][i]
   native_step=int(a['episode_length_buf'][i]) if 'episode_length_buf' in a else None
   state.update(completed_tick=tick,held_window=bool(state['conditions']['two_contact_fingers_every_substep'] and state['conditions']['height']),
    height_above_settled_m=float(pose[i,2]-height),goal_root_distance_m=float(np.linalg.norm(pose[i,:3]-goal[:3])),
    goal_quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[i,3:]*goal[3:]).sum()),0,1))),
    native_keypoint_max_distance_m=float(t['native_keypoints_max_dist'][i]),native_subgoals=int(a['successes'][i]))
   states.append(state)
  if switch is not None:
   state=next(s for s in states if s['completed_tick']==switch)
   assert state['held_window'] and state['stable_window']==phases['last_student_transition']['stable_window']
 rec.update(labels=labels,phases=phases,trigger_replay=replay,states=states,numeric_sha256=replay['numeric_sha256'],
   result_sha256=sha256(directory/'result.json'),capture_receipt_sha256=sha256(directory/'capture_complete.json'),
   capture_sha256=capture['dataset_capture']['sha256'])
 held.append(rec)
def group(values):
 return dict(attempts=len(values),legal=sum(r['legal'] for r in values),expert_takeovers=sum(r['expert_controls']>0 for r in values),
  expert_recoveries=sum(r['expert_recovery_success'] for r in values),accepted=sum(r['accepted'] for r in values),
  accepted_legal_expert_labels=sum(r['legal_expert_labels'] for r in values if r['accepted']),
  legal_expert_labels_all_outcomes=sum(r['legal_expert_labels'] for r in values))
heldsummary={}
for schedule in ('held60','held120'):
 subset=[r for r in held if r['schedule']==schedule];legal=[r for r in subset if r['legal']];triggered=[r for r in legal if r['expert_controls']>0]
 boundary=lambda r:next(s for s in r['states'] if s['completed_tick']==r['labels']['switch_action_index'])
 heldsummary[schedule]=dict(**group(subset),categories=dict(Counter(r['classification']['category'] for r in subset)),
  student_native_lift=sum(r['phases']['student_native_lift'] for r in legal),student_ever_old_stable=sum(r['phases']['first_student_stable'] is not None for r in legal),
  actual_boundary_held=sum(boundary(r)['held_window'] for r in triggered),actual_boundary_old_stable=sum(boundary(r)['stable_window'] for r in triggered),
  accepted_boundary_old_stable=sum(boundary(r)['stable_window'] for r in triggered if r['accepted']),
  accepted_boundary_not_old_stable=[r['case'] for r in triggered if r['accepted'] and not boundary(r)['stable_window']],
  actual_boundary_rotation_over_0p2rad=sum(boundary(r)['palm_relative_max_rotation_rad']>.2 for r in triggered),
  native_student_subgoals_max=max((r['phases']['authority']['student_max_subgoals'] for r in legal),default=None))
report=dict(version='closed_A64_partial_collection_review_v1',root=str(ROOT),started_utc=started,finished_utc=time.time(),
 role_complete=True,whole_collection_complete=False,summary=group(rows),by_schedule={k:group([r for r in rows if r['schedule']==k]) for k in sorted({r['schedule'] for r in rows})},
 attempted_distinct_families=len({r['family'] for r in rows}),accepted_distinct_families=len({r['family'] for r in rows if r['accepted']}),
 source_pair=plan['pairs']['A'],held_summaries=heldsummary,cases=rows,held_cases=held,
 evidence=dict(plan_sha256=sha256(ROOT/'plan.json'),results_snapshot_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=sha256(__file__),snapshot_sha256=sha256(OUT.with_name('snapshot.json'))),
 scope='A64 indexed opportunities only. All 24 held60/held120 cases are closed before numeric audit; no B/active HDF or RGB/DINO/model payloads opened.',
 definitions=['Held uses latest12 completed student transitions, >=2 fingers >0.1N in both120Hz samples and >=median student completed25–48 height+2cm.',
 'Old stable adds palm-relative translation<=1cm and rotation<=0.2rad; missing it is not a native fall/drop. Native lift is a separate original task criterion.',
 'No-trigger, no current held window, original native drop/fall, native timeout and expert recovery are different observations.',
 'Expert success is not autonomous student success; admitted legal labels are not optimizer exposure. Entrypoint schedule is not an action-stage label.',
 'This is a completed A-role slice of a still-active A/B collection, not a whole-collection throughput/final result.'],simulator_starts=0,optimizer_updates=0,large_rgb_hash_recomputed=False)
OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['summary','by_schedule','held_summaries','finished_utc']},indent=2))
