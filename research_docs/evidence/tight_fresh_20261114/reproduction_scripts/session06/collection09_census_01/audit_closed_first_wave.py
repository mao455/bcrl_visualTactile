"""Read only indexed/source-verified first-wave held records; never active HDF."""
from pathlib import Path
import json,time
import h5py,numpy as np
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_audit import audit_numeric
from tools.tight_fresh_takeover_pose_audit import window_components
from tools.tight_collection_summary import held_recheck
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/new_family_09')
OUT=Path(__file__).with_name('closed_first_wave_held.json')
assert not OUT.exists()
started=time.time();plan=json.loads((ROOT/'plan.json').read_text());cases={c['name']:c for c in plan['cases']};rows=[]
for indexed in json.loads((ROOT/'results.json').read_text()):
 name=indexed['case'];spec=cases[name];result=indexed['result'];directory=ROOT/name
 if spec['role']!='A' or spec['spec']['seed']>=28600016 or spec['expansion_schedule'] not in ('held60','held120'):continue
 capture=json.loads((directory/'capture_complete.json').read_text())
 assert capture['status']=='source_verified' and capture['dataset_capture']['verified_source_fields']==185
 assert capture['dataset_capture']==result['dataset_capture'] and result['software_complete']
 replay=audit_trigger_case(directory,result);labels,phases=audit_numeric(directory/'numeric.h5',result)
 with h5py.File(directory/'numeric.h5') as f:
  t=f['episode/transition'];a=t['diagnostics_after'];pose=a['object_pose_w'][:];palm=a['palm_body_pose_w'][:];force=t['object_contact_force_w'][:]
  height=float(np.median(pose[24:48,2]));first=replay['receipt']['first_contact_height_window_end_completed_tick'];switch=labels['switch_action_index']
  ticks=sorted({x for x in (first,switch,labels['student_history_ticks']) if x is not None});states=[]
  for tick in ticks:
   row=window_components(pose,palm,force,height,tick);i=tick-1;goal=t['diagnostics_before/goal_pose_w'][i]
   row.update(held_window=row['conditions']['two_contact_fingers_every_substep'] and row['conditions']['height'],
    height_above_settled_m=float(pose[i,2]-height),goal_root_distance_m=float(np.linalg.norm(pose[i,:3]-goal[:3])),
    goal_quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[i,3:]*goal[3:]).sum()),0,1))),
    native_keypoint_max_distance_m=float(t['native_keypoints_max_dist'][i]))
   states.append(row)
 rows.append(dict(case=name,schedule=spec['expansion_schedule'],accepted=result['accepted_for_training'],
  labels=labels,phases=phases,recheck=held_recheck(spec,result),trigger_replay=replay,
  states=states,capture_receipt_sha256=sha256(directory/'capture_complete.json'),
  result_sha256=sha256(directory/'result.json'),capture_profile=capture['profile']))
atomic_json(OUT,dict(version='closed_first_wave_held_review_v1',root=str(ROOT),partial_collection=True,
 started_utc=started,finished_utc=time.time(),plan_sha256=sha256(ROOT/'plan.json'),script_sha256=sha256(__file__),
 scope='First A wave only. Six already closed/indexed captures; no active HDF or RGB/DINO/weight payload read.',
 definitions='Reference is completed student25–48 median. Held requires latest12 transitions and both physics samples; old stable/native lift separate. Expert recovery is not autonomous success.',
 cases=rows))
print(json.dumps([dict(case=r['case'],accepted=r['accepted'],category=r['recheck']['category'],
 student_native_lift=r['phases']['student_native_lift'],student_stable=r['phases']['first_student_stable'] is not None,
 states=r['states']) for r in rows],indent=2))
