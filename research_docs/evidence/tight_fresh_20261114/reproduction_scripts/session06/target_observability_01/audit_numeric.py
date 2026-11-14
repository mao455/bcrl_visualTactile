"""Closed train corrections only: target/stage and late-takeover clock audit."""
from pathlib import Path
import json,time
import h5py,numpy as np
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_audit import audit_numeric
from tools.tight_fresh_takeover_pose_audit import window_components
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/new_family_09')
TREE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_collect09_20261114');OUT=Path(__file__).parent/'closed_target_clock_audit.json'
assert not OUT.exists()
read=lambda p:json.loads(p.read_text());rows=[]
selected=[(13,'held60'),(5,'held60'),(24,'held120'),(31,'held60'),(26,'contact0')]
for seed,schedule in selected:
 name=f'r09_A_expanded_session06_new64_s286000{seed:02d}_{schedule}';d=ROOT/name;x=read(d/'result.json');capture=read(d/'capture_complete.json')
 assert x['software_complete'] and capture['status']=='source_verified' and capture['dataset_capture']['verified_source_fields']==185
 assert x['dataset_capture']==capture['dataset_capture']
 bound=read(ROOT/f"wave_{x['batch_execution']['wave']:04d}_native_bound.json")
 replay=audit_trigger_case(d,x);labels,phases=audit_numeric(d/'numeric.h5',x);switch=labels['switch_action_index']
 with h5py.File(d/'numeric.h5','r') as f:
  t=f['episode/transition'];before=t['diagnostics_before'];after=t['diagnostics_after'];pose=after['object_pose_w'][:];palm=after['palm_body_pose_w'][:];force=t['object_contact_force_w'][:]
  height=float(np.median(pose[24:48,2]));i=switch-1;goal=after['goal_pose_w'][i];window=window_components(pose,palm,force,height,switch)
  stage_remaining=601-int(after['native_stage_step'][i]);assert int(before['native_stage_step'][switch])==int(after['native_stage_step'][i])
  boundary=dict(completed_student_tick=switch,native_stage_step=int(after['native_stage_step'][i]),native_goal=int(after['successes'][i]),
   timeout_at_stage_step=601,remaining_controls_if_no_subgoal=stage_remaining,remaining_seconds_if_no_subgoal=stage_remaining/60,
   height_above_settled_m=float(pose[i,2]-height),object_goal_root_distance_m=float(np.linalg.norm(pose[i,:3]-goal[:3])),
   object_goal_quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[i,3:]*goal[3:]).sum()),0,1))),
   native_keypoint_max_distance_m=float(t['native_keypoints_max_dist'][i]),window=window)
  events=[]
  for index in np.flatnonzero(after['successes'][:]!=before['successes'][:]):
   events.append(dict(completed_tick=int(index+1),expert_actions_since_takeover=int(index+1-switch),
    goals_before=int(before['successes'][index]),goals_after=int(after['successes'][index]),
    native_stage_step_before=int(before['native_stage_step'][index]),native_stage_step_after=int(after['native_stage_step'][index]),
    goal_before_w=before['goal_pose_w'][index].tolist(),goal_after_w=after['goal_pose_w'][index].tolist(),
    hole_before_w=before['hole_pose_w'][index].tolist(),object_after_w=pose[index].tolist(),
    near_goal_steps_after_goal_reset=int(t['native_success_diagnostic/near_goal_steps_after_goal_reset'][index]),
    keypoint_max_distance_m=float(t['native_keypoints_max_dist'][index]),
    retract_before=bool(before['retract_phase'][index]),retract_after=bool(after['retract_phase'][index]),
    retract_succeeded_after=bool(after['retract_succeeded'][index])))
  previous=before['successes'][:];near=t['native_success_diagnostic/near_goal'][:].astype(bool)
  goal_counts={str(int(s)):int((near&(previous==s)).sum()) for s in np.unique(previous)}
  diagnostic=dict(goal_valid_true=int(f['episode/obs/goal_valid'][:].sum()),task_ids=np.unique(f['episode/obs/task_id'][:]).tolist(),
   goal_commands_all_zero=bool((f['episode/obs/goal_command'][:]==0).all()),random_goal_any=bool(after['is_random_goal'][:].any()),
   native_subgoal_events=events,near_goal_hits_by_goal=goal_counts,terminal_native_stage_step=int(after['native_stage_step'][-1]),
   terminal_keypoint_max_distance_m=float(t['native_keypoints_max_dist'][-1]))
 rows.append(dict(case=name,accepted=x['accepted_for_training'],steps=x['steps'],reasons=x['reasons'],
  boundary=boundary,labels=labels,phases=phases,trigger_replay=replay,diagnostic=diagnostic,native_bound=bound,
  capture_sha256=x['dataset_capture']['sha256'],result_sha256=sha256(d/'result.json'),capture_receipt_sha256=sha256(d/'capture_complete.json')))
# Test only adjacent saved switch observations; do not invent identical histories.
path=Path(read(ROOT/rows[0]['case']/'result.json')['dataset_capture']['path'])
with h5py.File(path,'r') as f:
 g=f['episodes'][rows[0]['case']]
 switch_images=dict(indices=[404,405],rgb_bytewise_equal=bool(np.array_equal(g['frames/rgb_u8'][404],g['frames/rgb_u8'][405])),
  proprio_bytewise_equal=bool(np.array_equal(g['obs/proprio'][404],g['obs/proprio'][405])),
  interpretation='These two real frames differ. No empirical identical full-observation-history/different-target pair has been established.')
sources=['isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py','isaacsimenvs/tasks/play/utils/obs_utils.py',
 'isaacsimenvs/tasks/play/utils/reward_utils.py','isaacsimenvs/tasks/play/utils/scene_utils.py',
 'evaluation/problems/__init__.py','evaluation/problems/peg/problems.py','visuotactile/sim/environment.py',
 'visuotactile/native_controller.py','visuotactile/native_data.py','visuotactile/observations.py']
atomic_json(OUT,dict(version='closed_training_target_and_timeout_review_v1',created_utc=time.time(),
 frozen_tree=str(TREE),source_sha256={p:sha256(TREE/p) for p in sources},script_sha256=sha256(__file__),
 original_success_and_native_stage_clock_unchanged=True,simulator_starts=0,optimizer_updates=0,test_validation_payloads_opened=0,
 cases=rows,adjacent_switch_observations=switch_images,
 interpretations=['Goal sequence is fixed by visible fixture pose and hidden deterministic native stage; no independent random target in this task setting.',
 'Latest/current visible frame does not directly encode the internal accumulated10hit stage counter. History might infer it; no impossibility claim.',
 'Late takeover remaining time is conditional on no next subgoal; a true native subgoal resets stage clock. No manual reset or extra time was given.',
 'Expert timeout with7 or67 actions is not evidence of physical unrecoverability; comparison successful suffix lengths are descriptive, not a universal minimum.']))
for r in rows: print(r['case'],json.dumps(dict(boundary=r['boundary'],events=r['diagnostic']['native_subgoal_events'],labels=r['labels'],terminal_keypoint=r['diagnostic']['terminal_keypoint_max_distance_m']),ensure_ascii=False))
