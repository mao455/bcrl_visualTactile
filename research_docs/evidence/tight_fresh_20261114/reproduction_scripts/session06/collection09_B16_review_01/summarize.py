"""Compact interpretation of the fixed closed cohort; reads no HDF/RGB."""
from pathlib import Path
import json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
INPUT=HERE/'closed_B16.json';OUT=HERE/'interpretation.json'
assert not OUT.exists()
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(INPUT);root=Path(r['root']);bs=r['B_cases'];ars=r['A_same16_cases']
held=[c for c in bs if c['schedule'] in ('held60','held120')]
def compact_state(s):
    if s is None:return None
    return {k:s[k] for k in ('completed_tick','native_goal_before','native_goal_after','native_clock_after',
        'remaining_controls_if_no_subgoal','held12','old_stable12','native_lift','object_current_goal_root_distance_m',
        'object_current_goal_angle_deg','native_keypoint_max_distance_m','native_near_goal','native_alignment')}
def compact_case(c):
    return dict(case=c['case'],seed=c['seed'],schedule=c['schedule'],steps=c['steps'],accepted=c['accepted'],
        classification=c['held_recheck']['category'],student=c['student'],held_intervals=c['student_held_intervals'],
        first_held=compact_state(c['snapshots']['first_student_held']),takeover=compact_state(c['snapshots']['takeover']),
        earliest_allowed_completed_tick=c['held_recheck']['receipt']['earliest_allowed_completed_tick'],
        nonheld_windows_after_first=c['held_recheck']['receipt']['nonheld_windows_after_first'],
        legal_expert_actions=c['labels']['legal_expert_ticks'],
        legal_expert_actions_by_native_goal_before={str(s['native_goal_index_before_action']):s['legal_expert_controls'] for s in c['stage_coverage']},
        native_subgoal_completed_ticks=[s['completed_tick'] for s in c['native_subgoal_transitions']])
initial=[]
for c in r['same_family_observed_comparison']:
    ia=read(root/c['A']['case']/'initial_observable.json');ib=read(root/c['B']['case']/'initial_observable.json')
    a,b=np.asarray(ia['palm_body_pose_w']),np.asarray(ib['palm_body_pose_w'])
    initial.append(dict(seed=c['seed'],joint_max_absolute_difference_rad=c['initial_fields']['joint_position_rad']['max_abs_difference'],
        joint_velocity_max_absolute_difference_rad_s=c['initial_fields']['joint_velocity_rad_s']['max_abs_difference'],
        palm_root_translation_difference_m=float(np.linalg.norm(a[0,:3]-b[0,:3])),
        all_observable_fields_equal=c['same_saved_observable_state'],
        object_goal_hole_table_poses_equal=all(c['initial_fields'][k]['equal'] for k in ('object_pose_w','goal_pose_w','hole_pose_w','table_pose_w')),
        A_legal=c['A']['legal'],B_legal=c['B']['legal']))
long=next(c for c in bs if c['seed']==28600012)
stage_fields=('native_goal_index_before_action','all_controls','student_controls','expert_controls','legal_expert_controls',
    'expert_flush_masked','legal_expert_any_contact','legal_expert_two_fingers','legal_expert_held12','legal_expert_old_stable12',
    'legal_expert_native_near_goal','legal_expert_native_alignment','legal_expert_first_completed_tick','legal_expert_last_completed_tick')
out=dict(version='closed_B16_collection_interpretation_v1',scope=r['scope'],B_summary=r['B_summary'],
    A_same16_summary=r['A_same16_summary'],B_schedule_summary=r['B_by_schedule'],
    held_cases=[compact_case(c) for c in held],matched_A_held_cases=[compact_case(c) for c in ars if c['schedule'] in ('held60','held120')],
    same_family_initial_observations=initial,
    fixed80_long_recovery=dict(case=long['case'],steps=865,student_controls=80,expert_controls=785,legal_expert_actions=783,
        share_of_B16_admitted_expert_actions=783/r['B_summary']['accepted_legal_expert_actions'],
        student_boundary=compact_state(long['snapshots']['takeover']),
        native_subgoal_completed_ticks=[s['completed_tick'] for s in long['native_subgoal_transitions']],
        native_goal_before_action_coverage=[{k:s[k] for k in stage_fields} for s in long['stage_coverage']],
        interpretation='Early fixed80 rescue, not a student late-hold takeover. Expert supplies388 legal actions toward the first target and395 after the first native subgoal. Contact/held/near counts overlap;395 are not all contact or insertion actions. Each native stage retains601-step limit.'),
    conclusions=[
        'Closed B16 has15 legal starts,8 expert takeovers,8 actual recoveries and8 admitted trajectories/2493 legal expert labels. The illegal near60 case remains in the16 opportunities, with0 action, never counted as a task failure.',
        'Students before takeover or terminal:1/15 native lift,5/14 with an available student-only height reference ever held12,2/14 oldstable. Fixed40 lacks that reference. Native near/alignment/subgoal/autonomous success are all0.',
        'Delayed held schedules really reach later recoverable states: B01 waits211→271 and B08 waits133→253; both retained current held and oldstable at actual takeover. Expert then adds48+83 and30+66 legal phase0/phase1 actions. These are227 real late-state labels, not hypothetical rescues.',
        'The B01 and B08 boundaries retain330/348 native controls before any further stage reset; neither is near a spent601-stage budget. B08 has116 intervening nonheld windows yet again holds at253. Window failures alone are not native drops.',
        'Held60 failures are mixed: B05 briefly held205–209 with roughly81mm root distance and114deg orientation at firstheld, but after earliest265 never re-held; seeds09/13/15 never held. All4 timeout with no native fall/drop. Delayed collection can miss transient useful states, but no expert recovery from those missed states is established.',
        'The same A16 produce6 admitted recoveries/1451 labels. A05/A13 late rescues start with about104/91deg orientation error, whereas B01/B08 start near12/10deg. Both policy/state mixtures matter; late rescue does not mean all gathered labels address the same rotation error.',
        'All16 pairs share registered family/reset configuration and saved object/goal/hole/table poses, but none share the complete observable initial state: actual joints, velocities and palms differ. These are same-family opportunities, not exact physical/hidden-state causal pairs.',
        'Goal0 root is186mm above hole root; small goal0 distance is not arrival at the hole mouth. No student native alignment or subgoal is established by root distance or angle.'
    ],
    next_sampling_focus=[
        'Preserve a mixture of early contact/fixed rescue and delayed rechecked held rescue. The2 late recoveries justify continued measured coverage, not replacing all early schedules or favoring120 over60 from1 opportunity.',
        'Track transient-held cases separately from never-held cases. Existing contact0 can capture first held states; a future explicitly registered subset could target this coverage, but do not retrospectively count B05 as recovered or silently add fallback.',
        'Retain wrong-orientation held examples alongside nearly oriented examples; the separate held_root60 candidate addresses a documented A near-goal-root omission, but B05 firstheld root≈81mm would not satisfy60mm.',
        'During subsequent training, audit per-family and phase exposure: B12 alone contributes31.4% of this slice action labels. Avoid inferring diverse late-state coverage merely from this long expert suffix; later-phase actions from held cases and other families must actually be sampled.',
        'Await the rest of predeclared B cohort before final schedule rates/model ranking. This16-case slice provides sampling rationale, not a final autonomous success or throughput comparison.'
    ],
    evidence=dict(closed_numeric_review_sha256=sha(INPUT),audit_script_sha256=sha(HERE/'audit.py'),
        summary_script_sha256=sha(__file__),successful_stdout_sha256=sha(HERE/'stdout_retry01.log'),
        preserved_startup_import_error_sha256=sha(HERE/'stdout.log')),
    simulator_starts=0,optimizer_updates=0,active_payloads_opened=0,rgb_payloads_opened=0,rgb_hashes_recomputed=0)
with OUT.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(B_summary=out['B_summary'],held=[dict(seed=c['seed'],classification=c['classification'],
    first=c['first_held']['completed_tick'] if c['first_held'] else None,earliest=c['earliest_allowed_completed_tick'],
    switch=c['takeover']['completed_tick'] if c['takeover'] else None,expert_labels=c['legal_expert_actions']) for c in out['held_cases']],
    same_initial_object_goal_hole_table=sum(x['object_goal_hole_table_poses_equal'] for x in initial),
    same_all_observable=sum(x['all_observable_fields_equal'] for x in initial),
    joint_max_difference_range_rad=[min(x['joint_max_absolute_difference_rad'] for x in initial),max(x['joint_max_absolute_difference_rad'] for x in initial)],
    palm_translation_difference_range_m=[min(x['palm_root_translation_difference_m'] for x in initial),max(x['palm_root_translation_difference_m'] for x in initial)]),indent=2))
