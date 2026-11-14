"""Read completed causal takeover cases without touching active HDF5 files.

CPU diagnostics only. The committed aggregate results index is the completion
barrier: a case absent there is pending even if its result file already exists.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from tools.tight_fresh_phases import authority_phases, first
from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_takeover import classify, validate_plan, CONTACT_HEIGHT_CONTRACT
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import stable_windows, contact_height_windows
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def label_alignment(fields):
    """Independently derive actual authority, whole-queue flush and label masks."""
    source = np.asarray(fields['control_source']).reshape(-1)
    valid = np.asarray(fields['expert_label_valid'], dtype=bool).reshape(-1)
    tick = np.arange(len(source))
    student = source == 0
    require(np.isin(source, [0, 1]).all() and np.array_equal(student, tick < student.sum()),
        'Expected one complete student prefix followed by expert authority')
    switch = int(student.sum()) if (source == 1).any() else None
    queue = np.asarray(fields['expert_queue_length']).reshape(-1)
    delay = np.asarray(fields['expert_delay_index']).reshape(-1)
    source_step = np.asarray(fields['expert_source_step']).reshape(-1)
    require(len(queue) == len(tick) and (queue >= 1).all() and (queue == queue[0]).all(),
        'Actual expert queue must retain one positive length')
    require(((delay >= 0) & (delay < queue)).all(), 'Delay index outside actual queue')
    require(np.array_equal(source_step, np.maximum(tick-delay, 0)),
        'Expert suggestion source does not match the recorded actual delay')
    expected_clean = np.zeros(len(tick), dtype=bool) if switch is None else tick >= switch + queue - 1
    expected_source = np.zeros(len(tick), dtype=bool) if switch is None else source_step >= switch
    expected_valid = expected_clean & expected_source
    require(np.array_equal(fields['switch_clean'], expected_clean), 'Whole-queue flush disagrees')
    require(np.array_equal(fields['expert_source_is_expert_issued'], expected_source),
        'Expert source-authority flag disagrees')
    require(np.array_equal(valid, expected_valid), 'Expert supervision mask disagrees with real queue flush')
    phase = np.column_stack((~student, valid))
    segments = np.r_[0, np.cumsum((phase[1:] != phase[:-1]).any(-1))]
    require(np.array_equal(fields['segment_id'], segments), 'Actual authority/label segment boundaries disagree')
    labels = np.flatnonzero(valid)
    return dict(passed=True, switch_action_index=switch,
        first_legal_label_action_index=int(labels[0]) if len(labels) else None,
        queue_length=int(queue[0]), expert_flush_masked_ticks=int(((~student) & ~valid).sum()),
        student_history_ticks=int(student.sum()), expert_ticks=int((~student).sum()),
        legal_expert_ticks=int(valid.sum()), student_prefix_supervised=False,
        tick_index='Zero-based action/label index; stable window end uses completed transition count')


def audit_numeric(path, result):
    with h5py.File(path, 'r') as stream:
        transition = stream['episode/transition']
        d = transition['dagger']
        fields = {key:d[key][:] for key in ('control_source', 'expert_label_valid',
            'expert_queue_length', 'expert_delay_index', 'expert_source_step', 'switch_clean',
            'expert_source_is_expert_issued', 'segment_id')}
        alignment = label_alignment(fields)
        valid = fields['expert_label_valid'].astype(bool)
        require(np.array_equal(d['u_expert'][:][valid], transition['target_after_rad'][:][valid]),
            'Valid expert label is not the target actually issued on that transition')
        require(np.array_equal(d['q_reached'][:], transition['diagnostics_after/joint_position_rad'][:]),
            'Reached configuration does not match actual same-transition post-state')
        source = fields['control_source']; student = source == 0
        after = transition['diagnostics_after']
        pose = after['object_pose_w'][:]; palm = after['palm_body_pose_w'][:]
        force = transition['object_contact_force_w'][:]
        require(len(pose) >= 48, 'Settled reference requires complete first 48 actual transitions')
        height = float(np.median(pose[24:48, 2]))
        stable, contact = stable_windows(pose, palm, force, height)
        # The causal trigger cannot qualify before the reference is available.
        stable[:47] = False
        contact_height=contact_height_windows(pose,force,height)
        contact_height[:47]=False
        two = (contact.sum(-1) >= 2).all(-1)
        goal = after['goal_pose_w'][:]
        distance = np.linalg.norm(pose[:, :3] - goal[:, :3], axis=-1)
        angle = 2*np.arccos(np.clip(np.abs((pose[:, 3:]*goal[:, 3:]).sum(-1)), 0, 1))
        near = transition['native_success_diagnostic/near_goal'][:].astype(bool)
        align = transition['native_alignment_condition'][:].astype(bool)
        subgoals = after['successes'][:]
        authority = authority_phases(source, stable, near, align, subgoals)
        stable_student = np.flatnonzero(student & stable)
        late_student = student & (np.arange(len(student)) >= 48)
        elevated = pose[:,2] >= height+.02
        def full_window_count(mask):
            complete = np.convolve(np.asarray(mask, int), np.ones(12, int), mode='full')[:len(mask)] == 12
            return int((complete & student & (np.arange(len(student)) >= 47)).sum())
        def milestone(index):
            if index is None:
                return None
            return dict(completed_tick=int(index)+1, stable_window=bool(stable[index]),
                two_contact_fingers_both_substeps=bool(two[index]),
                goal_distance_m=float(distance[index]), goal_quaternion_angle_rad=float(angle[index]),
                native_near_goal=bool(near[index]), native_alignment=bool(align[index]),
                native_subgoals=int(subgoals[index]))
        first_stable = int(stable_student[0]) if len(stable_student) else None
        post_stable_student = student & (np.arange(len(student)) >= first_stable) if first_stable is not None else np.zeros(len(student), bool)
        same_goal = post_stable_student & (subgoals == subgoals[first_stable]) if first_stable is not None else post_stable_student
        indices = np.flatnonzero(same_goal)
        latest_student = int(np.flatnonzero(student)[-1]) if student.any() else None
        phase = dict(authority=authority, first_student_stable=milestone(first_stable),
            student_first_contact_height_window_end_tick=first(contact_height & student),
            last_student_transition=milestone(latest_student),
            student_after_stable_same_goal_closest_center=milestone(int(indices[np.argmin(distance[indices])])) if len(indices) else None,
            student_after_stable_same_goal_smallest_angle=milestone(int(indices[np.argmin(angle[indices])])) if len(indices) else None,
            student_native_lift=bool((after['lifted_object'][:].astype(bool) & student).any()),
            student_first_contact_completed_tick=first(contact.any(axis=(1,2)) & student),
            student_first_two_fingers_completed_tick=first(two & student),
            student_max_later_rise_over_settled_m=float((pose[late_student,2]-height).max()) if late_student.any() else None,
            student_max_later_rise_with_two_fingers_m=float((pose[late_student & two,2]-height).max()) if (late_student & two).any() else None,
            student_window_counts=dict(contact_at_least_two_fingers=full_window_count(two),
                elevation_at_least_2cm=full_window_count(elevated),
                contact_and_elevation=full_window_count(two & elevated),
                complete_stable_with_relative_pose_bounds=int((stable & student).sum())),
            expert_legal_stable_ticks=int((stable & valid).sum()),
            expert_legal_near_goal_ticks=int((near & valid).sum()),
            expert_legal_alignment_ticks=int((align & valid).sum()),
            expert_first_subgoal_completed_tick=first((subgoals > 0) & ~student),
            final_native_subgoals=int(subgoals.max()), settled_reference_m=height)
    dagger = result['dagger']
    require(alignment['switch_action_index'] == dagger['recovery_coverage']['switch_tick'],
        'Actual numeric switch differs from result')
    require(alignment['first_legal_label_action_index'] == dagger['recovery_coverage']['population']['first_valid_expert_tick'],
        'Actual first legal label differs from result')
    require(alignment['legal_expert_ticks'] == dagger['expert_valid_ticks'], 'Actual legal label count differs')
    require(alignment['expert_ticks'] == dagger['expert_ticks'], 'Actual expert authority count differs')
    return alignment, phase


def report(collection):
    collection = Path(collection)
    plan = read(collection/'plan.json'); validate_plan(plan)
    contact_contract=plan.get('stage_takeover_contract')==CONTACT_HEIGHT_CONTRACT
    event='contact_height' if contact_contract else 'stable'
    audit_name=event+'_stage_audit.json'
    # Snapshot the final marker first: a worker finishing during this audit must
    # not turn an earlier partial index into a purported final denominator.
    finished = (collection/'result.json').exists() and read(collection/'result.json').get('software_complete', False)
    index_bytes = (collection/'results.json').read_bytes() if (collection/'results.json').exists() else None
    entry_bytes = (collection/'correction_entries.json').read_bytes() if (collection/'correction_entries.json').exists() else None
    rows = json.loads(index_bytes) if index_bytes is not None else []
    entries = json.loads(entry_bytes) if entry_bytes is not None else {}
    cases = {c['name']:c for c in plan['cases']}
    names = [r['case'] for r in rows]
    require(len(set(names)) == len(names) and set(names) <= set(cases), 'Completed index has duplicate or unregistered cases')
    receipts = []
    for row in rows:
        name = row['case']; result = row['result']; directory = collection/name
        require(row['role'] == cases[name]['role'], 'Completed case role differs')
        require(result.get('software_complete'), 'Incomplete case must not enter completed audit')
        stage = read(directory/audit_name)
        require(stage['source_plan_sha256'] == sha256(collection/'plan.json'), 'Case audit is bound to another plan')
        trigger = audit_trigger_case(directory, result) if result.get('legal') else None
        category = classify(result, trigger)
        require(category == stage['category'] and trigger == stage['causal_trigger'], 'Causal stage audit differs')
        accepted = bool(result.get('accepted_for_training'))
        require(accepted == (name in entries), 'Recorded retained subset disagrees')
        require(not accepted or category == 'expert_rescue_after_'+event, 'Retained record is not a successful registered-stage rescue')
        item = dict(case=name, role=row['role'], category=category, accepted=accepted,
            success=bool(result.get('success')), native_terminal_reasons=result.get('reasons'),
            result_sha256=sha256(directory/'result.json'), stage_audit_sha256=sha256(directory/audit_name))
        if result.get('legal'):
            require(read(directory/'capture_audit.json')['passed'], 'Collector capture provenance failed')
            alignment, phase = audit_numeric(directory/'numeric.h5', result)
            first_window=(phase['student_first_contact_height_window_end_tick'] if contact_contract else
                phase['authority']['student_first_stable_window_end_tick'])
            require(first_window == trigger['receipt']['first_'+event+'_window_end_completed_tick'],
                'Full numeric registered window disagrees with causal trigger')
            item.update(label_alignment=alignment, phases=phase, trigger=trigger,
                numeric_sha256=sha256(directory/'numeric.h5'))
            if accepted:
                require(entries[name]['takeover_tick'] == alignment['switch_action_index'], 'Retained switch is not actual switch')
                item['capture_sha256'] = entries[name]['source_sha256']
        receipts.append(item)
    if finished:
        require(len(receipts) == len(cases) and set(entries) <= set(names), 'Final opportunity accounting is incomplete')
    return dict(version='completed_'+event+'_takeover_audit_v1', created_unix=time.time(),
        plan_sha256=sha256(collection/'plan.json'), collector_source_commit=plan['source_commit'],
        completed_index_snapshot_sha256=hashlib.sha256(index_bytes).hexdigest() if index_bytes is not None else None,
        retained_index_snapshot_sha256=hashlib.sha256(entry_bytes).hexdigest() if entry_bytes is not None else None,
        planned=len(cases), completed=len(receipts), pending=[name for name in cases if name not in names],
        collection_complete=bool(finished), cases=receipts,
        by_role={role:dict(opportunities=sum(c['role']==role for c in plan['cases']),
            completed=sum(r['role']==role for r in receipts),
            categories=dict(Counter(r['category'] for r in receipts if r['role']==role)),
            accepted=sum(r['accepted'] and r['role']==role for r in receipts),
            retained_legal_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in receipts if r['accepted'] and r['role']==role))
            for role in ('A','B')},
        scope='Only cases committed to results.json are opened; pending active HDF5 files are never read. No GPU, policy update or trial.',
        interpretation='Expert rescue is not autonomous success. Angles ignore shape symmetry; native alignment/near-goal remain separate. Delays are assigned to different families and do not identify a causal delay effect. Legal source ticks are not optimizer exposure.',
        test_used=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, report(args.collection))


if __name__ == '__main__':
    main()
