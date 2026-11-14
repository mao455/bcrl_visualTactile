"""CPU-only completed fixed/causal collection review; no policy execution."""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from tools.tight_fresh_stage_audit import report as stage_report
from tools.tight_fresh_takeover_pose_audit import window_components
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import contact_height_windows
from visuotactile.training_resources import atomic_json
from visuotactile.training_expansion import (COLLECTION_CONTRACT, read_registry,
    validate_capture_binding, validate_expansion_entry)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(rows):
    """Keep attempted, legal, triggered, recovered and retained denominators separate."""
    legal = [r for r in rows if r['legal']]
    retained = [r for r in legal if r['accepted']]
    causal = [r for r in rows if r['causal_trigger']]
    fixed = [r for r in rows if not r['causal_trigger']]
    def schedule_counts(group):
        admitted = [r for r in group if r['legal']]
        return dict(attempts=len(group), legal_initializations=len(admitted),
            takeovers=sum(r['boundary'] is not None for r in admitted),
            causal_condition_met=sum(r['causal_condition_met'] is True for r in admitted),
            expert_recovery_successes=sum(r['expert_recovery_success'] for r in admitted),
            accepted=sum(r['accepted'] for r in admitted))
    known_reference = [r for r in legal if r['student_settled_reference_available']]
    return dict(attempts=len(rows), legal_initializations=len(legal),
        invalid_initializations=len(rows)-len(legal),
        triggered=sum(r['boundary'] is not None for r in causal if r['legal']),
        causal_condition_met=sum(r['causal_condition_met'] is True for r in causal if r['legal']),
        fixed_takeovers=sum(r['boundary'] is not None for r in fixed if r['legal']),
        causal_opportunities=schedule_counts(causal), fixed_opportunities=schedule_counts(fixed),
        recovered=sum(r['expert_recovery_success'] for r in legal), accepted=len(retained),
        all_valid_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in legal),
        accepted_valid_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in retained),
        unretained_valid_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in legal if not r['accepted']),
        attempted_families=sorted({r['family'] for r in rows}),
        accepted_families=sorted({r['family'] for r in retained}),
        attempted_new_families=sorted({r['family'] for r in rows if r['registered_new_family']}),
        accepted_new_families=sorted({r['family'] for r in retained if r['registered_new_family']}),
        student_native_lift=sum(r['student_native_lift'] for r in legal),
        student_old_stable_window=sum(r['student_stable_window_seen'] for r in known_reference),
        student_window_reference_available=len(known_reference),
        student_window_reference_unavailable=len(legal)-len(known_reference),
        student_autonomous_successes=sum(r['student_autonomous_success'] for r in legal),
        categories=dict(Counter(r['category'] for r in rows)),
        by_schedule={schedule:schedule_counts([r for r in rows if r['schedule']==schedule])
            for schedule in sorted({r['schedule'] for r in rows})},
        retained_coverage_ticks={key:sum(r['coverage_ticks']['valid_expert'][key] or 0 for r in retained)
            for key in retained[0]['coverage_ticks']['valid_expert']} if retained else {},
        all_valid_expert_coverage_ticks={key:sum(r['coverage_ticks']['valid_expert'][key] or 0 for r in legal)
            for key in legal[0]['coverage_ticks']['valid_expert']} if legal else {},
        expert_window_reference_unavailable=sum(r.get('settled_reference_authority')=='unavailable' for r in legal))


def review(root):
    root = Path(root).resolve()
    plan = read(root/'plan.json')
    stage = stage_report(root)
    require(stage['collection_complete'] and not stage['pending'], 'Complete audited collection required')
    owner = read(root/'owner_finished.json')
    require(owner['software_complete'] and owner['returncode'] == 0, 'Completed owner required')
    frozen = Path(read(root/'owner_started.json')['cwd'])
    for path, digest in plan['source_sha256'].items():
        require(sha256(frozen/path) == digest, 'Frozen collection source changed: '+path)
    indexed = {r['case']:r['result'] for r in read(root/'results.json')}
    entries = read(root/'correction_entries.json')
    cases = {c['name']:c for c in plan['cases']}
    expanded = plan.get('stage_takeover_contract') == COLLECTION_CONTRACT
    rows = []
    for audit in stage['cases']:
        name = audit['case']; result = indexed[name]; case = cases[name]
        common = dict(case=name, role=case['role'], family=case['spec']['family'],
            category=audit['category'], accepted=audit['accepted'], legal=bool(result.get('legal')),
            causal_trigger=audit['causal_trigger'], schedule=case.get('expansion_schedule', 'contact_height'),
            registered_new_family=case['spec'].get('training_expansion') is not None,
            result_sha256=audit['result_sha256'])
        if not result.get('legal'):
            rows.append(dict(common, boundary=None, expert_recovery_success=False,causal_condition_met=None))
            continue
        receipt=(audit.get('trigger') or {}).get('receipt',{})
        event_key=next((key for key in ('first_held_near_goal_window_end_completed_tick',
            'first_contact_height_window_end_completed_tick','first_stable_window_end_completed_tick') if key in receipt),None)
        causal_condition_met=receipt[event_key] is not None if event_key is not None else None
        require(result['residual_ticks'] == result['steps'] and result['residual_forward_ticks'] == result['steps'],
            'Full DP/backend must process each actual observation')
        require((result['physics_hz'], result['simulation_step_hz'], result['plan_hz']) == (120, 60, 5),
            'Native timing contract changed')
        policy = result['complete_student_policy']
        pair = plan['pairs'][case['role']]
        require(all(policy[k] == pair[k] for k in ('frontend_sha256', 'backend_sha256')), 'Source policy differs')
        with h5py.File(root/name/'numeric.h5') as stream:
            if expanded:
                validate_capture_binding(case['spec'], json.loads(stream.attrs['metadata_json']))
                if audit['accepted'] and common['registered_new_family']:
                    validate_expansion_entry(entries[name], read(case['source_data']), case['spec']['training_expansion'])
                    validate_capture_binding(entries[name], json.loads(stream.attrs['metadata_json']))
                    require(entries[name]['source_sha256']==result['dataset_capture']['sha256'] and
                        entries[name]['episode']==case['spec']['identity'] and
                        all(entries[name][key]==pair[key] for key in ('frontend_sha256','backend_sha256')),
                        'Accepted capture/policy binding differs from completed opportunity')
            t = stream['episode/transition']; before = t['diagnostics_before']; after = t['diagnostics_after']
            pose = after['object_pose_w'][:]; palm = after['palm_body_pose_w'][:]
            force = t['object_contact_force_w'][:]; total = len(pose)
            times0, times1 = t['time_before_s'][:], t['time_after_s'][:]
            require(total == result['steps'] and np.array_equal(stream['episode/obs/time_s'][:], times0)
                and np.array_equal(times1[:-1], times0[1:]) and np.allclose(times1-times0, 1/60, rtol=0, atol=1e-9)
                and np.array_equal(t['step_id'][:], np.arange(total)), 'Actual observation/control clock differs')
            require((t['episode_id'][:] == t['episode_id'][0]).all(), 'Lane episode identity changed')
            for key in ('object_pose_w', 'palm_body_pose_w', 'goal_pose_w', 'successes'):
                require(np.array_equal(after[key][:-1], before[key][1:]), 'Adjacent state differs: '+key)
            source = t['dagger/control_source'][:]; student = source == 0
            valid = t['dagger/expert_label_valid'][:].astype(bool)
            height = float(np.median(pose[24:48, 2])) if total>=48 else float('inf')
            reference_available = audit['phases']['student_settled_reference_available']
            held = contact_height_windows(pose, force, height) if total>=48 else np.zeros(total,bool)
            held[:47] = False
            if not reference_available:
                held[student] = False
            contact = np.linalg.norm(force, axis=-1) > .1
            two = (contact.sum(-1) >= 2).all(-1)
            keypoint = t['native_keypoints_max_dist'][:]
            stage_before, stage_after = before['successes'][:], after['successes'][:]
            same_goal = (before['goal_pose_w'][:] == after['goal_pose_w'][:]).all(-1) & (stage_before == stage_after)
            goal_angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*before['goal_pose_w'][:,3:]).sum(-1)),0,1))
            prefix_goal=student & same_goal & (stage_before==0)
            eligible=np.flatnonzero(prefix_goal)
            closest=int(eligible[np.argmin(keypoint[eligible])]) if len(eligible) else None
            approach=None if closest is None else dict(completed_tick=closest+1,
                native_keypoint_distance_m=float(keypoint[closest]),quaternion_angle_rad=float(goal_angle[closest]),
                two_fingers_both_substeps=bool(two[closest]),
                held_window=bool(held[closest]) if reference_available else None)
            rotating = np.zeros(total, bool)
            for i in np.flatnonzero(held):
                rotating[i] = window_components(pose, palm, force, height, int(i)+1)['palm_relative_max_rotation_rad'] > .2
            first_contact = np.flatnonzero(contact.any(axis=(1, 2)))
            masks = dict(before_first_contact=np.arange(total) < (int(first_contact[0]) if len(first_contact) else total),
                two_fingers_each_physics_sample=two, completed_contact_height_window=held,
                held_palm_relative_rotation_over_0p2rad=held & rotating,
                held_same_goal_keypoint_over60mm=held & same_goal & (keypoint > .06),
                held_same_goal_keypoint_within60mm=held & same_goal & (keypoint <= .06),
                native_near_goal=t['native_success_diagnostic/near_goal'][:].astype(bool),
                native_alignment=t['native_alignment_condition'][:].astype(bool),
                after_first_subgoal=stage_before > 0, native_subgoal_increment=stage_after > stage_before)
            switch = audit['label_alignment']['switch_action_index']
            boundary = None
            if switch is not None:
                i = switch-1
                goal = before['goal_pose_w'][i]
                boundary = dict(completed_student_tick=switch, held_window=bool(held[i]) if reference_available else None,
                    original_stable_window=audit['phases']['last_student_transition']['stable_window'] if reference_available else None,
                    two_fingers_both_substeps=bool(two[i]),
                    height_above_settled_m=float(pose[i, 2]-height) if reference_available else None,
                    root_goal_distance_m=float(np.linalg.norm(pose[i, :3]-goal[:3])),
                    quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[i, 3:]*goal[3:]).sum()), 0, 1))),
                    native_keypoint_distance_m=float(keypoint[i]),
                    goal_pre_post_unchanged=bool(same_goal[i]), native_subgoals=int(stage_after[i]))
            counts = {authority:{key:int((mask & scope).sum()) for key,mask in masks.items()}
                for authority,scope in [('student', student), ('valid_expert', valid)]}
            if not reference_available:
                for key in counts['student']:
                    if key == 'completed_contact_height_window' or key.startswith('held_'):
                        counts['student'][key] = None
                        if total<48:
                            counts['valid_expert'][key] = None
            # Contact duration does not need an estimated settled-height baseline.
            run = longest = 0
            for value in two[student]:
                run = run+1 if value else 0
                longest = max(longest, run)
        if (root/name/'capture_complete.json').exists():
            capture_receipt = read(root/name/'capture_complete.json')
            require(capture_receipt['dataset_capture'] == result['dataset_capture'], 'Verified capture receipt differs')
        rows.append(dict(common,
            label_alignment=audit['label_alignment'], boundary=boundary,
            causal_condition_met=causal_condition_met,
            first_qualifying_condition_completed_tick=receipt.get(event_key) if event_key else None,
            expert_recovery_success=bool(result['dagger']['expert_takeover_success']),
            student_native_lift=audit['phases']['student_native_lift'],
            student_stable_window_seen=(audit['phases']['first_student_stable'] is not None) if reference_available else None,
            student_settled_reference_available=reference_available,
            settled_reference_authority=audit['phases']['settled_reference_authority'],
            student_longest_two_finger_contact_ticks=int(longest),
            student_closest_original_goal=approach,
            student_original_goal_min_quaternion_angle_rad=float(goal_angle[prefix_goal].min()) if prefix_goal.any() else None,
            student_autonomous_success=result['dagger']['student_autonomous_success'],
            coverage_ticks=counts, native_termination_reasons=result['reasons'],
            numeric_sha256=audit['numeric_sha256'],
            source_capture_sha256=result['dataset_capture']['sha256']))
    retained = [r for r in rows if r['accepted']]
    require({r['case'] for r in retained} == set(entries), 'Retained entries differ')
    summary = summarize(rows)
    registry = read_registry(plan['training_expansion_registry']) if expanded else None
    return dict(version='completed_training_expansion_collection_review_v1' if expanded else 'completed_contact_height_collection_review_v1', root=str(root), summary=summary,
        cases=rows, by_role=stage['by_role'],
        registered_new_families=sorted(registry['families']) if registry else [],
        by_role_summary={role:summarize([r for r in rows if r['role']==role]) for role in ('A','B')},
        source=dict(commit=plan['source_commit'], frozen_tree=str(frozen),
            checked_source_files=len(plan['source_sha256']), plan_sha256=sha256(root/'plan.json'),
            results_sha256=sha256(root/'results.json'), entries_sha256=sha256(root/'correction_entries.json'),
            owner_sha256=sha256(root/'owner_finished.json'), review_code_sha256=sha256(__file__)),
        assertions=dict(causal_trigger_replay=True, complete_queue_flush_labels=True,
            full_student_every_transition=True, native_clock_120_60_5=True,
            adjacent_state_continuity=True, source_identity=True),
        definitions=dict(coverage='Overlapping diagnostic masks, not mutually exclusive phases or optimizer exposure.',
            denominators='triggered counts actual causal takeovers; causal_condition_met also counts completed qualifying windows whose scheduled takeover was never reached. Fixed takeovers have separate attempted/legal denominators. Recovery success and retained training records are separate.',
            short_prefix='Before 48 student transitions, the settled-height reference is unavailable: dependent prefix windows/height are null, not negative findings. Expert-window coverage may use a mixed-authority reference and is descriptive only.',
            held='12 completed transitions, both physics samples >=2 fingers >0.1N and height>=settled+2cm; rotation allowed.',
            rotation='Palm-relative rotation over the same past12 window exceeds0.2rad; does not identify intent, slip or drop.',
            near60='Diagnostic coverage only; native keypoint15mm success unchanged.',
            after_first_subgoal='Pre-action native stage>0; task later-goal/insertion motions, not a new success definition.',
            authority='Expert recovery and its later insertion do not establish student autonomous success.'),
        optimizer_updates=0, simulator_starts=0, test_trajectories_opened=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier evidence')
    atomic_json(args.output, review(args.collection))


if __name__ == '__main__':
    main()
