"""CPU-only completed contact-height collection review; no policy execution."""
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


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    rows = []
    for audit in stage['cases']:
        name = audit['case']; result = indexed[name]; case = cases[name]
        require(result['residual_ticks'] == result['steps'] and result['residual_forward_ticks'] == result['steps'],
            'Full DP/backend must process each actual observation')
        require((result['physics_hz'], result['simulation_step_hz'], result['plan_hz']) == (120, 60, 5),
            'Native timing contract changed')
        policy = result['complete_student_policy']
        pair = plan['pairs'][case['role']]
        require(all(policy[k] == pair[k] for k in ('frontend_sha256', 'backend_sha256')), 'Source policy differs')
        with h5py.File(root/name/'numeric.h5') as stream:
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
            height = float(np.median(pose[24:48, 2]))
            held = contact_height_windows(pose, force, height); held[:47] = False
            contact = np.linalg.norm(force, axis=-1) > .1
            two = (contact.sum(-1) >= 2).all(-1)
            keypoint = t['native_keypoints_max_dist'][:]
            stage_before, stage_after = before['successes'][:], after['successes'][:]
            same_goal = (before['goal_pose_w'][:] == after['goal_pose_w'][:]).all(-1) & (stage_before == stage_after)
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
                boundary = dict(completed_student_tick=switch, held_window=bool(held[i]),
                    original_stable_window=audit['phases']['last_student_transition']['stable_window'],
                    two_fingers_both_substeps=bool(two[i]),
                    height_above_settled_m=float(pose[i, 2]-height),
                    root_goal_distance_m=float(np.linalg.norm(pose[i, :3]-goal[:3])),
                    quaternion_angle_rad=float(2*np.arccos(np.clip(abs((pose[i, 3:]*goal[3:]).sum()), 0, 1))),
                    native_keypoint_distance_m=float(keypoint[i]),
                    goal_pre_post_unchanged=bool(same_goal[i]), native_subgoals=int(stage_after[i]))
            counts = {authority:{key:int((mask & scope).sum()) for key,mask in masks.items()}
                for authority,scope in [('student', student), ('valid_expert', valid)]}
        if (root/name/'capture_complete.json').exists():
            receipt = read(root/name/'capture_complete.json')
            require(receipt['dataset_capture'] == result['dataset_capture'], 'Verified capture receipt differs')
        rows.append(dict(case=name, role=case['role'], family=case['spec']['family'],
            category=audit['category'], accepted=audit['accepted'],
            label_alignment=audit['label_alignment'], boundary=boundary,
            student_native_lift=audit['phases']['student_native_lift'],
            student_stable_window_seen=audit['phases']['first_student_stable'] is not None,
            student_autonomous_success=result['dagger']['student_autonomous_success'],
            coverage_ticks=counts, native_termination_reasons=result['reasons'],
            numeric_sha256=audit['numeric_sha256'], result_sha256=audit['result_sha256'],
            source_capture_sha256=result['dataset_capture']['sha256']))
    retained = [r for r in rows if r['accepted']]
    require({r['case'] for r in retained} == set(entries), 'Retained entries differ')
    summary = dict(attempts=len(rows), triggered=sum(r['boundary'] is not None for r in rows),
        recovered=len(retained), accepted=len(retained),
        all_valid_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in rows),
        accepted_valid_expert_ticks=sum(r['label_alignment']['legal_expert_ticks'] for r in retained),
        attempted_families=sorted({r['family'] for r in rows}),
        accepted_families=sorted({r['family'] for r in retained}),
        student_native_lift=sum(r['student_native_lift'] for r in rows),
        student_old_stable_window=sum(r['student_stable_window_seen'] for r in rows),
        student_autonomous_successes=sum(r['student_autonomous_success'] for r in rows),
        categories=dict(Counter(r['category'] for r in rows)),
        retained_coverage_ticks={key:sum(r['coverage_ticks']['valid_expert'][key] for r in retained)
            for key in retained[0]['coverage_ticks']['valid_expert']} if retained else {})
    return dict(version='completed_contact_height_collection_review_v1', root=str(root), summary=summary,
        cases=rows, by_role=stage['by_role'],
        source=dict(commit=plan['source_commit'], frozen_tree=str(frozen),
            checked_source_files=len(plan['source_sha256']), plan_sha256=sha256(root/'plan.json'),
            results_sha256=sha256(root/'results.json'), entries_sha256=sha256(root/'correction_entries.json'),
            owner_sha256=sha256(root/'owner_finished.json'), review_code_sha256=sha256(__file__)),
        assertions=dict(causal_trigger_replay=True, complete_queue_flush_labels=True,
            full_student_every_transition=True, native_clock_120_60_5=True,
            adjacent_state_continuity=True, source_identity=True),
        definitions=dict(coverage='Overlapping diagnostic masks, not mutually exclusive phases or optimizer exposure.',
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
