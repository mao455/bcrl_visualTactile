"""Read historical student prefixes to prepare a finite new collection cohort."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.tight_fresh_contact_height_coverage import causal_contact_height, schedule_scope
from tools.tight_fresh_expert_stability_coverage import require, read, first
from tools.tight_fresh_rollouts import completed_pair
from tools.tight_fresh_stage_takeover import validate_requests
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import stable_windows
from visuotactile.training_resources import atomic_json


PROPOSAL = [('formal_050_slow_3', 12), ('formal_051_slow_5', 0),
    ('formal_007_slow_3', 12), ('formal_053_slow_0', 0),
    ('formal_031_slow_4', 12), ('formal_047_slow_4', 0),
    ('formal_039_slow_5', 0), ('formal_046_slow_5', 0)]


def quat_angle(q, other):
    return float(2*np.arccos(np.clip(abs((q*other).sum()), 0, 1)))


def relative_pose(obj, palm):
    inverse = palm[3:].copy(); inverse[1:] *= -1
    qv = inverse[1:]; vector = obj[:3]-palm[:3]
    cross = 2*np.cross(qv, vector)
    location = vector+inverse[:1]*cross+np.cross(qv, cross)
    q = obj[3:]
    orientation = np.r_[inverse[0]*q[0]-(qv*q[1:]).sum(),
        inverse[0]*q[1:]+q[0]*qv+np.cross(qv, q[1:])]
    return location, orientation


def delay_state(arrays, windows, height, completed, delay, total):
    n = len(arrays['object'])
    scope = schedule_scope(completed, delay, n, total)
    if not scope['action_state_observed']:
        return scope
    action = scope['proposed_action_index']; i = action-1; start = completed-1
    obj, palm = arrays['object'], arrays['palm']
    pose, hand = obj[i], palm[i]
    # A candidate acts AFTER the last completed transition: its active target is goal_after.
    goal = arrays['goal_after'][i]
    two = ((np.linalg.norm(arrays['force'], axis=-1) > .1).sum(-1) >= 2).all(-1)
    interval = slice(completed, action)
    rel, rot = relative_pose(pose, hand)
    rel0, rot0 = relative_pose(obj[start], palm[start])
    stable, _ = stable_windows(obj[i-11:i+1], palm[i-11:i+1], arrays['force'][i-11:i+1], height)
    return dict(scope, preceding_completed_tick=action,
        contact_height_window_at_action_state=bool(windows[i]),
        frozen_stable_at_action_state=bool(stable[-1]),
        two_contact_fingers_both_substeps=bool(two[i]),
        post_trigger_contact_gap_ticks=int((~two[interval]).sum()),
        post_trigger_below_height_ticks=int((obj[interval, 2] < height+.02).sum()),
        object_rise_over_settled_m=float(pose[2]-height),
        object_to_active_goal_root_distance_m=float(np.linalg.norm(pose[:3]-goal[:3])),
        object_to_active_goal_angle_rad=quat_angle(pose[3:], goal[3:]),
        object_world_translation_since_trigger_m=float(np.linalg.norm(pose[:3]-obj[start, :3])),
        object_world_rotation_since_trigger_rad=quat_angle(pose[3:], obj[start, 3:]),
        palm_relative_translation_since_trigger_m=float(np.linalg.norm(rel-rel0)),
        palm_relative_rotation_since_trigger_rad=quat_angle(rot, rot0),
        active_goal_unchanged_since_trigger=bool(np.array_equal(goal, arrays['goal_after'][start])),
        native_subgoals=int(arrays['subgoals_after'][i]),
        previous_transition_native_near_goal=bool(arrays['near'][i]),
        previous_transition_native_lift=bool(arrays['lifted'][i]),
        before_first_native_near_goal=not bool(arrays['near'][:action].any()))


def validate_historical_result(raw, indexed, admitted):
    additions = {'accepted_for_training', 'actual_exposure'}
    require(not (set(raw) & additions) and raw == {key:value for key, value in indexed.items()
        if key not in additions} and indexed['accepted_for_training'] is admitted,
        'Historical result or actual admission differs')
    if 'actual_exposure' in indexed:
        # This accounting is also produced for unsuccessful expert suffixes.
        # Its presence alone never admits those labels to training.
        value = indexed['actual_exposure']
        require(value['episodes'] == 1 and value['history_ticks'] == raw['steps']
            and value['supervised_ticks'] == raw['dagger']['expert_valid_ticks']
            and value['supervised_anchors'] == raw['dagger']['recovery_coverage']['population']['dp_anchors'],
            'Historical potential supervision counts differ from the actual trajectory')


def historical_case(root, planned, indexed, entries, originals):
    spec = planned['spec']; name = planned['name']; result = indexed['result']
    family = spec['family']; identity = spec['source_identity']
    require(spec['split'] == result['split'] == 'train' and identity in originals
        and originals[identity]['family'] == family == result['family']
        and result['software_complete'] is True and result['legal'] is True,
        'Complete original30 training-family opportunity required before numeric read')
    case = root/name
    result_hash = sha256(case/'result.json'); numeric_hash = sha256(case/'numeric.h5')
    raw = read(case/'result.json')
    validate_historical_result(raw, result, name in entries)
    total = result['steps']
    with h5py.File(case/'numeric.h5', 'r') as stream:
        meta = json.loads(stream.attrs['metadata_json'])
        require(meta['dataset_split'] == 'train' and meta['initial_family'] == family
            and meta['source_identity'] == identity, 'Raw train identity differs')
        t = stream['episode/transition']; source = t['dagger/control_source'][:]
        valid = t['dagger/expert_label_valid'][:]; segment = t['dagger/segment_id'][:]
        n = int((source == 0).sum())
        require(source.shape == valid.shape == segment.shape == (total,) and np.isin(source, [0, 1]).all()
            and np.array_equal(source == 0, np.arange(total) < n) and n == result['dagger']['student_ticks']
            and total-n == result['dagger']['expert_ticks'] and not valid[:n].any()
            and (segment[:n] == segment[0]).all(), 'Real prefix/label authority differs')
        pair = read(root/'plan.json')['pairs'][planned['role']]
        require(meta['complete_student_policy'] == pair and result['checkpoint_sha256'] == pair['backend_sha256'],
            'Actual historical complete pair differs')
        fields = dict(object='diagnostics_after/object_pose_w', palm='diagnostics_after/palm_body_pose_w',
            force='object_contact_force_w', goal_after='diagnostics_after/goal_pose_w',
            subgoals_after='diagnostics_after/successes', near='native_success_diagnostic/near_goal',
            lifted='diagnostics_after/lifted_object')
        arrays = {key:t[path][:n] for key, path in fields.items()}
        require(all(np.isfinite(value).all() for value in arrays.values()), 'Finite prefix data required')
        for field in ('object_pose_w', 'palm_body_pose_w', 'goal_pose_w', 'successes'):
            require(np.array_equal(t['diagnostics_after/'+field][:n-1], t['diagnostics_before/'+field][1:n]),
                'Prefix adjacent actual state differs: '+field)
        before, after = t['time_before_s'][:n], t['time_after_s'][:n]
        require(np.array_equal(before, stream['episode/obs/time_s'][:n])
            and np.array_equal(after[:-1], before[1:])
            and np.array_equal(t['step_id'][:n], np.arange(n))
            and np.allclose(after-before, 1/60, rtol=0, atol=1e-9), 'Prefix60Hz timing differs')
        require(not any(value[:max(n-1, 0)].any() for value in t['termination_reasons'].values()),
            'A native terminal boundary occurs inside the prefix')
    windows, height = causal_contact_height(arrays['object'], arrays['force'])
    completed = first(windows)
    delay = {str(value):delay_state(arrays, windows, height, completed, value, total) for value in (0, 12, 24)}
    require(sha256(case/'result.json') == result_hash and sha256(case/'numeric.h5') == numeric_hash,
        'Historical evidence changed while reading')
    return dict(collection=root.name, case=name, role=planned['role'], family=family, source_identity=identity,
        total_ticks=total, actual_student_prefix_ticks=n, settled_reference_available=n >= 48,
        prefix_ending='native_termination' if n == total else 'actual_expert_intervention',
        contact_height_endpoints=int(windows.sum()), first_contact_height_completed_tick=completed,
        actual_expert_switch_action_index=n if n < total else None,
        actual_student_autonomous_success=result['dagger']['student_autonomous_success'],
        actual_expert_rescue_success=result['dagger']['expert_takeover_success'], admitted=name in entries,
        historical_pair={key:pair[key] for key in ('frontend_sha256', 'backend_sha256')},
        source_sha256=dict(result=result_hash, numeric=numeric_hash), delay_comparison=delay)


def delay_summary(rows, delay):
    states = [row['delay_comparison'][str(delay)] for row in rows]
    observed = [row for row in states if row['action_state_observed']]
    return dict(qualifying_prefixes=sum(row['first_completed_tick'] is not None for row in states),
        observed_action_states=len(observed), contact_height_window_still_true=sum(row['contact_height_window_at_action_state'] for row in observed),
        two_contact_fingers_still_true=sum(row['two_contact_fingers_both_substeps'] for row in observed),
        any_contact_gap_after_trigger=sum(row['post_trigger_contact_gap_ticks'] > 0 for row in observed),
        before_first_native_near_goal=sum(row['before_first_native_near_goal'] for row in observed),
        scopes=dict(Counter(row['scope'] for row in states)))


def proposal_rows(families):
    reasons = {
        'formal_050': 'A recent student continues rotating with continuous contact over12ticks; B loses two-finger contact over the same delay. Test both actual states, with no guaranteed rescue claim.',
        'formal_051': 'Recent B has a new early contact-height opportunity absent from stable-only collection; choose immediate takeover before its observed later contact gaps.',
        'formal_007': 'Recent B rotates during12ticks but its contact becomes intermittent; older B retained contact. Collect the actual delayed state to test recovery after student motion.',
        'formal_053': 'Recent B has an early contact-height opportunity but loses two-finger contact by12ticks; choose immediate takeover.',
        'formal_031': 'Only the original demonstration is admitted; older A qualified at132, rotated with continuous contact through144, while actual takeover at220 failed. This tests a distinct earlier state.',
        'formal_047': 'Older B qualified at149, then lost two-finger contact by161; actual fallback360 failed. Immediate takeover tests a distinct earlier state.',
        'formal_039': 'Older B first qualified at344 without a frozen stable window; actual fallback360 succeeded. Immediate state precedes that rescue by16ticks; newer-policy qualification remains unknown.',
        'formal_046': 'Older A qualified near the active goal root with large orientation error at259; actual delayed switch307 succeeded. Latest old-pair opportunity never qualified, so this is a conditional late-state opportunity, not guaranteed coverage.'}
    return [dict(identity=identity, family=identity.rsplit('_slow_', 1)[0], delay_ticks=delay,
        roles=['A', 'B'], planned_opportunities=2, reason=reasons[identity.rsplit('_slow_', 1)[0]],
        existing_family_evidence=families[identity.rsplit('_slow_', 1)[0]]) for identity, delay in PROPOSAL]


def run(source, collections, exposure_report, output):
    began = time.monotonic(); source = Path(source).resolve(); output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve earlier collection preparation evidence')
    torch.set_num_threads(2)
    data = {role:read(source/role/'state_data/data.json') for role in ('A', 'B')}
    require(data['A']['normalization_train_ids'] == data['B']['normalization_train_ids']
        and data['A']['entries'] == data['B']['entries'] and data['A']['splits'] == data['B']['splits'], 'Current paired population differs')
    original_ids = data['A']['normalization_train_ids']
    originals = {name:data['A']['entries'][name] for name in original_ids}
    require(len(originals) == len({entry['family'] for entry in originals.values()}) == 30, 'Distinct original30 required')
    identities, delays = zip(*PROPOSAL)
    for value in data.values():
        validate_requests(value, list(identities), list(delays))
    pairs = {role:completed_pair(source, role) for role in ('A', 'B')}
    exposure = read(exposure_report)
    for role, value in exposure['roles'].items():
        require(value['data_sha256'] == sha256(source/role/'state_data/data.json')
            and value['learning_sha256'] == sha256(source/role/'training/learning.jsonl')
            and value['backend_last_sha256'] == sha256(source/role/'backend'/role/'training/last.pt')
            and value['backend_finished_sha256'] == sha256(source/role/'backend'/role/'training/finished.json'),
            'Actual source family learning exposure changed')
    rows, roots = [], []
    for directory in collections:
        root = Path(directory).resolve(); plan, completed, final, entries = [read(root/name) for name in
            ('plan.json', 'results.json', 'result.json', 'correction_entries.json')]
        require(plan['kind'] == 'takeover' and plan['test_used'] is False
            and final['software_complete'] is True and final['worker_error'] is None
            and final['cases'] == len(completed) == len(plan['cases']), 'Completed finite historical collection required')
        index = {row['case']:row for row in completed}
        require(set(index) == {row['name'] for row in plan['cases']}, 'Completed population differs')
        for planned in plan['cases']:
            rows.append(historical_case(root, planned, index[planned['name']], entries, originals))
        roots.append(dict(path=str(root), hashes={name:sha256(root/name) for name in
            ('plan.json', 'results.json', 'result.json', 'correction_entries.json')}))
    require(len(rows) == 84 and len({row['case'] for row in rows}) == 84, 'Expected all84 opportunities in5 collections')
    counts = Counter(data['A']['entries'][name]['family'] for name in data['A']['splits']['train'])
    families = {}
    for name, original in originals.items():
        family = original['family']; past = [row for row in rows if row['family'] == family]
        qualified = [row for row in past if row['first_contact_height_completed_tick'] is not None]
        families[family] = dict(original_identity=name, admitted_training_episodes=counts[family],
            admitted_corrections=counts[family]-1, historical_opportunities=len(past),
            qualified_historical_prefixes=len(qualified), qualified_historical_cases=[row['case'] for row in qualified],
            full_native_student_failures=sum(row['prefix_ending'] == 'native_termination'
                and not row['actual_student_autonomous_success'] for row in past),
            prefixes_censored_by_expert=sum(row['prefix_ending'] == 'actual_expert_intervention' for row in past),
            longest_observed_student_prefix={role:max((row['actual_student_prefix_ticks'] for row in past if row['role'] == role), default=0) for role in ('A', 'B')},
            actual_current_training_exposure={role:exposure['roles'][role]['actual_family_exposure'][family] for role in ('A', 'B')})
    qualified = [row for row in rows if row['first_contact_height_completed_tick'] is not None]
    require(not any(row['historical_pair']['backend_sha256'] == pairs[row['role']]['backend_sha256'] for row in rows),
        'Report interpretation expects historical model prefixes, not current pair samples')
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA use')
    project = Path(__file__).resolve().parents[1]
    report = dict(version='next_contact_height_collection_coverage_v1', complete=True, proposed_source=str(source),
        source_pairs=pairs, source_data_sha256={role:sha256(source/role/'state_data/data.json') for role in ('A', 'B')},
        exposure_report_sha256=sha256(exposure_report), historical_collections=roots,
        historical_summary=dict(opportunities=len(rows), families=len(families),
            prefixes_with_reference=sum(row['settled_reference_available'] for row in rows),
            qualified_prefixes=len(qualified), qualified_families=len({row['family'] for row in qualified}),
            current_source_pair_prefixes=0, delays={str(delay):delay_summary(rows, delay) for delay in (0, 12, 24)}),
        proposed_cohort=dict(status='CPU preparation only; Agent selects after pending autonomous study finishes',
            opportunities=16, unique_original_families=8, source_identities=list(identities), assigned_delays=list(delays),
            no_fixed_time_fallback=True, excluded_formal029=True, delay24='read-only diagnostic; not an executable request',
            historical_delay_assignment_is_not_a_causal_delay_comparison=True, cases=proposal_rows(families)),
        family_coverage=families, cases=rows,
        semantics=dict(physical_scope='All84 completed training collection opportunities across5 old policy generations. Only recorded student prefixes supply physical state; admitted expert tails do not create candidate windows. An early actual intervention censors later student coverage.',
            triggers='Fixed contact-height12 rule, causal settled reference, no relative-pose admission. Delays0/12/24 compare states along the SAME original student prefix until its actual intervention or native termination. They do not simulate an alternate expert suffix.',
            goal='Candidate actionN uses state after completedN, including goal_after. Motion/angle are root-frame diagnostics, not symmetry-aware insertion success or a guarantee of intentional rotation.',
            selection='Training families only.16 finite opportunities with paired A/B, one assigned delay per unique family, one shared reset/event registration per pair. Retain nonqualification and failed rescues; no fixed-time fallback or outcome-based replacement.',
            model_limit='The proposed source is the completed round05 pair. Historical candidate counts are from older pairs and cannot estimate its qualification or success probability. No current-model training-family trajectories are present in this audit.',
            repetition='No formal029 opportunity. Previously observed family motion is evidence for selecting a distinct potential intervention phase, not proof that reset restores hidden state or the new model reproduces that phase.',
            train_labels='Full actual student prefix is history only. Expert uses new live observations after takeover; whole expert delay queue is masked; only successful actual legal suffixes may enter training. Source DP updates require refreshed backend plans/features before rear training.'),
        source_sha256={relative:sha256(project/relative) for relative in
            ('tools/tight_fresh_next_collection_coverage.py', 'tools/tight_fresh_contact_height_coverage.py',
             'tools/tight_fresh_expert_stability_coverage.py', 'tools/tight_fresh_stage_takeover.py',
             'tools/tight_fresh_collection_owner.py', 'visuotactile/takeover_trigger.py')},
        simulator_starts=0, optimizer_updates=0, cuda_initialized=False, validation_payloads_read=False, test_used=False,
        wall_seconds=time.monotonic()-began)
    atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--collections', type=Path, nargs='+', required=True)
    parser.add_argument('--exposure-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.collections, args.exposure_report, args.output)
    print(json.dumps(dict(output=str(args.output), summary=result['historical_summary'],
        source_identities=result['proposed_cohort']['source_identities'],
        assigned_delays=result['proposed_cohort']['assigned_delays']), indent=2))
