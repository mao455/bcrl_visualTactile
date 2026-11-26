"""CPU audit of candidate contact-height triggers on observed student prefixes."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.tight_fresh_expert_stability_coverage import require, read, first
from visuotactile.compact import sha256
from visuotactile import takeover_trigger as current
from visuotactile.training_resources import atomic_json


VERSION = 'observed_student_contact_height_coverage_v1'


def causal_contact_height(pose, force):
    """Independent fixed rule; no post-prefix state or pre-reference eligibility."""
    pose, force = np.asarray(pose), np.asarray(force)
    n = len(pose)
    require(pose.shape == (n, 7) and force.shape == (n, 2, 5, 3)
        and np.isfinite(pose).all() and np.isfinite(force).all(), 'Finite actual prefix required')
    eligible = np.zeros(n, dtype=bool)
    if n < 48:
        return eligible, None
    height = float(np.median(pose[24:48, 2]))
    two = ((np.linalg.norm(force, axis=-1) > .1).sum(-1) >= 2).all(-1)
    instantaneous = two & (pose[:, 2] >= height + .02)
    eligible = (np.convolve(instantaneous.astype(np.int64), np.ones(12, np.int64))[:n] == 12)
    eligible[:47] = False
    return eligible, height


def schedule_scope(completed, delay, student_ticks, total_ticks):
    """A terminal after-state is not an available next action in this episode."""
    if completed is None:
        return dict(first_completed_tick=None, proposed_action_index=None,
            action_state_observed=False, scope='no_qualifying_window_in_observed_prefix')
    require(48 <= completed <= student_ticks and delay >= 0 and total_ticks >= student_ticks,
        'Causal qualification and actual prefix length required')
    action = completed + delay
    if action < student_ticks:
        scope = 'observed_student_action_state'
    elif action == student_ticks and student_ticks < total_ticks:
        scope = 'observed_actual_expert_switch_state'
    elif student_ticks < total_ticks:
        scope = 'delay_censored_by_actual_expert_switch'
    else:
        scope = 'delay_reaches_or_exceeds_native_termination'
    return dict(first_completed_tick=completed, proposed_action_index=action,
        action_state_observed=scope.startswith('observed_'), scope=scope)


def frozen_trigger(source, expected_sha256):
    path = Path(source).resolve()/'visuotactile/takeover_trigger.py'
    require(sha256(path) == expected_sha256, 'Actual old trigger source changed')
    spec = importlib.util.spec_from_file_location('coverage_recorded_trigger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(path=str(path), sha256=sha256(path))


def validate_result_index(raw, indexed, prior):
    # The collection aggregator adds admission/accounting; physical result stays exact.
    additions = {'accepted_for_training', 'actual_exposure'}
    require(not (set(raw) & additions) and raw == {key:value for key, value in indexed.items()
        if key not in additions} and indexed['accepted_for_training'] == prior['accepted'],
        'Completed index/case result or reviewed admission differ')
    if 'actual_exposure' in indexed:
        value = indexed['actual_exposure']
        require(indexed['accepted_for_training'] is True and value['episodes'] == 1
            and value['history_ticks'] == raw['steps']
            and value['supervised_ticks'] == raw['dagger']['expert_valid_ticks']
            and value['supervised_anchors'] == raw['dagger']['recovery_coverage']['population']['dp_anchors'],
            'Added aggregate learning-accounting disagrees with actual trajectory')


def endpoint(index, pose, palm, force, goal, near, subgoals, stable, height):
    if index is None:
        return None
    i = index-1
    contact = ((np.linalg.norm(force[i-11:i+1], axis=-1) > .1).sum(-1) >= 2).all(-1)
    # Active pre-transition goal is retained across a native goal switch.
    return dict(completed_tick=index, frozen_stable=bool(stable[i]),
        contact_both_substeps_all12=bool(contact.all()),
        minimum_height_margin_m=float((pose[i-11:i+1, 2]-height-.02).min()),
        native_near_goal=bool(near[i]), native_subgoals_before=int(subgoals[i]),
        object_to_active_goal_root_distance_m=float(np.linalg.norm(pose[i, :3]-goal[i, :3])),
        object_to_active_goal_angle_rad=float(2*np.arccos(np.clip(abs((pose[i, 3:]*goal[i, 3:]).sum()), 0, 1))),
        object_to_palm_root_distance_m=float(np.linalg.norm(pose[i, :3]-palm[i, :3])))


def audit_case(directory, planned, result, prior, old):
    directory = Path(directory)
    require(result['identity'] == planned['name'] and result['split'] == 'train'
        and result['family'] == planned['spec']['family'] and result['software_complete'] is True
        and result['legal'] is True, 'Complete legal planned training trajectory required')
    before_hashes = {name:sha256(directory/name) for name in ('numeric.h5', 'result.json')}
    require(before_hashes['numeric.h5'] == prior['numeric_sha256']
        and before_hashes['result.json'] == prior['result_sha256'], 'Reviewed actual case changed')
    validate_result_index(read(directory/'result.json'), result, prior)
    total = result['steps']
    with h5py.File(directory/'numeric.h5', 'r') as stream:
        meta = json.loads(stream.attrs['metadata_json'])
        require(meta['dataset_split'] == 'train' and meta['initial_family'] == planned['spec']['family']
            and meta['source_identity'] == planned['spec']['source_identity']
            and meta['dagger']['trigger'] == planned['dagger']['trigger']
            and meta['complete_student_policy'] == planned['pair'], 'Raw training/student provenance differs')
        transition = stream['episode/transition']
        source = transition['dagger/control_source'][:]
        valid = transition['dagger/expert_label_valid'][:]
        segment = transition['dagger/segment_id'][:]
        student = source == 0
        n = int(student.sum())
        require(source.shape == valid.shape == segment.shape == (total,) and np.isin(source, [0, 1]).all()
            and np.array_equal(student, np.arange(total) < n) and not valid[:n].any()
            and (segment[:n] == segment[0]).all() and n == result['dagger']['student_ticks']
            and total-n == result['dagger']['expert_ticks'], 'Actual prefix authority differs')
        require(n >= 48, 'This audit requires the student-only settled reference')
        # No expert suffix physical state is used, even when it is available.
        after, before = transition['diagnostics_after'], transition['diagnostics_before']
        pose = after['object_pose_w'][:n]
        palm = after['palm_body_pose_w'][:n]
        force = transition['object_contact_force_w'][:n]
        goal = before['goal_pose_w'][:n]
        near = transition['native_success_diagnostic/near_goal'][:n].astype(bool)
        subgoals = before['successes'][:n]
        t0, t1 = transition['time_before_s'][:n], transition['time_after_s'][:n]
        episode = transition['episode_id'][:n]
        require(np.array_equal(stream['episode/obs/time_s'][:n], t0)
            and np.array_equal(t1[:-1], t0[1:]) and np.allclose(t1-t0, 1/60, rtol=0, atol=1e-9)
            and np.array_equal(transition['step_id'][:n], np.arange(n)) and (episode == episode[0]).all(),
            'Observed prefix native time/episode continuity differs')
        for key in ('object_pose_w', 'palm_body_pose_w', 'goal_pose_w', 'successes'):
            require(np.array_equal(after[key][:n-1], before[key][1:n]), 'Adjacent actual state differs: '+key)
        terminations = {key:np.asarray(value[:n], dtype=bool) for key, value in transition['termination_reasons'].items()}
        require(not any(value[:-1].any() for value in terminations.values()), 'Termination inside observed prefix')
    contact_height, height = causal_contact_height(pose, force)
    stable, _ = old.stable_windows(pose, palm, force, height)
    stable[:47] = False
    require(not (stable & ~contact_height).any(), 'Frozen stability must imply contact/height')
    implementation = current.contact_height_windows(pose, force, height)
    implementation[:47] = False
    require(np.array_equal(contact_height, implementation), 'New window implementation differs from independent rule')
    receipt = result['dagger']['trigger_receipt']
    trigger = old.StableGraspTrigger(planned['dagger']['trigger'], receipt['fallback_tick'])
    for i in range(n):
        trigger.observe(i, pose[i], palm[i], force[i])
    require(trigger.receipt() == receipt, 'Frozen live trigger receipt differs from actual prefix replay')
    require(first(stable) == receipt['first_stable_window_end_completed_tick'], 'Frozen first-window index differs')
    delay = planned['dagger']['trigger']['delay_ticks']
    scopes = {key:schedule_scope(first(mask), delay, n, total)
        for key, mask in (('frozen_stable', stable), ('contact_height', contact_height))}
    live = current.ContactHeightTrigger(dict(version=current.CONTACT_HEIGHT_VERSION, delay_ticks=delay), receipt['fallback_tick'])
    for i in range(n):
        if i >= live.scheduled_tick:
            break
        live.observe(i, pose[i], palm[i], force[i])
    require(live.first_contact_height_completed_tick == first(contact_height), 'New live first qualification differs')
    if contact_height.any():
        require(live.scheduled_tick == scopes['contact_height']['proposed_action_index'], 'New live schedule differs')
    first_goal = subgoals == 0
    before_near = first_goal & (np.cumsum(first_goal & near) == 0)
    ended = n == total
    require({name:sha256(directory/name) for name in before_hashes} == before_hashes, 'Case changed during audit')
    return dict(case=planned['name'], role=planned['role'], family=planned['spec']['family'],
        source_identity=planned['spec']['source_identity'], numeric_sha256=before_hashes['numeric.h5'],
        result_sha256=before_hashes['result.json'], total_ticks=total, observed_student_ticks=n,
        prefix_ending='native_termination' if ended else 'actual_expert_intervention',
        actual_expert_takeover_success=result['dagger']['expert_takeover_success'],
        hypothetical_new_takeover_success=None, assigned_delay_ticks=delay, settled_reference_m=height,
        frozen_live_receipt_reproduced=True, new_predicate_and_live_schedule_agree=True,
        all_prefix_clock_and_adjacent_state_checks=True, source_checkpoint_sha256=result['checkpoint_sha256'],
        windows={key:dict(observed_endpoints=int(mask.sum()),
            before_first_native_near_goal_endpoints=int((mask & before_near).sum()),
            first_before_near_completed_tick=first(mask & before_near), **scopes[key])
            for key, mask in (('frozen_stable', stable), ('contact_height', contact_height))},
        first_contact_height=endpoint(first(contact_height), pose, palm, force, goal, near, subgoals, stable, height),
        student_native_termination_flags={key:bool(value.any()) for key, value in terminations.items()})


def summarize(rows):
    return dict(prefixes=len(rows), observed_student_ticks=sum(row['observed_student_ticks'] for row in rows),
        window_qualified_prefixes={name:sum(row['windows'][name]['first_completed_tick'] is not None for row in rows)
            for name in ('frozen_stable', 'contact_height')},
        proposed_action_state_observed_prefixes={name:sum(row['windows'][name]['action_state_observed'] for row in rows)
            for name in ('frozen_stable', 'contact_height')},
        additional_qualified_prefixes=sum(row['windows']['contact_height']['first_completed_tick'] is not None
            and row['windows']['frozen_stable']['first_completed_tick'] is None for row in rows),
        contact_height_scope_counts=dict(Counter(row['windows']['contact_height']['scope'] for row in rows)),
        actual_prefix_endings=dict(Counter(row['prefix_ending'] for row in rows)))


def run(directory, execution_source, output):
    began = time.monotonic()
    directory, output = Path(directory).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve original coverage evidence')
    torch.set_num_threads(2)
    plan, completed, reviewed = [read(directory/name) for name in ('plan.json', 'results.json', 'reviewed_audit.json')]
    require(plan['stage_takeover_contract'] == 'stable_only_takeover_v1' and plan['round_index'] == 5
        and plan['test_used'] is False and len(plan['cases']) == len(completed) == 12
        and reviewed['collection_complete'] is True and reviewed['plan_sha256'] == sha256(directory/'plan.json'),
        'Complete reviewed stage05 training cohort required')
    index = {row['case']:row['result'] for row in completed}
    prior = {row['case']:row for row in reviewed['cases']}
    require(set(index) == set(prior) == {row['name'] for row in plan['cases']}, 'Case populations differ')
    old, frozen_binding = frozen_trigger(execution_source, plan['source_sha256']['visuotactile/takeover_trigger.py'])
    manifests = {}
    for role in ('A', 'B'):
        path = Path(plan['source_root'])/role/'state_data/data.json'
        require(sha256(path) == plan['source_data_sha256'][role], 'Source manifest changed')
        manifests[role] = read(path)
    rows = []
    for raw in plan['cases']:
        planned = dict(raw, pair=plan['pairs'][raw['role']])
        d = manifests[planned['role']]
        spec = planned['spec']
        require(spec['split'] == 'train' and spec['cohort'] == 'dagger_train'
            and spec['source_identity'] in d['normalization_train_ids']
            and spec['source_identity'] in d['splits']['train']
            and d['entries'][spec['source_identity']]['family'] == spec['family']
            and len(d['normalization_train_ids']) == 30, 'Original30 training-family provenance required before numeric access')
        rows.append(audit_case(directory/planned['name'], planned, index[planned['name']], prior[planned['name']], old))
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    project = Path(__file__).resolve().parents[1]
    result = dict(version=VERSION, complete=True, source_root=str(directory),
        evidence_sha256={name:sha256(directory/name) for name in ('plan.json', 'results.json', 'reviewed_audit.json')},
        source_data_sha256=plan['source_data_sha256'], frozen_trigger=frozen_binding,
        source_sha256={relative:sha256(project/relative) for relative in
            ('tools/tight_fresh_contact_height_coverage.py', 'visuotactile/takeover_trigger.py')},
        summary=summarize(rows), by_role={role:summarize([row for row in rows if row['role'] == role]) for role in ('A', 'B')},
        cases=rows, semantics=dict(rule='Same12ticks, both physics substeps >=2 fingers >0.1N and +0.02m above median completed25-48; omit relative-palm translation/rotation only. Qualification cannot precede completed48.',
            authority='Numeric physical tensors are sliced to the actual student prefix before computing either rule. Actual expert suffixes supply no candidate qualification. Whole numeric file hashes are checked against the reviewed evidence.',
            delay='Completed count N proposes zero-based action N+assigned_delay. Only a recorded student action or the actual intervention state counts as an observed candidate. Native terminal after-state is not another action opportunity.',
            screen='Main comparison is the raw versioned contact-height trigger. Strictly before first near_goal on first native goal is a separate reported diagnostic, not an extra live condition.',
            limitations='Six distinct original training families, paired A/B, not independent new test families. 11 complete student failures and one prefix censored by actual rescue. New candidate states were not expert rescue trials; rescue success is unknown. Later original student windows are descriptive and are not counterfactual trajectories after hypothetical earlier expert intervention.'),
        cpu_threads=2, cuda_initialized=False, optimizer_updates=0, simulator_starts=0,
        validation_payloads_read=False, test_used=False, wall_seconds=time.monotonic()-began)
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--execution-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.directory, args.execution_source, args.output)
    print(json.dumps(dict(output=str(args.output), summary=report['summary'], by_role=report['by_role']), indent=2))
