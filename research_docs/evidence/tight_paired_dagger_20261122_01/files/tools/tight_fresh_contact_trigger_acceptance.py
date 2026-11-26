"""CPU-only acceptance replay over every registered, actual student prefix.

No simulator, checkpoint, supervision or original result is modified. Each
hypothetical observer stops before its scheduled authority switch. The recorded
expert suffix is never supplied to either trigger.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from tools.tight_fresh_report import audit_trigger_case
from tools.tight_fresh_stage_takeover import NEVER_FALLBACK
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import CONTACT_HEIGHT_VERSION, VERSION, make_trigger, stable_windows


def read(path):
    return json.loads(Path(path).read_text())


def unchanged_original_predicate(plan):
    path = 'visuotactile/takeover_trigger.py'
    original = subprocess.check_output(['git', 'show', f"{plan['source_commit']}:{path}"])
    if hashlib.sha256(original).hexdigest() != plan['source_sha256'][path]:
        raise ValueError('Original registered trigger source hash does not match its commit')
    before, after = ast.parse(original), ast.parse(Path(path).read_bytes())
    def named(tree, names):
        for name in names:
            tree = next(n for n in tree.body if getattr(n, 'name', None) == name)
        return ast.dump(tree, include_attributes=False)
    checked = [['stable_windows'], ['StableGraspTrigger', 'observe'], ['StableGraspTrigger', 'receipt']]
    matches = {'.'.join(names): named(before, names) == named(after, names) for names in checked}
    if not all(matches.values()):
        raise ValueError('Established stable predicate or scheduling/receipt behavior changed')
    return dict(original_source_commit=plan['source_commit'], original_source_sha256=hashlib.sha256(original).hexdigest(),
        current_source_sha256=sha256(path), ast_equal=matches,
        constructor_change='Explicit version admission now rejects the other named trigger; original version still dispatches to the original class')


def reference_first_event(pose, palm, force, version):
    """Independent contact/height screen; causal reference excludes ticks1-24."""
    if len(pose) < 48:
        return None, None
    height = float(np.median(pose[24:48, 2]))
    if version == VERSION:
        flags = stable_windows(pose, palm, force, height)[0]
    else:
        flags = np.zeros(len(pose), dtype=bool)
        for end in range(47, len(pose)):
            window = slice(end-11, end+1)
            per_sample_contacts = (np.sqrt(np.sum(force[window] ** 2, axis=-1)) > .1).sum(axis=-1)
            flags[end] = bool(np.all(per_sample_contacts >= 2) and np.all(pose[window, 2] >= height + .02))
    qualifying = np.flatnonzero(flags & (np.arange(len(pose)) >= 47))
    return (int(qualifying[0]) + 1 if len(qualifying) else None), height


def replay(pose, palm, force, version, delay, total_steps):
    expected_first, expected_height = reference_first_event(pose, palm, force, version)
    trigger = make_trigger(dict(version=version, delay_ticks=delay), NEVER_FALLBACK)
    events = []
    for tick in range(len(pose)):
        if tick >= trigger.scheduled_tick:
            break
        event = trigger.observe(tick, pose[tick], palm[tick], force[tick])
        if event is not None:
            events.append(event)
    receipt = trigger.receipt()
    first_key = ('first_stable_window_end_completed_tick' if version == VERSION
        else 'first_contact_height_window_end_completed_tick')
    actual_first = receipt[first_key]
    expected_schedule = None if expected_first is None else expected_first + delay
    actual_schedule = None if trigger.scheduled_tick == NEVER_FALLBACK else trigger.scheduled_tick
    if (actual_first != expected_first or expected_schedule != actual_schedule
            or receipt['settled_height_m'] != expected_height
            or events != ([] if expected_schedule is None else [expected_schedule])):
        raise ValueError('Online trigger disagrees with the independent causal prefix screen')
    if len(trigger.poses) != min(len(pose), trigger.scheduled_tick):
        raise ValueError('Hypothetical observer continued after scheduled authority switch')
    return dict(configuration=receipt['configuration'], first_qualifying_completed_tick=actual_first,
        scheduled_action_tick=actual_schedule, observed_student_transitions=len(trigger.poses),
        excluded_available_student_transitions_after_hypothetical_switch=len(pose)-len(trigger.poses),
        causal_prefix_covers_switch=actual_schedule is not None and actual_schedule <= len(pose),
        scheduled_before_recorded_end=actual_schedule is not None and actual_schedule < total_steps,
        independent_reference_agrees=True, receipt=receipt)


def audit(collection):
    collection = Path(collection).resolve()
    plan = read(collection/'plan.json')
    if not read(collection/'result.json').get('software_complete'):
        raise ValueError('Use a completed original collection')
    registered = plan['cases']
    rows = read(collection/'results.json')
    if len(rows) != len(registered) or {x['case'] for x in rows} != {c['name'] for c in registered}:
        raise ValueError('All registered opportunities must stay in the denominator')
    originals = unchanged_original_predicate(plan)
    cases = []
    for case in registered:
        directory = collection/case['name']; result = read(directory/'result.json')
        if not result.get('legal') or not result.get('software_complete'):
            raise ValueError('This acceptance cohort requires complete legal numeric prefixes for every opportunity')
        old_audit = audit_trigger_case(directory, result)
        if old_audit is None or old_audit['receipt']['configuration']['version'] != VERSION:
            raise ValueError('Original cohort must retain the original stable trigger')
        with h5py.File(directory/'numeric.h5') as f:
            t = f['episode/transition']; sources = t['dagger/control_source'][:]
            if not np.isin(sources, [0, 1]).all():
                raise ValueError('Unknown authority source')
            prefix = int((sources == 0).sum())
            if not np.array_equal(sources == 0, np.arange(len(sources)) < prefix):
                raise ValueError('Actual student authority must be a contiguous prefix')
            switch = prefix if prefix < len(sources) else None
            if switch != result['dagger']['recovery_coverage']['switch_tick']:
                raise ValueError('Numeric authority and recorded actual switch differ')
            pose = t['diagnostics_after/object_pose_w'][:prefix]
            palm = t['diagnostics_after/palm_body_pose_w'][:prefix]
            force = t['object_contact_force_w'][:prefix]
        variants = {f'{label}_delay{delay}': replay(pose, palm, force, version, delay, len(sources))
            for label, version in [('original_stable', VERSION), ('contact_height', CONTACT_HEIGHT_VERSION)]
            for delay in (0, 12)}
        original_delay = case['dagger']['trigger']['delay_ticks']
        if variants[f'original_stable_delay{original_delay}']['receipt'] != result['dagger']['trigger_receipt']:
            raise ValueError('Actual registered original trigger receipt changed')
        cases.append(dict(case=case['name'], role=case['role'], family=case['spec']['family'],
            original_assigned_delay=original_delay, actual_student_prefix_ticks=prefix,
            actual_expert_suffix_ticks=len(sources)-prefix, actual_old_switch_tick=switch,
            actual_old_expert_success=result['dagger']['expert_takeover_success'],
            hashes={name: sha256(directory/name) for name in ('numeric.h5', 'result.json')},
            original_recorded_receipt_exactly_reproduced=True, replay_variants=variants))
    counts = {}
    for role in ('all', 'A', 'B'):
        selected = [c for c in cases if role == 'all' or c['role'] == role]
        counts[role] = dict(opportunities=len(selected), variants={key: dict(
            qualified=sum(c['replay_variants'][key]['first_qualifying_completed_tick'] is not None for c in selected),
            scheduled_with_available_student_prefix=sum(c['replay_variants'][key]['causal_prefix_covers_switch'] for c in selected),
            schedule_before_recorded_end=sum(c['replay_variants'][key]['scheduled_before_recorded_end'] for c in selected))
            for key in cases[0]['replay_variants']})
    sources = ['tools/tight_fresh_contact_trigger_acceptance.py', 'visuotactile/takeover_trigger.py',
        'tools/tight_fresh_stage_takeover.py', 'tools/tight_fresh_report.py', 'visuotactile/state_benchmark.py']
    return dict(version='actual_student_prefix_contact_trigger_acceptance_v1', passed=True,
        source_collection=collection.name, source_plan_sha256=sha256(collection/'plan.json'),
        source_result_sha256=sha256(collection/'result.json'), source_sha256={p: sha256(p) for p in sources},
        original_predicate_unchanged=originals, counts=counts, cases=cases,
        scope='CPU replay of all registered actual student prefixes. No expert suffix enters either trigger; each observer stops before its hypothetical switch. No simulation or new expert recovery occurred.',
        index_convention='Inputs use zero-based action indices; qualifying windows report completed transition counts N; first expert action would be N+delay.',
        limitations=['Coverage is not stable-grasp success, autonomous task success, or expert rescue feasibility.',
            'Delays are replayed on the same observed prefix; their causal effects after authority switch are unobserved.',
            'The frozen original stable predicate remains a separate diagnostic; it is not relaxed or renamed.',
            'No labels, native success condition, model inputs, policies, original files or frozen execution trees are changed.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior acceptance receipts')
    result = audit(args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], counts=result['counts'], output_sha256=sha256(args.output)), sort_keys=True))


if __name__ == '__main__':
    main()
