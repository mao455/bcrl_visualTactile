"""Finite causal stage collection; no fixed-time expert fallback or policy changes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import VERSION, CONTACT_HEIGHT_VERSION,HELD_NEAR_GOAL_VERSION,HELD_RECHECK_VERSION
from visuotactile.training_resources import atomic_json

CONTRACT = 'stable_only_takeover_v1'
CONTACT_HEIGHT_CONTRACT = 'contact_height_only_takeover_v1'
NEVER_FALLBACK = 1_000_000_000


def trigger_contract(version):
    if version == VERSION:
        return CONTRACT, 'stable_only', 'stable', 'first_stable_window_end_completed_tick'
    if version == CONTACT_HEIGHT_VERSION:
        return CONTACT_HEIGHT_CONTRACT, 'contact_height_only', 'contact_height', 'first_contact_height_window_end_completed_tick'
    if version == HELD_NEAR_GOAL_VERSION:
        return 'held_near_goal_only_takeover_v1','held_near_goal_only','held_near_goal','first_held_near_goal_window_end_completed_tick'
    if version == HELD_RECHECK_VERSION:
        return 'held_recheck_only_takeover_v1','held_recheck_only','held_recheck','first_rechecked_contact_height_window_end_completed_tick'
    raise ValueError('Unknown explicit stage trigger version')


def validate_requests(data, identities, delays):
    originals = data.get('normalization_train_ids', data['splits']['train'])
    if len(originals) != 30 or len(set(originals)) != 30:
        raise ValueError('Exactly the original train30 identities are required')
    if not identities or len(identities) != len(delays) or len(set(identities)) != len(identities):
        raise ValueError('Unique explicit identities and one delay per opportunity required')
    if any(isinstance(x, bool) or not isinstance(x, int) or x not in (0, 12) for x in delays):
        raise ValueError('Only preregistered delays 0 and 12 are supported')
    families = []
    for identity in identities:
        if identity not in originals or identity not in data['splits']['train']:
            raise ValueError('Source identity must belong to original train30')
        entry = data['entries'][identity]
        if entry.get('trajectory_kind') == 'expert_takeover_v1':
            raise ValueError('A correction cannot masquerade as an original demonstration')
        families.append(entry['family'])
    if len(set(families)) != len(families):
        raise ValueError('Each opportunity must use a different original family')


def prepare(source, output, identities, delays, round_index, event_seed, *, session_root=None,
            trigger_version=VERSION, first_plan_forensics=False):
    import h5py
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.state_benchmark import dagger_configuration
    source, output = Path(source).resolve(), Path(output).resolve()
    session_root = Path(session_root or source).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing collection identity')
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
        raise ValueError('Positive explicit round required')
    if isinstance(event_seed, bool) or not isinstance(event_seed, int) or event_seed < 0:
        raise ValueError('Nonnegative explicit event seed required')
    if type(first_plan_forensics) is not bool:
        raise ValueError('First-plan forensics must be explicitly enabled or disabled')
    session = read(session_root / 'session.json')
    contract, label, _, _ = trigger_contract(trigger_version)
    if session['deadline'] <= time.time():
        raise ValueError('Session deadline has already passed')
    data = {r: read(source / r / 'state_data/data.json') for r in ('A', 'B')}
    for value in data.values():
        validate_requests(value, identities, delays)
    if data['A']['splits'] != data['B']['splits']:
        raise ValueError('A/B splits differ')
    pairs = {r: completed_pair(source, r) for r in ('A', 'B')}
    cases = []
    for index, (identity, delay) in enumerate(zip(identities, delays)):
        entry = data['A']['entries'][identity]
        other = data['B']['entries'][identity]
        for key in ('path', 'family', 'source_sha256'):
            if entry[key] != other[key]:
                raise ValueError('A/B original demonstration differs')
        if sha256(entry['path']) != entry['source_sha256']:
            raise ValueError('Original demonstration bytes changed')
        with h5py.File(entry['path']) as f:
            original = json.loads(f.attrs['result_json'])
        if not original.get('success'):
            raise ValueError('Original must be a complete successful demonstration')
        for role in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
            name = f'r{round_index:02d}_{role}_{identity}_{label}_d{delay}'
            spec = dict(identity=name, family=entry['family'], split='train', cohort='dagger_train',
                source_identity=identity, seed=original['seed'], event_seed=event_seed + index,
                narrow_reset=copy.deepcopy(original['narrow_reset']),
                collision_gate='live_collision_mesh_bounds_v1', takeover_tick=NEVER_FALLBACK)
            bridge = dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme,
                trigger=dict(version=trigger_version, delay_ticks=delay))
            dagger_configuration(bridge, spec, data[role])
            cases.append(dict(name=name, role=role, spec=spec, dagger=bridge, video=index == 0,
                source_data=str(source / role / 'state_data/data.json')))
    # Build the complete immutable registration before creating any plan file.
    plan = dict(kind='takeover', stage_takeover_contract=contract, never_fallback=NEVER_FALLBACK,
        round_index=round_index, pairs=pairs, cases=cases, explicit_source_identities=list(identities),
        assigned_delays=list(delays), source_root=str(source), resource_root=str(session_root),
        source_data_sha256={r: sha256(source / r / 'state_data/data.json') for r in ('A', 'B')},
        source_session_sha256=sha256(session_root / 'session.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): sha256(p) for folder in ('tools', 'visuotactile')
            for p in sorted(Path(folder).rglob('*.py'))},
        test_used=False, inference_scheduler_index_mode='host', first_plan_forensics=first_plan_forensics,
        selection='Only naturally successful, valid real expert suffixes; retain every opportunity and failure',
        reason=f'Current complete student {label} window plus 0/12 ticks; no fixed-time expert fallback',
        interpretation='Different assigned families do not identify a causal effect of delay; no exact hidden-state pairing claim',
        utc=time.time())
    output.mkdir(parents=True)
    shutil.copyfile(session_root / 'session.json', output / 'session.json')
    atomic_json(output / 'plan.json', plan)
    return plan


def validate_plan(plan):
    if plan.get('stage_takeover_contract') not in (CONTRACT, CONTACT_HEIGHT_CONTRACT) or plan.get('never_fallback') != NEVER_FALLBACK:
        raise ValueError('Explicit never-fallback contract required')
    cases = plan['cases']
    if not cases or len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Unique finite opportunity registration required')
    for case in cases:
        if case['spec']['takeover_tick'] != NEVER_FALLBACK:
            raise ValueError('Fixed-time fallback is forbidden')
        version = VERSION if plan['stage_takeover_contract'] == CONTRACT else CONTACT_HEIGHT_VERSION
        if case['dagger'].get('trigger') not in [dict(version=version, delay_ticks=x) for x in (0, 12)]:
            raise ValueError('Registered trigger version and delays0/12 are required')


def native_bound_guard(env):
    """Same conservative loop bound as the existing collector, checked before action."""
    goals = int(env.env_max_goals.max())
    length = int(env.max_episode_length)
    bound = (goals + 2) * (length + 1)
    if goals < 1 or length < 1 or bound >= NEVER_FALLBACK:
        raise ValueError('Never-fallback sentinel must strictly exceed the native collector bound')
    return dict(native_max_goals=goals, native_max_episode_length=length,
        collector_max_control_ticks=bound, never_fallback=NEVER_FALLBACK,
        task_timeout_changed=False, success_definition_changed=False)


def classify(result, audit):
    if not result.get('legal'):
        return 'invalid_initialization'
    if not result.get('software_complete'):
        return 'software_incomplete'
    dagger = result['dagger']
    receipt = audit['receipt']
    _, _, event, first_key = trigger_contract(receipt['configuration']['version'])
    if audit['category'].startswith('fallback'):
        raise ValueError('Actual fixed-time fallback is forbidden')
    switched = audit['actual_switch_tick'] is not None
    if switched:
        first = receipt[first_key]
        if receipt['configuration']['version']==HELD_RECHECK_VERSION:
            original=receipt['first_contact_height_window_end_completed_tick']
            expected=first
            if (original is None or expected is None or receipt['latest_contact_height_window'] is not True
                    or receipt['earliest_allowed_completed_tick']!=original+receipt['configuration']['min_delay_ticks']
                    or expected<receipt['earliest_allowed_completed_tick']):
                raise ValueError('Expert authority lacks a rechecked held window after its minimum delay')
        else:expected = None if first is None else first + receipt['configuration']['delay_ticks']
        if (audit['category'] != f'student_{event}_then_takeover'
                or audit['actual_switch_tick'] != expected or expected >= NEVER_FALLBACK):
            raise ValueError('Expert authority lacks the registered causal stage trigger')
        return f'expert_rescue_after_{event}' if dagger['expert_takeover_success'] else f'expert_failed_after_{event}'
    if dagger['expert_ticks'] or dagger['expert_valid_ticks']:
        raise ValueError('Unswitched trajectory contains expert authority or labels')
    if dagger['student_autonomous_success']:
        return 'student_success_without_takeover'
    if receipt['configuration']['version']==HELD_RECHECK_VERSION:
        if receipt[first_key] is not None:return 'ended_after_held_recheck_before_scheduled_takeover'
        if receipt['first_contact_height_window_end_completed_tick'] is None:
            return 'ended_without_student_contact_height_window'
        if receipt['completed_student_transitions_observed']<receipt['earliest_allowed_completed_tick']:
            return 'ended_after_first_held_before_minimum_delay'
        return 'ended_after_minimum_delay_without_current_held_window'
    return (f'ended_after_{event}_before_scheduled_takeover'
        if receipt[first_key] is not None else f'ended_without_student_{event}_window')


def run(output):
    """Wrap existing runner only to guard/audit; the same trial performs every action."""
    from tools.tight_fresh_rollouts import run as existing_run
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile import state_benchmark
    output = Path(output).resolve()
    plan = read(output / 'plan.json')
    validate_plan(plan)
    for role, digest in plan['source_data_sha256'].items():
        if sha256(Path(plan['source_root']) / role / 'state_data/data.json') != digest:
            raise ValueError('Source data registration changed')
    if sha256(output / 'session.json') != plan['source_session_sha256']:
        raise ValueError('Registered session bytes changed')
    original_trial = state_benchmark.trial
    def guarded_trial(env, lease, directory, spec, *args, **kwargs):
        bound = native_bound_guard(env)
        result = original_trial(env, lease, directory, spec, *args, **kwargs)
        audit = audit_trigger_case(directory, result) if result.get('legal') else None
        category = classify(result, audit)
        audit_name = 'stable_stage_audit.json' if plan['stage_takeover_contract'] == CONTRACT else 'contact_height_stage_audit.json'
        atomic_json(Path(directory) / audit_name, dict(category=category,
            causal_trigger=audit, bound=bound, source_plan_sha256=sha256(output / 'plan.json'),
            no_expert_fallback=True, not_autonomous_success=bool(result.get('dagger', {}).get('expert_ticks', 0))))
        return result
    # Local process dispatch only: no source edits, observations, actions or timing changes.
    state_benchmark.trial = guarded_trial
    try:
        existing_run(output)
    finally:
        state_benchmark.trial = original_trial


def report(output, destination):
    plan = read(output / 'plan.json'); validate_plan(plan)
    if not read(output / 'result.json').get('software_complete'):
        raise ValueError('Incomplete collection must not be summarized as complete')
    rows = read(output / 'results.json')
    entries = read(output / 'correction_entries.json') if (output / 'correction_entries.json').exists() else {}
    if len(rows) != len(plan['cases']) or {x['case'] for x in rows} != {x['name'] for x in plan['cases']}:
        raise ValueError('All registered opportunities must remain in the denominator')
    cases = []
    for row in rows:
        audit_name = 'stable_stage_audit.json' if plan['stage_takeover_contract'] == CONTRACT else 'contact_height_stage_audit.json'
        audit = read(output / row['case'] / audit_name)
        if audit['source_plan_sha256'] != sha256(output / 'plan.json'):
            raise ValueError('Case audit belongs to another plan')
        category = classify(row['result'], audit['causal_trigger'])
        if category != audit['category']:
            raise ValueError('Classification changed')
        accepted = bool(row['result'].get('accepted_for_training'))
        expected = 'expert_rescue_after_stable' if plan['stage_takeover_contract'] == CONTRACT else 'expert_rescue_after_contact_height'
        if accepted != (row['case'] in entries) or (accepted and category != expected):
            raise ValueError('Retained subset differs from successful actual stable-state recoveries')
        cases.append(dict(case=row['case'], role=row['role'], category=category,
            accepted=accepted, audit=audit))
    atomic_json(destination, dict(contract=plan['stage_takeover_contract'], plan_sha256=sha256(output / 'plan.json'),
        completed=len(cases), cases=cases, test_used=False,
        counts={r: {key: sum(c['role'] == r and c['category'] == key for c in cases)
            for key in sorted({c['category'] for c in cases})} for r in ('A', 'B')}))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['prepare', 'run', 'report'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path); p.add_argument('--session-root', type=Path)
    p.add_argument('--identities', nargs='+'); p.add_argument('--delays', nargs='+', type=int)
    p.add_argument('--round', type=int); p.add_argument('--event-seed', type=int)
    p.add_argument('--report-output', type=Path)
    p.add_argument('--trigger-version', choices=[VERSION, CONTACT_HEIGHT_VERSION], default=VERSION)
    p.add_argument('--first-plan-forensics', action='store_true')
    a = p.parse_args()
    preparation_options = {'--source', '--session-root', '--identities', '--delays',
        '--round', '--event-seed', '--trigger-version', '--first-plan-forensics'}
    if a.stage != 'prepare' and any(token.split('=', 1)[0] in preparation_options for token in sys.argv[1:]):
        p.error('run/report use the immutable registered plan; preparation options require a new prepare identity')
    if a.stage == 'prepare':
        if any(x is None for x in (a.source, a.identities, a.delays, a.round, a.event_seed)):
            p.error('prepare requires source, identities, delays, round and event-seed')
        prepare(a.source, a.output, a.identities, a.delays, a.round, a.event_seed,
            session_root=a.session_root, trigger_version=a.trigger_version,
            first_plan_forensics=a.first_plan_forensics)
    elif a.stage == 'run':
        run(a.output)
    else:
        if a.report_output is None: p.error('report requires report-output')
        report(a.output, a.report_output)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import os
        import sys
        import traceback
        if len(sys.argv) > 1 and sys.argv[1] == 'run':
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        raise
