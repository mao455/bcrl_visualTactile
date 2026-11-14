"""Prepare a same-session collection source with explicitly separate data/policies.

No model selection, new resource window, monitor, collection or optimizer is run.
Links are read-only by convention; subsequent collection/training uses new outputs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from tools.tight_fresh_followup_session import (audit_data_extension, audit_role,
    file_binding)
from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

VERSION = 'same_session_policy_data_selection_v1'
RECEIPT = 'source_selection.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def evaluated_role(evaluation, role, pair):
    """Keep every registered opportunity, allowing only evidenced zero-action rejects."""
    from tools.tight_fresh_round_compare import (invalid_initialization_evidence,
        audit_numeric_clock, verify_execution_receipt)
    plan, rows, final = (read(evaluation/name) for name in ('plan.json', 'results.json', 'result.json'))
    require(plan.get('kind') == 'autonomous' and plan.get('test_used') is False
        and plan.get('source_sha256') and plan['pairs'][role] == pair,
        'Explicit completed autonomous source pair required')
    require(read(evaluation/'run_started.json')['plan_sha256'] == sha256(evaluation/'plan.json'),
        'Autonomous worker plan changed')
    cases = plan['cases']; names = [case['name'] for case in cases]
    require(cases and len(names) == len(set(names)) and all(Path(n).name == n and n not in ('.', '..') for n in names)
        and {case['role'] for case in cases} == {'A', 'B'}
        and all(case['spec'].get('split') == 'development' for case in cases)
        and final.get('software_complete') is True and not final.get('worker_error')
        and final.get('cases') == len(cases) == len(rows)
        and len({r['case'] for r in rows}) == len(rows) and {r['case'] for r in rows} == set(names),
        'Every registered autonomous opportunity must be resolved')
    indexed = {row['case']:row for row in rows}; receipts = {}; counts = {r:dict(legal=0, illegal=0) for r in ('A', 'B')}
    execution = verify_execution_receipt(evaluation, plan)
    assignments = {}; wave = 0
    if 'batch' in plan:
        for group in dict.fromkeys(case['role'] for case in cases):
            selected = [case for case in cases if case['role'] == group]
            for first in range(0, len(selected), plan['batch']['lanes']):
                for lane, case in enumerate(selected[first:first+plan['batch']['lanes']]):
                    assignments[case['name']] = (wave, lane)
                wave += 1
    for case in cases:
        name, group = case['name'], case['role']; path = evaluation/name
        result, started = read(path/'result.json'), read(path/'started.json')
        actual = plan['pairs'][group]
        require(indexed[name]['result'] == result and indexed[name]['role'] == group
            and started['spec'] == case['spec'] and all(result.get(k) == v for k,v in case['spec'].items())
            and case.get('dagger') is None and result.get('dagger') is None
            and not result.get('worker_error') and not result.get('censored'), 'Autonomous case provenance differs: '+name)
        if 'batch' in plan:
            metadata = read(path/'metadata.json')
            batch = result['batch_execution']; wave, lane = assignments[name]
            require(metadata['policy_checkpoint_sha256'] == actual['backend_sha256']
                and metadata['batch_execution'] == started['batch_execution'] == result['batch_execution']
                and metadata['execution_backend'] == execution['execution_backend']
                and (batch['wave'], batch['lane']) == (wave, lane) and batch['lanes'] == plan['batch']['lanes']
                and batch['vision_microbatch'] == execution['vision_microbatch']
                and batch.get('vision_tail_profile') == execution['vision_tail_profile']
                and batch['physics_seed'] == plan['batch']['seed']+wave
                and batch['event_seed'] == plan['batch']['seed']+100000+wave, 'Batch policy binding differs')
        else:
            require(Path(started['checkpoint']).resolve() == Path(actual['backend']).resolve(), 'Executed backend differs')
        if result.get('legal') is False:
            evidence = invalid_initialization_evidence(path, result)
            counts[group]['illegal'] += 1
        else:
            steps = result.get('steps')
            require(result.get('legal') is True and result.get('software_complete') is True
                and type(steps) is int and steps > 0 and result.get('residual_ticks') == steps
                and result.get('checkpoint_sha256') == actual['backend_sha256']
                and result.get('residual_version') == actual['residual_version']
                and result.get('simulation_step_hz') == 60 and result.get('physics_hz') == 120
                and result.get('plan_hz') == 5 and result.get('expert_shadow_computed') is False
                and result.get('action_control_mode') == 'absolute_joint_position_target_rad',
                'Incomplete autonomous complete-student execution: '+name)
            if 'batch' in plan:
                require(result.get('complete_student_policy') == actual, 'Complete policy identity differs')
            measured = result['native_inference']['summary']
            require(all(measured[k]['wall_s']['count'] == steps for k in
                ('dino_rgb', 'learned_visual_and_history', 'residual_forward_compose'))
                and measured['ddim_plan']['wall_s']['count'] == (steps+11)//12,
                'Real-frame history or planning count differs')
            evidence = audit_numeric_clock(path, steps)
            counts[group]['legal'] += 1
        receipts[name] = dict(role=group, result=file_binding(path/'result.json'),
            started=file_binding(path/'started.json'), evidence=evidence)
    require(counts[role]['legal'] > 0, 'Selected role has no complete legal autonomous execution')
    return dict(root=str(evaluation), selected_role=role, selected_pair=pair, counts=counts,
        cases=receipts, files={name:file_binding(evaluation/name) for name in
            ('plan.json', 'run_started.json', 'result.json', 'results.json')},
        execution=execution, source_commit=plan.get('source_commit'), source_sha256=plan['source_sha256'],
        interpretation='Illegal opportunities retained, not task failures/successes or replacement seeds')


def owner_receipts(roots, evaluations):
    paths = set()
    for root in roots:
        stages = root.with_name(root.name+'_stages')
        paths.update(stages.glob('*_worker.json'))
        paths.update(p for p in (stages/'registration.json', root/'launch.json', root/'active_owner.json') if p.exists())
    for root in evaluations:
        paths.update(p for p in (root/n for n in ('run_started.json', 'owner_started.json',
            'owner_worker_started.json', 'recovery_registration.json')) if p.exists())
    return sorted(paths)


def assert_owners_exited(receipts, roots, *, proc=Path('/proc')):
    """Inspect bound writer/owner PIDs, not all readers of a source checkpoint.

    Current session telemetry and independent stage-fit readers have no relevant
    writer receipt. Historical PID reuse is checked against start ticks, or the
    bound command-root arguments when old receipts lack those ticks.
    """
    from tools.tight_batch_owner import process_identity
    roots = tuple(Path(p).resolve() for p in roots); checked = []
    def identities(value):
        if not isinstance(value, dict):
            return
        for key in ('pid', 'worker_pid', 'owner_pid'):
            if type(value.get(key)) is int:
                yield value[key], value.get('proc_start_ticks', value.get('start_ticks')) if key == 'pid' else None
        for child in value.values():
            if isinstance(child, dict):
                yield from identities(child)
    records = [(path, pid, start) for path in receipts for pid, start in identities(read(path))]
    known_starts = {}
    for _, pid, start in records:
        if start is not None:known_starts.setdefault(pid, set()).add(start)
    for path in receipts:
        for _, pid, start in (row for row in records if row[0] == path):
            if pid == os.getpid():
                continue
            current = process_identity(pid, proc)
            row = dict(receipt=str(path), pid=pid, expected_start_ticks=start, observed=current)
            if current is None or current['state'] == 'Z':
                row['status'] = 'exited'
            elif start is not None or pid in known_starts:
                expected = {start} if start is not None else known_starts[pid]
                require(current['start_ticks'] not in expected, 'Related source writer/owner is still live: '+str(path))
                row['status'] = 'pid_reused'
            else:
                directory = proc/str(pid)
                argv = directory.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
                cwd = directory.joinpath('cwd').resolve(strict=True)
                values = []
                for index,arg in enumerate(argv):
                    for flag in ('--root', '--output', '--source-root'):
                        value = argv[index+1] if arg == flag and index+1 < len(argv) else arg[len(flag)+1:] if arg.startswith(flag+'=') else None
                        if value is not None:
                            p = Path(value); values.append((cwd/p).resolve() if not p.is_absolute() else p.resolve())
                require(not any(p == r or p.is_relative_to(r) for p in values for r in roots),
                    'Related source writer/owner still names source artifacts: '+str(path))
                row.update(status='legacy_pid_no_longer_bound_to_source', observed_argv=argv)
            checked.append(row)
    return checked


def snapshot(roots):
    """Stat audited immutable inputs before/after preparation; no raw/DINO reads."""
    paths = set()
    for root in roots:
        paths.update(root/name for name in ('session.json', 'metadata.json', 'gate.json'))
        for role in ('A', 'B'):
            base = root/role; rear = base/'backend'/role
            paths.update(base/name for name in ('cache_index.json', 'state_data/data.json', 'state_data/statistics.pt',
                'state_data/numeric.pt', 'state_data/loss_context.pt', 'training/best.pt', 'training/finished.json'))
            paths.update(rear/name for name in ('training/best.pt', 'training/finished.json', 'offline/result.json'))
            paths.update((rear/'nominal').glob('*'))
    def stat(path):
        s = path.stat()
        return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    return {str(path.resolve()):stat(path) for path in sorted(paths) if path.is_file()}


def prepare(session_root, data_source, output, *, role_sources, role_evaluations):
    require(set(role_sources) == set(role_evaluations) == {'A', 'B'}, 'Explicit A/B policy sources and evaluations required')
    session_root, pool = Path(session_root).resolve(strict=True), Path(data_source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve existing collection source identity')
    output = output.resolve()
    require(output != session_root and output.is_relative_to(session_root), 'Source view must be a new child of the active session')
    roots = {role:Path(path).resolve(strict=True) for role,path in role_sources.items()}
    evaluations = {role:Path(path).resolve(strict=True) for role,path in role_evaluations.items()}
    require(not any(output == p or output.is_relative_to(p) for p in {*roots.values(), pool, *evaluations.values()}),
        'Source view cannot modify an artifact or evaluation root')
    session = read(session_root/'session.json')
    require(session.get('resource_policy') == 'full_gpu_v1' and session.get('visual_stride') == 1
        and session.get('control_hz') == 60 and session.get('planning_hz') == 5, 'Native continuous-vision session required')
    def check():
        require(time.time() < session['deadline'], 'Original session deadline has passed')
        require(not (session_root/'STOP').exists(), 'Original session STOP is set')
    check()
    require((session_root/'shared_resources.json').is_file(), 'Active original telemetry file required')
    require(read(session_root/'gate.json').get('passed') is True, 'Original session input gate must have passed')
    for root in {*roots.values(), pool}:
        other = read(root/'session.json')
        require(all(other.get(k) == session.get(k) for k in ('version', 'domain', 'visual_stride',
            'control_hz', 'planning_hz', 'resource_policy', 'expert_sha256', 'dino_sha256')),
            'Policy/data physical or sensor session contract differs')
        require(read(root/'metadata.json') == read(session_root/'metadata.json')
            and read(root/'gate.json').get('passed') is True, 'Task metadata/input gate differs')
    writers = {*roots.values(), pool, *evaluations.values()}
    receipts = owner_receipts({*roots.values(), pool}, set(evaluations.values()))
    owners = assert_owners_exited(receipts, writers)
    before = snapshot({*roots.values(), pool, session_root})
    pairs = {role:completed_pair(root, role) for role,root in roots.items()}
    roles = {}; data = {}; evaluated = {}
    for role in ('A', 'B'):
        check()
        front_finished = file_binding(roots[role]/role/'training/finished.json')
        finished = read(front_finished['path'])
        require(finished.get('complete') is True and finished.get('checkpoint_sha256') == pairs[role]['frontend_sha256'],
            'Frontend training must be complete and bind its actual selected checkpoint')
        roles[role] = dict(source_root=str(roots[role]), selected_pair=pairs[role], frontend_finished=front_finished,
            audit=audit_role(roots[role], role, pairs[role], session))
        data[role] = audit_data_extension(roots[role], pool, role, session, allow_equal=True)
        evaluated[role] = evaluated_role(evaluations[role], role, pairs[role])
    require(all(data['A'][k] == data['B'][k] for k in ('splits', 'normalization_train_ids'))
        and read(pool/'A/state_data/data.json')['entries'] == read(pool/'B/state_data/data.json')['entries'],
        'Shared data pool A/B populations differ')
    check(); owners_after = assert_owners_exited(receipts, writers)
    require(before == snapshot({*roots.values(), pool, session_root}), 'Source inputs changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(session_root/'session.json', output/'session.json')
    shutil.copyfile(session_root/'metadata.json', output/'metadata.json')
    atomic_json(output/'gate.json', dict(passed=True, inherited_input_gate=str(pool/'gate.json'),
        inherited_input_gate_sha256=sha256(pool/'gate.json'), meaning='Inherited input gate only; no new task success claim'))
    links = {}
    for role in ('A', 'B'):
        base = output/role; rear = base/'backend'/role
        for p in (base/'training', rear/'training', rear/'offline'):
            p.mkdir(parents=True)
        targets = {'state_data':pool/role/'state_data', 'cache_index.json':pool/role/'cache_index.json',
            'training/best.pt':Path(pairs[role]['frontend']),
            'training/finished.json':roots[role]/role/'training/finished.json',
            f'backend/{role}/state_data':roots[role]/role/'state_data',
            f'backend/{role}/nominal':roots[role]/role/f'backend/{role}/nominal',
            f'backend/{role}/training/best.pt':Path(pairs[role]['backend']),
            f'backend/{role}/training/finished.json':roots[role]/role/f'backend/{role}/training/finished.json',
            f'backend/{role}/offline/result.json':roots[role]/role/f'backend/{role}/offline/result.json'}
        for relative,target in targets.items():
            target = target.resolve(strict=True)
            (base/relative).symlink_to(target, target_is_directory=target.is_dir())
            links[str(Path(role)/relative)] = str(target)
        for directory in (base, base/'backend', rear):
            shutil.copyfile(session_root/'session.json', directory/'session.json')
            (directory/'shared_resources.json').symlink_to(session_root/'shared_resources.json')
    (output/'shared_resources.json').symlink_to(session_root/'shared_resources.json')
    require({role:completed_pair(output, role) for role in ('A', 'B')} == pairs, 'Selected pair changed in source view')
    check()
    receipt = dict(version=VERSION, complete=True, view_root=str(output), resource_root=str(session_root),
        session=file_binding(session_root/'session.json'), deadline=session['deadline'],
        selected_pairs=pairs, policy_sources=roles, data_pool=dict(root=str(pool), roles=data),
        evaluations=evaluated, inherited_artifact_links=links, immutable_input_stat=before,
        owner_checks_before=owners, owner_checks_after=owners_after, created_utc=time.time(),
        scope='Same original session and deadline; collection data only, no checkpoint data/cache relabeling',
        owner_scope='Concrete source writer/owner receipts; session monitor and independent read-only diagnostics are not quiescence targets',
        new_workers=0, optimizer_updates=0, heldout_training_payloads_opened=0,
        development_numeric_audited=True)
    atomic_json(output/RECEIPT, receipt)
    atomic_json(output/'prepared.json', dict(kind=VERSION, complete=True, source_selection_sha256=sha256(output/RECEIPT)))
    binding_for(output)
    return receipt


def binding_for(source):
    source = Path(source).resolve(strict=True)
    path = source/RECEIPT
    if not path.exists():
        require(read(source/'prepared.json').get('kind') != VERSION if (source/'prepared.json').exists() else True,
            'Prepared source selection lost its receipt')
        return None
    binding = dict(version=VERSION, path=str(path), sha256=sha256(path))
    validate_binding(binding, source)
    return binding


def validate_binding(binding, source, *, plan=None):
    source = Path(source).resolve(strict=True); path = source/RECEIPT
    require(binding == dict(version=VERSION, path=str(path), sha256=sha256(path)), 'Source-selection binding changed')
    receipt = read(path); prepared = read(source/'prepared.json')
    require(receipt.get('version') == VERSION and receipt.get('complete') is True and receipt['view_root'] == str(source)
        and prepared == dict(kind=VERSION, complete=True, source_selection_sha256=binding['sha256']),
        'Incomplete selected source view')
    session_path = Path(receipt['resource_root'])/'session.json'
    require(sha256(source/'session.json') == sha256(session_path) == receipt['session']['sha256']
        and read(session_path)['deadline'] == receipt['deadline']
        and (source/'shared_resources.json').resolve(strict=True) == (session_path.parent/'shared_resources.json').resolve(strict=True),
        'Source view changed original session/resource identity')
    for relative,target in receipt['inherited_artifact_links'].items():
        require((source/relative).is_symlink() and (source/relative).resolve(strict=True) == Path(target),
            'Source-view artifact link changed: '+relative)
    data_sha = {role:receipt['data_pool']['roles'][role]['files']['data.json']['sha256'] for role in ('A', 'B')}
    require(all(sha256(source/role/'state_data/data.json') == digest for role,digest in data_sha.items()),
        'Selected source data manifest changed')
    if plan is not None:
        require(plan['pairs'] == receipt['selected_pairs'] and plan['source_data_sha256'] == data_sha
            and plan['resource_root'] == receipt['resource_root']
            and plan['source_session_sha256'] == receipt['session']['sha256'], 'Collection lost selected policy/data/session binding')
    return receipt


def role_paths(values):
    result = {}
    for value in values:
        role, separator, path = value.partition('=')
        require(separator and role in ('A', 'B') and path and role not in result, 'Use unique A=PATH and B=PATH arguments')
        result[role] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session-root', 'data-source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--role-source', action='append', required=True, metavar='ROLE=ROOT')
    parser.add_argument('--role-evaluation', action='append', required=True, metavar='ROLE=DEVELOPMENT')
    args = parser.parse_args()
    receipt = prepare(args.session_root, args.data_source, args.output,
        role_sources=role_paths(args.role_source), role_evaluations=role_paths(args.role_evaluation))
    print(json.dumps(dict(view_root=receipt['view_root'], deadline=receipt['deadline'], selected_pairs=receipt['selected_pairs'])), flush=True)


if __name__ == '__main__':
    main()
