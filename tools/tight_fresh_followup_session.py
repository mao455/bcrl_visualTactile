"""Prepare an independent follow-up window from an evaluated complete A/B pair.

No worker is started and no source file is modified. Artifact links are intended
for reading; they are not an operating-system write barrier. Collection and cycle
outputs must use new children of the destination, never the inherited artifacts.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def file_binding(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Artifact digest changed: ' + str(path))
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def evaluated_pairs(source, evaluation, pairs):
    """Require every registered case, including natural failures, to be complete."""
    plan = read(evaluation / 'plan.json')
    started = read(evaluation / 'run_started.json')
    finished = read(evaluation / 'result.json')
    rows = read(evaluation / 'results.json')
    if (plan.get('kind') != 'autonomous' or plan.get('test_used') is not False
            or plan.get('pairs') != pairs or not plan.get('source_sha256')
            or Path(plan.get('resource_root', '')).resolve() != source):
        raise ValueError('Evaluation must bind the selected autonomous non-test A/B pair')
    if started.get('plan_sha256') != sha256(evaluation / 'plan.json'):
        raise ValueError('Evaluation plan differs from its worker registration')
    if (finished.get('software_complete') is not True or finished.get('worker_error')
            or finished.get('cases') != len(rows)):
        raise ValueError('Source autonomous evaluation is incomplete')
    cases = plan.get('cases', [])
    names = [case['name'] for case in cases]
    if (not cases or len(set(names)) != len(names)
            or any(Path(name).name != name or name in ('.', '..') for name in names)
            or {case['role'] for case in cases} != {'A', 'B'}
            or len(rows) != len(cases) or {row['case'] for row in rows} != set(names)):
        raise ValueError('Source evaluation cases are missing or duplicated')
    by_name = {row['case']: row for row in rows}
    receipts = {}
    for case in cases:
        row = by_name[case['name']]
        result = row['result']
        pair = pairs[case['role']]
        receipt = evaluation / case['name'] / 'result.json'
        if (row['role'] != case['role'] or result != read(receipt)
                or not result.get('legal') or not result.get('software_complete')
                or result.get('censored') or result.get('worker_error')
                or result.get('dagger') is not None or case.get('dagger') is not None
                or result.get('steps', 0) < 1
                or result.get('residual_ticks') != result['steps']
                or result.get('checkpoint_sha256') != pair['backend_sha256']
                or result.get('residual_version') != pair['residual_version']
                or result.get('simulation_step_hz') != 60 or result.get('plan_hz') != 5
                or result.get('action_control_mode') != 'absolute_joint_position_target_rad'
                or any(result.get(key) != value for key, value in case['spec'].items())):
            raise ValueError('Case lacks complete autonomous full-student evidence: ' + case['name'])
        receipts[case['name']] = file_binding(receipt)
    return dict(root=str(evaluation), cases=len(cases), case_results=receipts,
        files={name:file_binding(evaluation/name) for name in
               ('plan.json', 'run_started.json', 'result.json', 'results.json')},
        source_commit=plan.get('source_commit'), source_sha256=plan['source_sha256'])


def context_root(source):
    """Use local input evidence or the immediate study's explicit registration."""
    if (source/'metadata.json').is_file() and (source/'gate.json').is_file():
        return source, None
    registration = source.parent/'registration.json'
    registered = read(registration)
    context = Path(registered['source_root']).resolve(strict=True)
    protected = registered.get('protected', {})
    for role in ('A', 'B'):
        for name in ('data.json', 'statistics.pt'):
            original = context/role/'state_data'/name
            if (source/role/'state_data'/name).resolve() != original.resolve():
                raise ValueError('Study input evidence does not refer to the selected data')
            expected = protected.get(str(original))
            if expected is None:
                raise ValueError('Study registration does not protect its data and affine')
            file_binding(original, expected)
    return context, file_binding(registration)


def source_ownership(source, evaluation, context):
    """Include the original telemetry owner, not only a study variant child."""
    telemetry = source/'shared_resources.json'
    resource_root = telemetry.resolve().parent if telemetry.is_symlink() else source
    roots = {source, evaluation, context, resource_root}
    receipts = set()
    for root in (source, context):
        for name in ('run_started.json', 'owner_started.json', 'monitor.json'):
            receipts.update(root.rglob(name))
    ancestors = [p for p in source.parents if p == resource_root or resource_root in p.parents]
    for root in (evaluation, resource_root, *ancestors):
        for name in ('run_started.json', 'owner_started.json', 'monitor.json'):
            if (root/name).is_file():
                receipts.add(root/name)
    return roots, sorted(receipts)


def assert_quiescent(roots, receipts, proc=Path('/proc')):
    """Read process state only. Never signal a source process."""
    from visuotactile.state_resources import owned_command_root
    def live(directory):
        try:
            return directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    # Historical PIDs can be reused. A live PID alone is never worker identity;
    # confirm the current argv's bound root, including receipts with recycled PIDs.
    candidates = set(proc.iterdir())
    for receipt in receipts:
        pid = read(receipt).get('pid')
        if isinstance(pid, int):
            candidates.add(proc/str(pid))
    for directory in candidates:
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue  # Research workers are owned by this user; do not inspect unrelated accounts.
            if not live(directory):
                continue
            command = directory.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            # Owners sometimes name their bound experiment --output or --source-root.
            for flag in ('--root', '--output', '--source-root'):
                normalized = []
                for index, value in enumerate(command):
                    if value == flag and index+1 < len(command):
                        normalized += ['--root', command[index+1]]
                    elif value.startswith(flag+'='):
                        normalized += ['--root', value[len(flag)+1:]]
                if not normalized:
                    continue
                cwd = (directory.joinpath('cwd').resolve(strict=True)
                    if any(not Path(value).is_absolute() for value in normalized[1::2]) else Path('/'))
                if owned_command_root(normalized, roots, cwd=cwd):
                    raise ValueError('Live process still owns source artifacts: pid=' + directory.name)
        except (FileNotFoundError, ProcessLookupError):
            continue


def audit_role(source, role, pair, session):
    import torch
    base = source/role
    state = base/'state_data'
    rear = base/'backend'/role
    files = {}
    for name in ('data.json', 'statistics.pt', 'numeric.pt'):
        files['state_data/'+name] = file_binding(state/name)
    if role == 'A' or (state/'loss_context.pt').exists():
        files['state_data/loss_context.pt'] = file_binding(state/'loss_context.pt')
    files['cache_index.json'] = file_binding(base/'cache_index.json')
    files['nominal/index.json'] = file_binding(rear/'nominal/index.json')
    files['nominal/scales.json'] = file_binding(rear/'nominal/scales.json')
    for name in ('training/best.pt', 'training/finished.json', 'offline/result.json'):
        files['backend/'+name] = file_binding(rear/name,
            pair['backend_sha256'] if name == 'training/best.pt' else None)
    files['frontend'] = file_binding(base/'training/best.pt', pair['frontend_sha256'])
    data = read(state/'data.json')
    visual = read(base/'cache_index.json')
    nominal = read(rear/'nominal/index.json')
    expected = dict(data_sha256=files['state_data/data.json']['sha256'],
        statistics_sha256=files['state_data/statistics.pt']['sha256'], dino_sha256=session['dino_sha256'])
    if (data.get('visual_stride') != 1 or data.get('supervision') !=
            ('q_reached' if role == 'A' else 'u_expert')
            or not visual.get('complete') or visual.get('stride') != 1
            or visual.get('data_sha256') != expected['data_sha256']
            or visual.get('dino_sha256') != expected['dino_sha256']
            or visual.get('trainable_projection_cached') is not False):
        raise ValueError('Data or continuous-vision cache contract mismatch for ' + role)
    identities = [identity for split in data['splits'].values() for identity in split]
    if (len(identities) != len(set(identities)) or set(identities) != set(data['entries'])
            or set(visual['entries']) != set(identities) or set(nominal['episodes']) != set(identities)):
        raise ValueError('Data and cache episode populations differ')
    if (nominal.get('base_sha256') != pair['frontend_sha256']
            or any(nominal.get(key) != expected[key] for key in ('data_sha256', 'statistics_sha256'))
            or nominal.get('residual_statistics_sha256') != sha256(source/'B/state_data/statistics.pt')):
        raise ValueError('Backend plan cache no longer belongs to this pair/data/affine')
    for kind, path, cache_digest in [('frontend', base/'training/best.pt', files['cache_index.json']['sha256']),
            ('backend', rear/'training/best.pt', files['nominal/index.json']['sha256'])]:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        config = saved['configuration']
        if (any(config.get(key) != value for key, value in expected.items())
                or config.get('cache_index_sha256') != cache_digest
                or saved['contract'].get('visual_stride') != 1):
            raise ValueError(kind + ' checkpoint data/statistics/cache binding changed')
        if kind == 'backend' and (config.get('base_sha256') != pair['frontend_sha256']
                or config.get('residual_statistics_sha256') != nominal['residual_statistics_sha256']):
            raise ValueError('Backend pair or shared command affine changed')
        del saved
    numeric = torch.load(state/'numeric.pt', map_location='cpu', weights_only=False)
    if numeric.get('data_sha256') != expected['data_sha256'] or set(numeric['records']) != set(identities):
        raise ValueError('Numeric history cache data binding changed')
    del numeric
    large = {}
    for identity in identities:
        if Path(identity).name != identity or identity in ('.', '..'):
            raise ValueError('Local episode names required')
        entry = data['entries'][identity]
        cached = visual['entries'][identity]
        if (cached.get('source_sha256') != entry['source_sha256']
                or cached.get('frames') != entry['length']
                or (state/'perception'/(identity+'.pt')).resolve() != Path(cached['path']).resolve()):
            raise ValueError('Visual cache source/time population changed: ' + identity)
        large[identity] = {}
        for kind, path, digest in [('raw', Path(entry['path']), entry['source_sha256']),
                ('visual', Path(cached['path']), cached['sha256'])]:
            stat = path.stat()
            if not path.is_file() or stat.st_size < 1:
                raise ValueError('Missing source/cache artifact: ' + str(path))
            large[identity][kind] = dict(path=str(path.resolve()), declared_sha256=digest,
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, bytes_rehashed=False)
        files['nominal/'+identity+'.pt'] = file_binding(rear/'nominal'/(identity+'.pt'),
            nominal['episodes'][identity]['sha256'])
    return dict(files=files, large_artifacts=large, splits=data['splits'],
        normalization_train_ids=data.get('normalization_train_ids', data['splits']['train']))


def prepare(source, output, hours, *, evaluation=None):
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('New independent session destination already exists')
    output = output.resolve()
    if isinstance(hours, bool) or not math.isfinite(hours) or hours <= 0:
        raise ValueError('New session hours must be finite and positive')
    evaluation = Path(evaluation).resolve(strict=True) if evaluation else source/'development'
    session = read(source/'session.json')
    if (session.get('visual_stride') != 1 or session.get('control_hz') != 60
            or session.get('planning_hz') != 5 or session.get('resource_policy') != 'full_gpu_v1'):
        raise ValueError('Expected current every-frame full-GPU research contract')
    context, registration = context_root(source)
    roots, receipts = source_ownership(source, evaluation, context)
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise ValueError('Independent output must be outside the original session and artifact roots')
    assert_quiescent(roots, receipts)
    pairs = {role:completed_pair(source, role) for role in ('A', 'B')}
    evaluated = evaluated_pairs(source, evaluation, pairs)
    gate = read(context/'gate.json')
    if gate.get('passed') is not True:
        raise ValueError('Source every-frame input gate did not pass')
    inputs = {name:file_binding(context/name) for name in ('metadata.json', 'gate.json')}
    source_session = file_binding(source/'session.json')
    roles = {role:audit_role(source, role, pairs[role], session) for role in ('A', 'B')}
    if (roles['A']['splits'] != roles['B']['splits'] or
            roles['A']['normalization_train_ids'] != roles['B']['normalization_train_ids']):
        raise ValueError('A/B data or original normalization cohorts differ')
    code_root = Path(__file__).resolve().parent.parent
    code = {str(path.relative_to(code_root)):sha256(path) for folder in ('tools', 'visuotactile')
        for path in sorted((code_root/folder).rglob('*.py'))}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_root, text=True).strip()
    assert_quiescent(roots, receipts)
    started = time.time()
    deadline = started+float(hours)*3600
    if not math.isfinite(deadline):
        raise ValueError('New deadline must be finite')
    fresh = {key:copy.deepcopy(value) for key, value in session.items()
        if not (key.endswith('deadline') or key in ('owned_worker_roots', 'resource_root',
            'source_root', 'source_commit', 'source_head', 'source_sha256', 'started', 'started_utc',
            'trainable_initialization', 'authorization', 'followup'))}
    fresh.update(session_kind='independent_followup_v1', started_utc=started,
        deadline=deadline, cache_deadline=deadline, training_deadline=deadline, offline_deadline=deadline,
        resource_root=str(output), owned_worker_roots=[str(output)], source_commit=commit,
        trainable_initialization='Continuation from reviewed complete A/B pairs; fresh initial30 run is preserved',
        authorization='User-authorized continuing autonomous DAgger research; independent session, not an extension',
        followup=dict(source_root=str(source), source_session_sha256=source_session['sha256'],
            previous_deadline=session['deadline'], selected_pairs=pairs, hours=float(hours)))
    output.mkdir(parents=True, exist_ok=False)
    (output/'provenance').mkdir()
    atomic_json(output/'session.json', fresh)
    shutil.copyfile(source/'session.json', output/'provenance/source_session.json')
    shutil.copyfile(context/'metadata.json', output/'metadata.json')
    shutil.copyfile(context/'gate.json', output/'provenance/source_gate.json')
    atomic_json(output/'gate.json', dict(passed=True, inherited_input_gate=inputs['gate.json']['path'],
        inherited_input_gate_sha256=inputs['gate.json']['sha256'],
        meaning='Inherited reviewed input gate; no new simulation gate or task-success claim',
        upstream_metadata_sha256=inputs['metadata.json']['sha256']))
    links = {}
    for role in ('A', 'B'):
        base = output/role
        rear = base/'backend'/role
        for directory in (base/'training', rear/'training', rear/'offline'):
            directory.mkdir(parents=True)
        artifacts = {
            'state_data':source/role/'state_data', 'cache_index.json':source/role/'cache_index.json',
            'training/best.pt':Path(pairs[role]['frontend']),
            'backend/'+role+'/state_data':source/role/'state_data',
            'backend/'+role+'/nominal':source/role/'backend'/role/'nominal',
            'backend/'+role+'/training/best.pt':Path(pairs[role]['backend']),
            'backend/'+role+'/training/finished.json':source/role/'backend'/role/'training/finished.json',
            'backend/'+role+'/offline/result.json':source/role/'backend'/role/'offline/result.json'}
        for name, target in artifacts.items():
            target = target.resolve(strict=True)
            (base/name).symlink_to(target, target_is_directory=target.is_dir())
            links[str(Path(role)/name)] = str(target)
        for directory in (base, base/'backend', rear):
            atomic_json(directory/'session.json', dict(fresh, modality=role))
            (directory/'shared_resources.json').symlink_to(output/'shared_resources.json')
    # Root telemetry is intentionally absent until this root's own monitor starts.
    if {role:completed_pair(output, role) for role in ('A', 'B')} != pairs:
        raise ValueError('Prepared links changed selected complete policy identities')
    binding = dict(version='independent_followup_binding_v1', source_root=str(source),
        context_root=str(context), source_session=source_session, context_inputs=inputs,
        upstream_study_registration=registration, selected_pairs=pairs, evaluation=evaluated,
        roles=roles, inherited_artifact_links=links, source_commit=commit, source_sha256=code,
        source_ownership_roots=sorted(str(root) for root in roots),
        source_workers_checked_exited_utc=started, source_pid_receipts=[str(p) for p in receipts],
        artifact_verification='All listed files and nominal shards rehashed; raw and frozen visual '
            'shards bound to declared manifest SHA256 and current stat, not rehashed',
        mutation_scope='Destination only; artifact symlinks are read-only by convention, not filesystem enforcement',
        previous_deadline=session['deadline'], started_utc=started, deadline=deadline,
        deadline_relation='New independent window; source deadline and evidence unchanged')
    atomic_json(output/'provenance/binding.json', binding)
    atomic_json(output/'prepared.json', dict(kind='independent_followup_v1', complete=True,
        source_root=str(source), selected_pairs=pairs, source_commit=commit,
        binding_sha256=sha256(output/'provenance/binding.json'), session_sha256=sha256(output/'session.json'),
        source_workers_exited=True, monitor_started=False, workers_started=False, utc=started))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, required=True)
    parser.add_argument('--evaluation', type=Path, help='Completed autonomous evaluation, default source/development')
    args = parser.parse_args()
    binding = prepare(args.source_root, args.output, args.hours, evaluation=args.evaluation)
    print(json.dumps(dict(output=str(args.output.resolve()), deadline=binding['deadline'],
        selected_pairs=binding['selected_pairs'], workers_started=False)), flush=True)


if __name__ == '__main__':
    main()
