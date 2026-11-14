"""Matched final-head reset study of nominal versus actual-target output anchors.

Preparation/execution is explicit. This module never selects or promotes a model.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.tight_fresh_backend import prepare as prepare_backend
from tools.tight_fresh_eval_recovery import wait_owned
from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

ARMS = ('control', 'previous')


def sources():
    return {str(p): sha256(p) for folder in ('tools', 'visuotactile')
            for p in sorted(Path(folder).rglob('*.py'))}


def source_evidence(source, session, pairs):
    from tools.tight_fresh_followup_session import evaluated_pairs
    origin = source if (source / 'development/result.json').exists() else Path(
        session.get('followup', {}).get('source_root', source))
    origin = origin.resolve()
    development = origin / 'development'
    if origin != source and session['followup']['selected_pairs'] != pairs:
        raise ValueError('Independent session selected another source pair')
    audit = evaluated_pairs(origin, development, pairs)
    return dict(root=str(origin), plan_sha256=sha256(development / 'plan.json'),
                results_sha256=sha256(development / 'results.json'), audit=audit)


def check_session_binding(output, config):
    """Keep every stage on the registered domain, deadline and telemetry owner."""
    binding = config['session_binding']
    if (str(output.resolve()) != binding['output_root']
            or config['source_root'] != binding['source_root']
            or config['deadline'] != binding['deadline']):
        raise ValueError('Study session identity changed')
    for path, digest in binding['files'].items():
        if sha256(path) != digest:
            raise ValueError('Registered study session changed: ' + path)
    session = read(output / 'session.json')
    if (session['deadline'] != binding['deadline']
            or session['domain'] != binding['domain']
            or session['resource_root'] != binding['resource_root']):
        raise ValueError('Registered deadline, domain or resource owner changed')
    target = Path(binding['shared_resources_target'])
    for root in [Path(config['source_root']), *(output / arm for arm in ARMS),
                 Path(binding['resource_root'])]:
        if (root / 'shared_resources.json').resolve(strict=True) != target:
            raise ValueError('Study telemetry ownership changed')
    return session


def prepare(source, output, epochs, count, seed):
    from visuotactile.residual_output import REACHABLE_PREVIOUS_TARGET_OUTPUT, MATCHED_FINAL_HEAD_RESET
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve every study attempt')
    output = output.resolve()
    if (type(epochs) is not int or type(count) is not int or type(seed) is not int
            or not 1 <= epochs <= 20 or not 1 <= count <= 8
            or not 0 <= seed <= 2**32 - 100000 - count):
        raise ValueError('Positive bounded training and development budgets required')
    session = read(source / 'session.json')
    if time.time() >= session['deadline']:
        raise TimeoutError('Registered session expired')
    pairs = {role: completed_pair(source, role) for role in ('A', 'B')}
    evidence = source_evidence(source, session, pairs)
    protected = {item['path']:item['sha256'] for section in ('files', 'case_results')
                 for item in evidence['audit'][section].values()}
    resource_target = (source / 'shared_resources.json').resolve(strict=True)
    resource_root = Path(session.get('resource_root', resource_target.parent)).resolve()
    if (resource_root / 'shared_resources.json').resolve(strict=True) != resource_target:
        raise ValueError('Source session and telemetry have different owners')
    session['resource_root'] = str(resource_root)
    populations = {}
    for role, pair in pairs.items():
        for key in ('frontend', 'backend'):
            protected[pair[key]] = pair[key + '_sha256']
        data = read(source / role / 'state_data/data.json')
        populations[role] = data['splits']
        for name in ('state_data/data.json', 'state_data/statistics.pt', 'cache_index.json',
                     f'backend/{role}/nominal/index.json', f'backend/{role}/nominal/scales.json'):
            path = source / role / name
            protected[str(path)] = sha256(path)
        index = read(source / role / 'backend' / role / 'nominal/index.json')
        if (index['base_sha256'] != pair['frontend_sha256']
                or index['data_sha256'] != sha256(source / role / 'state_data/data.json')
                or index.get('statistics_sha256') != sha256(source / role / 'state_data/statistics.pt')
                or index.get('residual_statistics_sha256') != sha256(source / 'B/state_data/statistics.pt')):
            raise ValueError('Selected source DP/cache/data/statistics do not match')
    if populations['A'] != populations['B'] or populations['A'].get('test'):
        raise ValueError('Matched A/B non-test populations required')
    output.mkdir(parents=True)
    atomic_json(output / 'session.json', session)
    for arm in ARMS:
        root = output / arm
        root.mkdir()
        atomic_json(root / 'session.json', session)
        (root / 'shared_resources.json').symlink_to(source / 'shared_resources.json')
        for role, pair in pairs.items():
            directory = root / role
            directory.mkdir()
            for name in ('state_data', 'training', 'cache_index.json'):
                (directory / name).symlink_to(source / role / name)
            atomic_json(directory / 'continuation.json', dict(path=pair['frontend'],
                sha256=pair['frontend_sha256'], previous_backend=pair['backend'],
                previous_backend_sha256=pair['backend_sha256'],
                statistics_sha256=sha256(source / role / 'state_data/statistics.pt'),
                initialization='Same selected trunk and buffers; matched zero final Linear; fresh optimizer'))
        for role in ('A', 'B'):
            prepare_backend(root, role, epochs, continue_rear=True,
                output_contract=REACHABLE_PREVIOUS_TARGET_OUTPUT if arm == 'previous' else None,
                continuation_head_reset=MATCHED_FINAL_HEAD_RESET)
            nominal = root / role / 'backend' / role / 'nominal'
            nominal.rmdir()
            original = (source / role / 'backend' / role / 'nominal').resolve()
            nominal.symlink_to(original, target_is_directory=True)
            atomic_json(root / role / 'backend/cache_complete.json', dict(complete=True,
                recomputed=False, index_sha256=sha256(original / 'index.json'),
                unchanged_frontend_sha256=pairs[role]['frontend_sha256'],
                unchanged_data_sha256=sha256(source / role / 'state_data/data.json'),
                reason='Frozen identical DP, data and statistics; trainer revalidates every cache shard'))
    session_binding = dict(source_root=str(source), output_root=str(output.resolve()),
        deadline=session['deadline'], domain=session['domain'], resource_root=str(resource_root),
        shared_resources_target=str(resource_target),
        files={str(path):sha256(path) for path in
               [source / 'session.json', *output.rglob('session.json')]})
    protected.update(session_binding['files'])
    for pattern in ('configuration.json', 'continuation.json', 'cache_complete.json'):
        protected.update({str(path):sha256(path) for path in output.rglob(pattern)})
    registration = dict(version='matched_rear_output_anchor_study_v1', source_root=str(source),
        source_pairs=pairs, source_evidence=evidence, source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=sources(),
        protected=protected, session_binding=session_binding,
        epochs=epochs, count_per_role=count, development_seed=seed,
        deadline=session['deadline'], utc=time.time(), test_used=False,
        arms=list(ARMS), matched_head_reset=MATCHED_FINAL_HEAD_RESET,
        training_population=populations['A'], loss_weight=1.,
        method='Frozen complete DP/data/cache. Same role-specific trained trunk, matched zero final Linear, fresh optimizers and equal real exposure. Only output anchor differs; both still read full DP plans. Original final execution constraints unchanged.',
        interpretation='Zero raw prior changes from clipped nominal to previous actual actuator target. Inherited trunk previously learned nominal coordinates. No exact hidden-state paired rollout claim. No automatic model promotion.')
    atomic_json(output / 'registration.json', registration)
    atomic_json(output / 'status.json', dict(stage='prepared', utc=time.time()))


def development_cases(domain, count, seed):
    from visuotactile.narrow_data import direct_configuration
    cases = []
    order = [('control', 'A'), ('previous', 'B'), ('previous', 'A'), ('control', 'B')]
    for index in range(count):
        value = seed + index
        spec = dict(identity=f'development_{value}', family=f'fresh_development_{value}',
            split='development', seed=value, event_seed=value + 100000, cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1', narrow_reset=direct_configuration(domain, value))
        for arm, role in order if index % 2 == 0 else list(reversed(order)):
            pair_key = arm + '_' + role
            cases.append(dict(name=pair_key + '_' + spec['identity'], role=pair_key,
                study_arm=arm, student_role=role, spec=copy.deepcopy(spec), video=index == 0))
    return cases


def prepare_development(output):
    config = read(output / 'registration.json')
    session = check_session_binding(output, config)
    pairs = {arm + '_' + role: completed_pair(output / arm, role)
             for arm in ARMS for role in ('A', 'B')}
    destination = output / 'development'
    destination.mkdir()
    atomic_json(destination / 'session.json', session)
    atomic_json(destination / 'plan.json', dict(kind='autonomous', pairs=pairs,
        cases=development_cases(session['domain'], config['count_per_role'], config['development_seed']),
        resource_root=session['resource_root'], source_commit=config['source_commit'],
        source_sha256=config['source_sha256'], inference_scheduler_index_mode='host',
        test_used=False, study='matched_rear_output_anchor_study_v1', utc=time.time(),
        cohort='Repeated development opportunities, alternating reversed arm/role order; no exact hidden-state pairing'))


def run(output):
    config = read(output / 'registration.json')
    check_session_binding(output, config)
    if (output / 'run_started.json').exists():
        raise FileExistsError('No duplicate study owner')
    if time.time() >= config['deadline']:
        raise TimeoutError('Registered session expired')
    atomic_json(output / 'run_started.json', dict(pid=os.getpid(), utc=time.time(),
        registration_sha256=sha256(output / 'registration.json')))
    ledger = output / 'stages'
    ledger.mkdir()
    receipts = []
    def execute(name, args, completion, startup=False):
        check_session_binding(output, config)
        for path, digest in {**config['protected'], **config['source_sha256']}.items():
            if sha256(path) != digest:
                raise ValueError('Frozen study dependency changed: ' + path)
        if time.time() >= config['deadline']:
            raise TimeoutError('Original registered deadline expired')
        command = [sys.executable, '-m', *map(str, args)]
        atomic_json(output / 'status.json', dict(stage=name, command=command, utc=time.time()))
        began = time.time()
        with (ledger / (name + '.log')).open('x') as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONFAULTHANDLER='1'))
            reason = wait_owned(worker, config['deadline'],
                startup_marker=completion.parent / 'initialization.json' if startup else None,
                startup_seconds=180 if startup else None)
        receipt = dict(stage=name, returncode=worker.returncode, stop_reason=reason,
            seconds=time.time() - began, command=command, completion=str(completion),
            completion_sha256=sha256(completion) if completion.exists() else None)
        atomic_json(ledger / (name + '.json'), receipt)
        receipts.append(receipt)
        if reason or worker.returncode or not completion.exists():
            raise RuntimeError('Study stage failed: ' + name)
        complete = read(completion)
        required = ('target_reached' if 'residual-train' in args else
                    'complete' if 'residual-offline' in args else 'software_complete')
        if (complete.get(required) is not True or complete.get('disposable')
                or any(complete.get(key) is False for key in ('complete', 'software_complete', 'target_reached'))):
            raise RuntimeError('Incomplete stage receipt: ' + name)
        print(json.dumps(receipt), flush=True)
    try:
        for arm, role in [('control', 'A'), ('previous', 'B'), ('previous', 'A'), ('control', 'B')]:
            rear = output / arm / role / 'backend' / role
            for stage, completion in [('residual-train', rear / 'training/finished.json'),
                                      ('residual-offline', rear / 'offline/result.json')]:
                execute(arm + '_' + role + '_' + stage, ['visuotactile.anchor_continue', stage,
                    '--root', rear, '--job', 'image', '--output', completion.parent], completion)
        prepare_development(output)
        execute('development', ['tools.tight_fresh_rollouts', 'run', '--root', output,
            '--output', output / 'development'], output / 'development/result.json', startup=True)
        atomic_json(output / 'finished.json', dict(software_complete=True, stages=receipts,
            utc=time.time(), research_conclusion='Pending Agent review; no automatic selection or next experiment'))
        atomic_json(output / 'status.json', dict(stage='complete', utc=time.time()))
    except BaseException as error:
        atomic_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            stages=receipts, utc=time.time()))
        atomic_json(output / 'status.json', dict(stage='failed', error=repr(error), utc=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--seed', type=int, default=28140000)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.source is None:
            p.error('Preparation requires reviewed complete source pairs')
        prepare(a.source.resolve(), a.output.resolve(), a.epochs, a.count, a.seed)
    else:
        run(a.output.resolve())


if __name__ == '__main__':
    main()
