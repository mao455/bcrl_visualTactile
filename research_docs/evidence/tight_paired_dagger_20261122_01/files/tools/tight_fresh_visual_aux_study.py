"""Prepare and own one matched, calibrated visual-pose auxiliary study.

Preparation never launches workers. Execution never promotes a model or retries
an attempt. The source is an explicitly reviewed nominal-anchor complete pair.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tools.tight_fresh_anchor_study import source_evidence, sources
from tools.tight_fresh_eval_recovery import stop_owned
from tools.tight_fresh_followup_session import audit_role
from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.visual_aux_training import SPEC_VERSION, load_auxiliary_spec

VERSION = 'matched_visual_pose_aux_study_v1'
ARMS = ('control', 'auxiliary')
ROUTES = dict(control='detached_readout', auxiliary='shared_vision')
ORDER = (('control', 'A'), ('auxiliary', 'B'), ('auxiliary', 'A'), ('control', 'B'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_configuration(source, role, pair):
    import torch
    from visuotactile.position_control import continuous_image_contract
    from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
    from visuotactile.target_windows import validate_target_window
    frontend = torch.load(pair['frontend'], map_location='cpu', weights_only=False)
    backend = torch.load(pair['backend'], map_location='cpu', weights_only=False)
    expected = continuous_image_contract('q_reached' if role == 'A' else 'u_expert')
    require(frontend['contract'] == expected and backend['contract'] == expected
        and backend['configuration']['output_contract'] == REACHABLE_CENTERED_OUTPUT
        and backend['residual_output_version'] == REACHABLE_CENTERED_OUTPUT['version'],
        'Reviewed v6 source with nominal-centered backend required; no automatic anchor selection')
    config = frontend['configuration']
    require(config == read(source/role/'training/configuration.json')
        and frontend['updates'] > 0, 'Selected source configuration differs')
    return dict(target_window_contract=validate_target_window(
        config.get('continuation', {}).get('target_window_contract')))


def acceptance_binding(calibration, acceptance, code):
    measured = read(calibration); accepted = read(acceptance)
    require(accepted.get('version') == 'visual_pose_aux_acceptance_v1'
        and accepted.get('passed') is True and accepted.get('formal_training_updates') == 0
        and accepted.get('device') == 'cuda:0' and accepted.get('optimizer_updates') == 4
        and accepted.get('source_unchanged') is True and accepted.get('roles') == ['A','B']
        and accepted.get('routes') == list(ROUTES.values())
        and accepted.get('validation_used') is False and accepted.get('test_used') is False
        and accepted.get('calibration_sha256') == sha256(calibration)
        and accepted.get('source_sha256') == code,
        'Passed disposable CUDA acceptance bound to this source and calibration is required')
    require(measured.get('passed') is True and measured.get('optimizer_updates') == 0
        and measured.get('simulator_starts') == 0 and measured.get('validation_used') is False
        and measured.get('test_used') is False and measured.get('source_sha256') == code,
        'Train-only zero-update calibration must bind the exact execution source')
    require(all(accepted.get(key) == measured.get(key) for key in
        ('head_seed', 'loss_weight', 'frontend_sha256', 'data_sha256', 'statistics_sha256', 'target_window_contract')),
        'Acceptance and calibrated source/head/weight identities differ')
    return measured


def prepare(source, output, calibration, acceptance, epochs, rear_epochs, count, seed, *, learning_rate=3e-5, session_root=None):
    source = source.resolve(strict=True); output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve every study attempt')
    output = output.resolve(); calibration = calibration.resolve(strict=True); acceptance = acceptance.resolve(strict=True)
    require(all(type(v) is int for v in (epochs, rear_epochs, count, seed))
        and 1 <= epochs <= 20 and 1 <= rear_epochs <= 20 and 1 <= count <= 8
        and 0 <= seed <= 2**32-100000-count
        and type(learning_rate) in (float, int) and math.isfinite(learning_rate) and 0 < learning_rate <= 1e-4,
        'Finite matched training and development budgets required')
    code = sources(); measured = acceptance_binding(calibration, acceptance, code)
    require(read(acceptance).get('learning_rate') == learning_rate,
        'Formal learning rate differs from the preregistered disposable acceptance')
    session_root = Path(session_root).resolve(strict=True) if session_root else source
    session = read(session_root/'session.json')
    require(session['domain'] == read(source/'session.json')['domain'], 'Source and active session domains differ')
    if time.time() >= session['deadline']:
        raise TimeoutError('Original registered session expired')
    require(session.get('visual_stride') == 1 and session.get('control_hz') == 60
        and session.get('planning_hz') == 5 and session.get('resource_policy') == 'full_gpu_v1',
        'Current full-frame 60Hz/5Hz research session required')
    resource = (session_root/'shared_resources.json').resolve(strict=True)
    owner = Path(session.get('resource_root', resource.parent)).resolve(strict=True)
    require((owner/'shared_resources.json').resolve(strict=True) == resource, 'Telemetry owner differs')
    session['resource_root'] = str(owner)
    pairs = {role:completed_pair(source, role) for role in ('A', 'B')}
    evidence = source_evidence(source, session, pairs)
    audits = {role:audit_role(source, role, pairs[role], session) for role in ('A', 'B')}
    configurations = {role:source_configuration(source, role, pairs[role]) for role in ('A', 'B')}
    require(audits['A']['splits'] == audits['B']['splits'] and not audits['A']['splits'].get('test')
        and audits['A']['normalization_train_ids'] == audits['B']['normalization_train_ids']
        and len(audits['A']['normalization_train_ids']) == 30,
        'Matched original30 normalization and aggregate non-test populations required')
    require(read(source/'gate.json').get('passed') is True, 'Inherited real-frame input gate required')
    protected = {str(p):sha256(p) for p in
        (source/'session.json', session_root/'session.json', source/'metadata.json', source/'gate.json', calibration, acceptance)}
    for section in ('files', 'case_results'):
        protected.update({item['path']:item['sha256'] for item in evidence['audit'][section].values()})
    for role in ('A', 'B'):
        for name, item in audits[role]['files'].items():
            # Old learned plan shards are audited source evidence, never reused.
            if not name.startswith('nominal/'):
                protected[item['path']] = item['sha256']
        for key in ('frontend', 'backend'):
            protected[pairs[role][key]] = pairs[role][key+'_sha256']
    output.mkdir(parents=True)
    atomic_json(output/'session.json', session)
    (output/'shared_resources.json').symlink_to(resource)
    for arm in ARMS:
        root = output/arm; root.mkdir()
        for name in ('session.json', 'metadata.json'):
            atomic_json(root/name, session if name == 'session.json' else read(source/name))
        atomic_json(root/'gate.json', dict(passed=True, inherited_input_gate=str(source/'gate.json'),
            inherited_input_gate_sha256=sha256(source/'gate.json'),
            auxiliary_acceptance_sha256=sha256(acceptance),
            meaning='Inherited every-frame input gate plus explicit disposable auxiliary CUDA acceptance'))
        (root/'shared_resources.json').symlink_to(resource)
        for role in ('A', 'B'):
            directory = root/role; directory.mkdir()
            for name in ('state_data', 'cache_index.json'):
                (directory/name).symlink_to((source/role/name).resolve(strict=True))
            (directory/'shared_resources.json').symlink_to(resource)
            atomic_json(directory/'session.json', session)
            pair = pairs[role]
            continuation = dict(path=pair['frontend'], sha256=pair['frontend_sha256'],
                statistics_sha256=sha256(source/role/'state_data/statistics.pt'),
                source_root=str(source/role), reuse_index=str(source/role/'cache_index.json'),
                target_window_contract=configurations[role]['target_window_contract'], learning_rate=learning_rate,
                initialization='Same selected v6 EMA; matched auxiliary head; fresh paired AdamW/EMA; fixed original30 affine',
                previous_backend=pair['backend'], previous_backend_sha256=pair['backend_sha256'])
            atomic_json(directory/'continuation.json', continuation)
            atomic_json(directory/'visual_auxiliary.json', dict(version=SPEC_VERSION, route=ROUTES[arm],
                head_seed=measured['head_seed'], loss_weight=measured['loss_weight'],
                calibration=dict(path=str(calibration), sha256=sha256(calibration))))
            load_auxiliary_spec(root, role, continuation)
    own_files = {str(path):sha256(path) for path in output.rglob('*.json') if not path.is_symlink()}
    protected.update(own_files)
    registration = dict(version=VERSION, source_root=str(source), output_root=str(output), source_pairs=pairs,
        source_evidence=evidence, source_audits=audits, source_configurations=configurations,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=code,
        protected=protected, arms=list(ARMS), routes=ROUTES, frontend_epochs=epochs, backend_epochs=rear_epochs,
        frontend_learning_rate=learning_rate, count_per_role=count, development_seed=seed,
        head_seed=measured['head_seed'], loss_weight=measured['loss_weight'],
        calibration=dict(path=str(calibration), sha256=sha256(calibration)),
        acceptance=dict(path=str(acceptance), sha256=sha256(acceptance)),
        training_population=audits['A']['splits'], normalization_train_ids=audits['A']['normalization_train_ids'],
        session_binding=dict(session_root=str(session_root), deadline=session['deadline'], domain=session['domain'], resource_root=str(owner),
            shared_resources_target=str(resource), session_sha256=sha256(output/'session.json')),
        deadline=session['deadline'], utc=time.time(), test_used=False,
        cache_policy='Reuse only frozen DINO raw patches/depth. Each updated DP regenerates its own learned plan/features before backend training.',
        method='Equal source EMA, data, affine, auxiliary head/weight, optimizer budgets and selection; only detached versus shared-vision gradient route differs. Both retain nominal-centered backend continuation.',
        interpretation='Repeated development opportunities, no exact hidden-state pairing; no automatic promotion or next experiment.')
    atomic_json(output/'registration.json', registration)
    atomic_json(output/'prepared.json', dict(complete=True, workers_started=False,
        registration_sha256=sha256(output/'registration.json'), utc=time.time()))
    atomic_json(output/'status.json', dict(stage='prepared', workers_started=False, utc=time.time()))


def check_binding(output, config):
    require(read(output/'prepared.json')['registration_sha256'] == sha256(output/'registration.json'),
        'Registered study bytes changed after preparation')
    require(config['version'] == VERSION and config['arms'] == list(ARMS) and config['routes'] == ROUTES
        and str(output.resolve()) == config['output_root'], 'Study identity differs')
    for path, digest in {**config['protected'], **config['source_sha256']}.items():
        require(sha256(path) == digest, 'Frozen study dependency changed: '+path)
    binding = config['session_binding']; session = read(output/'session.json')
    require(config['deadline'] == binding['deadline'] == session['deadline']
        and session['domain'] == binding['domain'] and session['resource_root'] == binding['resource_root']
        and sha256(output/'session.json') == binding['session_sha256'], 'Registered session changed')
    for root in (output, Path(binding['session_root']), Path(binding['resource_root']),
                 *(output/arm for arm in ARMS), *(output/arm/role for arm, role in ORDER)):
        require(str((root/'shared_resources.json').resolve(strict=True)) == binding['shared_resources_target'],
            'Telemetry owner changed')
    return session


def development_cases(domain, count, seed):
    from visuotactile.narrow_data import direct_configuration
    result = []
    for index in range(count):
        value = seed+index
        spec = dict(identity=f'development_{value}', family=f'fresh_development_{value}', split='development',
            seed=value, event_seed=value+100000, cohort='fresh', collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=direct_configuration(domain, value))
        for arm, role in ORDER if index % 2 == 0 else reversed(ORDER):
            key = arm+'_'+role
            result.append(dict(name=key+'_'+spec['identity'], role=key, study_arm=arm, student_role=role,
                spec=copy.deepcopy(spec), video=index == 0))
    return result


def prepare_development(output, config):
    session = check_binding(output, config)
    pairs = {arm+'_'+role:completed_pair(output/arm, role) for arm, role in ORDER}
    destination = output/'development'; destination.mkdir()
    atomic_json(destination/'session.json', session)
    atomic_json(destination/'plan.json', dict(kind='autonomous', pairs=pairs,
        cases=development_cases(session['domain'], config['count_per_role'], config['development_seed']),
        resource_root=session['resource_root'], source_commit=config['source_commit'], source_sha256=config['source_sha256'],
        inference_scheduler_index_mode='host', test_used=False, study=VERSION, utc=time.time(),
        cohort='Interleaved full DP plus backend; repeated development opportunities, not exact hidden-state pairs'))


def wait_child(process, deadline, *, stop_paths, stop_requested, startup_marker=None):
    """Only signal the process group created by this study owner."""
    began = time.time(); reason = None
    try:
        while process.poll() is None:
            now = time.time()
            if stop_requested() or any(path.exists() for path in stop_paths):
                reason = 'requested_stop'; break
            if now >= deadline:
                reason = 'original_session_deadline'; break
            if startup_marker is not None and not startup_marker.exists() and now-began >= 180:
                reason = 'pre_control_startup_timeout'; break
            time.sleep(min(.5, max(.001, deadline-now)))
    finally:
        if process.poll() is None:
            stop_owned(process)
    return reason


def run(output):
    config = read(output/'registration.json'); check_binding(output, config)
    if (output/'run_started.json').exists():
        raise FileExistsError('No duplicate study owner')
    stop_paths = [output/'STOP', Path(config['session_binding']['resource_root'])/'STOP',
        Path(config['session_binding']['session_root'])/'STOP']
    if any(path.exists() for path in stop_paths):
        raise InterruptedError('Study stop requested before launch')
    if time.time() >= config['deadline']:
        raise TimeoutError('Original registered session expired')
    with (output/'run_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), utc=time.time(),
            registration_sha256=sha256(output/'registration.json')), stream)
        stream.write('\n')
    ledger = output/'stages'; ledger.mkdir(); receipts = []; stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    handlers = {sig:signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    def execute(name, arguments, completion, required, *, startup=False):
        check_binding(output, config)
        if stopped or any(path.exists() for path in stop_paths):
            raise InterruptedError('Study stop requested')
        if time.time() >= config['deadline']:
            raise TimeoutError('Original registered session expired')
        command = [sys.executable, '-m', *map(str, arguments)]; began = time.time()
        atomic_json(output/'status.json', dict(stage=name, command=command, utc=began))
        with (ledger/(name+'.log')).open('x') as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONFAULTHANDLER='1'))
            reason = wait_child(worker, config['deadline'], stop_paths=stop_paths, stop_requested=lambda:stopped,
                startup_marker=completion.parent/'initialization.json' if startup else None)
        receipt = dict(stage=name, command=command, returncode=worker.returncode, stop_reason=reason,
            seconds=time.time()-began, completion=str(completion),
            completion_sha256=sha256(completion) if completion.exists() else None)
        atomic_json(ledger/(name+'.json'), receipt); receipts.append(receipt)
        if reason or worker.returncode or not completion.exists():
            raise RuntimeError('Study stage failed: '+name)
        proof = read(completion)
        require(proof.get(required) is True and not proof.get('disposable')
            and not proof.get('worker_error')
            and not any(proof.get(key) is False for key in ('complete', 'software_complete', 'target_reached')),
            'Incomplete stage receipt: '+name)
        print(json.dumps(receipt), flush=True)
    try:
        for arm, role in ORDER:
            root = output/arm
            execute(arm+'_'+role+'_frontend', ['tools.tight_fresh_train', 'train', '--root', root,
                '--role', role, '--epochs', config['frontend_epochs']], root/role/'training/finished.json', 'complete')
        for arm, role in ORDER:
            root = output/arm; rear = root/role/'backend'
            execute(arm+'_'+role+'_backend_prepare', ['tools.tight_fresh_backend', '--root', root, '--role', role,
                '--epochs', config['backend_epochs'], '--continue-rear'], rear/'prepared.json', 'inherited_student_parameters')
            execute(arm+'_'+role+'_backend_cache', ['tools.tight_finish', 'residual_cache', '--root', rear],
                rear/'cache_complete.json', 'complete')
            for stage, destination, required in [('residual-train', rear/role/'training/finished.json', 'target_reached'),
                                                  ('residual-offline', rear/role/'offline/result.json', 'complete')]:
                execute(arm+'_'+role+'_'+stage, ['visuotactile.anchor_continue', stage, '--root', rear/role,
                    '--job', 'image', '--output', destination.parent], destination, required)
        prepare_development(output, config)
        execute('development', ['tools.tight_fresh_rollouts', 'run', '--root', output,
            '--output', output/'development'], output/'development/result.json', 'software_complete', startup=True)
        atomic_json(output/'finished.json', dict(software_complete=True, stages=receipts, utc=time.time(),
            research_conclusion='Pending actual source/gradient/exposure and full-student behavior review; no automatic promotion'))
        atomic_json(output/'status.json', dict(stage='complete', utc=time.time()))
    except BaseException as error:
        atomic_json(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), stages=receipts, utc=time.time()))
        atomic_json(output/'status.json', dict(stage='failed', error=repr(error), utc=time.time()))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'run']); parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-root', type=Path, help='Existing live session/telemetry owner; source evidence remains immutable')
    parser.add_argument('--calibration', type=Path); parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--epochs', type=int, default=5); parser.add_argument('--rear-epochs', type=int, default=5)
    parser.add_argument('--count', type=int, default=4); parser.add_argument('--seed', type=int, default=28140000)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if any(value is None for value in (args.source, args.calibration, args.acceptance)):
            parser.error('Reviewed source, calibration and disposable CUDA acceptance are required')
        prepare(args.source, args.output, args.calibration, args.acceptance,
            args.epochs, args.rear_epochs, args.count, args.seed, session_root=args.session_root)
    else:
        run(args.output.resolve())


if __name__ == '__main__':
    main()
