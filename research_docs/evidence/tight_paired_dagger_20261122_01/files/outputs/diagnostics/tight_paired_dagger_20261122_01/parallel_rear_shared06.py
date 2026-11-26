"""Run-local capacity adapter; external presence is observed without a veto.

Native trainers, serial DPs/caches/offline work, own data, epoch60, RNG, resource
leases, and child cleanup are delegated unchanged to the frozen rear proposal.
Importing this module performs no writes, launches, signals, or GPU queries.
"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time

from outputs.diagnostics.tight_paired_dagger_20261122_01 import parallel_rear_proposal as base

HERE = Path(__file__).resolve().parent
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.parallel_rear_shared06'
ROOT, CONFIG, PROJECT = base.ROOT, base.CONFIG, base.PROJECT
JOB = HERE/'proposed_main_train_shared06.json'
REGISTRATION = HERE/'parallel_rear_shared06_registration.json'
RUNTIME = HERE/'parallel_rear_shared06_runtime'
BASE_WRAPPER_SHA = '5d314b23d935cf82672e3a1a3ba2d53d68aebe12c308fb224b47aafa7b4c944b'
BASE_REGISTRATION_SHA = 'eb56d9d2dc5f8e15c2dc5185f480e40aa9761f421970e84f298be28162cb0683'


def proposed_job(root=ROOT):
    job = base.proposed_job(root)
    job['stages'][-1]['args'][-1] = str(REGISTRATION)
    return job


def expected_registration():
    """Pin the reviewed proposal and derive only this adapter's new identity."""
    if (base.digest(base.__file__) != BASE_WRAPPER_SHA
            or base.digest(base.REGISTRATION) != BASE_REGISTRATION_SHA):
        raise ValueError('Frozen rear proposal or registration changed')
    old = base.read(base.REGISTRATION)
    for path, expected in old['source_sha256'].items():
        if base.digest(path) != expected:
            raise ValueError('Frozen rear source changed: '+path)
    value = deepcopy(old)
    value.update(version='paired_rear_shared_capacity06_v1', job=str(JOB),
        wrapper_sha256=base.digest(__file__),
        base_wrapper_sha256=BASE_WRAPPER_SHA,
        base_registration_sha256=BASE_REGISTRATION_SHA,
        external_presence_is_veto=False,
        cache_and_offline_vram_per_worker_mib=1024,
        semantics=old['semantics']+' External CPU/GPU processes remain recorded in original '
                  'telemetry. Only their presence veto is removed; telemetry errors, actual '
                  'remaining capacity, source checks, native leases, and finite cleanup remain.')
    value['source_sha256'][str(Path(base.__file__).resolve())] = BASE_WRAPPER_SHA
    value['source_sha256'][str(base.REGISTRATION)] = BASE_REGISTRATION_SHA
    return value


def prepare():
    """Explicitly write two new registration artifacts; never overwrite a file."""
    registration = expected_registration()
    if JOB.exists() or REGISTRATION.exists():
        raise ValueError('Preserve existing shared06 job/registration')
    base.write_new(JOB, proposed_job())
    base.write_new(REGISTRATION, registration)


def admission(row, registration, now, workers):
    """Use the original row and every original capacity/error check unchanged."""
    if type(workers) is not int or workers not in (1, 2):
        return dict(allowed=False, workers=workers, reasons=['invalid_worker_count'])
    result = base.admission(row, registration, now, workers)
    reasons = [reason for reason in result['reasons'] if reason != 'external_gpu_present']
    return dict(result, allowed=not reasons, reasons=reasons,
                external_gpu_processes=row.get('external_gpu_processes'),
                external_presence_is_veto=False)


def readiness(row, scan, registration, now):
    """Fresh real telemetry and remaining reserves; retain all external records."""
    reasons = []
    try:
        if (not isinstance(scan, dict) or not isinstance(scan.get('processes'), list)
                or not isinstance(scan.get('errors'), list)):
            reasons.append('malformed_external_scan')
        elif scan['errors']:
            reasons.append('external_scan_errors')
        age = now-float(row['utc_epoch'])
        if not math.isfinite(age) or not -2 <= age < registration['max_telemetry_age_s']:
            reasons.append('stale_telemetry')
        for key in ('gpu_processes', 'external_gpu_processes', 'process_telemetry_errors'):
            if not isinstance(row.get(key), list):
                reasons.append('malformed_'+key)
        if row.get('process_telemetry_errors'):
            reasons.append('process_telemetry_errors')
        for key, minimum in (('gpu_free_mib', registration['min_gpu_free_mib']),
                             ('ram_available_bytes', registration['min_ram_bytes'])):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < minimum):
                reasons.append('remaining_'+key)
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append('invalid_telemetry_or_registration')
    return dict(allowed=not reasons, reasons=reasons, external_scan=scan,
                external_gpu_processes=row.get('external_gpu_processes'),
                external_presence_is_veto=False)


def validate(root, registration_path):
    root = Path(root).resolve()
    if Path(registration_path).resolve() != REGISTRATION:
        raise ValueError('Exact shared06 registration path required')
    registration = base.read(registration_path)
    if registration != expected_registration():
        raise ValueError('Immutable shared06 registration changed')
    if root != Path(registration['root']).resolve():
        raise ValueError('Exact original learning root required')
    if base.read(JOB) != proposed_job(root.parent):
        raise ValueError('Exact shared06 job changed')
    if base.read(registration['serial_job']) != registration['expected_serial_job']:
        raise ValueError('Original serial fallback job changed')
    if (root/'paired_episode_io.json').exists():
        raise ValueError('Distinct own-data pools cannot share an I/O rendezvous')
    for role in 'AB':
        if (root/role/'backend').exists():
            raise ValueError('Preserve any previous backend attempt')
        if base.read(root/role/'training/deployment.json')['epoch'] != 60:
            raise ValueError('Both final DPs must complete first')
    return registration


def run(root, registration_path):
    registration = validate(root, registration_path)
    root = Path(root).resolve()
    affinity = set(registration['cpu_affinity'])
    if not affinity <= os.sched_getaffinity(0):
        raise ValueError('Registered CPU8-15 not available')
    os.sched_setaffinity(0, affinity)
    from outputs.diagnostics.tight_paired_dagger_20261122_01.resource_watch_v4 import external_test_processes
    owner = Path(registration['owner'])
    session = base.read(root/'session.json')
    deadline = min(float(session['deadline']), float(registration['deadline']))
    stop_paths = [root.parent/'STOP', root/'STOP', owner/'STOP', root.parent/'STOP_CYCLE']
    if (not math.isfinite(deadline) or time.time() >= deadline
            or any(path.exists() for path in stop_paths)):
        raise TimeoutError('Original deadline or owned STOP before rear preparation')
    RUNTIME.mkdir(exist_ok=False)
    # Each delayed child launch gets its own current peak-plus-reserve check.
    # Parent cache/offline peaks were 746/444 MiB; 1024 MiB, then the original
    # 1.25 safety factor, covers their CUDA residence. Keep the original 6GiB
    # RAM worker bound, raised from actual nominal-cache bytes once available.
    launch_registration = dict(registration, vram_per_worker_mib=
                               registration['cache_and_offline_vram_per_worker_mib'])

    def row():
        return base.read(owner/'shared_resources.json')

    def ready():
        try:
            telemetry, scan = row(), external_test_processes()
            decision = readiness(telemetry, scan, registration, time.time())
            capacity = admission(telemetry, launch_registration, time.time(), 1)
            decision.update(allowed=decision['allowed'] and capacity['allowed'],
                            launch_capacity=capacity, launch_registration=launch_registration,
                            raw_monitor_row=telemetry)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            decision = dict(allowed=False, reasons=[repr(exc)])
        with (RUNTIME/'readiness.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(dict(utc=time.time(), decision=decision))+'\n')
        return decision['allowed']

    def await_ready():
        while not ready():
            if time.time() >= deadline or any(path.exists() for path in stop_paths):
                raise TimeoutError('Finite wait before rear admission')
            time.sleep(.5)

    with base.termination_signals():
        await_ready()
        commands = {role:base.stage_commands(root, role) for role in 'AB'}
        preparation = RUNTIME/'serial_caches'
        preparation.mkdir()
        base.run_roles({role:commands[role][:2] for role in 'AB'}, False, deadline,
                       stop_paths, ready, preparation)
        sizes = {role:sum(path.stat().st_size for path in
                 (root/role/'backend'/role/'nominal').glob('*.pt')) for role in 'AB'}
        actual = dict(registration)
        actual['ram_per_worker_bytes'] = max(registration['ram_per_worker_bytes'],
                                             2*2**30+3*max(sizes.values()))
        launch_registration = actual
        await_ready()
        snapshot = row()
        pair = admission(snapshot, actual, time.time(), 2)
        single = admission(snapshot, actual, time.time(), 1)
        if not pair['allowed'] and not single['allowed']:
            raise RuntimeError('Insufficient actual remaining headroom for serial rear; no trainer launched')
        decision = dict(parallel=pair['allowed'], pair=pair, single=single, utc=time.time(),
            actual_nominal_cache_bytes=sizes, admission_registration=actual,
            registration_sha256=base.digest(registration_path),
            cpu_affinity=sorted(os.sched_getaffinity(0)), raw_monitor_row=snapshot)
        base.write_new(RUNTIME/'admission.json', decision)
        training = RUNTIME/'rear_training'
        training.mkdir()
        events = base.run_roles({role:[commands[role][2]] for role in 'AB'}, decision['parallel'],
                               deadline, stop_paths, ready, training)
        offline = RUNTIME/'serial_offline'
        offline.mkdir()
        launch_registration = dict(actual, vram_per_worker_mib=
                                   registration['cache_and_offline_vram_per_worker_mib'])
        base.run_roles({role:[commands[role][3]] for role in 'AB'}, False,
                       deadline, stop_paths, ready, offline)
        base.write_new(RUNTIME/'complete.json', dict(utc=time.time(), events=len(events),
            parallel=decision['parallel'], caches_serial=True, offline_serial=True))


def project():
    from tools import tight_project
    tight_project.ENTRIES.update(paired_frontend=('train', 'tools.tight_paired_learning', ['frontend']),
        paired_rear_parallel=('train', MODULE, ['rear-pair']))
    sys.argv.pop(1)
    tight_project.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'project':
        return project()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'rear-pair'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        run(args.root, args.registration)


if __name__ == '__main__':
    main()
