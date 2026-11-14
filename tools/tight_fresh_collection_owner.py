"""Supervise one already registered causal-stage collection in its frozen tree."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from tools.tight_fresh_eval_recovery import stop_owned, wait_owned
from tools.tight_fresh_stage_takeover import CONTRACT, CONTACT_HEIGHT_CONTRACT, validate_plan
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def check_complete(output, plan, worker, started, deadline, plan_sha256, session_sha256):
    """Completion means every registered opportunity, including unsuccessful ones."""
    validate_plan(plan)
    audit_name = {CONTRACT: 'stable_stage_audit.json',
                  CONTACT_HEIGHT_CONTRACT: 'contact_height_stage_audit.json'}[plan['stage_takeover_contract']]
    if sha256(output / 'plan.json') != plan_sha256 or sha256(output / 'session.json') != session_sha256:
        raise ValueError('Registration changed during collection')
    launch = read(output / 'run_started.json')
    initialization = read(output / 'initialization.json')
    result = read(output / 'result.json')
    rows = read(output / 'results.json')
    if (launch.get('pid') != worker.pid or launch.get('plan_sha256') != plan_sha256
            or not started <= launch['utc'] <= deadline or initialization['seconds'] < 0):
        raise ValueError('Worker startup identity differs from the owned process')
    if (result.get('software_complete') is not True or result.get('worker_error') is not None
            or not started <= result['utc'] <= deadline
            or result.get('cases') != len(plan['cases']) or len(rows) != len(plan['cases'])
            or {row['case'] for row in rows} != {case['name'] for case in plan['cases']}):
        raise ValueError('Registered opportunities did not all complete inside the deadline')
    for case in plan['cases']:
        row = next(row for row in rows if row['case'] == case['name'])
        audit = read(output / case['name'] / audit_name)
        if (row['role'] != case['role'] or row['result'].get('software_complete') is not True
                or audit['source_plan_sha256'] != plan_sha256 or not audit['no_expert_fallback']):
            raise ValueError('Case completion or causal-stage audit differs from its registration')
    return result


def run(output, worker_tree):
    output = output.resolve()
    worker_tree = worker_tree.resolve()
    plan = read(output / 'plan.json')
    session = read(output / 'session.json')
    validate_plan(plan)
    if plan.get('kind') != 'takeover' or plan.get('test_used') is not False:
        raise ValueError('Registered training-only takeover collection required')
    if any(Path(case['name']).name != case['name'] for case in plan['cases']):
        raise ValueError('Cases require local directory names')
    previous = ('owner_started.json', 'owner_worker_started.json', 'owner_finished.json', 'worker.log',
                'run_started.json', 'initialization.json', 'result.json', 'results.json', 'correction_entries.json')
    if any((output / name).exists() for name in previous) or any((output / c['name']).exists() for c in plan['cases']):
        raise FileExistsError('Preserve an existing collection attempt')
    if time.time() >= session['deadline']:
        raise TimeoutError('Registered collection window has expired')
    session_sha256 = sha256(output / 'session.json')
    if (session_sha256 != plan['source_session_sha256']
            or sha256(Path(plan['resource_root']) / 'session.json') != session_sha256):
        raise ValueError('Registered session or resource deadline changed')
    actual = {str(p.relative_to(worker_tree)): p for folder in ('tools', 'visuotactile')
              for p in (worker_tree / folder).rglob('*.py')}
    if not plan['source_sha256'] or set(actual) != set(plan['source_sha256']):
        raise ValueError('Frozen worker source population changed')
    for path, digest in plan['source_sha256'].items():
        if sha256(actual[path]) != digest:
            raise ValueError('Frozen worker source changed: ' + path)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worker_tree, text=True).strip() != plan['source_commit']:
        raise ValueError('Frozen worker commit changed')
    if time.time() >= session['deadline']:
        raise TimeoutError('Registered collection window expired during source verification')
    command = [str(worker_tree / '.venv_isaacsim/bin/python'), '-m',
               'tools.tight_fresh_stage_takeover', 'run', '--output', str(output)]
    started = time.time()
    plan_sha256 = sha256(output / 'plan.json')
    registration = dict(pid=os.getpid(), started=started, command=command, cwd=str(worker_tree),
        source_sha256=sha256(__file__), plan_sha256=plan_sha256, session_sha256=session_sha256,
        deadline=session['deadline'], startup_seconds=180)
    # Exclusive creation also prevents two owners from racing past the preflight.
    with (output / 'owner_started.json').open('x') as stream:
        json.dump(registration, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    worker = None
    reason = None
    failure = None
    complete = False
    previous_handlers = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        with (output / 'worker.log').open('x') as stream:
            worker = subprocess.Popen(command, cwd=worker_tree, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONFAULTHANDLER='1'))
            atomic_json(output / 'owner_worker_started.json', dict(worker_pid=worker.pid,
                created_process_group=worker.pid, command=command, owner_pid=os.getpid(), utc=time.time()))
            reason = wait_owned(worker, session['deadline'], startup_marker=output / 'initialization.json',
                                startup_seconds=180)
        if worker.returncode == 0 and reason is None:
            check_complete(output, plan, worker, started, session['deadline'], plan_sha256, session_sha256)
            complete = True
    except BaseException as error:
        failure = error
        reason = reason or ('owner_interrupted' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'owner_error')
    finally:
        if worker is not None:
            stop_owned(worker)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    result_path = output / 'result.json'
    receipt = dict(software_complete=complete, worker_pid=worker.pid if worker else None,
        returncode=worker.returncode if worker else None, stop_reason=reason or ('complete' if complete else 'worker_failure'),
        seconds=time.time() - started, error=None if failure is None else repr(failure),
        result_sha256=sha256(result_path) if result_path.exists() else None, ended=time.time())
    atomic_json(output / 'owner_finished.json', receipt)
    print(json.dumps(receipt), flush=True)
    if failure is not None:
        raise failure
    if not complete:
        raise RuntimeError('Collection incomplete; preserve evidence for Agent review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-tree', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.worker_tree)
