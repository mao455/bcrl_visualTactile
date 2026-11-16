"""Finite four-pair native evaluation; one coordinator owns the monitor and Kit.

No training, collection, inherited deadline, or automatic failed-job retry.
Completed jobs are immutable and skipped on resume; interrupted attempts remain.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

TREE = Path(__file__).resolve().parents[2]
RUNS = TREE / 'outputs/visuotactile/run'
ROOT = RUNS / 'tight_reliable_eval_20261116_01'
OLD = RUNS / 'tight_fulltask_20261116_01'
DIAG = RUNS / 'tight_stability_20261116_01'
SEEDS = [28500100, 28500101, 28500105, 28500106]


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    import torch
    ROOT.mkdir()
    session = read(OLD / 'session.json')
    now = time.time()
    session.update(started_utc=now, deadline=now+5400, training_deadline=now+5400,
                   offline_deadline=now+5400, cache_deadline=now+5400,
                   resource_root=str(ROOT.resolve()), owned_worker_roots=[str(ROOT.resolve())],
                   authorization='Finite existing full-policy evaluation after 8b946ca; no training or DAgger')
    session.pop('resource_clock_parent', None)
    atomic_json(ROOT / 'session.json', session)
    pairs = {}
    identities = {}
    for source, groups in [('candidates', ['A_long', 'B_short']), ('fallbacks', ['A11', 'B15'])]:
        for role, group in zip('AB', groups):
            pair = read(DIAG / source / 'plan.json')['pairs'][role]
            pairs[group] = pair
            identities[group] = {}
            for part in ('frontend', 'backend'):
                assert sha256(pair[part]) == pair[part+'_sha256']
                ck = torch.load(pair[part], map_location='cpu', mmap=True, weights_only=False)
                identities[group][part] = dict(path=pair[part], sha256=pair[part+'_sha256'],
                    epoch=ck.get('epoch'), updates=ck.get('updates'), supervision=ck.get('supervision'),
                    selected_file=Path(pair[part]).name)
                del ck
    # A occupies the first two lanes; all four models receive identical starts.
    pairs = {k: pairs[k] for k in ['A_long', 'A11', 'B_short', 'B15']}
    atomic_json(ROOT / 'model_identities.json', identities)
    base = read(OLD / 'baseline_evaluation/plan.json')
    jobs = []
    for seed in SEEDS:
        directory = ROOT / str(seed)
        directory.mkdir()
        plan = copy.deepcopy(base)
        template = next(c for c in base['cases'] if c['spec']['seed'] == seed)
        plan.update(pairs=pairs, cases=[], resource_root=str(ROOT.resolve()), utc=now)
        plan['batch'].update(lanes=4, seed=28600200)
        plan['evaluation_scope'] = dict(seeds=[seed], split='Existing development; not training or final test',
            protocol='Native termination for every legal start; no hold-time gate or grasp censor',
            reset='Private family native reset stream, pre-action actual observable pairing',
            limitation='Hidden simulator state and future lane random events are not bitwise paired')
        for group in pairs:
            case = copy.deepcopy(template)
            case.update(name=group+'_'+str(seed), role=group, video=False)
            case.pop('grasp_evaluation_ticks', None)
            plan['cases'].append(case)
        plan['source_sha256'] = {str(p.relative_to(TREE)): sha256(p)
            for folder in ['tools', 'visuotactile', 'scripts/tight_fulltask_20261116']
            for p in (TREE / folder).rglob('*.py')}
        atomic_json(directory / 'plan.json', plan)
        atomic_json(directory / 'session.json', session)
        jobs.append(dict(seed=seed, path=str(directory), plan_sha256=sha256(directory/'plan.json')))
    atomic_json(ROOT/'registration.json', dict(utc=now, jobs=jobs, identities=identities,
        total_registered=16, first_job_is_included_full_task_validation=True,
        original_software_interruption=str(OLD/'paired_sources'),
        rule='Four fixed jobs, serial Kit, no outcome-dependent replacements; completed receipts resume unchanged',
        training_decision='No resume: both latest DP independent validation worsened; preserve two corrections and all checkpoints'))


def terminate_child(child):
    """Signal only the still-unreaped subprocess group created by this owner."""
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def completed(directory):
    if not (directory/'result.json').exists():
        return False
    result = read(directory/'result.json')
    cases = read(directory/'plan.json')['cases']
    return (result.get('software_complete') is True and result['cases'] == len(cases)
            and all((directory/c['name']/'result.json').exists() for c in cases))


def run():
    # Lock is held until all owned children exit. No worker writes coordinator state.
    with (ROOT/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = read(ROOT/'registration.json')
        for job in registration['jobs']:
            assert sha256(Path(job['path'])/'plan.json') == job['plan_sha256']
        if (ROOT/'complete.json').exists():
            assert all(completed(Path(job['path'])) for job in registration['jobs'])
            assert read(ROOT/'owner_exit.json')['children_reaped']
            receipt = dict(utc=time.time(), stage='complete', launched_workers=0,
                skipped_completed_jobs=[job['seed'] for job in registration['jobs']],
                result_sha256={str(job['seed']):sha256(Path(job['path'])/'result.json') for job in registration['jobs']},
                original_complete=read(ROOT/'complete.json'), original_owner_exit=read(ROOT/'owner_exit.json'))
            atomic_json(ROOT/'resume_check.json', receipt)
            atomic_json(ROOT/'coordinator_status.json', receipt)
            return
        session = read(ROOT/'session.json')
        assert time.time() < session['deadline']
        assert not (ROOT/'STOP').exists(), 'STOP is retained; review before resuming'
        (ROOT/'STOP_MONITOR').unlink(missing_ok=True)
        monitor = child = None
        try:
            with (ROOT/'monitor.log').open('a') as log:
                monitor = subprocess.Popen([sys.executable, '-m', 'visuotactile.state_resources', '--root', str(ROOT)],
                    cwd=TREE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            until = time.time()+20
            while True:
                assert monitor.poll() is None and time.time() < until
                if (ROOT/'shared_resources.json').exists() and time.time()-read(ROOT/'shared_resources.json')['utc_epoch'] < 3:
                    break
                time.sleep(.25)
            for registered in registration['jobs']:
                directory = Path(registered['path'])
                assert sha256(directory/'plan.json') == registered['plan_sha256']
                if completed(directory):
                    continue
                # Do not overwrite an interrupted native execution or silently rerun it.
                assert not (directory/'run_started.json').exists(), 'Interrupted job needs explicit remaining-case registration'
                row = dict(seed=registered['seed'], started=time.time(), monitor_pid=monitor.pid,
                           output=str(directory), owner_pid=os.getpid())
                with (directory/'worker.log').open('x') as log:
                    child = subprocess.Popen([sys.executable, '-m', __spec__.name, 'worker',
                        '--root', str(ROOT), '--output', str(directory)], cwd=TREE, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                row['pid'] = child.pid
                while child.poll() is None:
                    row['heartbeat'] = time.time()
                    atomic_json(directory/'worker_status.json', row)
                    atomic_json(ROOT/'coordinator_status.json', dict(**row, stage='evaluation'))
                    if monitor.poll() is not None:
                        raise RuntimeError('Owned monitor exited while a job was active; preserve partial result')
                    if time.time() >= session['deadline']-20 or (ROOT/'STOP').exists():
                        raise TimeoutError('Finite session deadline or STOP')
                    # Long startup/progress silence gets a saved traceback, then bounded stop.
                    progress = list(directory.glob('*progress*.json')) + list(directory.glob('*/result.json'))
                    latest = max([row['started']] + [p.stat().st_mtime for p in progress])
                    if time.time()-latest > 300:
                        if (directory/'runtime_diagnostics_ready.json').exists():
                            os.kill(child.pid, signal.SIGUSR2)
                        raise TimeoutError('No recorded progress for 300 seconds')
                    time.sleep(2)
                row.update(ended=time.time(), exit_code=child.returncode)
                atomic_json(directory/'worker_status.json', row)
                assert child.returncode == 0 and completed(directory), 'Worker failed or incomplete; no task-failure substitution'
                assert read(directory/'paired_initial_check.json')['passed']
                atomic_json(directory/'owner_accepted.json', dict(utc=time.time(), plan_sha256=registered['plan_sha256'],
                    result_sha256=sha256(directory/'result.json'), monitor_alive=monitor.poll() is None))
                child = None
            atomic_json(ROOT/'complete.json', dict(complete=True, utc=time.time(), jobs=len(SEEDS)))
            atomic_json(ROOT/'coordinator_status.json', dict(stage='complete', utc=time.time(), jobs=len(SEEDS)))
        except BaseException:
            atomic_json(ROOT/'failure.json', dict(utc=time.time(), error=traceback.format_exc()))
            atomic_json(ROOT/'coordinator_status.json', dict(stage='failed', utc=time.time(), error=traceback.format_exc()))
            raise
        finally:
            # Monitor outlives every simulation; it cannot stop other sessions.
            terminate_child(child)
            (ROOT/'STOP_MONITOR').touch()
            if monitor is not None:
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    terminate_child(monitor)
            atomic_json(ROOT/'owner_exit.json', dict(utc=time.time(), owner_pid=os.getpid(),
                monitor_pid=None if monitor is None else monitor.pid,
                monitor_returncode=None if monitor is None else monitor.poll(), children_reaped=True))


def worker():
    from scripts.tight_fulltask_20261116.paired_eval import install
    from tools.tight_fresh_batch_run import run as evaluate
    output = Path(sys.argv[sys.argv.index('--output')+1])
    install(output)
    evaluate(output)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'worker': worker}[sys.argv[1]]()
