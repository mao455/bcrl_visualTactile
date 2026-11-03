"""Classify a proven pre-scene native abort without editing frozen H24 workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from tools import h24_comparison_run as owner
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

RECOVERY = 'runtime_recovery_01'
ORIGINAL_CLASSIFIER = owner.startup_failure_evidence


def native_startup_evidence(root, name):
    evidence = ORIGINAL_CLASSIFIER(root, name)
    if evidence is not None:
        return evidence
    output = root/name
    log = root/(name+'.log')
    exit_path = root/(name+'_process_exit.json')
    if not output.is_dir() or not log.exists() or not exit_path.exists():
        return None
    exited = owner.read(exit_path)
    if exited.get('code') != -6 or not exited.get('reaped'):
        return None
    allowed = {'runtime_controls.json', 'runtime_before.json', 'runtime_ready.json',
        'lifecycle.json', 'app_ready.json'}
    files = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if not {'runtime_controls.json', 'lifecycle.json'} <= files or not files <= allowed:
        return None
    if owner.read(output/'lifecycle.json').get('state') != 'launching':
        return None
    text = log.read_text(errors='replace')
    if any(s in text for s in ('[INFO]: Completed setting up the environment', 'Traceback (most recent call last)')):
        return None
    signatures = ('[Fatal]', 'libc.so.6!abort', 'libomniclient.so!omniClientFreeContent',
        'omniclient::provider_file::FileThread')
    if not all(s in text for s in signatures):
        return None
    return dict(classification='pre-scene FileThread/omniClientFreeContent SIGABRT; exception text absent',
        confirmed_sigabrt=True, native_stack_available=True, task_started=False,
        root_cause_proven=False, retry_authorized=False, log_sha256=sha256(log))


def register(root):
    folder = root/RECOVERY
    if folder.exists():
        raise ValueError('Recovery already registered; never reset it')
    owner.verify_frozen(root)
    session = owner.read(root/'session.json')
    if time.time() >= session['deadline']:
        raise TimeoutError('Original total deadline passed')
    stages = ['sample_3000_loop_worker_01', 'sample_3000_loop_worker_02']
    classified = {name: native_startup_evidence(root, name) for name in stages}
    if not all(classified.values()):
        raise ValueError('Both failed attempts must be classified before recovery')
    for name in stages:
        pid = owner.read(root/(name+'_process_exit.json'))['pid']
        if Path('/proc', str(pid)).exists():
            raise RuntimeError('Failed worker PID still exists')
    folder.mkdir()
    archive = folder/'prior_state';archive.mkdir()
    for name in ('blocked.json', 'status.json', 'owner_exit.json', 'monitor_exit.json',
            'sample_3000_loop/result.json', 'sample_3000_loop/progress.json'):
        path = root/name
        if path.exists():
            target = archive/name;target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    failed = {}
    for name in stages:
        paths = [p for p in (root/name).rglob('*') if p.is_file()]
        paths += [root/(name+s) for s in ('.log', '_launch.json', '_job.json', '_process_exit.json', '_exit.json')]
        failed.update({str(p.relative_to(root)): sha256(p) for p in paths if p.exists()})
    protected = ['session.json', 'run_plan.json', 'setup_complete.json', 'fairness.json',
        'sample_resume/h24_train/configuration.json', 'sample_resume/h24_train/milestone_3000.pt',
        'h24_epsilon_train/configuration.json', 'h24_epsilon_train/milestone_3000.pt']
    service = Path('tools/h24-comparison-20261103-01-recovery.service')
    registration = dict(utc=time.time(), launcher_sha256=sha256(Path(__file__)),
        service_sha256=sha256(service), source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        preserved_sha256={name:sha256(root/name) for name in protected}, failed_sha256=failed,
        archive_sha256={str(p.relative_to(folder)):sha256(p) for p in archive.rglob('*') if p.is_file()},
        classifications=classified, original_deadline=session['deadline'], original_training_deadline=session['training_deadline'],
        behavior='Classify pre-scene abort; retain original two-consecutive-fault cohort cap. No failed attempt retried.',
        frozen_sources_changed=False, worker_changes=False, new_model_updates=0)
    atomic_json(folder/'registration.json', registration)


def verify(root):
    folder = root/RECOVERY
    registration = owner.read(folder/'registration.json')
    if sha256(Path(__file__)) != registration['launcher_sha256']:
        raise ValueError('Recovery source drift')
    if sha256(Path('tools/h24-comparison-20261103-01-recovery.service')) != registration['service_sha256']:
        raise ValueError('Recovery service drift')
    owner.verify_frozen(root)
    for field in ('preserved_sha256', 'failed_sha256'):
        for name, digest in registration[field].items():
            if sha256(root/name) != digest:
                raise ValueError('Recovery protected evidence changed: '+name)
    for name, digest in registration['archive_sha256'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Archived failure evidence changed')
    return registration


def reuse_fairness(root):
    """The fixed 3000 checkpoints were already checked; later logs may exceed 3000."""
    evidence = owner.read(root/'fairness.json')
    if evidence.get('scope') != 'actual paired3000 training verified' or not all(evidence['actual_3000'].values()):
        raise ValueError('Verified paired3000 fairness evidence required')
    for group, path in [('sample', 'sample_resume/h24_train'), ('epsilon', 'h24_epsilon_train')]:
        if sha256(root/path/'milestone_3000.pt') != evidence[group+'_checkpoint_sha256']:
            raise ValueError('Paired3000 checkpoint changed')


def run(root):
    registration = verify(root)
    # Clear the stale active error only after preserving/checking its exact bytes.
    blocked = root/'blocked.json'
    old = root/RECOVERY/'prior_state/blocked.json'
    if blocked.exists():
        if not old.exists() or sha256(blocked) != sha256(old):
            raise RuntimeError('A new blocking condition needs review')
        blocked.rename(root/RECOVERY/'resolved_blocked.json')
    owner.startup_failure_evidence = native_startup_evidence
    owner.fairness3000 = reuse_fairness
    atomic_json(root/RECOVERY/'startup.json', dict(pid=os.getpid(), utc=time.time(),
        registration_sha256=sha256(root/RECOVERY/'registration.json'), sources_verified=True,
        deadline=registration['original_deadline']))
    owner.monitored(root, lambda: owner.run(root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('register', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args();root = args.root.resolve()
    def interrupt(signum, frame):
        raise InterruptedError('Recovery owner signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupt);signal.signal(signal.SIGINT, interrupt)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register(root) if args.command == 'register' else run(root)
