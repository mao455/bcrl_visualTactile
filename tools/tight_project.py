"""One configuration-driven entrance; existing workers retain all research math.

A job describes explicit ordered stages, prepared artifacts and an existing finite
resource session. No dates, model promotion, data expansion or retries are inferred.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ENTRIES = {
    'frontend': ('train', 'tools.tight_fresh_train', ['train']),
    'backend_prepare': ('train', 'tools.tight_fresh_backend', []),
    'backend_cache': ('train', 'tools.tight_finish', ['residual_cache']),
    'backend_train': ('train', 'visuotactile.anchor_continue', ['residual-train']),
    'backend_offline': ('train', 'visuotactile.anchor_continue', ['residual-offline']),
    'aggregate': ('prepare', 'tools.tight_fresh_dagger', ['aggregate']),
    'visual_cache': ('prepare', 'tools.tight_fresh_dagger', ['cache']),
    'dagger_prepare': ('prepare', 'tools.tight_fresh_batch_run', ['prepare']),
    'dagger': ('dagger', 'tools.tight_fresh_batch_run', ['run']),
    'evaluate': ('evaluate', 'tools.tight_evaluate', []),
}

def read(path):
    return json.loads(Path(path).read_text())

def commands(job, operation):
    if job.get('version') != 'tight_project_job_v1' or job.get('operation') != operation:
        raise ValueError('Explicit version and matching operation required')
    stages = job.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('Register finite ordered stages')
    result = []
    for stage in stages:
        kind, module, prefix = ENTRIES[stage['entry']]
        if kind != operation:
            raise ValueError('Separate preparation, learning, collection and evaluation jobs')
        args = stage.get('args')
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError('Arguments must be literal strings, never shell commands')
        result.append([sys.executable, '-m', module, *prefix, *args])
    return result

def stop(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)

@contextmanager
def termination_signals():
    """Unwind owned children on service termination; restore caller handlers."""
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Owner received signal {signum}')
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for sig in previous:
            signal.signal(sig, terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute(job, operation):
    from visuotactile.training_resources import atomic_json
    import fcntl
    import shutil
    queue = commands(job, operation)
    root = Path(job['resource_root']).resolve()
    session = read(root / 'session.json')
    if session['deadline'] <= time.time():
        raise ValueError('Expired session; historical deadlines are never reset')
    if shutil.disk_usage(root).free < session.get('reserve_bytes', 12 * 2**30):
        raise ValueError('Resource session disk reserve not met')
    if Path.cwd().resolve() != PROJECT or PROJECT.name != 'bc+rl_RSS':
        raise ValueError('Run current work from the canonical bc+rl_RSS checkout')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT), str(PROJECT/'rl_games')])
    child = monitor = None
    with termination_signals(), (root/'project_owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = root/'project_job_started.json'
        if receipt.exists() or (root/'STOP_MONITOR').exists():
            raise ValueError('Completed or attempted identity must be preserved; no automatic retry')
        atomic_json(receipt, dict(job=job, commands=queue, pid=os.getpid(), utc=time.time(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        try:
            with (root/'project_monitor.log').open('x') as log:
                monitor = subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(root)],
                    cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            until = min(session['deadline'], time.time()+30)
            while not (root/'shared_resources.json').exists():
                if monitor.poll() is not None or time.time() >= until:
                    raise RuntimeError('Resource monitor did not start')
                time.sleep(.2)
            for index, command in enumerate(queue):
                with (root/f'project_stage_{index:02d}.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(root/'project_status.json',dict(stage=index,command=command,pid=child.pid,utc=time.time()))
                while child.poll() is None:
                    if monitor.poll() is not None or time.time() >= session['deadline'] or (root/'STOP').exists():
                        raise RuntimeError('Monitor/deadline/STOP ended this job')
                    time.sleep(1)
                if child.returncode:
                    raise RuntimeError(f'Stage {index} returned {child.returncode}; evidence retained')
            atomic_json(root/'project_complete.json',dict(stages=len(queue),utc=time.time()))
        except BaseException as exc:
            atomic_json(root/'project_failure.json',dict(error=repr(exc),utc=time.time()))
            raise
        finally:
            stop(child)
            (root/'STOP_MONITOR').touch()
            if monitor:
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop(monitor)
            atomic_json(root/'project_exit.json',dict(children_reaped=True,utc=time.time()))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['status','prepare','train','dagger','evaluate'])
    p.add_argument('--config',type=Path,default=PROJECT/'configs/tight_project.json')
    p.add_argument('--job',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    if a.operation == 'status':
        registry = read(a.config)
        print(json.dumps(dict(project=str(PROJECT),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),registry=registry),indent=2))
        return
    if a.job is None:
        p.error('--job is required; never infer a new experiment from historical scripts')
    job=read(a.job)
    if a.dry_run:
        print(json.dumps(commands(job,a.operation),indent=2))
    else:
        execute(job,a.operation)

if __name__ == '__main__':
    main()
