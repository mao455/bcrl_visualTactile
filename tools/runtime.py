"""Shared durable I/O and owned-process management for the current queues."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
CODE=Path(__file__).resolve().parents[1]
RUN=["bash","visuotactile/run_cpu.sh"]

def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def supervised_owner(function):
    """Turn supervisor SIGTERM into a normal exception so owned cleanup runs."""
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous=signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            raise InterruptedError('Owner received SIGTERM; preserve evidence and stop owned workers')
        signal.signal(signal.SIGTERM,terminate)
        try:return function(*args, **kwargs)
        finally:signal.signal(signal.SIGTERM,previous)
    return wrapped


def kill_owned(worker):
    if worker.poll() is None:
        os.killpg(worker.pid,signal.SIGTERM)
        try:worker.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=15)
def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (p/'status').read_text().splitlines() if ':' in line)
        watches = instances = 0
        missing=[]
        try: fds = list((p/'fd').iterdir())
        except PermissionError:
            fds=[];missing.append('fd_directory_permission_denied')
        for fd in fds:
            try:
                if 'inotify' in os.readlink(fd):
                    instances += 1
                    watches += sum(s.startswith('inotify wd:') for s in (p/'fdinfo'/fd.name).read_text().splitlines())
            except (FileNotFoundError, ProcessLookupError): pass
            except PermissionError: missing.append('fdinfo_permission_denied')
        return dict(pid=pid, ppid=int(fields[1]), process_group=int(fields[2]), start_ticks=int(fields[19]),
            state=fields[0], rss_kib=int(status.get('VmRSS', '0 kB').split()[0]),
            threads=int(status['Threads']), fds=None if missing else len(fds),
            watches=None if missing else watches, inotify_instances=None if missing else instances,
            telemetry_missing=missing)
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid):
    # Kernel task child lists avoid scanning/signalling unrelated user processes.
    found = set(); pending = [pid]
    while pending:
        parent = pending.pop()
        for task in Path('/proc', str(parent), 'task').glob('*'):
            try: children = [int(x) for x in (task/'children').read_text().split()]
            except (FileNotFoundError, ProcessLookupError, PermissionError): continue
            for child in children:
                if child not in found: found.add(child); pending.append(child)
    return found


def gpu_pids():
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
    return {int(line.strip()) for line in lines if line.strip().isdigit()}
def classify(output, code, timeout, text, valid):
    if valid:
        return 'normal_sdk_return' if code==0 and (output/'closed.json').exists() else 'sdk_fast_exit_zero' if code==0 else 'shutdown_fault_after_complete_runtime'
    if 'FileThread' in text and code in (-6,-11):return 'startup_FileThread_before_scene' if not (output/'environment.json').exists() else 'runtime_FileThread'
    if timeout:return timeout
    return 'worker_error' if (output/'worker_error.json').exists() else 'startup_or_runtime_software_missing'
