"""Unactivated run-local proposal: serial DP/cache, parallel rear training, serial offline.

No GPU imports, processes, schedule writes or experiment mutations on import.
The existing project owner owns the only monitor. Existing trainers are invoked
unchanged as independent process groups, with their own output roots and RNG.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
MODULE = 'outputs.diagnostics.tight_paired_dagger_20261122_01.parallel_rear_proposal'
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
CONFIG = PROJECT/'configs/experiments/paired_dagger_20261122'
JOB = HERE/'proposed_main_train_parallel.json'
REGISTRATION = HERE/'parallel_rear_registration.json'
SOURCES = ('tools/tight_paired_learning.py', 'tools/tight_fresh_backend.py',
           'tools/tight_finish.py', 'tools/tight_fresh_train.py',
           'visuotactile/anchor_continue.py', 'visuotactile/state_resources.py',
           'visuotactile/latest_takeover_sampling.py', 'visuotactile/paired_training_decks.py',
           'visuotactile/trajectory_training.py', 'visuotactile/paired_curve_diagnostics.py',
           'tools/tight_project.py',
           'outputs/diagnostics/tight_paired_dagger_20261122_01/resource_watch_v4.py',
           'outputs/diagnostics/tight_paired_dagger_20261122_01/resource_watch_parallel_proposal.py')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stage_commands(root, role):
    root = Path(root)
    rear = root/role/'backend'
    child = rear/role
    return [
        [sys.executable, '-m', 'tools.tight_paired_learning', 'rear-prepare', '--root', str(root), '--role', role],
        [sys.executable, '-m', 'tools.tight_finish', 'residual_cache', '--root', str(rear)],
        [sys.executable, '-m', 'tools.tight_paired_learning', 'rear-train', '--root', str(child)],
        [sys.executable, '-m', 'visuotactile.anchor_continue', 'residual-offline', '--root', str(child),
         '--job', 'image', '--output', str(child/'offline')],
    ]


def serial_job(root=ROOT):
    learning = Path(root)/'learning'
    stages = [dict(entry='paired_frontend', args=['--root', str(learning), '--role', role]) for role in 'AB']
    for role in 'AB':
        rear = learning/role/'backend'; child = rear/role
        stages.extend([
            dict(entry='paired_rear_prepare', args=['--root', str(learning), '--role', role]),
            dict(entry='backend_cache', args=['--root', str(rear)]),
            dict(entry='paired_rear_train', args=['--root', str(child)]),
            dict(entry='backend_offline', args=['--root', str(child), '--job', 'image', '--output', str(child/'offline')]),
        ])
    return dict(version='tight_project_job_v1', operation='train', resource_root=str(Path(root)/'owners/main_train'), stages=stages)


def proposed_job(root=ROOT):
    job = serial_job(root)
    job['stages'] = job['stages'][:2]+[dict(entry='paired_rear_parallel', args=[
        '--root', str(Path(root)/'learning'), '--registration', str(REGISTRATION)])]
    return job


def prepare():
    """Write only new AUDIT proposal files; do not alter the active job/driver."""
    session = read(ROOT/'session.json')
    registration = dict(version='paired_rear_parallel_proposal_v1', activated=False,
        root=str(ROOT/'learning'), owner=str(ROOT/'owners/main_train'), job=str(JOB),
        serial_job=str(CONFIG/'main_train.json'), expected_serial_job=serial_job(),
        source_sha256={str(PROJECT/p): digest(PROJECT/p) for p in SOURCES},
        wrapper_sha256=digest(__file__), deadline=session['deadline'],
        cpu_affinity=list(range(8,16)), threads=2, epochs=60, deploy_epoch=60,
        ram_per_worker_bytes=6*2**30, vram_per_worker_mib=1536,
        min_ram_bytes=max(10,session['resources']['min_ram_gib'])*2**30,
        min_gpu_free_mib=max(4096,session['resources']['min_gpu_free_mib']),
        max_telemetry_age_s=3., safety_factor=1.25,
        semantics='Same complete trainers, schedules, targets, source, RNG seeds, ledgers and epoch60. '
                  'Independent processes; scheduling only; no bitwise or speedup claim. '
                  'DP serial because measured combined peaks exceed card. '
                  'Own nominal caches run serial before either rear trainer; offline runs serial after both. ' 'Rear pair admitted before either starts; otherwise both trainers run serial. '
                  'Any launched failure is terminal, never a training retry.',
        limitations=['Parent process_rss telemetry is monitor RSS, not trainer RSS; 6GiB/worker is a conservative admission bound, not a measured parent peak.',
            'Current v4 exact owner argv does not recognize this caller; use proposed watcher adapter after old writer exits.',
            'Cooperative pause retains resident memory and cannot preempt submitted CUDA; no zero-interference guarantee.',
            'Keep actual presealed_guard --mode main before evaluation.'])
    write_new(JOB, proposed_job())
    write_new(REGISTRATION, registration)
    print(json.dumps(dict(job=str(JOB), registration=str(REGISTRATION), activated=False)))


def admission(row, registration, now, workers):
    reasons = []
    try:
        numeric = ('ram_available_bytes', 'gpu_free_mib')
        if any(isinstance(row.get(k),bool) or not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k] < 0 for k in numeric):
            raise ValueError('Finite nonnegative resource quantities required')
        if any(not math.isfinite(float(registration[k])) or float(registration[k]) <= 0 for k in ('min_ram_bytes','ram_per_worker_bytes','vram_per_worker_mib','min_gpu_free_mib','safety_factor','max_telemetry_age_s')):
            raise ValueError('Finite positive admission bounds required')
        age = now-float(row['utc_epoch'])
        if not math.isfinite(age) or not -2 <= age < registration['max_telemetry_age_s']:
            reasons.append('stale_telemetry')
        for key in ('gpu_processes','external_gpu_processes','process_telemetry_errors'):
            if not isinstance(row.get(key), list): reasons.append('malformed_'+key)
        if row.get('process_telemetry_errors'): reasons.append('process_telemetry_errors')
        if row.get('external_gpu_processes'): reasons.append('external_gpu_present')
        required_ram = registration['min_ram_bytes']+math.ceil(workers*registration['ram_per_worker_bytes']*registration['safety_factor'])
        required_gpu = registration['min_gpu_free_mib']+math.ceil(workers*registration['vram_per_worker_mib']*registration['safety_factor'])
        if row['ram_available_bytes'] < required_ram: reasons.append('combined_ram_headroom')
        if row['gpu_free_mib'] is None or row['gpu_free_mib'] < required_gpu: reasons.append('combined_vram_headroom')
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append('invalid_telemetry_or_registration')
    return dict(allowed=not reasons, workers=workers, reasons=reasons)


def identity(pid):
    try:
        parts=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return int(parts[19])
    except (OSError, ValueError, IndexError): return None


def stop_all(children, *, grace_s=5., kill_s=3.):
    """Signal only owned live process groups; common deadline fits outer 15s grace."""
    active = [p for p in children if p.poll() is None]
    identities = {p.pid: identity(p.pid) for p in active}
    def send(p, sig):
        if p.poll() is None and identities[p.pid] is not None and identity(p.pid)==identities[p.pid]:
            try: os.killpg(p.pid,sig)
            except ProcessLookupError: pass
    for p in active: send(p,signal.SIGTERM)
    until=time.monotonic()+grace_s
    while any(p.poll() is None for p in active) and time.monotonic()<until: time.sleep(.05)
    for p in active: send(p,signal.SIGKILL)
    until=time.monotonic()+kill_s
    for p in active:
        if p.poll() is None:
            p.wait(timeout=max(.01,until-time.monotonic()))


@contextmanager
def termination_signals():
    def terminate(signum, frame): raise KeyboardInterrupt(signum)
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
    try:
        for s in previous: signal.signal(s,terminate)
        yield
    finally:
        for s,h in previous.items(): signal.signal(s,h)


def run_roles(commands, parallel, deadline, stop_paths, ready, output, *,
              spawn=subprocess.Popen, cleanup=stop_all, now=time.time, sleep=time.sleep):
    """Finite independent pipelines. Failure never starts a replacement attempt."""
    waiting=list(commands); cursors={r:0 for r in waiting}; active={}; children=[]; logs=[]; events=[]
    try:
        while waiting or active:
            if now() >= deadline or any(Path(p).exists() for p in stop_paths):
                raise TimeoutError('Original deadline or owned STOP')
            for role,p in list(active.items()):
                code=p.poll()
                if code is None: continue
                events.append(dict(role=role,stage=cursors[role]-1,returncode=code,utc=now()))
                del active[role]
                if code: raise RuntimeError(f'{role} stage {cursors[role]-1} returned {code}; no retry')
                if cursors[role]<len(commands[role]): waiting.insert(0,role)
            capacity=2 if parallel else 1
            if waiting and len(active)<capacity and ready():
                # One launch per telemetry/process scan; no speculative queues.
                role=waiting.pop(0); index=cursors[role]
                log=(Path(output)/f'{role}_stage_{index:02d}.log').open('x');logs.append(log)
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                p=spawn(commands[role][index],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(p);active[role]=p;cursors[role]+=1
                events.append(dict(role=role,stage=index,pid=p.pid,command=commands[role][index],utc=now()))
            sleep(.5)
        return events
    finally:
        cleanup(children)
        for log in logs: log.close()
        write_new(Path(output)/'pipeline_events.json',events)


def run(root, registration_path):
    registration=read(registration_path); root=Path(root).resolve()
    if root != Path(registration['root']).resolve() or registration['epochs'] != 60 or registration['deploy_epoch'] != 60:
        raise ValueError('Exact registered learning root and sixty epochs required')
    if registration['wrapper_sha256'] != digest(__file__): raise ValueError('Wrapper source changed')
    for p,expected in registration['source_sha256'].items():
        if digest(p)!=expected: raise ValueError('Trainer source changed: '+p)
    if read(registration['serial_job'])!=registration['expected_serial_job']:
        raise ValueError('Original serial fallback job changed')
    if read(registration['job'])!=proposed_job(root.parent): raise ValueError('Proposed job changed')
    if (root/'paired_episode_io.json').exists(): raise ValueError('Concurrent I/O rendezvous cannot be used for distinct own-data pools')
    for role in 'AB':
        if (root/role/'backend').exists(): raise ValueError('Preserve any previous backend attempt')
        d=read(root/role/'training/deployment.json')
        if d['epoch']!=60: raise ValueError('Both final DPs must complete first')
    affinity=set(registration['cpu_affinity'])
    if not affinity <= os.sched_getaffinity(0): raise ValueError('Registered CPU8-15 not available')
    os.sched_setaffinity(0,affinity)
    from outputs.diagnostics.tight_paired_dagger_20261122_01.resource_watch_v4 import external_test_processes
    owner=Path(registration['owner']); session=read(root/'session.json')
    deadline=min(float(session['deadline']),float(registration['deadline']))
    output=HERE/'parallel_rear_runtime'; output.mkdir(exist_ok=False)
    def row(): return read(owner/'shared_resources.json')
    def ready():
        scan=external_test_processes()
        if scan['processes'] or scan['errors']: return False
        telemetry=row()
        # This is startup admission only; existing FullLease and v4 all-group
        # pauses retain per-block resource/deadline checks throughout training.
        try:
            age=time.time()-float(telemetry['utc_epoch'])
            return (-2<=age<registration['max_telemetry_age_s'] and
                    telemetry['process_telemetry_errors']==[] and telemetry['external_gpu_processes']==[])
        except (KeyError,TypeError,ValueError): return False
    with termination_signals():
        while not ready():
            if time.time()>=deadline or (root.parent/'STOP').exists() or (owner/'STOP').exists():
                raise TimeoutError('Finite wait before rear admission')
            time.sleep(.5)
        commands={r:stage_commands(root,r) for r in 'AB'}
        # Cache the two distinct current DPs serially. A cache stage is DP
        # inference, not a rear trainer: do not admit it from rear-only bounds.
        preparation=output/'serial_caches';preparation.mkdir()
        run_roles({r:commands[r][:2] for r in 'AB'},False,deadline,
                  [root.parent/'STOP',root/'STOP',owner/'STOP'],ready,preparation)
        sizes={r:sum(p.stat().st_size for p in (root/r/'backend'/r/'nominal').glob('*.pt')) for r in 'AB'}
        # residual_setup eagerly loads every nominal record on the CPU. Bound
        # from actual cache bytes, not old episode counts or monitor RSS.
        admission_registration=dict(registration)
        admission_registration['ram_per_worker_bytes']=max(registration['ram_per_worker_bytes'],
            2*2**30+3*max(sizes.values()))
        while not ready():
            if time.time()>=deadline or (root.parent/'STOP').exists() or (owner/'STOP').exists():
                raise TimeoutError('Finite wait before actual rear admission')
            time.sleep(.5)
        pair=admission(row(),admission_registration,time.time(),2)
        single=admission(row(),admission_registration,time.time(),1)
        if not pair['allowed'] and not single['allowed']:
            raise RuntimeError('Insufficient headroom even for serial rear; caches preserved, no trainer launched')
        decision=dict(parallel=pair['allowed'],pair=pair,single=single,utc=time.time(),
            actual_nominal_cache_bytes=sizes,admission_registration=admission_registration,
            registration_sha256=digest(registration_path),cpu_affinity=sorted(os.sched_getaffinity(0)))
        write_new(output/'admission.json',decision)
        training=output/'rear_training';training.mkdir()
        events=run_roles({r:[commands[r][2]] for r in 'AB'},decision['parallel'],deadline,
                        [root.parent/'STOP',root/'STOP',owner/'STOP'],ready,training)
        offline=output/'serial_offline';offline.mkdir()
        run_roles({r:[commands[r][3]] for r in 'AB'},False,deadline,
                  [root.parent/'STOP',root/'STOP',owner/'STOP'],ready,offline)
        write_new(output/'complete.json',dict(utc=time.time(),events=len(events),parallel=decision['parallel'],
            caches_serial=True,offline_serial=True))



def project():
    from tools import tight_project
    tight_project.ENTRIES.update(paired_frontend=('train','tools.tight_paired_learning',['frontend']),
        paired_rear_parallel=('train',MODULE,['rear-pair']))
    sys.argv.pop(1); tight_project.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project': return project()
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['prepare','rear-pair'])
    p.add_argument('--root',type=Path);p.add_argument('--registration',type=Path,default=REGISTRATION)
    a=p.parse_args()
    if a.mode=='prepare': prepare()
    else: run(a.root,a.registration)


if __name__=='__main__': main()
