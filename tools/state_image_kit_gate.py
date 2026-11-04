"""Persistent, serial Kit lifecycle gate for the new State/Image baseline run.

No parameter training. Each episode is a fresh process; SDK close, process exit,
descendant release and native task termination are required independently.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


CODE = Path(__file__).resolve().parents[1]
BASE = CODE/'outputs/visuotactile/native_dp_v2/h24_diagnosis_20261104_01'
RUN = ['bash', 'visuotactile/run_cpu.sh']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def proc(pid):
    p = Path('/proc')/str(pid)
    try:
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        status = dict(line.split(':', 1) for line in (p/'status').read_text().splitlines() if ':' in line)
        watches = instances = 0
        fds = list((p/'fd').iterdir())
        for fd in fds:
            try:
                if 'inotify' in os.readlink(fd):
                    instances += 1
                    watches += sum(s.startswith('inotify wd:') for s in (p/'fdinfo'/fd.name).read_text().splitlines())
            except (FileNotFoundError, ProcessLookupError): pass
        return dict(pid=pid, ppid=int(fields[1]), process_group=int(fields[2]), start_ticks=int(fields[19]),
            state=fields[0], rss_kib=int(status.get('VmRSS', '0 kB').split()[0]),
            threads=int(status['Threads']), fds=len(fds), watches=watches, inotify_instances=instances)
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid):
    # Kernel task child lists avoid scanning/signalling unrelated user processes.
    found = set(); pending = [pid]
    while pending:
        parent = pending.pop()
        for task in Path('/proc', str(parent), 'task').glob('*'):
            try: children = [int(x) for x in (task/'children').read_text().split()]
            except (FileNotFoundError, ProcessLookupError): continue
            for child in children:
                if child not in found: found.add(child); pending.append(child)
    return found


def gpu_pids():
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()
    return {int(line.strip()) for line in lines if line.strip().isdigit()}


def graceful_acceptance(output, code, released):
    output = Path(output)
    if code != 0 or not released or not (output/'closed.json').exists():
        return False
    if (output/'worker_error.json').exists() or (output/'process_exit_requested.json').exists():
        return False
    if not (output/'result.json').exists(): return False
    result = read(output/'result.json'); trials = result.get('trials', [])
    return (result.get('software_complete') is True and len(trials) == 1
        and trials[0].get('legal') is True and trials[0].get('steps', 0) > 1
        and not trials[0].get('action_exhausted', True)
        and any(trials[0].get('reasons', {}).values())
        and trials[0].get('actuator_target_max_error') == 0.)


def prepare(root):
    root = root.resolve(); gate = root/'kit_gate'
    if gate.exists(): raise FileExistsError('Never replace a registered gate')
    preflight = read(root/'preflight/03_after_external_resolution.json')
    if not preflight['resource_admitted']: raise RuntimeError('Resource preflight failed')
    old = read(BASE/'session.json'); original_plan = read(BASE/'plan.json')
    gate.mkdir(); now = time.time()
    session = {key:old[key] for key in ('source_root','expert_checkpoint','expert_sha256','data_sha256','resource_policy')}
    session.update(started_epoch=now, deadline=now+2*3600, training_deadline=now+2*3600,
        resource_target_percent=40., owned_worker_roots=[str(gate)],
        kit_gate=dict(shutdown='graceful_sdk_close', exit_on_error=False),
        training_authorized_in_this_process=False)
    write(gate/'session.json', session)
    write(gate/'freeze/benchmark.json', read(BASE/'freeze/benchmark.json'))
    sources = {name:sha(CODE/name) for name in old['implementation_sha256']}
    sources.update({name:sha(CODE/name) for name in ('tools/state_image_kit_gate.py','tools/kit_resource_preflight.py')})
    checkpoint = original_plan['checkpoint_manifest']['sample/milestone_8000.pt']
    if sha(checkpoint['path']) != checkpoint['sha256']: raise ValueError('Protected checkpoint drift')
    stages = [dict(name='01_expert', mode='expert'), dict(name='02_sample8000', mode='dp'), dict(name='03_expert_repeat', mode='expert')]
    for stage in stages:
        job = dict(specs=[original_plan['train']], video=False)
        if stage['mode']=='dp':
            job.update(checkpoint=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'], inference=original_plan['inference'])
        write(gate/'jobs'/(stage['name']+'.json'), job)
    write(gate/'plan.json', dict(stages=stages, maximum_starts=5, required_consecutive=3,
        worker_wall_s=900, shutdown_wall_s=60, spec=original_plan['train'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        source_sha256=sources, session_sha256=sha(gate/'session.json'),
        original_reference=str(BASE/'stages/V3'/original_plan['train']['identity']),
        success_not_required_for_student=True, require_sdk_close=True, retries_automatic=False))
    write(gate/'status.json',dict(state='pending',starts=0,passed=False,utc=now))


def stop_owned(worker):
    if worker.poll() is None:
        os.killpg(worker.pid,signal.SIGTERM)
        try: worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid,signal.SIGKILL); worker.wait(timeout=15)


def run(root):
    from tools.kit_resource_preflight import inspect
    root = root.resolve(); gate = root/'kit_gate'; plan=read(gate/'plan.json')
    owner_lock=(gate/'owner.lock').open('a'); fcntl.flock(owner_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (gate/'owner_exit.json').exists(): raise RuntimeError('Completed or blocked gate cannot silently restart')
    monitor = worker = None; rows=[]; failure=None
    def status(state, **values):
        record=dict(state=state, owner_pid=os.getpid(), starts=len(rows), passed=False, utc=time.time(), **values)
        write(gate/'status.json',record)
        previous=read(root/'status.json')
        previous.update(state=state,stage='kit_reliability',kit_gate=state,kit_starts=len(rows),pid=os.getpid(),
            heartbeat=time.time(),service='state-image-kit-20261104-01.service',publication_state='pending',
            resource_preflight='passed',reason=values.get('reason'),worker_pid=values.get('worker_pid'))
        write(root/'status.json',previous)
    try:
        for name,digest in plan['source_sha256'].items():
            if sha(CODE/name)!=digest: raise ValueError('Frozen gate source drift: '+name)
        if sha(gate/'session.json')!=plan['session_sha256']: raise ValueError('Gate session drift')
        with (gate/'monitor.log').open('x') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(gate)],cwd=CODE,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        for stage in plan['stages']:
            if len(rows)>=plan['maximum_starts']: raise RuntimeError('Five-start cap exhausted')
            if monitor.poll() is not None: raise RuntimeError('Resource monitor exited')
            resource=inspect(256,gate); write(gate/(stage['name']+'_preflight.json'),resource)
            if not resource['resource_admitted']: raise RuntimeError('Resource admission no longer safe')
            name=stage['name']; output=gate/'stages'/name; output.parent.mkdir(exist_ok=True)
            if output.exists(): raise RuntimeError('Attempt directory already exists')
            command=RUN+['-m','visuotactile.state_benchmark','--root',str(gate),'--output',str(output),
                '--mode',stage['mode'],'--job',str(gate/'jobs'/(name+'.json'))]
            began=time.time(); observed={}; launch=dict(name=name,mode=stage['mode'],command=command,utc=began)
            rows.append(launch); write(gate/'attempts.json',rows)
            with (gate/'stages'/(name+'.log')).open('x') as log:
                worker=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                launch['pid']=worker.pid; write(gate/'attempts.json',rows)
                while worker.poll() is None:
                    now=time.time()
                    members={worker.pid,*descendants(worker.pid)}
                    samples=[p for pid in members if (p:=proc(pid)) is not None]
                    for p in samples: observed[p['pid']]=p['start_ticks']
                    lifecycle=read(output/'lifecycle.json') if (output/'lifecycle.json').exists() else {}
                    with (gate/'stages'/(name+'_processes.jsonl')).open('a') as out:
                        out.write(json.dumps(dict(utc=now,lifecycle=lifecycle,processes=samples))+'\n')
                    status('preflight',attempt=name,worker_pid=worker.pid,lifecycle=lifecycle)
                    if now-began>plan['worker_wall_s'] or (gate/'STOP').exists():
                        raise TimeoutError('Owned worker deadline or STOP reached')
                    if lifecycle.get('state')=='closing' and now-lifecycle['utc']>plan['shutdown_wall_s']:
                        raise TimeoutError('SDK shutdown did not finish within60s')
                    if monitor.poll() is not None: raise RuntimeError('Resource monitor exited during worker')
                    time.sleep(2)
                worker.wait(); code=worker.returncode
            still=[pid for pid,start in observed.items() if (p:=proc(pid)) is not None and p['start_ticks']==start]
            remaining_gpu=sorted(set(observed)&gpu_pids())
            released=not still and not remaining_gpu
            passed=graceful_acceptance(output,code,released)
            launch.update(exit_code=code,reaped=True,remaining_owned_pids=still,remaining_owned_gpu_pids=remaining_gpu,
                passed=passed,elapsed_s=time.time()-began,completed_utc=time.time())
            write(gate/'attempts.json',rows); worker=None
            post=inspect(256,gate); write(gate/(name+'_after.json'),post)
            if not passed: raise RuntimeError('Kit full lifecycle gate failed: '+name)
            if not post['resource_admitted']: raise RuntimeError('Post-exit resource pressure')
        import numpy as np
        identity=plan['spec']['identity']; first=gate/'stages/01_expert'/identity
        comparisons={}
        for name,path in [('repeat',gate/'stages/03_expert_repeat'/identity),('historical',Path(plan['original_reference']))]:
            with np.load(first/'trace.npz') as a, np.load(path/'trace.npz') as b:
                comparisons[name]={key:a[key].shape==b[key].shape and a[key].dtype==b[key].dtype and a[key].tobytes()==b[key].tobytes()
                    for key in a.files if key in b.files and key!='expert_shadow'}
            if not comparisons[name] or not all(comparisons[name].values()):
                raise RuntimeError('Expert control/physical trajectory changed: '+name)
        write(gate/'parity.json',dict(fields=comparisons,passed=True,hidden_sim_state_identity_claimed=False))
        final=inspect(256,gate); write(gate/'final_preflight.json',final)
        if not final['resource_admitted']: raise RuntimeError('Final resource admission failed')
        write(gate/'result.json',dict(passed=True,consecutive=len(rows),required=3,maximum_starts=5,
            graceful_sdk_exit=True,expert_and_student=True,parity=True,new_optimizer_updates=0,utc=time.time()))
        status('completed')
    except BaseException as error:
        failure=repr(error)
        write(gate/'owner_error.json',dict(error=failure,traceback=traceback.format_exc(),utc=time.time()))
        if worker is not None:
            stop_owned(worker)
            rows[-1].update(exit_code=worker.returncode,watchdog=True,reaped=True)
            write(gate/'attempts.json',rows)
        status('blocked',reason=failure)
    finally:
        (gate/'STOP_MONITOR').touch()
        if monitor is not None:
            try: monitor.wait(timeout=15)
            except subprocess.TimeoutExpired: stop_owned(monitor)
        passed=failure is None
        write(gate/'owner_exit.json',dict(state='completed' if passed else 'blocked',passed=passed,
            starts=len(rows),reason=failure,pid=os.getpid(),utc=time.time()))
        current=read(root/'status.json'); current.update(pid=None,worker_pid=None,heartbeat=None,
            state='preflight' if passed else 'blocked',kit_gate='completed' if passed else 'blocked',
            reason='Kit gate passed; training implementation/supervision still pending' if passed else failure,utc=time.time())
        write(root/'status.json',current)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.root)
