"""Durable finite owner for a registered batch, including startup and partial census."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from tools.tight_fresh_eval_recovery import stop_owned,wait_owned
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def process_identity(pid, proc=Path('/proc')):
    try:
        directory = proc/str(pid)
        fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
            sid=int(fields[3]), start_ticks=int(fields[19]), uid=directory.stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def bind_worker_group(pid, proc=Path('/proc')):
    """Bind only the new session explicitly created by this owner's Popen."""
    identity = process_identity(pid, proc)
    if (identity is None or identity['uid'] != os.getuid()
            or identity['pgid'] != pid or identity['sid'] != pid):
        raise ValueError('Worker must lead the owner-created process group and session')
    return identity


def owned_group_members(identity, proc=Path('/proc')):
    """A reused leader PID or foreign session never authorizes a group signal."""
    if (identity.get('pid', 0) <= 1 or identity.get('pgid') != identity['pid']
            or identity.get('sid') != identity['pid'] or identity.get('uid') != os.getuid()
            or type(identity.get('start_ticks')) is not int):
        raise ValueError('Missing owner-created process-group binding')
    leader = process_identity(identity['pid'], proc)
    if leader is not None and any(leader[key] != identity[key]
            for key in ('start_ticks', 'uid', 'pgid', 'sid')):
        raise ValueError('Worker leader PID was reused or changed its ownership/session')
    members = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        member = process_identity(int(directory.name), proc)
        if member is None or member['pgid'] != identity['pgid']:
            continue
        if (member['sid'] != identity['sid'] or member['uid'] != identity['uid']
                or member['start_ticks'] < identity['start_ticks']):
            raise ValueError('Registered group contains an unbound session/owner/process')
        members.append(member)
    return sorted(members, key=lambda member:member['pid'])


def cleanup_owned_group(identity, *, grace_seconds=3., proc=Path('/proc')):
    """Remove descendants even if a native abort already reaped the leader.

    Spawn verifiers inherit the worker's PGID/SID. Polling only the leader misses
    these surviving readers; the creation binding plus live identities is the
    signal boundary. Zombies are recorded but are not runnable work.
    """
    began = time.monotonic()
    before = owned_group_members(identity, proc)
    signals = []
    def live(members):
        return [member for member in members if member['state'] not in ('Z', 'X')]
    remaining = before
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not live(remaining):
            break
        # Revalidate immediately before each signal; never target a recycled
        # historical PID solely because it appears in a persisted receipt.
        remaining = owned_group_members(identity, proc)
        if not live(remaining):
            break
        try:
            os.killpg(identity['pgid'], sig)
            signals.append(int(sig))
        except ProcessLookupError:
            pass
        until = time.monotonic()+grace_seconds
        while True:
            remaining = owned_group_members(identity, proc)
            if not live(remaining) or time.monotonic() >= until:
                break
            time.sleep(.05)
    if live(remaining):
        raise RuntimeError('Owned worker process-group descendants did not exit')
    return dict(binding=identity, before=before, after=remaining, signals=signals,
        live_descendants_exited=True, seconds=time.monotonic()-began,
        scope='Only this owner-created PGID/SID; leader exit does not imply child exit')


def census(output,plan):
    rows=[]
    for case in plan['cases']:
        directory=output/case['name'];result=directory/'result.json'
        value=read(result) if result.exists() else {}
        execution=(read(directory/'execution_complete.json')
            if (directory/'execution_complete.json').exists() else {})
        recording_complete=bool(value.get('software_complete'))
        admission_complete=(plan.get('kind')!='takeover' or not value.get('legal')
            or 'accepted_for_training' in value)
        complete=recording_complete and admission_complete
        rows.append(dict(case=case['name'],role=case['role'],
            state=('complete' if complete else 'execution_complete_pending_admission' if recording_complete
                else 'execution_complete_pending_verification' if execution.get('software_complete')
                else 'in_progress' if directory.exists() else 'not_started'),
            recording_complete=recording_complete,admission_complete=bool(recording_complete and admission_complete),
            execution_complete=bool(recording_complete or execution.get('software_complete')),
            result_sha256=sha256(result) if result.exists() else None,
            accepted=value.get('accepted_for_training',False)))
    return rows


def run(output,worker_tree,startup_seconds=300):
    output=Path(output).resolve();worker_tree=Path(worker_tree).resolve()
    plan=read(output/'plan.json');session=read(output/'session.json')
    from tools.tight_fresh_batch_run import validate_capture_audit_request
    validate_capture_audit_request(plan)
    if plan.get('batch',{}).get('version')!='whole_wave_complete_student_v1' or not 1<=startup_seconds<=600:
        raise ValueError('Registered batch and finite startup bound required')
    if time.time()>=session['deadline']:raise TimeoutError('Registered session expired')
    expected={str(p.relative_to(worker_tree)):sha256(p) for folder in ('tools','visuotactile')
        for p in (worker_tree/folder).rglob('*.py')}
    if expected!=plan['source_sha256']:raise ValueError('Frozen source bytes/population differ')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker_tree,text=True).strip()!=plan['source_commit']:
        raise ValueError('Frozen source commit differs')
    if any((output/name).exists() for name in ('owner_started.json','run_started.json','worker.log','result.json')):
        raise FileExistsError('No duplicate start or overwritten attempt')
    if any((output/case['name']).exists() for case in plan['cases']):raise FileExistsError('Preserve partial cases')
    command=[str(worker_tree/'.venv_isaacsim/bin/python'),'-m','tools.tight_fresh_batch_run','run','--output',str(output)]
    start=time.time();digest=sha256(output/'plan.json');worker=None;worker_group=None;reason=None;error=None;complete=False
    with (output/'owner_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),started=start,deadline=session['deadline'],startup_seconds=startup_seconds,
            command=command,cwd=str(worker_tree),plan_sha256=digest),stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    handlers={}
    def interrupted(signum,frame):raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        with (output/'worker.log').open('x') as stream:
            worker=subprocess.Popen(command,cwd=worker_tree,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
            worker_group=bind_worker_group(worker.pid)
            atomic_json(output/'owner_worker_started.json',dict(worker_pid=worker.pid,owner_pid=os.getpid(),
                worker_group=worker_group,start_new_session=True,utc=time.time()))
            reason=wait_owned(worker,session['deadline'],startup_marker=output/'initialization.json',startup_seconds=startup_seconds)
        if worker.returncode==0 and reason is None:
            result=read(output/'result.json');rows=census(output,plan)
            complete=(result.get('software_complete') is True and result.get('cases')==len(rows)
                and all(row['state']=='complete' for row in rows) and sha256(output/'plan.json')==digest)
            if not complete:raise ValueError('Batch exited without complete registered evidence')
    except BaseException as caught:error=repr(caught);reason=reason or 'owner_exception'
    finally:
        if worker is not None:
            try:
                stop_owned(worker)
                cleanup=(cleanup_owned_group(worker_group) if worker_group is not None
                    else dict(live_descendants_exited=False,error='Worker group creation binding unavailable'))
            except BaseException as caught:
                cleanup=dict(live_descendants_exited=False,binding=worker_group,error=repr(caught))
            atomic_json(output/'owner_group_cleanup.json',cleanup)
            if not cleanup['live_descendants_exited']:
                complete=False;reason='owned_group_cleanup_failed';error=cleanup.get('error')
        for sig,handler in handlers.items():signal.signal(sig,handler)
        atomic_json(output/'partial_census.json',census(output,plan))
        receipt=dict(software_complete=complete,worker_pid=worker.pid if worker else None,
            returncode=worker.returncode if worker else None,stop_reason=reason or ('complete' if complete else 'worker_failure'),
            error=error,seconds=time.time()-start,ended=time.time(),plan_sha256=digest)
        atomic_json(output/'owner_finished.json',receipt);print(json.dumps(receipt),flush=True)
    if not complete:raise RuntimeError('Batch incomplete; preserved census requires evidence-based recovery')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-tree',type=Path,required=True)
    p.add_argument('--startup-seconds',type=int,default=300);a=p.parse_args();run(a.output,a.worker_tree,a.startup_seconds)
