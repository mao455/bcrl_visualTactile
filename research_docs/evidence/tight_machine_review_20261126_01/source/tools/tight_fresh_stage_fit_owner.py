"""Own one already registered four-arm stage-fit diagnostic; never prepare or retry it."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from tools.tight_batch_owner import bind_worker_group,cleanup_owned_group
from tools.tight_fresh_cycle import wait_backend_owned
from tools.tight_fresh_eval_recovery import stop_owned
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def source_files(tree):
    return {str(path.relative_to(tree)):sha256(path) for folder in ('tools','visuotactile')
        for path in sorted((tree/folder).rglob('*.py'))}


def stage_specs(output,worker_tree):
    python=str(worker_tree/'.venv_isaacsim/bin/python')
    prefix=[python,'-m','tools.tight_fresh_stage_fit']
    stages=[]
    for role in ('A','B'):
        for arm in ('source','selected'):
            stages.append(dict(name=role+'_'+arm,role=role,arm=arm,
                command=prefix+['run-arm','--registration',str(output/role/'registration.json'),'--arm',arm],
                completion=output/role/arm/'result.json',exclusive=output/role/arm))
    for role in ('A','B'):
        stages.append(dict(name=role+'_compare',role=role,arm=None,
            command=prefix+['compare','--registration',str(output/role/'registration.json')],
            completion=output/role/'comparison.json',exclusive=output/role/'comparison.json'))
    return stages


def preflight(output,session_root,worker_tree,worker_commit,wall_seconds):
    if (type(wall_seconds) not in (int,float) or not math.isfinite(wall_seconds)
            or not 0<wall_seconds<=1800):raise ValueError('Explicit positive wall bound at most1800 seconds required')
    if not re.fullmatch(r'[0-9a-f]{7,40}',worker_commit):raise ValueError('Explicit frozen worker commit required')
    if (output/'owner_started.json').exists() or (output/'owner_finished.json').exists():
        raise FileExistsError('Preserve the previous diagnostic owner identity')
    session=read(session_root/'session.json')
    if not output.is_relative_to(session_root):raise ValueError('Diagnostic output must belong to the registered session')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker_tree,text=True).strip()
    expected=subprocess.check_output(['git','rev-parse',worker_commit+'^{commit}'],cwd=worker_tree,text=True).strip()
    if head!=expected:raise ValueError('Frozen evaluator commit differs')
    sources=source_files(worker_tree)
    evaluator=sources.get('tools/tight_fresh_stage_fit.py')
    if evaluator is None:raise ValueError('Frozen evaluator is absent')
    registrations={role:read(output/role/'registration.json') for role in ('A','B')}
    roots=set()
    for role,plan in registrations.items():
        root=Path(plan['root']).resolve(strict=True);roots.add(root)
        if (plan['role']!=role or plan['version']!='train_only_selected_frontend_stage_fit_v3'
                or plan['evaluator_sha256']!=evaluator or read(root/'session.json')!=session
                or not root.is_relative_to(session_root)):
            raise ValueError('Role, evaluator or original resource session differs')
    if len(roots)!=1 or registrations['A']['identities']!=registrations['B']['identities']:
        raise ValueError('Both roles must use the same registered training population and cycle')
    stages=stage_specs(output,worker_tree)
    if any(stage['exclusive'].exists() for stage in stages):
        raise FileExistsError('Preserve every existing arm or comparison; no implicit repeat')
    return session,registrations,sources,head,stages


def check_result(stage,registrations,digests):
    result=read(stage['completion']);role=stage['role'];arm=stage['arm']
    if (result.get('registration_sha256')!=digests[role]
            or result.get('optimizer_updates')!=0
            or any(result.get(key) is not False for key in ('rear_evaluated','validation_evaluated','test_evaluated'))):
        raise ValueError('Diagnostic completion binding or no-training scope differs')
    if arm is not None:
        if (result.get('complete') is not True or result.get('arm')!=arm
                or result.get('checkpoint_sha256')!=registrations[role][arm]['sha256']):
            raise ValueError('Arm completion or checkpoint differs')
    elif result.get('version')!=registrations[role]['version']:
        raise ValueError('Comparison version differs')


def run(output,session_root,worker_tree,worker_commit,wall_seconds=1800):
    began=time.time();mono=time.monotonic()
    output=Path(output).resolve(strict=True);session_root=Path(session_root).resolve(strict=True)
    worker_tree=Path(worker_tree).resolve(strict=True)
    session,registrations,sources,head,stages=preflight(output,session_root,worker_tree,worker_commit,wall_seconds)
    deadline=min(session['deadline'],began+wall_seconds)
    if time.time()>=deadline:raise TimeoutError('Original session or owner wall deadline exhausted')
    digests={role:sha256(output/role/'registration.json') for role in ('A','B')}
    session_digest=sha256(session_root/'session.json')
    stop_paths=(session_root/'STOP',output/'STOP',Path(registrations['A']['root'])/'STOP')
    started=dict(version='train_only_stage_fit_four_arm_owner_v1',pid=os.getpid(),started=began,
        original_session_deadline=session['deadline'],deadline=deadline,wall_seconds=wall_seconds,
        session_root=str(session_root),session_sha256=session_digest,worker_tree=str(worker_tree),worker_commit=head,
        worker_source_sha256=sources,registration_sha256=digests,owner_source_sha256=sha256(__file__),
        order=[stage['name'] for stage in stages],stop_paths=list(map(str,stop_paths)),
        telemetry_note='Existing session monitor does not recognize run-arm --registration ownership; use these bound child PIDs for corrected attribution; preserve original samples',
        timing_scope='Owner/arm wall includes possible overlap with the independent formal cycle; not an isolated throughput measurement')
    with (output/'owner_started.json').open('x') as stream:
        json.dump(started,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    handlers={};receipts=[];error=None;complete=False
    def interrupted(signum,frame):raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        for stage in stages:
            if any(path.exists() for path in stop_paths):raise InterruptedError('Registered diagnostic STOP')
            if time.time()>=deadline:raise TimeoutError('Original session or owner wall deadline exhausted')
            if sha256(session_root/'session.json')!=session_digest:raise ValueError('Original session changed')
            if any(sha256(output/role/'registration.json')!=digests[role] for role in ('A','B')):
                raise ValueError('Registered diagnostic changed')
            if source_files(worker_tree)!=sources:raise ValueError('Frozen worker source drift')
            if stage['exclusive'].exists():raise FileExistsError('Diagnostic stage already exists: '+stage['name'])
            name=stage['name'];start=time.time();process=None;binding=None;reason=None;stage_error=None;ok=False
            atomic_json(output/'owner_status.json',dict(stage=name,started=start,deadline=deadline))
            try:
                with (output/('owner_'+name+'.log')).open('x') as stream:
                    process=subprocess.Popen(stage['command'],cwd=worker_tree,stdin=subprocess.DEVNULL,stdout=stream,
                        stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
                    binding=bind_worker_group(process.pid)
                    atomic_json(output/('owner_'+name+'_started.json'),dict(stage=name,command=stage['command'],
                        cwd=str(worker_tree),worker_group=binding,started=start,deadline=deadline,
                        registration_sha256=digests[stage['role']],completion=str(stage['completion'])))
                    reason=wait_backend_owned(process,deadline,stop_paths)
                    if reason=='original_session_deadline' and deadline<session['deadline']:
                        reason='owner_wall_limit'
                if reason or process.returncode or not stage['completion'].exists():
                    raise RuntimeError('Diagnostic worker incomplete: '+str(reason or process.returncode))
                check_result(stage,registrations,digests);ok=True
            except BaseException as caught:
                stage_error=repr(caught);raise
            finally:
                cleanup=dict(live_descendants_exited=True,worker_started=False)
                if process is not None:
                    try:
                        stop_owned(process)
                        if binding is None:raise RuntimeError('Worker group identity unavailable')
                        cleanup=cleanup_owned_group(binding)
                    except BaseException as caught:
                        cleanup=dict(live_descendants_exited=False,error=repr(caught),binding=binding)
                ok=ok and cleanup['live_descendants_exited']
                receipt=dict(stage=name,complete=ok,started=start,seconds=time.time()-start,ended=time.time(),
                    returncode=process.returncode if process else None,stop_reason=reason,error=stage_error,
                    completion=str(stage['completion']),completion_sha256=sha256(stage['completion']) if stage['completion'].exists() else None,
                    cleanup=cleanup)
                atomic_json(output/('owner_'+name+'.json'),receipt);receipts.append(receipt)
            if not ok:raise RuntimeError('Owned diagnostic group cleanup failed')
        complete=True
    except BaseException as caught:error=repr(caught)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        result=dict(software_complete=complete,error=error,started=began,ended=time.time(),
            seconds=time.monotonic()-mono,deadline=deadline,original_session_deadline=session['deadline'],
            registration_sha256=digests,stages=receipts,
            pending=[stage['name'] for stage in stages if stage['name'] not in {r['stage'] for r in receipts}],
            optimizer_updates=0,autonomous_evaluation=False)
        atomic_json(output/'owner_finished.json',result)
        atomic_json(output/'owner_status.json',dict(stage='complete' if complete else 'failed',software_complete=complete,error=error))
    if not complete:raise RuntimeError('Stage-fit owner incomplete; preserve all arms and stage receipts: '+str(error))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('output','session-root','worker-tree'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker-commit',required=True);parser.add_argument('--wall-seconds',type=float,default=1800)
    args=parser.parse_args();print(json.dumps(run(**vars(args))))


if __name__=='__main__':main()
