"""One bounded engineering probe; preserve failures and never retry implicitly."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.tight_fresh_eval_recovery import wait_owned, stop_owned
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def run(output, worker_tree, *, startup_seconds=300, wall_seconds=600):
    output=Path(output).resolve();worker_tree=Path(worker_tree).resolve()
    plan=read(output/'plan.json');session=read(output/'session.json')
    if (plan.get('kind')!='engineering_batch_throughput_probe' or plan.get('test_used') is not False
            or not 1<=startup_seconds<=600 or not startup_seconds<=wall_seconds<=3600):
        raise ValueError('Registered diagnostic and finite startup/wall budgets required')
    expected={str(p.relative_to(worker_tree)):sha256(p) for folder in ('tools','visuotactile')
        for p in (worker_tree/folder).rglob('*.py')}
    if expected!=plan['source_sha256']:raise ValueError('Probe frozen source differs')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker_tree,text=True).strip()!=plan['source_commit']:
        raise ValueError('Probe frozen commit differs')
    if any((output/name).exists() for name in ('owner_started.json','worker.log','started.json','result.json')):
        raise FileExistsError('Preserve previous probe identity')
    began=time.time();deadline=min(session['deadline'],began+wall_seconds)
    if deadline<=began:raise TimeoutError('Session expired')
    command=[str(worker_tree/'.venv_isaacsim/bin/python'),'-m','tools.tight_fresh_batch_eval','run',
        '--root',plan['resource_root'],'--output',str(output)]
    binding=dict(command=command,cwd=str(worker_tree),owner_pid=os.getpid(),started=began,
        deadline=deadline,session_deadline=session['deadline'],startup_seconds=startup_seconds,
        plan_sha256=sha256(output/'plan.json'),owner_script=str(Path(__file__).resolve()),
        owner_sha256=sha256(Path(__file__)),no_automatic_retry=True)
    atomic_json(output/'owner_started.json',binding)
    worker=None;reason=None;error=None;complete=False
    try:
        with (output/'worker.log').open('x') as stream:
            worker=subprocess.Popen(command,cwd=worker_tree,start_new_session=True,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
            atomic_json(output/'owner_worker_started.json',dict(pid=worker.pid,utc=time.time()))
            reason=wait_owned(worker,deadline,startup_marker=output/'startup_timing.json',startup_seconds=startup_seconds)
        if worker.returncode==0 and reason is None:
            result=read(output/'result.json');rows=result.get('results',[])
            complete=(result.get('software_complete') is True and result.get('diagnostic_only') is True
                and result.get('autonomous_evaluation_eligible') is False
                and len(rows)==len(plan['pairs']) and {row['role'] for row in rows}==set(plan['pairs'])
                and all(len(row['results'])==len(plan['cases']) and
                    all(r.get('success') is None and r.get('autonomous_evaluation_eligible') is False
                        for r in row['results']) for row in rows)
                and sha256(output/'plan.json')==binding['plan_sha256'])
            if not complete:raise ValueError('Probe completion population/contract mismatch')
    except BaseException as caught:error=repr(caught)
    finally:
        if worker is not None:stop_owned(worker)
        atomic_json(output/'owner_finished.json',dict(software_complete=complete,stop_reason=reason,
            error=error,returncode=worker.returncode if worker else None,seconds=time.time()-began,
            ended=time.time(),startup_complete=(output/'startup_timing.json').exists(),
            partial_roles=[role for role in plan['pairs'] if (output/role/'result.json').exists()],
            result_sha256=sha256(output/'result.json') if (output/'result.json').exists() else None))
    if not complete:raise RuntimeError('Probe incomplete; inspect retained original evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker-tree',type=Path,required=True);p.add_argument('--startup-seconds',type=int,default=300)
    p.add_argument('--wall-seconds',type=int,default=600);a=p.parse_args()
    run(a.output,a.worker_tree,startup_seconds=a.startup_seconds,wall_seconds=a.wall_seconds)
