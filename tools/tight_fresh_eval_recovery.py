"""Run one fresh evaluation attempt with a bounded pre-control Kit startup.

The previous attempt must not have begun any case. Training is never repeated.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def stop_owned(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def run(root,previous,output,source,startup_seconds):
    if output.exists():raise FileExistsError('Preserve prior evaluation identities')
    if (previous/'initialization.json').exists() or list(previous.glob('*/result.json')):
        raise ValueError('Recovery is only for a pre-control initialization failure')
    old=json.loads((previous/'plan.json').read_text())
    if old['kind']!='autonomous':raise ValueError('Only autonomous evaluation recovery')
    session=json.loads((root/'session.json').read_text())
    if time.time()>=session['deadline']:raise TimeoutError('Original session expired')
    old_pid=json.loads((previous/'run_started.json').read_text())['pid']
    if Path(f'/proc/{old_pid}').exists():raise ValueError('Previous worker must exit first')
    cases=old['cases'];count=sum(c['role']=='A' for c in cases)
    seeds=[c['spec']['seed'] for c in cases if c['role']=='A']
    python=source/'.venv_isaacsim/bin/python'
    prepare=[str(python),'-m','tools.tight_fresh_rollouts','prepare','--root',str(root),
        '--output',str(output),'--count',str(count),'--seed',str(min(seeds))]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1')
    subprocess.run(prepare,cwd=source,env=env,check=True,timeout=60)
    plan=json.loads((output/'plan.json').read_text())
    if plan['cases']!=old['cases'] or plan['pairs']!=old['pairs']:
        raise ValueError('Recovery changed registered cases or selected complete pairs')
    wrapper="import faulthandler,runpy;faulthandler.dump_traceback_later(90,repeat=True);runpy.run_module('tools.tight_fresh_rollouts',run_name='__main__')"
    command=[str(python),'-c',wrapper,'run','--root',str(root),'--output',str(output)]
    began=time.time()
    with (output/'owner_worker.log').open('x') as log:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(output/'recovery_registration.json',dict(command=command,pid=process.pid,
            prior=str(previous),prior_plan_sha256=sha256(previous/'plan.json'),plan_sha256=sha256(output/'plan.json'),
            startup_seconds=startup_seconds,deadline=session['deadline'],started=began,
            scope='One pre-control recovery, exact same cases/pairs, no new training; original failed startup remains separate'))
        reason=None
        try:
            while process.poll() is None:
                initialized=(output/'initialization.json').exists()
                if time.time()>=session['deadline']:reason='original_session_deadline';break
                if not initialized and time.time()-began>=startup_seconds:reason='pre_control_startup_timeout';break
                time.sleep(2)
        finally:
            if process.poll() is None:stop_owned(process)
        complete=(output/'result.json').exists() and json.loads((output/'result.json').read_text()).get('software_complete',False)
        atomic_json(output/'recovery_finished.json',dict(returncode=process.returncode,software_complete=complete,
            stop_reason=reason or ('complete' if complete else 'worker_failure'),seconds=time.time()-began))
        if not complete:raise RuntimeError('Evaluation recovery did not complete: '+str(reason or process.returncode))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--startup-seconds',type=int,default=180);a=p.parse_args()
    if not 60<=a.startup_seconds<=600:p.error('Bounded startup of 60 to 600 seconds required')
    run(a.root.resolve(),a.previous.resolve(),a.output.resolve(),a.source.resolve(),a.startup_seconds)
