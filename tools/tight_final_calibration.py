"""Last registered three-start midpoint calibration; no further search edge."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tools.runtime import CODE,RUN,read,write,supervised_owner,kill_owned


def worker(root):
    from tools.tight_state import capture
    from tools.tight_state import environment
    from visuotactile.narrow_data import direct_configuration
    from visuotactile.tight_gate import paired_admission,require
    s=read(root/'session.json');p=read(root/'final_calibration_plan.json')
    require(hashlib.sha256(json.dumps(s['domain'],sort_keys=True).encode()).hexdigest()==p['domain_sha256'],'Registered final domain changed')
    require(s['development_deadline']==p['deadline'],'Original deadline changed')
    require(p['maximum_additional_expert_starts']==3 and p['no_more_calibration_after_this'],'Finite final calibration required')
    refs=read(root/'refined_development_results.json')['references'];require(len(refs)==3,'Three frozen native references required')
    app,env,player,lease=environment(root);lease.deadline=p['deadline'];rows=[]
    for j,seed in enumerate(p['seeds']):
        family=f'development_refined_{j:02d}'
        spec=dict(identity=family+'_a0525',family=family,seed=seed,split='development',cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed))
        row=capture(root,env,player,lease,spec,p['alpha']);row['admission']=paired_admission(row,refs[j],s['development'])
        for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
            if row.get(key)!=refs[j].get(key):row['admission']['errors'].append(key+'_mismatch');row['admission']['passed']=False
        rows.append(row);write(root/'final_calibration_results.json',dict(references=refs,slow=rows))
    previous=root/'selected_development_gate.json'
    write(root/'selected_development_gate_before_midpoint.json',read(previous))
    write(previous,dict(passed=all(r['admission']['passed'] for r in rows),alpha=p['alpha'],references=refs,slow=rows,
        protocol='Last3 registered midpoint candidates, all prior failures retained; no further calibration'))


@supervised_owner
def run(root):
    plan=read(root/'final_calibration_plan.json')
    while not read(root/'status.json').get('owned_workers_exited'):
        if time.time()>=plan['deadline']:raise TimeoutError('Original development deadline')
        time.sleep(2)
    require_path=root/'actuator_gate.json'
    if not require_path.exists() or not read(require_path)['passed']:raise ValueError('Actuator gate must finish before final calibration')
    passed=False
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'stages/final_midpoint_calibration';path.mkdir(parents=True,exist_ok=False)
        monitor=child=None
        try:
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            with (root/'monitor.log').open('a') as log:monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (path/'worker.log').open('w') as log:child=subprocess.Popen(RUN+['-m','tools.tight_final_calibration','worker','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'status.json',dict(state='running',stage='final_midpoint_calibration',formal_updates=0,formal_episodes=0,owned_workers_exited=False))
            while child.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Continuous monitor stopped')
                if time.time()>=plan['deadline'] or (root/'STOP').exists():raise TimeoutError('Original deadline/STOP')
                time.sleep(2)
            write(path/'exit.json',dict(returncode=child.returncode,utc=time.time()))
            if child.returncode:raise RuntimeError('Final midpoint worker failed')
            passed=read(root/'selected_development_gate.json')['passed']
            write(root/'status.json',dict(state='development_passed' if passed else 'blocked',stage='final_development_gate',
                reason=None if passed else 'Final registered speed still fails per-episode slowing; no more calibration or formal training',formal_updates=0,formal_episodes=0))
        except BaseException as error:
            write(path/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            write(root/'status.json',dict(state='blocked',stage='final_midpoint_calibration',reason=repr(error),formal_updates=0,formal_episodes=0))
            raise
        finally:
            if child is not None:kill_owned(child)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            state=read(root/'status.json');state['owned_workers_exited']=True;write(root/'status.json',state)
    if passed:
        from tools.tight_finish import pipeline
        pipeline(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.mode=='run':run(root)
    else:
        code=0
        try:worker(root)
        except BaseException:write(root/'final_midpoint_error.json',dict(traceback=traceback.format_exc()));code=1
        sys.stdout.flush();sys.stderr.flush();os._exit(code)
