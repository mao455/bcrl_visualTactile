"""Current standalone StateDP entry; frozen development -> data -> training.

Only this run's evidence can advance a stage. Attempt rows are committed before
physical starts; failures are never retried or replaced to increase success rate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE,RUN,read,write,supervised_owner,sha,kill_owned,proc
from visuotactile.tight_gate import validate_config,paired_admission,numeric_admission


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root,config_path):
    config=validate_config(read(config_path));root.mkdir(parents=True,exist_ok=False)
    expert=(CODE/config['expert_checkpoint']).resolve()
    if sha(expert)!=config['expert_sha256']:raise ValueError('Frozen expert SHA mismatch')
    now=time.time();r=config['resources']
    session=dict(config,**r,expert_path=str(expert),configuration_sha256=digest(config),
        started=now,deadline=now+r['maximum_hours']*3600,
        collection_deadline=now+r['collection_hours']*3600,
        development_deadline=now+r['development_hours']*3600,
        training_deadline=now+r['maximum_hours']*3600,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        owned_worker_roots=[str(root/'training'),str(root/'audit_smoke')])
    # Existing simulation/training constructors need the expert filename, not an
    # old session or a student checkpoint. Other data paths below are protection only.
    session['expert_checkpoint']=str(expert)
    write(root/'configuration.json',config);write(root/'session.json',session)
    write(root/'source_freeze.json',{str(p):sha(p) for folder in ('tools','visuotactile','isaacsimenvs','evaluation','rl_games')
        for p in sorted((CODE/folder).rglob('*.py'))})
    protected={str((CODE/p).resolve()):sha(CODE/p) for p in config['source_protection']}
    write(root/'protected.json',protected)
    write(root/'status.json',dict(state='prepared',stage='development',formal_updates=0,formal_episodes=0))


def beat(root,**fields):
    write(root/'heartbeat.json',dict(utc=time.time(),**fields))


def capture(root,env,player,lease,spec,alpha,*,speed_scheme=None):
    from visuotactile.state_benchmark import trial
    from visuotactile.camera_validity import inspect_capture
    s=read(root/'session.json');out=root/'episodes'/spec['identity']
    if out.exists():raise FileExistsError('No implicit retry of a physical attempt: '+str(out))
    if shutil.disk_usage(root).free<s['reserve_bytes']:raise RuntimeError('Disk reserve reached before capture')
    speed=dict(speed_scheme) if speed_scheme is not None else dict(kind='closed_loop',arm=alpha,hand=alpha,phase='all')
    attempt=dict(identity=spec['identity'],family=spec['family'],seed=spec['seed'],alpha=alpha,speed_scheme=speed,
        split=spec['split'],event_seed=spec.get('event_seed'),started=time.time(),classification='started')
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps(attempt)+'\n');f.flush();os.fsync(f.fileno())
    beat(root,stage='capture',identity=spec['identity'])
    env.speed_config=speed
    try:
        result=trial(env,lease,out,dict(spec,alpha=alpha),{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True,
            capture_compression=s.get('capture_compression','lzf'))
        row=dict(attempt,success=result['success'],legal=result['legal'],software_complete=result['software_complete'],
            configuration_sha256=s['configuration_sha256'],reset_sha256=digest(spec['narrow_reset']),
            expert_sha256=s['expert_sha256'],classification='illegal_initial' if not result['legal'] else 'task_success' if result['success'] else 'task_failure')
        if result['legal']:
            path=Path(result['dataset_capture']['path'])
            visual=inspect_capture(path,preview=out/'three_views.jpg')
            _,meta,_,numeric=numeric_admission(path)
            row.update(numeric,path=str(path),source_sha256=sha(path),visual_passed=visual['passed'],
                visual_acceptance_sha256=sha(path.parent/'visual_acceptance.json'),
                actual_native_configuration_sha256=digest(meta['config']),
                initial_observable_sha256=sha(out/'initial_observable.json'))
            if not visual['passed']:row['classification']='sensor_failure'
        write(out/'admission_inputs.json',row)
    except BaseException as error:
        row=dict(attempt,classification='software_or_sensor_failure',success=False,error=repr(error),traceback=traceback.format_exc())
        write(root/'attempt_failures'/f"{spec['identity']}.json",row)
        with (root/'completed_attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        raise
    with (root/'completed_attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return row


def environment(root,evaluation=False,*,camera_content_validation=None,lease=None):
    from visuotactile.native_evaluation import client_startup_controls,make_env
    from visuotactile.state_resources import lease_for
    from visuotactile.expert import create_expert
    s=read(root/'session.json')
    if lease is None:lease=lease_for(root,deadline=s['deadline'] if evaluation else s['collection_deadline'])
    if not evaluation and (camera_content_validation is False or not s.get('camera_content_validation',False)):
        raise ValueError('Collection requires explicit camera_content_validation=true and fixed camera configuration')
    if s.get('precision_policy') == 'unified_fp32':
        from visuotactile.precision import state_fp32
        write(root/'precision.json',state_fp32())
    write(root/('eval_startup.json' if evaluation else 'startup.json'),client_startup_controls('kit_managed'))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cpu',fast_shutdown=False,kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    import isaacsimenvs
    env=make_env(True,lease,scene_queries=True,expert_speed=None if evaluation else dict(kind='closed_loop',arm=1.,hand=1.,phase='all'),
        camera_content_validation=(s.get('camera_content_validation',False) if camera_content_validation is None else camera_content_validation))
    if sha(s['expert_checkpoint'])!=s['expert_sha256']:raise RuntimeError('Expert source drift')
    with lease.block():player,_=create_expert(env,s['expert_checkpoint'])
    return app,env,player,lease


def development(root):
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json')
    if s.get('speed_scheme',{}).get('hand_target_cap_rad_s') is not None:
        raise ValueError('Validated Hand cap requires tight_finish gates; legacy scalar-alpha development is not eligible')
    app,env,player,lease=environment(root)
    lease.deadline=s['development_deadline']
    refs={};rows=[];candidates=[]
    for i,seed in enumerate(s['development']['seeds']):
        family=f'development_{i:02d}'
        spec=dict(identity=family+'_native',family=family,seed=seed,split='development',cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed))
        refs[family]=capture(root,env,player,lease,spec,1.)
    for alpha in s['development']['alphas']:
        current=[]
        for i,seed in enumerate(s['development']['seeds']):
            family=f'development_{i:02d}'
            spec=dict(identity=family+f'_a{round(alpha*100):03d}',family=family,seed=seed,split='development',cohort='fresh',
                collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(s['domain'],seed))
            slow=capture(root,env,player,lease,spec,alpha)
            admission=paired_admission(slow,refs[family],s['development'])
            # Repeated reset geometry and native config must also match exactly.
            for key in ('initial_observable_sha256','actual_native_configuration_sha256'):
                if slow.get(key)!=refs[family].get(key):admission['errors'].append(key+'_mismatch');admission['passed']=False
            slow['admission']=admission;current.append(slow);rows.append(slow)
            write(root/'development_results.json',dict(references=list(refs.values()),slow=rows,candidates=candidates))
        passed=all(r['admission']['passed'] for r in current)
        candidates.append(dict(alpha=alpha,passed=passed,starts=len(current),accepted=sum(r['admission']['passed'] for r in current)))
        if passed:
            write(root/'development_gate.json',dict(passed=True,alpha=alpha,candidates=candidates,
                references=list(refs.values()),slow=rows,selection='First preregistered constant alpha passing every development family'))
            return
    write(root/'development_gate.json',dict(passed=False,candidates=candidates,references=list(refs.values()),slow=rows,
        reason='No preregistered speed passed every development family; no formal collection/training permitted'))


def check_protected(root):
    for path,expected in read(root/'protected.json').items():
        if sha(path)!=expected:raise RuntimeError('Protected source manifest changed: '+path)


@supervised_owner
def run(root):
    import fcntl
    from tools.kit_resource_preflight import inspect
    passed=False
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(root/'status.json')['state']!='prepared':raise ValueError('No automatic restart of an existing attempt root')
        pre=inspect(parent=root);write(root/'kit_preflight.json',pre)
        if not pre['resource_admitted']:raise RuntimeError('Kit resources unavailable')
        monitor=child=None
        try:
            with (root/'monitor.log').open('w') as log:
                monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with (root/'development.log').open('w') as log:
                child=subprocess.Popen(RUN+['-m','tools.tight_state','development','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'owner.json',dict(owner=proc(os.getpid()),worker=proc(child.pid),monitor=proc(monitor.pid)))
            write(root/'status.json',dict(state='running',stage='development',formal_updates=0,formal_episodes=0))
            deadline=read(root/'session.json')['development_deadline']
            while child.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Continuous monitor exited')
                if time.time()>=deadline:raise TimeoutError('Development deadline reached')
                if (root/'STOP').exists():raise InterruptedError('Owned STOP')
                time.sleep(2)
            write(root/'development_exit.json',dict(returncode=child.returncode,utc=time.time()))
            if child.returncode:raise RuntimeError('Development worker failed; preserve attempt denominator')
            gate=read(root/'development_gate.json');passed=gate['passed']
            write(root/'selected_development_gate.json',gate)
            if not gate['passed']:
                write(root/'status.json',dict(state='blocked',stage='development_gate',reason=gate['reason'],formal_updates=0,formal_episodes=0))
            else:
                write(root/'status.json',dict(state='development_passed',stage='data_freeze_pending',formal_updates=0,formal_episodes=0))
        except BaseException as error:
            write(root/'owner_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
            write(root/'status.json',dict(state='blocked',stage='development',reason=repr(error),formal_updates=0,formal_episodes=0))
            raise
        finally:
            if child is not None:kill_owned(child)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            check_protected(root)
            status=read(root/'status.json');status['owned_workers_exited']=True;write(root/'status.json',status)

    if passed:
        from tools.tight_finish import pipeline
        pipeline(root)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','development','status'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--config',type=Path,default=CODE/'configs/tight_state.json')
    args=p.parse_args();root=args.root.resolve()
    if args.mode=='prepare':prepare(root,args.config)
    elif args.mode=='run':run(root)
    elif args.mode=='status':print(json.dumps(read(root/'status.json'),indent=2))
    else:
        code=0
        try:development(root)
        except BaseException:
            write(root/'development_error.json',dict(traceback=traceback.format_exc()));code=1
        sys.stdout.flush();sys.stderr.flush();os._exit(code)


if __name__=='__main__':main()
