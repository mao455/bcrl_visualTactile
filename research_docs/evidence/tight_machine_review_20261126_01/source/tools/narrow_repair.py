"""Finite camera/history repair, slow-data contrast and fresh StateDP owner."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, proc, kill_owned


def beat(root, **fields):
    write(root/'heartbeat.json',dict(utc=time.time(),**fields))


def plans(root):
    """Frozen geometry-only choices before new student/expert outcomes."""
    import numpy as np
    from tools.narrow_collection import GATE
    from visuotactile.narrow_data import direct_configuration
    s=read(root/'session.json');parent=Path(s['parent'])
    families=read(parent/'families.json');train=sorted([f for f in families if f['split']=='train'],key=lambda f:f['identity'])
    pilot=[train[i] for i in (0,len(train)//2,len(train)-1)]
    def fresh(prefix,seed,n):
        return [dict(identity=f'{prefix}_{i:02d}',seed=seed+i,cohort='fresh',split='unseen',collision_gate=GATE,
            narrow_reset=direct_configuration(s['domain'],seed+i)) for i in range(n)]
    plan=dict(pilot_families=pilot,alphas=[1.,.55,.60,.65],
        candidates=fresh('repair_family',26164000,80),final_unseen=fresh('independent_unseen',26165000,12),
        recheck=[x for x in read(parent/'evaluation_plan.json') if x['identity'] in
            ('train_family_000','train_family_004','unseen_00','unseen_04')],
        final_train=[x for x in read(parent/'evaluation_plan.json') if x['split']=='train'])
    if (root/'plan.json').exists():raise FileExistsError('Plan already frozen')
    write(root/'plan.json',plan)


def cameras(root):
    """One persistent Kit: missing-view A/B, real expert, matched-precision DP review."""
    import numpy as np
    import torch
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import prepare, trial
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.camera_validity import frame_metrics, inspect_capture
    from visuotactile.narrow_data import record, admission_conflict
    from visuotactile.precision import state_fp32, snapshot
    from visuotactile.recording import model_fingerprint
    from visuotactile.video import verify_video
    s=read(root/'session.json');plan=read(root/'plan.json');parent=Path(s['parent'])
    app,env,player,lease=environment(root,camera_content_validation=False)
    fingerprint=model_fingerprint(player.model)
    f=plan['pilot_families'][0];spec=dict(f['spec'],family=f['identity'],split='train')
    probe=root/'camera_probe';probe.mkdir()
    measurements=[]
    for mode in ('missing_configuration','configured'):
        out=probe/mode;out.mkdir()
        with lease.block():
            if mode=='configured':
                env.camera_content_validation=True;env.configure_camera_views()
            _,legal=prepare(env,IsolatedEnvironment(env,spec['seed']),spec,{},out)
            packet=env.research_packet
            rgb=np.rint(packet['image'][0,:,:3].cpu().numpy()*255).astype('uint8')
            measure=frame_metrics(rgb,packet['depth_m'][0].cpu().numpy(),packet['depth_valid'][0].cpu().numpy())
            measure.update(mode=mode,legal=legal['legal'],camera_valid=packet['camera_valid'][0].tolist(),
                pose=packet['camera_pose_w'][0].tolist(),tick=int(packet['step_id'][0]),physics_steps=int(env.research_tick))
            from PIL import Image
            Image.fromarray(np.concatenate(rgb.transpose(0,2,3,1),axis=1)).save(out/'views.jpg')
            measurements.append(measure)
    diff=admission_conflict(read(probe/'missing_configuration/initial_observable.json'),read(probe/'configured/initial_observable.json'),True,True)
    report=dict(measurements=measurements,observable_difference=diff,passed=all(measurements[-1]['content_valid']) and max(diff['observable_max_abs'].values())<1e-6)
    write(root/'camera_probe.json',report)
    if not report['passed']:raise RuntimeError('Camera configuration A/B did not pass')
    speed=dict(kind='closed_loop',arm=1.,hand=1.,phase='all');env.speed_config=speed
    spec.update(identity=f['identity']+'_repair_native',alpha=1.)
    out=root/'episodes'/spec['identity'];beat(root,stage='camera_expert',identity=spec['identity'])
    r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],expert_speed=speed,capture_dataset=True,video=True,
        video_title='Frozen RL expert | camera acceptance | native speed | 1x')
    if not r['legal']:raise RuntimeError('Pilot family rejected')
    path=Path(r['dataset_capture']['path']);record(path)
    visual=inspect_capture(path,preview=out/'three_view_review.jpg')
    write(root/'camera_expert.json',dict(result=r,visual=visual,video=verify_video(out/'rollout.mp4',r['steps']),sha256=sha(path)))
    if not visual['passed'] or not r['success']:raise RuntimeError('First full expert camera acceptance failed')
    if model_fingerprint(player.model)!=fingerprint:raise RuntimeError('Expert weights changed')
    write(root/'camera_acceptance_complete.json',dict(passed=True,expert_parameters_unchanged=True))


def recheck(root):
    import torch
    from tools.narrow_collection import environment
    from visuotactile.state_benchmark import trial
    from visuotactile.precision import state_fp32, snapshot
    from visuotactile.video import verify_video
    s=read(root/'session.json');plan=read(root/'plan.json');parent=Path(s['parent'])
    app,env,player,lease=environment(root,evaluation=True)
    # Both arithmetic modes see repaired cameras and the same fixed starts. No
    # policy updates; arithmetic gains are not attributed to new demonstrations.
    saved=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
    results=[]
    for precision in ('legacy_default','unified_fp32'):
        if precision=='legacy_default':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        else:state_fp32()
        for spec in plan['recheck']:
            # Explicit reviewed pre-action software failure is retained, consumes
            # its original start, and is not retried or silently called timeout.
            failure=root/'reviewed_preaction_failure.json'
            if failure.exists():
                missing=read(failure)
                if missing['identity']==spec['identity'] and missing['precision_name']==precision:
                    results.append(missing);write(root/'recheck_results.json',results);continue
            out=root/'recheck_evaluation'/(precision+'__'+spec['identity'])
            beat(root,stage='old_model_recheck',precision=precision,identity=spec['identity'])
            r=trial(env,lease,out,spec,{},player,s['expert_checkpoint'],saved=saved,position=True,video=True,
                inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091),
                video_title=f"State200 update5735 | {precision} | {spec['identity']} | diagnostic | 1x")
            results.append(dict(r,precision=snapshot(),precision_name=precision,
                video=verify_video(out/'rollout.mp4',r['steps']) if r['legal'] else None))
            write(root/'recheck_results.json',results)
    state_fp32()
    write(root/'recheck_complete.json',dict(passed=True,allocated=len(results),
        software_failures=sum(not r['software_complete'] for r in results),completed=sum(r['software_complete'] for r in results)))


def status(root, **fields):
    old=read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json',dict(old,**fields,utc=time.time()))


def protected(root):
    for p,h in read(root/'protected.json').items():
        if sha(p)!=h:raise RuntimeError('Protected artifact changed: '+p)


def execute(root, phase, jobs, deadline):
    """One reviewed phase start; no retry/rebudget path, every child owned."""
    s=read(root/'session.json');worker=monitor=None
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        directory=root/'phases'/phase;directory.mkdir(parents=True,exist_ok=False)
        write(directory/'source.json',{str(p):sha(p) for folder in ('tools','visuotactile','isaacsimenvs','evaluation')
            for p in (CODE/folder).rglob('*.py')})
        write(directory/'budget.json',dict(started=time.time(),deadline=deadline,session_deadline=s['deadline'],
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))
        def interrupted(*_):raise InterruptedError('Owner stop signal')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            protected(root)
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            with (directory/'monitor.log').open('w') as log:
                monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def run(name,args,limit=None,kit=False):
                nonlocal worker
                if time.time()>=min(deadline,s['deadline']) or (root/'STOP').exists():raise TimeoutError('Original deadline/STOP')
                for p,h in read(directory/'source.json').items():
                    if sha(p)!=h:raise RuntimeError('Active phase source drift: '+p)
                job=directory/name;job.mkdir()
                if kit:
                    from tools.kit_resource_preflight import inspect
                    pre=inspect(parent=root);write(job/'inotify.json',pre)
                    if not pre['resource_admitted']:raise RuntimeError('Kit inotify gate')
                with (job/'worker.log').open('w') as log:
                    worker=subprocess.Popen(RUN+args,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status(root,state='running',phase=phase,stage=name,owner=proc(os.getpid()),worker=proc(worker.pid),
                    monitor=proc(monitor.pid),owned_workers_exited=False,error=None)
                last_disk=0;began=time.time()
                while worker.poll() is None:
                    if time.time()>=min(deadline,s['deadline'],limit or deadline) or (root/'STOP').exists():raise TimeoutError('Fixed stage deadline/STOP')
                    if monitor.poll() is not None:raise RuntimeError('Monitor exited')
                    if time.time()-last_disk>30:
                        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                        if size>s['maximum_output_bytes'] or shutil.disk_usage(root).free<s['reserve_bytes']:raise RuntimeError('Disk budget/reserve')
                        last_disk=time.time()
                    progress=[root/'heartbeat.json',*root.glob('expanded_training/progress.json'),*root.glob('expanded_training/validation.json'),
                        *root.glob('episodes/*/progress.json'),*root.glob('*evaluation/*/progress.json')]
                    latest=max([began]+[p.stat().st_mtime for p in progress if p.exists()])
                    if time.time()-latest>900:raise TimeoutError('No durable progress for 900s')
                    time.sleep(2)
                write(job/'exit.json',dict(returncode=worker.returncode,utc=time.time(),worker_absent=proc(worker.pid) is None))
                if worker.returncode:raise RuntimeError(name+' failed; no automatic retry')
            jobs(run)
            protected(root);write(directory/'complete.json',dict(complete=True,utc=time.time()))
            status(root,state='phase_complete',stage=phase)
        except BaseException as error:
            write(directory/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=time.time()))
            status(root,state='blocked',error=repr(error));raise
        finally:
            if worker is not None:kill_owned(worker)
            if monitor is not None:
                (root/'STOP_MONITOR').touch()
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
                if (root/'monitor_exit.json').exists():shutil.copy2(root/'monitor_exit.json',directory/'monitor_exit.json')
            status(root,owned_workers_exited=True,worker=None,monitor=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('audit','plans','cameras','camera-worker','recheck-worker','status'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.mode=='status':print(read(root/'status.json'));return
    if a.mode=='plans':plans(root);return
    if a.mode in ('camera-worker','recheck-worker'):
        try:(cameras if a.mode=='camera-worker' else recheck)(root)
        except BaseException:
            write(root/(a.mode+'_error.json'),dict(traceback=traceback.format_exc(),utc=time.time()))
            sys.stdout.flush();sys.stderr.flush();os._exit(1)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    if a.mode=='cameras':
        assert read(root/'gpu_parity.json')['passed'] and read(root/'data_diagnosis.json')['passed']
        execute(root,'camera_and_recheck',lambda run:run('camera_worker',['-m','tools.narrow_repair','camera-worker','--root',str(root)],kit=True),
            read(root/'session.json')['gate_deadline']);return
    def jobs(run):
        for mode in ('gpu','dataset'):
            run(mode,['-m','tools.narrow_repair_audit',mode,'--root',str(root)])
    execute(root,'audit',jobs,read(root/'session.json')['gate_deadline'])


if __name__=='__main__':main()
