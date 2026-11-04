"""Persistent serial offline training, then separately classified Kit evaluation."""
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

CODE=Path(__file__).resolve().parents[1]
RUN=['bash','visuotactile/run_cpu.sh']


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


def prepare(root):
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    old=CODE/'outputs/visuotactile/native_dp_v2/h24_parameterization_20261103_01'
    history=read(old/'session.json');now=time.time()
    session={k:history[k] for k in ('source_root','data_sha256','expert_checkpoint','expert_sha256','resource_policy')}
    session.update(run_id=root.name,protocol='offline_training_with_isolated_evaluation_v2',started_epoch=now,
        deadline=now+72*3600,training_deadline=now+72*3600,resource_target_percent=40.,
        numeric_cache=history['profiles']['h24_epsilon']['numeric_cache'],position_statistics=str(old/'position_train_statistics.pt'),
        kit_gate=dict(shutdown='graceful_sdk_close',exit_on_error=False),
        specs=history['motion_specs'],milestones=[1000,3000,6000,10000],supervised=[50,100,200],
        old_blocked_root_untouched='state_image_20261104_01',training_requires_sdk_normal_exit=False,
        kit_diagnostic_starts=6,evaluation_maximum_starts=96,evaluation_retries=0,training_resume_limit_per_job=1,
        owned_worker_roots=[str(root)],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip())
    assert sha(session['numeric_cache'])==history['profiles']['h24_epsilon']['numeric_cache_sha256']
    assert sha(Path(session['source_root'])/'data.json')==session['data_sha256']
    code_paths=list(history['implementation_sha256'])+['visuotactile/state_image_loss.py','visuotactile/state_image_offline.py']
    offline={str((CODE/p).resolve()):sha(CODE/p) for p in code_paths}
    for p in (session['numeric_cache'],session['position_statistics'],str(Path(session['source_root'])/'data.json'),
            str(Path(session['source_root'])/'native_perception.json'),session['expert_checkpoint']):offline[p]=sha(p)
    write(root/'offline_freeze.json',offline)
    session['implementation_sha256']={p:sha(CODE/p) for p in code_paths}
    write(root/'session.json',session)
    write(root/'orchestration_freeze.json',{p:sha(CODE/p) for p in ('tools/state_image_queue.py','tools/state_image_kit_diagnostic.py','tools/state_image_report.py')})
    previous=CODE/'outputs/visuotactile/native_dp_v2/state_image_20261104_01/kit_gate'
    write(root/'freeze/benchmark.json',read(previous/'freeze/benchmark.json'))
    write(root/'prior_evidence.json',dict(blocked_review_sha256=sha(previous/'completion_review.json'),
        expert_trajectory_parity_sha256=sha(previous/'completed_expert_trajectory_parity.json'),
        offline_gate_is_separate=True,old_sdk_gate_passed=False))
    write(root/'status.json',dict(state='pending',training_state='pending',evaluation_state='pending',pid=None,utc=now))


def kill_owned(worker):
    if worker.poll() is None:
        os.killpg(worker.pid,signal.SIGTERM)
        try:worker.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=15)


class Owner:
    def __init__(self,root):
        self.root=root.resolve();self.session=read(root/'session.json');self.monitor=None;self.worker=None
        self.attempts=read(root/'attempts.json') if (root/'attempts.json').exists() else []
    def status(self,**kwargs):
        current=read(self.root/'status.json');current.update(utc=time.time(),pid=os.getpid(),heartbeat=time.time())
        current.update(kwargs);write(self.root/'status.json',current)
    def stage(self,name,module,arguments,*,kit=False):
        final=self.root/'stages'/name/'committed.json'
        if final.exists():return read(final)
        parent=final.parent;parent.mkdir(parents=True,exist_ok=True)
        if kit:
            from tools.kit_resource_preflight import inspect
            preflight=inspect(256,self.root);write(parent/'resource_preflight.json',preflight)
            if not preflight['resource_admitted']:raise RuntimeError('Evaluation-only inotify resource admission failed; training checkpoints preserved')
        number=1+sum(r['name']==name for r in self.attempts)
        # Existing incomplete attempt consumes budget; never reinitialize silently.
        if number>1:
            if kit: return dict(state='software_missing',reason='prior attempt already consumed')
            if number>2 or 'train_' not in name:raise RuntimeError('Unrecoverable stage; preserve evidence: '+name)
            job=name.split('_')[0]
            if not (self.root/(job+'_train')/'last.pt').exists():raise RuntimeError('No resumable checkpoint')
        output=parent/f'attempt_{number}'
        command=RUN+['-m',module,'--root',str(self.root),'--output',str(output),*arguments]
        record=dict(name=name,number=number,output=str(output),kit=kit,utc=time.time())
        self.attempts.append(record);write(self.root/'attempts.json',self.attempts)
        from tools.state_image_kit_gate import proc,descendants,gpu_pids
        observed={};started=time.time();timed_out=None;last_progress=started
        with (parent/f'attempt_{number}.log').open('x') as log:
            self.worker=subprocess.Popen(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            record['pid']=self.worker.pid;write(self.root/'attempts.json',self.attempts)
            while self.worker.poll() is None:
                if self.monitor.poll() is not None:raise RuntimeError('Continuous monitor died')
                phase=read(output/'shutdown_phase.json') if (output/'shutdown_phase.json').exists() else {}
                for pid in {self.worker.pid,*descendants(self.worker.pid)}:
                    p=proc(pid)
                    if p:observed[pid]=p['start_ticks']
                self.status(state='evaluating' if kit else 'supervised' if name.endswith(('50','100','200')) else 'running',
                    stage=name,worker_pid=self.worker.pid,training_state='completed' if kit else 'running',
                    evaluation_state='running' if kit else 'pending',shutdown_phase=phase)
                now=time.time()
                if now>self.session['deadline'] or (self.root/'STOP').exists():timed_out='session_bound_or_STOP'
                if kit and now-started>900:timed_out='Kit_worker_900s'
                if kit and phase and now-phase['utc']>45:timed_out='SDK_shutdown_45s'
                if not kit:
                    job=name.split('_')[0];progress=self.root/(job+'_train')/'progress.json'
                    if progress.exists():last_progress=max(last_progress,progress.stat().st_mtime)
                    waiting=any(self.root.glob('wait_*.json'))
                    if now-last_progress>1800 and not waiting:timed_out='training_no_progress_1800s'
                if timed_out:kill_owned(self.worker);break
                time.sleep(2)
            self.worker.wait();code=self.worker.returncode;self.worker=None
        live=[pid for pid,tick in observed.items() if (p:=proc(pid)) and p['start_ticks']==tick]
        gpu=sorted(set(observed)&gpu_pids());record.update(exit_code=code,finished=time.time(),timeout=timed_out,
            remaining_owned_pids=live,remaining_owned_gpu_pids=gpu,reaped=True)
        relative=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
        cgroup=Path('/sys/fs/cgroup')/relative.lstrip('/')
        if not relative.endswith('/state-image-offline-20261104-02.service'):raise RuntimeError('Dedicated service cgroup required')
        members={int(p) for f in [cgroup/'cgroup.procs',*cgroup.glob('**/cgroup.procs')] for p in f.read_text().split()}
        unexpected=members-{os.getpid(),self.monitor.pid,*descendants(self.monitor.pid)}
        record['unexpected_cgroup_pids']=sorted(unexpected)
        write(self.root/'attempts.json',self.attempts)
        if live or gpu or unexpected:raise RuntimeError('Owned process resources not fully released')
        if kit:
            result=read(output/'result.json') if (output/'result.json').exists() else None
            runtime_ok=bool(result and result.get('software_complete') and not (output/'worker_error.json').exists())
            traces=[]
            if runtime_ok:
                import numpy as np
                for trial in result.get('trials',[]):
                    if not trial.get('legal'):continue
                    trace=output/trial['identity']/'trace.npz'
                    with np.load(trace) as z:
                        assert all(np.isfinite(z[k]).all() for k in z.files)
                        assert len(z['joint'])==trial['steps'] and trial['actuator_target_max_error']==0.
                        assert not trial['action_exhausted']
                    traces.append(dict(path=str(trace),sha256=sha(trace)))
            closed=code==0 and (output/'closed.json').exists()
            summary=dict(state='completed' if runtime_ok else 'software_missing',runtime_complete=runtime_ok,
                sdk_graceful_close=closed,shutdown_state='normal_return' if closed else 'fast_or_abnormal_exit',
                exit_code=code,timeout=timed_out,resources_released=True,result=result,output=str(output),traces=traces)
        else:
            if code!=0 or not (output/'result.json').exists():raise RuntimeError('Offline worker failed: '+name)
            result=read(output/'result.json')
            if not result.get('passed',result.get('software_complete')):raise RuntimeError('Offline acceptance failed')
            summary=dict(state='completed',result=result,output=str(output),resources_released=True)
        write(final,summary);return summary
    def supervise(self,job):
        records=[read(self.root/'stages'/f'{job}_train_{n}'/'committed.json') for n in (50,100,200)]
        rows=[json.loads(s) for s in (self.root/(job+'_train')/'train.jsonl').read_text().splitlines()]
        assert rows[-1]['update']==200 and len({r['update'] for r in rows})==200
        assert all(math_finite(r['loss']) and math_finite(r['gradient_norm']) and r['history_grad']>0 for r in rows)
        for n,r in zip((50,100,200),records):
            assert r['result']['updates']==n and r['result']['checkpoint_readback']
            assert r['result']['kit_dependency']['kit_loaded'] is False
            if n>50:
                resume=read(Path(r['output'])/'resume.json');assert resume['from_updates']==(50 if n==100 else 100)
                assert resume['optimizer_ema_rng_sampler_restored']
        write(self.root/(job+'_supervised.json'),dict(passed=True,updates=200,monitoring_cycles=3,
            checkpoint_reloaded_in_fresh_process=True,validation_completed=True,kit_required=False,
            first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],utc=time.time()))
    def run(self):
        root=self.root
        for name,digest in read(root/'orchestration_freeze.json').items():
            if sha(CODE/name)!=digest:raise RuntimeError('Orchestrator source changed: '+name)
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        with (root/'monitor.log').open('a') as log:self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for job in ('state','image'):
                if not (root/(job+'_audit')/'result.json').exists():
                    result=self.stage(job+'_audit','visuotactile.state_image_offline',['audit','--job',job])
                    # Copy only lightweight acceptance; calibration retains original path.
                    path=Path(result['output']);(root/(job+'_audit')).mkdir(exist_ok=True)
                    import shutil
                    for name in ('result.json','calibration.json'):shutil.copyfile(path/name,root/(job+'_audit')/name)
                for target in (*self.session['supervised'],*self.session['milestones']):
                    self.stage(f'{job}_train_{target}','visuotactile.state_image_offline',['train','--job',job,'--target',str(target)])
                    if target==200:self.supervise(job)
                    self.report()
            self.status(training_state='completed')
            evaluation_complete=self.kit_work()
            self.status(state='completed' if evaluation_complete else 'software_missing',training_state='completed',
                evaluation_state='completed' if evaluation_complete else 'software_missing',worker_pid=None)
        except BaseException as e:
            write(root/'owner_error.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=time.time()))
            self.status(state='blocked',reason=repr(e))
            if self.worker:kill_owned(self.worker)
        finally:
            (root/'STOP_MONITOR').touch()
            try:self.monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(self.monitor)
            self.status(pid=None,heartbeat=None,worker_pid=None)
            write(root/'owner_exit.json',read(root/'status.json'));self.report()
            try:
                subprocess.run(RUN+['-m','tools.state_image_report','--root',str(root),'--publish'],cwd=CODE,check=True)
            except Exception as error:write(root/'publication.json',dict(state='blocked',reason=repr(error),utc=time.time()))
    def report(self):
        subprocess.run(RUN+['-m','tools.state_image_report','--root',str(self.root)],cwd=CODE,check=True)
    def kit_work(self):
        root=self.root;spec=self.session['specs'][2]
        cases=[('minimal_full','startup',False,False),('minimal_fast','startup',True,False),
            ('base_callback','smoke',False,False),('base_disarmed','smoke',False,True),
            ('expert_no_diagnostic_thread','expert',False,True),('expert_fast','expert',True,True)]
        missing=0
        for name,mode,fast,disarm in cases:
            path=root/'jobs'/(name+'.json');write(path,dict(mode=mode,specs=[spec],video=False,lifecycle=dict(fast_shutdown=fast,disarm_stop=disarm)))
            result=self.stage('kit_'+name,'tools.state_image_kit_diagnostic',['--job',str(path)],kit=True);self.report()
            missing=0 if result.get('runtime_complete') else missing+1
            if missing>=2:return False
        all_complete=True
        for job in ('state','image'):
            for update in self.session['milestones']:
                checkpoint=root/(job+'_train')/f'milestone_{update}.pt';digest=sha(checkpoint)
                for ordinal,spec in enumerate(self.session['specs']):
                    name=f'{job}_loop_{update}_{ordinal:02d}';path=root/'jobs'/(name+'.json')
                    write(path,dict(mode='dp',specs=[spec],checkpoint=str(checkpoint),checkpoint_sha256=digest,
                        video=update==10000 and ordinal==4,inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091),
                        lifecycle=dict(fast_shutdown=True,disarm_stop=True)))
                    result=self.stage(name,'tools.state_image_kit_diagnostic',['--job',str(path)],kit=True)
                    complete=result.get('runtime_complete',False);all_complete &= complete
                    missing=0 if complete else missing+1
                    if missing>=2:return False
                self.report()
        return all_complete


def math_finite(value):
    import math
    return math.isfinite(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    else:
        lock=(a.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.root/'owner_exit.json').exists():raise RuntimeError('Ended queue requires explicit audited recovery')
        Owner(a.root).run()
