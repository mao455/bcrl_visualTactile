"""Bounded, persistent final-model-only evaluation owner; never trains."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.state_image_queue import CODE, RUN, read, write, sha, kill_owned
from tools.state_image_kit_gate import proc, descendants, gpu_pids

SERVICE='final-dp-evaluation-20261104-01.service'
OLD=CODE/'outputs/visuotactile/native_dp_v2/state_image_20261104_02'


def classify(output, code, timeout, text, valid):
    if valid:
        return 'normal_sdk_return' if code==0 and (output/'closed.json').exists() else 'sdk_fast_exit_zero' if code==0 else 'shutdown_fault_after_complete_runtime'
    if 'FileThread' in text and code in (-6,-11):return 'startup_FileThread_before_scene' if not (output/'environment.json').exists() else 'runtime_FileThread'
    if timeout:return timeout
    return 'worker_error' if (output/'worker_error.json').exists() else 'startup_or_runtime_software_missing'


def prepare(root):
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    original=read(OLD/'session.json')
    for unit in ('state-image-offline-20261104-02.service',SERVICE):
        assert subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode!=0
    assert read(OLD/'status.json')['training_state']=='completed'
    checkpoints={}
    for job,model in [('state','StateDP'),('image','ImageDP')]:
        p=OLD/(job+'_train')/'milestone_10000.pt'
        checkpoints[model]=dict(path=str(p.resolve()),sha256=sha(p),updates=10000,
            cpu_strict_load_passed=True,all_ema_finite=True,
            evidence='2026-11-04 supervised CPU torch.load and strict model.load_state_dict; both all keys matched')
    expected=read(OLD/'offline_freeze.json')
    # Source/data identities only; no cache re-audit or numerical test replay.
    for p,digest in expected.items():
        if p.endswith('.py') or p.endswith('/data.json'):assert sha(p)==digest, p
    video=CODE/'outputs/visuotactile/native_dp_v2/h24_diagnosis_20261104_01/videos/V3_diagnostic.mp4'
    assert video.exists()
    now=time.time();session={k:original[k] for k in ('source_root','data_sha256','expert_checkpoint','expert_sha256','resource_policy','resource_target_percent','specs')}
    session.update(run_id=root.name,started_epoch=now,deadline=now+12*3600,training_updates=0,
        maximum_starts=30,software_retry_limit=6,per_initial_maximum_attempts=2,
        checkpoints=checkpoints,supervised_model_rollouts=4,
        old_root=str(OLD.resolve()),old_owner_exit_sha256=sha(OLD/'owner_exit.json'),
        kit_gate=dict(shutdown='graceful_sdk_close',exit_on_error=False),
        expert_video=str(video.resolve()),expert_video_sha256=sha(video),
        owned_worker_roots=[str(root)],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip())
    write(root/'session.json',session)
    write(root/'freeze/benchmark.json',read(OLD/'freeze/benchmark.json'))
    user_config=Path('/home/CNS2025331827/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json')
    shutil.copyfile(user_config,root/'freeze/user.config.json')
    sources=dict(original['implementation_sha256'])
    for name in ('tools/final_dp_eval.py','tools/final_dp_worker.py','tools/final_dp_report.py','tools/state_image_kit_diagnostic.py',
                 'tools/state_image_queue.py','tools/state_image_kit_gate.py','tools/kit_resource_preflight.py','tools/h24_diagnosis_report.py','tools/h24_diagnosis_common.py'):
        sources[name]=sha(CODE/name)
    write(root/'source_freeze.json',sources)
    write(root/'status.json',dict(state='pending',stage='supervised_first_four',pid=None))
    write(root/'attempts.json',[])
    # Preserve only existing evidence copies; never mutate old attempts.
    for stage in ('state_loop_6000_08','state_loop_6000_09'):
        target=root/'previous_failure_evidence'/stage;target.mkdir(parents=True)
        for name in ('resource_preflight.json','committed.json','attempt_1.log'):
            shutil.copyfile(OLD/'stages'/stage/name,target/name)
    write(root/'startup_mitigation.json',dict(
        prior_signature='libomniclient provider_file::FileThread SIGABRT before scene',
        prior_watch_probes='256/256; no demonstrated ENOSPC in these attempts',
        change='Per-job portable Kit data/cache/log roots; identical copied persistent configuration; no shared cache removal',
        causal_claim=False,source='https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/Kernel/CommandLineOptions.html'))
    from tools.final_dp_report import report
    report(root)


def equal_values(a,b):
    import torch
    import numpy as np
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_values(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal_values(x,y) for x,y in zip(a,b))
    return a==b


class Owner:
    def __init__(self,root):
        self.root=root.resolve();self.session=read(root/'session.json')
        self.attempts=read(root/'attempts.json');self.monitor=None;self.worker=None
    def status(self,state,**more):
        write(self.root/'status.json',dict(state=state,pid=os.getpid(),heartbeat=time.time(),**more))
    def report(self,publish=False):
        subprocess.run(RUN+['-m','tools.final_dp_report','--root',str(self.root)]+(['--publish'] if publish else []),cwd=CODE,check=True)
    def run_one(self,model,ordinal):
        root=self.root;spec=self.session['specs'][ordinal]
        number=1+sum(a['model']==model and a['ordinal']==ordinal for a in self.attempts)
        assert number<=2 and len(self.attempts)<self.session['maximum_starts']
        checkpoint=self.session['checkpoints'][model]
        assert sha(checkpoint['path'])==checkpoint['sha256']
        from tools.kit_resource_preflight import inspect
        preflight=inspect(256,root)
        preflight['limitation']='Resource admission only; no repeated normal-exit prerequisite'
        tag=f'{model}_{ordinal:02d}_attempt_{number}';parent=root/'stages'/tag;parent.mkdir(parents=True,exist_ok=False)
        write(parent/'resource_preflight.json',preflight)
        if not preflight['resource_admitted']:raise RuntimeError('Resource admission blocked: '+str(preflight['reasons']))
        job=dict(id=tag,model=model,mode='dp',specs=[spec],checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],
            video=True,inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091),
            lifecycle=dict(fast_shutdown=True,disarm_stop=True))
        path=root/'jobs'/(tag+'.json');write(path,job);output=parent/'output'
        record=dict(id=tag,model=model,ordinal=ordinal,number=number,output=str(output),started=time.time())
        self.attempts.append(record);write(root/'attempts.json',self.attempts)
        observed={};timeout=None;began=time.time()
        with (parent/'worker.log').open('x') as log,(parent/'process.jsonl').open('x') as samples:
            command=RUN+['-m','tools.final_dp_worker','--root',str(root),'--output',str(output),'--job',str(path)]
            self.worker=subprocess.Popen(command,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=self.worker.pid;write(root/'attempts.json',self.attempts)
            while self.worker.poll() is None:
                if self.monitor.poll() is not None:raise RuntimeError('Continuous resource monitor exited')
                snapshots=[]
                for pid in {self.worker.pid,*descendants(self.worker.pid)}:
                    p=proc(pid)
                    if p:observed[pid]=p['start_ticks'];snapshots.append(p)
                samples.write(__import__('json').dumps(dict(utc=time.time(),processes=snapshots))+'\n');samples.flush()
                phase=read(output/'shutdown_phase.json') if (output/'shutdown_phase.json').exists() else {}
                self.status('supervised' if not (root/'supervision_accepted.json').exists() else 'evaluating',stage=tag,
                    worker_pid=self.worker.pid,completed=sum(a.get('valid',False) for a in self.attempts),planned=24)
                now=time.time()
                if now>self.session['deadline'] or (root/'STOP').exists():timeout='session_deadline_or_STOP'
                elif now-began>1200:timeout='worker_1200s'
                elif phase and now-phase['utc']>45:timeout='shutdown_45s'
                if timeout:kill_owned(self.worker);break
                time.sleep(2)
            self.worker.wait();code=self.worker.returncode;self.worker=None
        live=[pid for pid,tick in observed.items() if (p:=proc(pid)) and p['start_ticks']==tick]
        gpu=sorted(set(observed)&gpu_pids())
        relative=next(s.split('::',1)[1] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
        assert relative.endswith('/'+SERVICE),relative
        cgroup=Path('/sys/fs/cgroup')/relative.lstrip('/')
        members={int(p) for f in [cgroup/'cgroup.procs',*cgroup.glob('**/cgroup.procs')] for p in f.read_text().split()}
        unexpected=members-{os.getpid(),self.monitor.pid,*descendants(self.monitor.pid)}
        record.update(finished=time.time(),exit_code=code,timeout=timeout,remaining_owned_pids=live,
            remaining_owned_gpu_pids=gpu,unexpected_cgroup_pids=sorted(unexpected),resources_released=not(live or gpu or unexpected),reaped=True,valid=False)
        if not record['resources_released']:write(root/'attempts.json',self.attempts);raise RuntimeError('Owned worker resource release failed')
        result=read(output/'result.json') if (output/'result.json').exists() else {}
        if result.get('software_complete') and not (output/'worker_error.json').exists():
            import numpy as np
            trials=result.get('trials',[]);assert len(trials)==1
            trial=trials[0];record['trial']=trial
            if trial.get('legal'):
                assert trial['checkpoint_sha256']==checkpoint['sha256']
                assert trial['environment_sha256']==read(root/'freeze/benchmark.json')['configuration_sha256']
                assert not trial['action_exhausted'] and trial['actuator_target_max_error']==0.
                assert trial['submitted_target_max_error']==0.
                audit=read(output/'controller_audit.json')
                assert audit['completed'] and audit['act_calls']==trial['steps'] and audit['execution_acks']==trial['steps']
                folder=output/trial['identity'];trace=folder/'trace.npz'
                with np.load(trace) as z:
                    assert len(z['joint'])==trial['steps'] and all(np.isfinite(z[k]).all() for k in z.files)
                video=folder/'rollout.mp4'
                # Decode every frame, not just inspect the container header.
                subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                frames=int(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(video)],text=True))
                assert frames==trial['steps']
                for file in folder.iterdir():
                    if file.is_file():
                        with file.open('rb') as stream:os.fsync(stream.fileno())
                record.update(valid=True,trace_sha256=sha(trace),controller_audit=audit,
                    video=dict(path=str(video),sha256=sha(video),decoded_frames=frames),result_sha256=sha(output/'result.json'))
        record['classification']=classify(output,code,timeout,(parent/'worker.log').read_text(errors='replace'),record['valid'])
        write(parent/'committed.json',record);write(root/'attempts.json',self.attempts)
        if record['valid']:self.analyze_one(record)
        self.check_pairs();self.report()
        return record
    def check_pairs(self):
        import torch
        pairs=[]
        for i in range(12):
            rows=[a for a in self.attempts if a['ordinal']==i and a.get('valid')]
            if len(rows)!=2:continue
            paths=[Path(a['output'])/a['trial']['identity'] for a in rows]
            observable=read(paths[0]/'initial_observable.json')==read(paths[1]/'initial_observable.json')
            rng=equal_values(*[torch.load(p/'environment_rng_initial.pt',map_location='cpu',weights_only=False) for p in paths])
            # Actual physical properties and delay buffers exported by the old runner.
            initial=equal_values(*[torch.load(p/'initial_export.pt',map_location='cpu',weights_only=False) for p in paths])
            pairs.append(dict(ordinal=i,passed=observable and rng and initial,observable_equal=observable,
                environment_rng_equal=rng,initial_export_equal=initial))
        write(self.root/'pairs.json',pairs)
        if any(not p['passed'] for p in pairs):raise RuntimeError('Paired initial-state contract mismatch; preserve both traces')
    def analyze_one(self,record):
        import numpy as np
        from tools.h24_diagnosis_report import describe
        folder=Path(record['output'])/record['trial']['identity']
        summary,series,trace=describe(folder)
        details=read(self.root/'failure_analysis.json') if (self.root/'failure_analysis.json').exists() else []
        details.append(dict(model=record['model'],ordinal=record['ordinal'],cohort=record['trial']['cohort'],
            success=record['trial']['success'],reasons=record['trial']['reasons'],metrics=summary,
            before_termination=dict(last_ticks=min(30,len(trace['joint'])),
                object_z_m=trace['object'][-30:,2].tolist(),contact_force_N=series['object_contact_N'][-30:].tolist(),
                hand_target_jump_rad=series['hand_jump_rad'][-30:].tolist(),arm_target_jump_rad=series['arm_jump_rad'][-30:].tolist()),
            maximum_hand_jump=dict(tick=int(np.argmax(series['hand_jump_rad'])),rad=float(np.max(series['hand_jump_rad'])))))
        write(self.root/'failure_analysis.json',details)
    def run(self):
        root=self.root
        for p,digest in read(root/'source_freeze.json').items():assert sha(CODE/p)==digest,p
        assert sha(OLD/'owner_exit.json')==self.session['old_owner_exit_sha256']
        with (root/'monitor.log').open('a') as log:
            self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            failed_in_row=[]
            for ordinal in range(12):
                for model in ('StateDP','ImageDP'):
                    while not any(a.get('valid') and a['model']==model and a['ordinal']==ordinal for a in self.attempts):
                        prior=[a for a in self.attempts if a['model']==model and a['ordinal']==ordinal]
                        if len(prior)>=2:break
                        if prior and sum(a['number']>1 for a in self.attempts)>=6:break
                        record=self.run_one(model,ordinal)
                        if record['valid']:failed_in_row=[]
                        else:
                            failed_in_row.append(record['classification'])
                            if len(failed_in_row)>=2 and failed_in_row[-1]==failed_in_row[-2]:
                                raise RuntimeError('Two consecutive '+record['classification']+'; targeted repair required before further starts')
                            if record['classification']!='startup_FileThread_before_scene':
                                raise RuntimeError('Non-startup fault requires targeted repair: '+record['classification'])
                if ordinal==1:
                    first=[a for a in self.attempts if a['ordinal']<2 and a.get('valid')]
                    assert len(first)==4,'First two per model not valid'
                    self.status('awaiting_supervised_review',stage='first_four_complete',completed=4,planned=24)
                    self.report()
                    while not (root/'supervision_accepted.json').exists():
                        if time.time()>self.session['deadline'] or (root/'STOP').exists():raise RuntimeError('Supervised review wait ended at session bound')
                        time.sleep(3)
                    assert read(root/'supervision_accepted.json')['passed'] is True
            count=sum(a.get('valid',False) for a in self.attempts)
            self.status('completed' if count==24 else 'software_missing',stage='final',completed=count,planned=24)
        except BaseException as e:
            if self.worker:kill_owned(self.worker)
            write(root/'owner_error.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=time.time()))
            self.status('blocked',stage=read(root/'status.json').get('stage'),reason=str(e),completed=sum(a.get('valid',False) for a in self.attempts),planned=24)
        finally:
            (root/'STOP_MONITOR').touch()
            try:self.monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:kill_owned(self.monitor)
            final=read(root/'status.json');final.update(pid=None,worker_pid=None,heartbeat=None,finished=time.time())
            write(root/'status.json',final);write(root/'owner_exit.json',final)
            try:self.report(publish=True)
            except BaseException as e:write(root/'publication.json',dict(state='blocked',reason=repr(e)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    else:
        lock=(a.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (a.root/'owner_exit.json').exists(),'Ended run requires audited recovery, no automatic budget reset'
        Owner(a.root).run()
