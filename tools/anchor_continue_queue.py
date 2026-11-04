"""Single persistent owner: supervised resume, anchor milestones, isolated evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.state_image_queue import CODE,RUN,read,write,sha,kill_owned
from tools.state_image_kit_gate import proc,descendants,gpu_pids


class Owner:
    def __init__(self,root):
        self.root=root.resolve();self.s=read(root/'session.json');self.worker=None;self.monitor=None
        self.attempts=read(root/'attempts.json') if (root/'attempts.json').exists() else []
    def status(self,**kw):
        state=read(self.root/'status.json');state.update(pid=os.getpid(),heartbeat=time.time());state.update(kw);write(self.root/'status.json',state)
    def report(self):
        subprocess.run(RUN+['-m','tools.anchor_continue_report','--root',str(self.root)],cwd=CODE,check=True)
    def stage(self,name,module,args,kit=False):
        root=self.root;committed=root/'stages'/name/'committed.json'
        if committed.exists():return read(committed)
        for recovery in range(1 if kit else 3):
            number=1+sum(x['name']==name for x in self.attempts)
            if number>(2 if kit else 3):raise RuntimeError('Bounded recovery exhausted: '+name)
            if kit:
                used=[x for x in self.attempts if x['kit']]
                assert len(used)<self.s['kit_maximum_starts']
                from tools.kit_resource_preflight import inspect
                preflight=inspect(256,root);write(root/'preflights'/f'{name}_{number}.json',preflight)
                if not preflight['resource_admitted']:raise RuntimeError('Kit resource admission failed; training preserved')
            folder=root/'stages'/name/f'attempt_{number}';folder.mkdir(parents=True,exist_ok=False)
            output=folder/'output';began=time.time();observed={};reason=None
            record=dict(name=name,number=number,kit=kit,output=str(output),started=began,valid=False)
            self.attempts.append(record);write(root/'attempts.json',self.attempts)
            with (folder/'worker.log').open('x') as log,(folder/'process.jsonl').open('x') as samples:
                self.worker=subprocess.Popen(RUN+['-m',module,'--root',str(root),'--output',str(output),*args],cwd=CODE,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=self.worker.pid;write(root/'attempts.json',self.attempts)
                try:
                    while self.worker.poll() is None:
                        current=[]
                        for pid in {self.worker.pid,*descendants(self.worker.pid)}:
                            p=proc(pid)
                            if p:observed[pid]=p['start_ticks'];current.append(p)
                        samples.write(json.dumps(dict(utc=time.time(),processes=current))+'\n');samples.flush()
                        self.status(state='evaluating' if kit else 'running',stage=name,worker_pid=self.worker.pid)
                        phase=read(output/'shutdown_phase.json') if (output/'shutdown_phase.json').exists() else {}
                        if self.monitor.poll() is not None:reason='monitor_exited'
                        elif (root/'STOP').exists() or time.time()>self.s['deadline']:reason='STOP_or_deadline'
                        elif kit and time.time()-began>1200:reason='Kit1200s'
                        elif phase and time.time()-phase['utc']>45:reason='shutdown45s'
                        elif not kit and time.time()-began>1800:
                            progress=root/(name.split('_')[0]+'_train')/'progress.json'
                            if not progress.exists() or time.time()-progress.stat().st_mtime>1800:
                                if not list(root.glob('wait_*.json')):reason='offline_no_progress1800s'
                        if reason:kill_owned(self.worker);break
                        time.sleep(2)
                    self.worker.wait();code=self.worker.returncode
                finally:
                    if self.worker.poll() is None:kill_owned(self.worker)
                self.worker=None
            live=[pid for pid,t in observed.items() if (p:=proc(pid)) and p['start_ticks']==t]
            gpu=sorted(set(observed)&gpu_pids());assert not live and not gpu,'Owned worker resource leak'
            result=read(output/'result.json') if (output/'result.json').exists() else {}
            record.update(exit_code=code,timeout=reason,finished=time.time(),resources_released=True,
                remaining_owned_pids=live,remaining_owned_gpu_pids=gpu)
            record['valid']=bool(result.get('software_complete' if kit else 'passed')) and not (output/'worker_error.json').exists()
            record['result']=result;write(folder/'exit.json',record);write(root/'attempts.json',self.attempts)
            if record['valid']:
                if kit:self.validate_kit(record)
                write(committed,record);self.report();return record
            if kit:
                # A cold-cache timeout is retained, never blindly retried.
                from tools.final_dp_eval import classify
                record['classification']=classify(output,code,bool(reason),(folder/'worker.log').read_text(errors='replace'),False)
                write(root/'attempts.json',self.attempts)
                recent=[x for x in self.attempts if x['kit']][-2:]
                if len(recent)==2 and all(x.get('classification')==record['classification'] for x in recent):
                    raise RuntimeError('Two same-stage Kit failures; specific repair required')
                recoveries=sum(x['kit'] and x['number']>1 for x in self.attempts)
                if record['classification']=='startup_FileThread_before_scene' and recoveries<2 and number==1:
                    return self.stage(name,module,args,kit=True)
                raise RuntimeError('Kit result missing; retained '+str(folder))
            log=(folder/'worker.log').read_text(errors='replace')
            if reason or 'Nonfinite' in log or 'FloatingPointError' in log:raise RuntimeError('Unsafe automatic continuation; retained '+str(folder))
            if 'train' not in name or not (root/(name.split('_')[0]+'_train')/'last.pt').exists():
                raise RuntimeError('No resumable offline checkpoint: '+name)
            # Resumption repeats only uncommitted steps, with original optimizer/RNG.
        raise RuntimeError('Offline retries exhausted')
    def validate_kit(self,record):
        import numpy as np
        from tools.final_dp_video import verify_video
        from tools.h24_diagnosis_report import compare_initial,describe
        result=record['result'];assert len(result['trials'])==1;trial=result['trials'][0]
        assert trial['legal'] and not trial['action_exhausted']
        assert trial['actuator_target_max_error']==trial['submitted_target_max_error']==0
        folder=Path(record['output'])/trial['identity']
        with np.load(folder/'trace.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        job=read(self.root/'jobs'/(record['name']+'.json'))
        model='StateDP' if job['model'].startswith('StateDP') else 'ImageDP'
        original=next(x for x in read(Path(self.s['baseline_root'])/'attempts.json') if x.get('valid') and x['trial']['identity']==trial['identity'] and x['model']==model)
        initial=compare_initial(folder,Path(original['output'])/trial['identity'])
        assert initial['observable_equal'] and initial['environment_rng_equal'] and not initial['exported_differences']
        record['initial']=initial;record['video']=verify_video(folder/'rollout.mp4',trial['steps'])
        record['metrics']=describe(folder)[0];record['trace_sha256']=sha(folder/'trace.npz')
        record['native_outcome_valid']=True;record['sdk_exit']=dict(code=record['exit_code'],graceful=(Path(record['output'])/'closed.json').exists(),fast_shutdown=True)
    def train_job(self,job):
        self.stage(job+'_contract','visuotactile.anchor_continue',['contract','--job',job])
        for n in (5,15,50):self.stage(f'{job}_train_smoke_{n}','visuotactile.anchor_continue',['train','--job',job,'--target',str(n)])
        if job=='state':
            self.status(state='supervised_ready',stage='state50_waiting_review',worker_pid=None)
            self.report()
            while not (self.root/'SUPERVISION_ACCEPTED.json').exists():
                if time.time()>self.s['deadline'] or (self.root/'STOP').exists():raise RuntimeError('Stopped before formal continuation release')
                time.sleep(2)
            assert read(self.root/'SUPERVISION_ACCEPTED.json')['passed']
        for epoch in self.s['milestones']:
            target=self.s['targets'][job][str(epoch)]
            self.stage(f'{job}_train_epoch_{epoch}','visuotactile.anchor_continue',['train','--job',job,'--target',str(target)])
            metrics=[x for x in read(self.root/(job+'_train')/'metrics.json') if x['validation_scope']=='validation40']
            original=read(Path(self.s['old_root'])/(job+'_train')/'metrics.json')[-1]['validation']['selection_mse']
            serious=len(metrics)>=2 and all(x['validation']['selection_mse']>original*1.2 for x in metrics[-2:])
            plateau=False
            if epoch>=150 and len(metrics)>=3:
                values=[x['validation']['selection_mse'] for x in metrics[-3:]]
                plateau=all((a-b)/a<.005 for a,b in zip(values,values[1:]))
            if serious or plateau:
                write(self.root/(job+'_stop_decision.json'),dict(epoch=epoch,reason='two_validations_20percent_regression' if serious else 'two_intervals_under_0.5percent',
                    select_best=True,values=[x['validation']['selection_mse'] for x in metrics[-3:]],no_discarded_checkpoint=True))
                break
        self.report()
    def kit(self,name,job):
        if not (self.root/'private_runtime').exists():
            source=CODE/'outputs/visuotactile/native_dp_v2/grasp_diagnosis_20261104_01/private_runtime'
            shutil.copytree(source,self.root/'private_runtime')
        path=self.root/'jobs'/(name+'.json');write(path,job)
        return self.stage(name,'tools.anchor_continue_worker',['--job',str(path)],kit=True)
    def evaluate(self,job):
        best=read(self.root/(job+'_train')/'best.json');model='StateDP' if job=='state' else 'ImageDP';rows=[]
        for ordinal in self.s['closed_loop_ordinals']:
            spec=self.s['specs'][ordinal]
            payload=dict(model=model,updates=best['update'],mode='dp',specs=[spec],checkpoint=best['path'],checkpoint_sha256=best['sha256'],
                video=True,inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091,overlap_steps=0),
                lifecycle=dict(fast_shutdown=True,disarm_stop=True),capture_pair=False)
            rows.append(self.kit(f'{job}_evaluation_{ordinal:02d}',payload))
        original=[x for x in read(Path(self.s['baseline_root'])/'attempts.json') if x.get('valid') and x['model']==model and x['ordinal'] in self.s['closed_loop_ordinals']]
        trials=[x['result']['trials'][0] for x in rows]
        improved=any(t['success'] or t['native_subgoals'] for t in trials) or sum(t['native_lift'] for t in trials)>sum(x['trial']['native_lift'] for x in original)
        summary=dict(valid=len(rows),planned=4,successes=sum(t['success'] for t in trials),improved=improved,
            rule=self.s['state_improvement_rule'],selected_checkpoint=best,rows=rows,original_reference=[dict(ordinal=x['ordinal'],trial=x['trial']) for x in original])
        write(self.root/(job+'_closed_loop.json'),summary)
        if improved:
            # Complete the existing frozen twelve, not new or outcome-selected seeds.
            for ordinal,spec in enumerate(self.s['specs']):
                if ordinal in self.s['closed_loop_ordinals']:continue
                payload=dict(payload,specs=[spec])
                rows.append(self.kit(f'{job}_evaluation_{ordinal:02d}',payload))
                summary.update(valid=len(rows),planned=12,rows=rows,successes=sum(x['result']['trials'][0]['success'] for x in rows))
                write(self.root/(job+'_closed_loop.json'),summary)
        self.report();return improved
    def image_pair(self):
        from tools.anchor_continue_pair import compare
        cp=self.s['checkpoints']['image'];rows=[]
        for n in (1,2):
            rows.append(self.kit(f'image_pair_{n}',dict(model='ImageDP',updates=10000,mode='dp',specs=[self.s['specs'][4]],
                checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],video=True,capture_pair=True,
                inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091,overlap_steps=0),lifecycle=dict(fast_shutdown=True,disarm_stop=True))))
        result=compare(self.root,rows);self.report();return result['passed']
    def run(self):
        root=self.root
        with (root/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for p,digest in read(root/'runtime_freeze.json').items():assert sha(CODE/p)==digest,'Source drift: '+p
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            with (root/'monitor.log').open('a') as log:
                self.monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                self.train_job('state')
                if not self.evaluate('state'):
                    self.status(state='completed_state_no_closed_loop_improvement',stage='image_pending_state_evidence',worker_pid=None)
                    return
                if not self.image_pair():
                    self.status(state='blocked_image_pair',stage='image_not_trained',worker_pid=None)
                    return
                self.train_job('image');self.evaluate('image')
                self.status(state='completed',stage='all_authorized_stages',worker_pid=None)
            except BaseException:
                write(root/'owner_error.json',dict(error=traceback.format_exc(),utc=time.time()))
                self.status(state='blocked',worker_pid=None)
                raise
            finally:
                if self.worker:kill_owned(self.worker)
                (root/'STOP_MONITOR').touch()
                try:self.monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(self.monitor)
                self.status(pid=None,worker_pid=None,finished=time.time())
                write(root/'owner_exit.json',read(root/'status.json'));self.report()
                subprocess.run(RUN+['-m','tools.anchor_continue_report','--root',str(root),'--publish'],cwd=CODE)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();Owner(a.root).run()
