"""Persistent bounded diagnostics owner. No optimizer or old-owner restart."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from tools.state_image_queue import CODE, RUN, read, write, kill_owned, sha
from tools.state_image_kit_gate import proc, descendants, gpu_pids


def run_ablations(root,monitor):
    import numpy as np
    from tools.kit_resource_preflight import inspect
    from tools.final_dp_eval import classify
    from tools.final_dp_video import verify_video
    from tools.h24_diagnosis_report import compare_initial,describe
    session=read(root/'session.json')
    assert read(root/'replay_evidence.json')['passed']
    assert read(root/'analysis.json')['identical_recorded_initials']
    assert read(root/'parity_precision.json')['passed']
    for p,digest in read(root/'source_freeze_ablations.json').items():assert sha(CODE/p)==digest,p
    baseline=[x for x in read(Path(session['baseline_root'])/'attempts.json') if x.get('valid')]
    attempts=read(root/'ablation_attempts.json') if (root/'ablation_attempts.json').exists() else []
    failures=[]
    for variant in ('precision_only','precision_K4'):
        for ordinal in (0,4):
            spec=session['specs'][ordinal]
            for model in ('StateDP','ImageDP'):
                for number in (1,2):
                    if any(x.get('valid') and x['variant']==variant and x['ordinal']==ordinal and x['model']==model for x in attempts):break
                    assert len(attempts)<10,'Total bounded startup budget exhausted'
                    if number==2 and sum(x['number']==2 for x in attempts)>=2:raise RuntimeError('Two recoveries exhausted')
                    cp=session['checkpoints'][model];assert sha(cp['path'])==cp['sha256']
                    tag=f'{variant}_{model}_{ordinal:02d}_attempt_{number}'
                    preflight=inspect(256,root);write(root/'preflights'/(tag+'.json'),preflight)
                    assert preflight['resource_admitted'],preflight['reasons']
                    job=dict(model=model+' '+variant,mode='dp',specs=[spec],checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],
                        video=True,inference=dict(inference_steps=16,clip_sample=False,sampler_seed=7091,overlap_steps=0),
                        ablation=variant,lifecycle=dict(fast_shutdown=True,disarm_stop=True))
                    path=root/'jobs'/(tag+'.json');write(path,job);output=root/'stages'/tag/'output'
                    record=dict(variant=variant,model=model,ordinal=ordinal,number=number,tag=tag,output=str(output),
                        started=time.time(),valid=False)
                    attempts.append(record);write(root/'ablation_attempts.json',attempts)
                    exit_result=stage(root,monitor,tag,'tools.grasp_diagnosis_worker',['--output',str(output),'--job',str(path)])
                    record.update(exit_result)
                    result=read(output/'result.json') if (output/'result.json').exists() else {}
                    if result.get('software_complete') and not (output/'worker_error.json').exists():
                        assert len(result['trials'])==1;trial=result['trials'][0];folder=output/trial['identity']
                        assert trial['legal'] and not trial['action_exhausted'] and trial['checkpoint_sha256']==cp['sha256']
                        assert trial['environment_sha256']==read(root/'freeze/benchmark.json')['configuration_sha256']
                        assert trial['actuator_target_max_error']==trial['submitted_target_max_error']==0.
                        audit=read(output/'controller_audit.json');assert audit['completed'] and audit['act_calls']==trial['steps']
                        with np.load(folder/'trace.npz') as trace:assert all(np.isfinite(trace[k]).all() for k in trace.files)
                        original=next(x for x in baseline if x['ordinal']==ordinal and x['model']==model)
                        initial=compare_initial(folder,Path(original['output'])/trial['identity'])
                        assert initial['observable_equal'] and initial['environment_rng_equal'] and not initial['exported_differences']
                        video=verify_video(folder/'rollout.mp4',trial['steps'])
                        summary,_,_=describe(folder)
                        record.update(valid=True,trial=trial,initial=initial,video=video,metrics=summary,
                            trace_sha256=sha(folder/'trace.npz'),ablation_plans_sha256=sha(folder/'ablation_plans.npz'))
                        if variant=='precision_K4':
                            ref=next(x for x in attempts if x.get('valid') and x['variant']=='precision_only' and x['ordinal']==ordinal and x['model']==model)
                            with np.load(Path(ref['output'])/trial['identity']/'trace.npz') as a,np.load(folder/'trace.npz') as b:
                                prefix={k:np.array_equal(a[k][:12],b[k][:12]) for k in ('command','target','joint','object','actor','object_contact')}
                            record['unchanged_first12_prefix']=prefix
                            assert all(prefix.values()),'Single-factor pre-intervention prefix mismatch'
                    record['classification']=classify(output,record['exit_code'],record['timeout'],
                        (output.parent/'worker.log').read_text(errors='replace'),record['valid'])
                    write(root/'ablation_attempts.json',attempts);write(output.parent/'committed.json',record)
                    from tools.grasp_diagnosis_report import report
                    report(root)
                    if record['valid']:
                        failures=[];break
                    failures.append(record['classification'])
                    if len(failures)>=2 and failures[-1]==failures[-2]:raise RuntimeError('Two consecutive same-stage crashes; no blind restart')
                    if record['classification']!='startup_FileThread_before_scene':raise RuntimeError('Diagnostic worker fault; inspect evidence')
                if not any(x.get('valid') and x['variant']==variant and x['ordinal']==ordinal and x['model']==model for x in attempts):
                    raise RuntimeError('Per-condition recovery exhausted')
    write(root/'ablation_complete.json',dict(valid=sum(x.get('valid',False) for x in attempts),planned=8,attempts=len(attempts),training_updates=0))
    return 'completed'


def stage(root, monitor, name, module, arguments, limit=1200):
    folder=root/'stages'/name;folder.mkdir(parents=True,exist_ok=False)
    output=folder/'output';observed={};began=time.time();session=read(root/'session.json');reason=None
    with (folder/'worker.log').open('x') as log, (folder/'process.jsonl').open('x') as samples:
        worker=subprocess.Popen(RUN+['-m',module,'--root',str(root),*arguments],cwd=CODE,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(folder/'launch.json',dict(pid=worker.pid,started=began,command=worker.args))
        try:
            while worker.poll() is None:
                current=[]
                for pid in {worker.pid,*descendants(worker.pid)}:
                    item=proc(pid)
                    if item:observed[pid]=item['start_ticks'];current.append(item)
                samples.write(__import__('json').dumps(dict(utc=time.time(),processes=current))+'\n');samples.flush()
                write(root/'status.json',dict(state='running',stage=name,pid=os.getpid(),worker_pid=worker.pid,
                    heartbeat=time.time(),training_updates=0))
                phase=read(output/'shutdown_phase.json') if (output/'shutdown_phase.json').exists() else {}
                if monitor.poll() is not None:reason='monitor_exited'
                elif (root/'STOP').exists() or time.time()>session['deadline']:reason='STOP_or_deadline'
                elif time.time()-began>limit:reason='worker_timeout'
                elif phase and time.time()-phase['utc']>45:reason='shutdown_timeout'
                if reason:kill_owned(worker);break
                time.sleep(2)
            worker.wait()
        finally:
            if worker.poll() is None:kill_owned(worker)
        live=[pid for pid,start in observed.items() if (p:=proc(pid)) and p['start_ticks']==start]
        gpu=sorted(set(observed)&gpu_pids())
        result=dict(name=name,exit_code=worker.returncode,timeout=reason,finished=time.time(),
            remaining_owned_pids=live,remaining_owned_gpu_pids=gpu,resources_released=not(live or gpu))
        write(folder/'exit.json',result)
        assert result['resources_released'],'Owned process resource leak'
        return result


def run(root, mode):
    root=root.resolve();monitor=None
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOP_MONITOR').exists(),'Explicit recovery must preserve prior monitor evidence'
        with (root/'monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if mode in ('parity','parity_precision'):
                number=1+len(list((root/'stages').glob('parity_*')))
                result=stage(root,monitor,f'parity_{number:02d}','tools.grasp_diagnosis_parity',
                    ['--precision'] if mode=='parity_precision' else [],1800)
                assert result['exit_code']==0, 'Numerical audit worker failed; inspect retained log'
                parity=read(root/(mode+'.json'))
                state='parity_passed' if parity['passed'] else 'contract_failed'
            else:
                state=run_ablations(root,monitor)
            write(root/'status.json',dict(state=state,stage=mode,pid=None,worker_pid=None,finished=time.time(),training_updates=0))
            if mode=='ablations':
                from tools.grasp_diagnosis_report import report
                report(root)
        except BaseException:
            write(root/('owner_error_'+mode+'.json'),dict(error=traceback.format_exc(),utc=time.time()))
            write(root/'status.json',dict(state='blocked',stage=mode,pid=None,worker_pid=None,training_updates=0))
            raise
        finally:
            (root/'STOP_MONITOR').touch()
            try:monitor.wait(timeout=8)
            except subprocess.TimeoutExpired:kill_owned(monitor)
            write(root/('owner_exit_'+mode+'.json'),dict(pid=os.getpid(),finished=time.time(),monitor_exit=monitor.returncode,
                state=read(root/'status.json')['state']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','parity_precision','ablations']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.root,a.mode)
