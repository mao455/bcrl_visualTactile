"""Bounded H24 parameterization experiment using the existing trainer and workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from visuotactile.compact import sha256
from visuotactile.state_full_run import Runner, startup_failure_evidence
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def stop_owned(process):
    if process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def verify_frozen(root):
    prepared=read(root/'setup_complete.json')
    if prepared['plan_sha256']!=sha256(root/'run_plan.json') or prepared['session_sha256']!=sha256(root/'session.json'):
        raise ValueError('Immutable plan/session drift')
    from visuotactile.state_history_training import require_foundation
    require_foundation(root)


def recover_interrupted_training(root, name, directory):
    """One recovery from a durable checkpoint; retain every original attempt/log."""
    import torch
    archive = root/'recovery'/name
    checkpoint = directory/'last.pt'
    if archive.exists() or not checkpoint.exists():
        raise RuntimeError('Recovery bound reached or no durable checkpoint: '+name)
    exited = root/(name+'_process_exit.json')
    if exited.exists() and read(exited)['code'] not in (-15, -9, None):
        raise RuntimeError('Unclassified training error requires review: '+name)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = {'optimizer','ema','model','ema_optimization_step','schedule_rng','noise_rng',
        'torch_rng','cuda_rng','schedule_order','schedule_cursor','history'}
    if not required <= saved.keys() or saved['config_sha256'] != sha256(directory/'configuration.json'):
        raise ValueError('Incomplete or incompatible recovery checkpoint')
    archive.mkdir(parents=True)
    (root/name).rename(archive/'attempt')
    for suffix in ('.log','_launch.json','_process_exit.json','_exit.json','_watchdog.json'):
        path=root/(name+suffix)
        if path.exists():path.rename(archive/path.name)
    # An abrupt stop may leave log rows beyond the last atomic checkpoint. Keep
    # the original log and replay only those uncommitted optimizer steps.
    log=directory/'train.jsonl'
    if log.exists():
        shutil.copy2(log,archive/'train_before_resume.jsonl')
        retained=[]
        for line in log.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row['update']<=saved['updates']:retained.append(line)
        updates=[json.loads(line)['update'] for line in retained]
        if updates != list(range(1,saved['updates']+1)):
            raise ValueError('Committed training log is not contiguous')
        temporary=directory/'train.resume.tmp'
        temporary.write_text('\n'.join(retained)+'\n');temporary.replace(log)
    atomic_json(archive/'recovery.json',dict(from_updates=saved['updates'],
        checkpoint_sha256=sha256(checkpoint),deadline_unchanged=True,maximum_recoveries=1,utc=time.time()))


def worker(root, name, module, extra, cwd=None, worker_root=None, trace=False, timeout=900):
    output = root/name
    marker = root/(name+'_committed.json')
    if marker.exists():
        if sha256(output/'result.json') != read(marker)['sha256']:
            raise ValueError('Committed result changed: '+name)
        return read(output/'result.json')
    session = read(root/'session.json')
    if (root/'setup_complete.json').exists():verify_frozen(root)
    if output.exists():
        status=read(root/'status.json') if (root/'status.json').exists() else {}
        pid=status.get('worker_pid') if status.get('stage')==name else None
        if pid and Path('/proc',str(pid),'cmdline').exists():
            command=Path('/proc',str(pid),'cmdline').read_bytes()
            if str(output).encode() in command:raise RuntimeError('Existing stage worker is still alive')
        result_path=output/'result.json'
        if result_path.exists() and read(result_path).get('software_complete') and not (output/'worker_error.json').exists():
            if module=='visuotactile.state_benchmark':Runner(root).verify_worker_exit(name)
            atomic_json(marker,dict(sha256=sha256(result_path),recovered_completion=True,utc=time.time()))
            return read(result_path)
        if module=='visuotactile.state_full_training':
            profile=extra[extra.index('--profile')+1]
            recover_interrupted_training(root,name,(worker_root or root)/(profile+'_train'))
        else:
            raise RuntimeError('Preserve existing attempt without retry: '+name)
    deadline = min(session['deadline'], time.time()+timeout)
    if module=='visuotactile.state_full_training':
        target=int(extra[extra.index('--target')+1])
        if target>read(root/'run_plan.json')['maximum_cumulative_updates']:
            raise ValueError('Owner cumulative update ceiling exceeded')
        deadline=min(deadline,session['training_deadline'])
    if time.time()>=deadline or (root/'STOP').exists():raise TimeoutError('Fixed deadline or STOP before launch')
    command = ['bash', 'visuotactile/run_cpu.sh', '-m', module, '--root', str(worker_root or root),
        '--output', str(output), *extra]
    if trace:
        command = ['strace', '-f', '-e', 'trace=inotify_init1,inotify_add_watch,inotify_rm_watch,prlimit64',
            '-o', str(root/(name+'.strace')), *command]
    atomic_json(root/(name+'_launch.json'), dict(command=command, cwd=str(cwd or Path.cwd()), utc=time.time()))
    with (root/(name+'.log')).open('a') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        atomic_json(root/'status.json', dict(state='running', stage=name, pid=os.getpid(),
            worker_pid=process.pid, utc=time.time()))
        last_heartbeat=time.time()
        try:
            while process.poll() is None:
                if time.time()-last_heartbeat>=60:
                    progress_path=(worker_root or root)/(extra[extra.index('--profile')+1]+'_train/progress.json') if '--profile' in extra else output/'progress.json'
                    progress=read(progress_path) if progress_path.exists() else {}
                    atomic_json(root/'status.json',dict(state='running',stage=name,pid=os.getpid(),worker_pid=process.pid,
                        updates=progress.get('updates'),episodes_complete=progress.get('episodes_complete'),utc=time.time()))
                    last_heartbeat=time.time()
                error = output/'worker_error.json'
                lifecycle = read(output/'lifecycle.json') if (output/'lifecycle.json').exists() else {}
                reason = None
                if time.time() >= deadline or (root/'STOP').exists():reason='fixed deadline or STOP'
                if error.exists() and time.time()-error.stat().st_mtime > 10:reason='durable error cleanup stalled'
                if lifecycle.get('state')=='closing' and time.time()-lifecycle['utc'] > 30:reason='closing stalled'
                if reason:
                    atomic_json(root/(name+'_watchdog.json'), dict(reason=reason, utc=time.time(), retry=False))
                    stop_owned(process)
                    break
                time.sleep(1)
        finally:
            stop_owned(process)
        requested=read(output/'process_exit_requested.json') if (output/'process_exit_requested.json').exists() else {}
        atomic_json(root/(name+'_process_exit.json'), dict(pid=requested.get('pid',process.pid) if trace else process.pid,
            launcher_pid=process.pid, code=process.returncode, reaped=True, utc=time.time()))
    if process.returncode or not (output/'result.json').exists() or (output/'worker_error.json').exists():
        atomic_json(root/(name+'_exit.json'), dict(error=f'{name} failed: exit={process.returncode}', utc=time.time()))
        if (module=='visuotactile.state_full_training' and process.returncode in (-15,-9)
                and time.time()<deadline and not (root/'STOP').exists()):
            return worker(root,name,module,extra,cwd,worker_root,trace,max(1,deadline-time.time()))
        raise RuntimeError(f'{name} failed: exit={process.returncode}')
    result = read(output/'result.json')
    if not result.get('software_complete'):
        raise RuntimeError('Incomplete result: '+name)
    if module=='visuotactile.state_benchmark':
        instance=Runner(root);instance.verify_worker_exit(name)
    atomic_json(marker, dict(sha256=sha256(output/'result.json'), utc=time.time()))
    return result


def monitored(root, action):
    (root/'STOP_MONITOR').unlink(missing_ok=True)
    with (root/'monitor.log').open('a') as stream:
        monitor=subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m','visuotactile.state_resources',
            '--root',str(root)], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        error_seen=None
        try:
            action()
        except BaseException as error:
            error_seen=error
            import traceback
            atomic_json(root/'blocked.json',dict(reason=repr(error),utc=time.time(),traceback=traceback.format_exc()))
            atomic_json(root/'status.json',dict(state='blocked',reason=repr(error),utc=time.time()))
        finally:
            (root/'STOP_MONITOR').touch()
            try:monitor.wait(timeout=20)
            except subprocess.TimeoutExpired:stop_owned(monitor)
            atomic_json(root/'owner_exit.json',dict(utc=time.time(),monitor_exited=True))
            if (root/'setup_complete.json').exists():
                publish(root)
        if error_seen is not None:raise error_seen


def kit_gate(root):
    session=read(root/'session.json')
    for name, specs in [('kit_gate_02',session['motion_specs'][:2]),
                        ('kit_gate_03',list(reversed(session['motion_specs'][:2])))]:
        if time.time() >= session['gate_deadline']:
            raise TimeoutError('Fixed gate deadline')
        job=root/(name+'_job.json');atomic_json(job,dict(specs=specs,video=False))
        result=worker(root,name,'visuotactile.state_benchmark',['--mode','expert','--job',str(job)],trace=True)
        if len(result['trials']) != 2 or not all(row.get('legal') and row.get('success') for row in result['trials']):
            raise RuntimeError('Full expert episodes did not pass')
    pairs=[]
    previous=Path(read(root/'run_plan.json')['old_root'])
    original=Path(read(previous/'session.json')['previous_root'])/'expert_learning'
    for spec in session['motion_specs'][:2]:
        identity=spec['identity'];directories=[root/name/identity for name in ('kit_gate_02','kit_gate_03')]
        directories.append(original/identity)
        initial_equal=all(read(directory/'initial_observable.json')==read(directories[0]/'initial_observable.json') for directory in directories)
        with np.load(directories[0]/'trace.npz') as first, np.load(directories[1]/'trace.npz') as second, np.load(directories[2]/'trace.npz') as historical:
            fields={key:np.array_equal(first[key],second[key]) and np.array_equal(first[key],historical[key])
                for key in ('actor','command','target','joint','object','lifted','subgoals','alignment')}
        pairs.append(dict(identity=identity,initial_equal=initial_equal,fields=fields))
    persistent=all(row['initial_equal'] and all(row['fields'].values()) for row in pairs)
    atomic_json(root/'kit_acceptance.json',dict(passed=True,successful_full_episodes=4,
        persistent_reset_parity_passed=persistent,pairs=pairs,
        evaluation_mode='persistent_cohort' if persistent else 'isolated_singleton',
        sdk_root_cause_fixed=False,remaining_risk='Shared inotify ENOSPC observed; successful finite probes do not prove long-run reliability',
        startup_attempts=4,failed_launches=1,automatic_retries=0,utc=time.time()))
    atomic_json(root/'status.json',dict(state='gate_completed',utc=time.time(),persistent_reset_parity_passed=persistent))


def setup(root):
    plan=read(root/'run_plan.json');old=Path(plan['old_root'])
    if not read(root/'kit_acceptance.json')['passed'] or not read(root/'batch_audit.json')['passed']:
        raise ValueError('Kit and measured batch gates required')
    if (root/'setup_complete.json').exists():raise ValueError('Never reset a prepared run')
    if time.time()>=plan['gate_deadline']:raise TimeoutError('Fixed setup Gate deadline')
    pinned=read(root/'pinned_sample_signature.json');current=read(root/'sample_signature.json');epsilon=read(root/'epsilon_signature.json')
    if pinned!=current or current['initial_model_sha256']!=epsilon['initial_model_sha256']:
        raise ValueError('Sample numerical parity or identical initialization gate failed')
    if dict(epsilon['contract'],prediction_type='sample')!=current['contract']:
        raise ValueError('Parameterization is not the only contract difference')
    atomic_json(root/'fairness.json',dict(passed=True,scope='pretraining; actual 3000 comparison pending',
        pinned_sample=current,epsilon_initial=epsilon,pinned_sample_forward_gradient_prediction_exact=True,
        plan_sha256=sha256(root/'run_plan.json')))
    from visuotactile.position_control import contract
    old_session=read(old/'session.json');session=read(root/'session.json')
    sources={str(path):sha256(path) for path in [*Path('visuotactile').rglob('*.py'),*Path('isaacsimenvs/tasks').rglob('*.py'),
        *Path('tools').glob('h24_comparison_*.py'),Path('tools/run_h24_comparison.sh'),
        Path('tools/h24-comparison-20261103-01.service'),Path('tools/state_fk_audit.py'),
        Path('research_docs/evidence/state_motion_20261103/fk_readonly/fk_validation.json')]}
    study=dict(old_session['profiles']['h24'],contract=contract(24,True,'epsilon'),updates=10000)
    session.update(implementation_sha256=sources,profiles={'h24_epsilon':study},motion_study=True,
        owned_worker_roots=[str(root/'sample_resume')],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        kit_gate=dict(old_session['kit_gate'],exit_on_error=True),kit_retries_per_stage=0,kit_retries_total=0)
    atomic_json(root/'session.json',session)
    for name in ['position_audit','motion_smoke']:
        shutil.copytree(old/name,root/name)
    for name in ['position_train_statistics.pt','preserved.json']:
        shutil.copy2(old/name,root/name)
    shutil.copy2(old/'foundation.json',root/'foundation.json')
    foundation=read(root/'foundation.json')
    foundation['evidence_sha256']['kit_acceptance.json']=sha256(root/'kit_acceptance.json')
    foundation['evidence_sha256']['batch_audit.json']=sha256(root/'batch_audit.json')
    for name in ('pinned_sample_signature','sample_signature','epsilon_signature'):
        foundation['evidence_sha256'][name+'.json']=sha256(root/(name+'.json'))
    atomic_json(root/'foundation.json',foundation)
    sample=root/'sample_resume';sample.mkdir()
    for name in ['position_audit','motion_smoke','freeze']:
        shutil.copytree(old/name,sample/name)
    for name in ['position_train_statistics.pt','preserved.json','foundation.json']:
        shutil.copy2(old/name,sample/name)
    pinned_session=dict(old_session,started_epoch=session['started_epoch'],deadline=session['deadline'],
        training_deadline=session['training_deadline'],gate_deadline=session['gate_deadline'])
    atomic_json(sample/'session.json',pinned_session)
    directory=sample/'h24_train';directory.mkdir()
    for name in ['configuration.json','last.pt','best.pt','milestone_3000.pt','metrics.json','train.jsonl']:
        shutil.copy2(old/'h24_train'/name,directory/name)
    if sha256(directory/'milestone_3000.pt')!=plan['sample_checkpoint_sha256']:
        raise ValueError('Immutable Sample checkpoint copy mismatch')
    for name in ['shared_resources.json','gpu.lock','STOP']:
        (sample/name).symlink_to(root/name)
    atomic_json(root/'setup_complete.json',dict(plan_sha256=sha256(root/'run_plan.json'),session_sha256=sha256(root/'session.json'),
        sample_configuration_sha256=sha256(directory/'configuration.json'),
        sample_milestone_sha256=sha256(directory/'milestone_3000.pt'),
        sample_runtime=plan['sample_pinned_worktree'],training_budget_enforced_by_owner=10000,utc=time.time()))
    verify_frozen(root)


def fairness3000(root):
    import torch
    sample_path=root/'sample_resume/h24_train';epsilon_path=root/'h24_epsilon_train'
    sample_config=read(sample_path/'configuration.json');epsilon_config=read(epsilon_path/'configuration.json')
    expected=dict(sample_config,contract=dict(sample_config['contract'],prediction_type='epsilon'),
        maximum_updates=10000,source_sha256=epsilon_config['source_sha256'])
    if epsilon_config!=expected:raise ValueError('Unregistered difference in paired training configuration')
    sample=torch.load(sample_path/'milestone_3000.pt',map_location='cpu',weights_only=False)
    epsilon=torch.load(epsilon_path/'milestone_3000.pt',map_location='cpu',weights_only=False)
    fields=['updates','ema_optimization_step','schedule_order','schedule_cursor','train_ids','validation_ids','seen_ids']
    equal={key:sample[key]==epsilon[key] for key in fields}
    equal.update({key:torch.equal(sample[key],epsilon[key]) for key in ('schedule_rng','noise_rng')})
    logs={name:[read_line for read_line in map(json.loads,(directory/'train.jsonl').read_text().splitlines())]
        for name,directory in [('sample',sample_path),('epsilon',epsilon_path)]}
    equal['lr_trajectory']=len(logs['sample'])==len(logs['epsilon'])==3000 and all(
        (first['update'],first['lr'])==(second['update'],second['lr']) for first,second in zip(logs['sample'],logs['epsilon']))
    if not all(equal.values()) or sample['updates']!=3000:raise ValueError('Measured 3000-update fairness gate failed: '+str(equal))
    result=read(root/'fairness.json')
    result.update(scope='actual paired3000 training verified',actual_3000=equal,
        sample_checkpoint_sha256=sha256(sample_path/'milestone_3000.pt'),
        epsilon_checkpoint_sha256=sha256(epsilon_path/'milestone_3000.pt'),
        declared_configuration_differences=['prediction_type','source_sha256','maximum_updates ceiling; optimizer_horizon remains19000'])
    atomic_json(root/'fairness.json',result)


def offline(root, label, checkpoint):
    session=read(root/'session.json');source=read(Path(session['source_root'])/'data.json')
    job=root/(label+'_offline_job.json')
    atomic_json(job,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        identities=sorted(source['splits']['validation']),samplers={'A0':dict(inference_steps=16,clip_sample=False)},
        seeds=[7091],split='validation',timing=True,overlap_steps=0,save_numeric_plans=True))
    return worker(root,label+'_offline','visuotactile.state_motion',['--job',str(job)],timeout=3600)


def cohort(root, label, checkpoint):
    specs=read(root/'run_plan.json')['formal_cohort'];persistent=read(root/'kit_acceptance.json')['persistent_reset_parity_passed']
    chunks=[specs] if persistent else [[spec] for spec in specs]
    destination=root/(label+'_loop');destination.mkdir(exist_ok=True)
    marker=root/(label+'_loop_committed.json')
    if marker.exists():
        if sha256(destination/'result.json')!=read(marker)['sha256']:
            raise ValueError('Committed cohort changed: '+label)
        return read(destination/'result.json')
    rows=[];missing=[];startup_faults=0
    for index, chunk in enumerate(chunks):
        name=label+f'_loop_worker_{index:02d}'
        job=root/(name+'_job.json')
        atomic_json(job,dict(specs=chunk,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            inference=dict(inference_steps=16,clip_sample=False,overlap_steps=0,sampler_seed=7091),video=index==0))
        try:
            result=worker(root,name,'visuotactile.state_benchmark',['--mode','dp','--job',str(job)],timeout=2400)
            rows+=result['trials'];startup_faults=0
        except RuntimeError:
            evidence=startup_failure_evidence(root,name)
            progress=root/name/'progress.json'
            if progress.exists():rows+=read(progress)['trials']
            if evidence is None:
                atomic_json(destination/'result.json',dict(software_complete=False,trials=rows,registered=12,
                    missing=[dict(spec=spec,classification='worker_error',native_outcome=None) for spec in chunk if spec['identity'] not in {row['identity'] for row in rows}]))
                raise
            startup_faults+=1
            missing += [dict(spec=spec,classification='pre_scene_native_fault',native_outcome=None) for spec in chunk]
            if startup_faults>=2:
                missing += [dict(spec=spec,classification='not_started_after_two_consecutive_startup_faults',native_outcome=None)
                    for remainder in chunks[index+1:] for spec in remainder]
                break
        for row in rows:
            link=destination/row['identity'];source=root/name/row['identity']
            if not link.exists() and source.exists():link.symlink_to(source,target_is_directory=True)
        atomic_json(destination/'progress.json',dict(trials=rows,missing=missing,registered=12))
    result=dict(software_complete=not missing,trials=rows,missing=missing,registered=12)
    atomic_json(destination/'result.json',result)
    atomic_json(marker,dict(sha256=sha256(destination/'result.json'),utc=time.time()))
    return result


def selection(root, results):
    from visuotactile.h24_selection import choose
    common=set(row['identity'] for row in results['sample']['loop']['trials']) & set(
        row['identity'] for row in results['epsilon']['loop']['trials'])
    for identity in common:
        if read(root/'sample_3000_loop'/identity/'initial_observable.json') != read(
                root/'epsilon_3000_loop'/identity/'initial_observable.json'):
            raise ValueError('Paired observable reset mismatch: '+identity)
    result=choose(results)
    result.update(rule_sha256=sha256(root/'run_plan.json'),test_used=False,utc=time.time(),
        paired_observable_initial_states_verified=sorted(common),exact_internal_simulator_state_proven=False)
    atomic_json(root/'selection.json',result)
    return result['selected']


def run(root):
    setup_info=read(root/'setup_complete.json');plan=read(root/'run_plan.json')
    if setup_info['plan_sha256']!=sha256(root/'run_plan.json') or setup_info['session_sha256']!=sha256(root/'session.json'):
        raise ValueError('Immutable plan/session drift')
    results={}
    for target in (1000,3000):
        worker(root,'h24_epsilon_'+str(target),'visuotactile.state_full_training',
            ['--kind','dp','--profile','h24_epsilon','--target',str(target)],timeout=3600)
        publish(root)
    fairness3000(root)
    for group,checkpoint in [('sample',root/'sample_resume/h24_train/milestone_3000.pt'),
                             ('epsilon',root/'h24_epsilon_train/milestone_3000.pt')]:
        label=group+'_3000'
        results[group]=dict(offline=offline(root,label,checkpoint),loop=cohort(root,label,checkpoint))
        publish(root)
    selected=selection(root,results);publish(root)
    for target in (5000,8000,10000):
        if selected=='sample':
            worker(root,f'h24_sample_{target}','visuotactile.state_full_training',
                ['--kind','dp','--profile','h24','--target',str(target)],
                cwd=plan['sample_pinned_worktree'],worker_root=root/'sample_resume',timeout=3600)
            checkpoint=root/'sample_resume/h24_train'/f'milestone_{target}.pt'
        else:
            worker(root,f'h24_epsilon_{target}','visuotactile.state_full_training',
                ['--kind','dp','--profile','h24_epsilon','--target',str(target)],timeout=3600)
            checkpoint=root/'h24_epsilon_train'/f'milestone_{target}.pt'
        label=selected+'_'+str(target)
        offline(root,label,checkpoint);cohort(root,label,checkpoint);publish(root)
    missing=any(read(path).get('missing') for path in root.glob('*_loop/result.json'))
    atomic_json(root/'status.json',dict(state='software_missing' if missing else 'completed',selected=selected,updates=10000,
        training_state='completed',evaluation_state='software_missing' if missing else 'completed',utc=time.time()))
    publish(root)


def publish(root):
    from tools.h24_comparison_report import report
    try:report(root)
    except Exception as error:
        # Reporting must not discard valid training or suppress later milestones.
        import traceback
        atomic_json(root/'report_error.json',dict(reason=repr(error),traceback=traceback.format_exc(),utc=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('gate','setup','run','report'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    def interrupt(signum,frame):raise InterruptedError('Owner signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='setup':setup(root)
        elif args.command=='report':publish(root)
        else:monitored(root,lambda:kit_gate(root) if args.command=='gate' else run(root))
