"""Bounded owner for paired loss experiments and registered downstream stages."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .compact import sha256
from .training_resources import atomic_json


def initialize(root, source):
    root.mkdir(exist_ok=False)
    started = datetime.fromisoformat('2026-10-31T03:01:47+00:00').timestamp()
    if time.time() >= started+6*3600:
        raise TimeoutError('Original study window already ended')
    session = dict(version=1,run_id=root.name,started_epoch=started,deadline=started+6*3600,
        training_deadline=started+5*3600,dp_deadline=started+3*3600,dp_epochs=12,
        source_root=str(source),dp_lines=['G00-A','G00-B','G10-A','G10-B'],seeds=[981001,981002,981004],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='new paired loss/residual development; historical datasets and denominators untouched')
    atomic_json(root/'session.json',session)
    atomic_json(root/'gpu_schedule.json',dict(target_percent=38,admission_percent=42,min_gpu_free_mib=6144,min_ram_gib=16))
    for name in ('cache_index.json','metadata.json','splits.json'):
        atomic_json(root/name,json.loads((source/name).read_text()))
    (root/'cache').symlink_to((source/'cache').resolve(),target_is_directory=True)
    atomic_json(root/'initial_sources.json',{group:sha256(source/(group+'-new_full')/'best.pt') for group in ('G00','G10')})


def run(root, resume=False):
    session = json.loads((root/'session.json').read_text())
    status_path = root/'pipeline.json'
    if status_path.exists() and not resume:
        raise ValueError('Existing pipeline requires explicit continuation')
    children,handles = [],[]
    def status(state,**values):
        atomic_json(status_path,dict(state=state,pid=os.getpid(),utc_epoch=time.time(),**values))
    def launch(module,arguments,label):
        handle = (root/(label+'_'+str(time.time_ns())+'.log')).open('x')
        handles.append(handle)
        process = subprocess.Popen(['bash','visuotactile/run_cpu.sh','-m',module,'--root',str(root),*arguments],
            stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        children.append(process)
        atomic_json(root/f'worker_{process.pid}.json',dict(pid=process.pid,module=module,arguments=arguments,
            launched_epoch=time.time(),module_sha256=sha256(Path(*module.split('.')).with_suffix('.py'))))
        return process
    def wait(processes,deadline):
        signalled = False
        grace = None
        while any(process.poll() is None for process in processes):
            if time.time()>=deadline or (root/'STOP').exists() or any(process.poll() not in (None,0) for process in processes):
                if not signalled:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    signalled,grace = True,time.monotonic()
                elif time.monotonic()-grace>90:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
            time.sleep(2)
        codes = [process.returncode for process in processes]
        if any(codes):
            raise RuntimeError(f'Stage stopped with exit codes {codes}; preserve artifacts, no automatic retry')
    monitor = None
    try:
        if (root/'monitor_exit.json').exists():
            raise ValueError('Completed monitor cannot be silently restarted')
        if (root/'monitor.json').exists():
            monitor_pid = json.loads((root/'monitor.json').read_text())['pid']
            command = Path(f'/proc/{monitor_pid}/cmdline').read_bytes()
            if b'visuotactile.study_resources' not in command:
                raise ValueError('Independent monitor ownership missing')
        else:
            monitor = launch('visuotactile.study_resources',[],'monitor')
        if not (root/'interface_audit.json').exists():
            status('interface_audit')
            wait([launch('visuotactile.study_audit',[],'interface_audit')],session['dp_deadline'])
        if not json.loads((root/'interface_audit.json').read_text())['passed']:
            raise ValueError('Interface audit failed')
        for group in ('G00','G10'):
            for pilot in (True,False):
                phase = group+('_pilot' if pilot else '_training')
                workers = []
                for loss in ('A','B'):
                    line = group+'-'+loss
                    directory = root/(line+('_pilot' if pilot else ''))
                    if (directory/'report.json').exists():
                        continue
                    arguments = ['--line',line]+(['--pilot'] if pilot else [])
                    if directory.exists():
                        if not resume or not (directory/'last.pt').exists():
                            raise ValueError('Partial trainer requires explicit resumable state')
                        arguments += ['--resume']
                    workers.append(launch('visuotactile.train_loss_study',arguments,line+('_pilot' if pilot else '')))
                status(phase,workers=[worker.pid for worker in workers])
                wait(workers,session['dp_deadline'])
                for loss in ('A','B'):
                    directory = root/(group+'-'+loss+('_pilot' if pilot else ''))
                    result = json.loads((directory/'report.json').read_text())
                    expected = 11 if pilot else 9984
                    if result['state']!='complete' or result['updates']!=expected:
                        raise ValueError('Incomplete paired update budget')
        selected = {}
        for group in ('G00','G10'):
            candidates = {}
            for loss in ('A','B'):
                line = group+'-'+loss
                history = json.loads((root/line/'metrics.json').read_text())
                candidates[line] = min(row['validation']['selection_mse'] for row in history)
            selected[group] = min(candidates,key=candidates.get)
        atomic_json(root/'selected.json',dict(selected=selected,criterion='EMA validation prefix12 equal arm/hand MSE',
            checkpoint_sha256={group:sha256(root/line/'best.pt') for group,line in selected.items()}))
        for line in session['dp_lines']:
            if (root/line/'test.json').exists():
                continue
            status('dp_test',line=line)
            wait([launch('visuotactile.train_loss_study',['--line',line,'--test'],line+'_test')],session['training_deadline'])
        status('waiting_registered_downstream',selected=selected)
        while not (root/'downstream.json').exists():
            if time.time()>=session['training_deadline'] or (root/'STOP').exists():
                raise TimeoutError('No reviewed downstream implementation before cutoff')
            time.sleep(5)
        stages = json.loads((root/'downstream.json').read_text())['stages']
        allowed = {'visuotactile.evaluate_study','visuotactile.transformer_data',
                   'visuotactile.train_transformer_residual','visuotactile.plot_loss_study'}
        for stage in stages:
            if stage['module'] not in allowed:
                raise ValueError('Unknown registered downstream module')
            if (root/stage['completion_file']).exists():
                continue
            status(stage['name'])
            wait([launch(stage['module'],stage['arguments'],stage['name'])],session['deadline'])
            if not (root/stage['completion_file']).exists():
                raise ValueError('Exit0 without expected stage evidence')
        status('complete',scope='execution complete; scientific outcomes require report review')
    except BaseException as error:
        status('interrupted',error=repr(error))
        raise
    finally:
        for child in children:
            if child is not monitor and child.poll() is None:
                child.terminate()
        for child in children:
            if child is not monitor:
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        atomic_json(root/'STOP_MONITOR',dict(utc_epoch=time.time()))
        if monitor is not None:
            monitor.wait(timeout=10)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('outputs/visuotactile/run/long_horizon_20261030_v2'))
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.init:
        initialize(args.root,args.source)
    elif args.status:
        print((args.root/'pipeline.json').read_text())
    else:
        with (args.root/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.root,args.resume)
