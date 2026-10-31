"""One bounded owner for pilot admission, training and the six diagnostics.

This runner never retries a failed stage or extends a deadline. Existing pilots
are observed; only its newly launched children may be signalled.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .compact import sha256
from .dp_objectives import DEFAULT_DP_OBJECTIVE, recorded_objective
from .training_resources import atomic_json


GROUPS = ('G00-new', 'G10-new')


def pilot_admission(root, objective=DEFAULT_DP_OBJECTIVE):
    root = Path(root)
    rows = {}
    causal = json.loads((root/'causal_acceptance.json').read_text())
    index = json.loads((root/'cache_index.json').read_text())
    for group in GROUPS:
        directory = root/(group+'_pilot_sample3300')
        fit = json.loads((directory/'fit.json').read_text())
        report = json.loads((directory/'report.json').read_text())
        config = json.loads((directory/'config.json').read_text())
        learned, initial = fit['trained'], fit['initial']
        checks = dict(completed=report['updates']==3300 and report['completed_epochs']==300,
            objective=recorded_objective(config)==objective,
            prediction_type=config['model']['prediction_type']=='sample',
            correct_train=config['train_episodes']==index['splits']['train'][:4],
            finite=all(math.isfinite(value) for value in learned.values() if isinstance(value,(float,int))))
        for part in ('arm','hand'):
            checks[part+'_command'] = learned[f'command_{part}_rmse'] < learned[f'zero_command_{part}_rmse']
            checks[part+'_reference'] = learned[f'reference_{part}_rmse'] < learned[f'copy_q_{part}_rmse']
            checks[part+'_improvement'] = learned[f'command_{part}_rmse'] < initial[f'command_{part}_rmse']
        rows[group] = dict(passed=all(checks.values()),checks=checks,metrics=learned,
            fit_sha256=sha256(directory/'fit.json'),source_hashes=config['source_hashes'])
    return dict(passed=bool(causal['passed']) and all(row['passed'] for row in rows.values()),
        prediction_type='sample',loss=objective,groups=rows,causal_sha256=sha256(root/'causal_acceptance.json'),
        scope='small complete-trajectory fitting gate only; no task success or convergence claim',created_epoch=time.time())


def active_trainers(root, stage, tag=None):
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
            if ('visuotactile.train_long' in args and '--stage' in args and
                    args[args.index('--stage')+1]==stage and
                    (tag is None or ('--tag' in args and args[args.index('--tag')+1]==tag)) and '--root' in args and
                    Path(args[args.index('--root')+1]).resolve()==root.resolve()):
                found.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):
            continue
    return found


def active_pilots(root):
    return active_trainers(root,'pilot','sample3300')


def decode_videos(directory):
    import av
    result = []
    for path in sorted(directory.glob('*.mp4')):
        with av.open(str(path)) as container:
            frames, previous = 0, None
            for frame in container.decode(video=0):
                if frame.pts is None or (previous is not None and frame.pts<=previous):
                    raise ValueError('Video timestamps are not strictly increasing')
                previous = frame.pts
                frames += 1
            if frames < 1:
                raise ValueError('Empty video')
        result.append(dict(file=path.name,frames=frames,bytes=path.stat().st_size,sha256=sha256(path)))
    atomic_json(directory/'video_validation.json',dict(videos=result,scope='all frames decoded on CPU'))
    return result


def run(root, resume_training=False, objective=DEFAULT_DP_OBJECTIVE):
    session = json.loads((root/'session.json').read_text())
    children = []
    handles = []
    def status(state, **details):
        atomic_json(root/'pipeline.json',dict(state=state,pid=os.getpid(),utc_epoch=time.time(),**details))
    def launch(module, arguments, label):
        handle = (root/(label+'.log')).open('x')
        handles.append(handle)
        command = ['bash','visuotactile/run_cpu.sh','-m',module,'--root',str(root),*arguments]
        process = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        children.append(process)
        return process
    def wait(processes, deadline):
        signalled = False
        stop_time = None
        while any(process.poll() is None for process in processes):
            if time.time()>=deadline or (root/'STOP').exists():
                if not signalled:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                    signalled,stop_time = True,time.monotonic()
                elif time.monotonic()-stop_time>90:
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
            time.sleep(2)
        return [process.returncode for process in processes]
    try:
        status('waiting_pilots',observed_pids=active_pilots(root))
        while not all((root/(group+'_pilot_sample3300')/'report.json').exists() for group in GROUPS):
            if time.time()>=session['training_deadline'] or (root/'STOP').exists():
                raise TimeoutError('Pilot waiting ended at original training cutoff')
            if not active_pilots(root):
                raise RuntimeError('A pilot exited without its completion report; no automatic retry')
            time.sleep(5)
        admission = pilot_admission(root,objective)
        atomic_json(root/'pilot_admission.json',admission)
        if not admission['passed']:
            raise RuntimeError('Complete-trajectory pilot gate failed; full training not admitted')
        if time.time()>=session['training_deadline']:
            raise TimeoutError('No training time remains')
        for group in GROUPS:
            directory = root/(group+'_full')
            if directory.exists() and not resume_training:
                raise ValueError('Full training output already exists; no implicit restart')
            if resume_training and not (directory/'last.pt').exists():
                raise ValueError('Explicit continuation requires both existing training checkpoints')
        status('training',admission=admission)
        workers = {}
        for index,group in enumerate(GROUPS):
            if resume_training and (root/(group+'_full')/'report.json').exists():
                continue
            arguments = ['--group',group,'--stage','full','--epochs','40','--seed',str(20261031+index),
                         '--prediction-type','sample','--loss',objective]
            label = group+'_full'
            if resume_training:
                arguments += ['--resume','--resume-runtime-update']
                label += '_resume_'+str(time.time_ns())
            workers[group] = launch('visuotactile.train_long',arguments,label)
        status('training',workers={group:worker.pid for group,worker in workers.items()},
               epochs_each=40,updates_each=33280,resumed=resume_training)
        codes = wait(list(workers.values()),session['training_deadline'])
        completed = [(root/(group+'_full')/'report.json').exists() for group in GROUPS]
        if any(codes) and time.time()<session['training_deadline']:
            raise RuntimeError(f'Trainer error before cutoff: {codes}; no automatic retry')
        if (root/'STOP').exists():
            raise InterruptedError('STOP requested; no further GPU stages')
        for group in GROUPS:
            if not (root/(group+'_full')/'best.pt').exists():
                raise RuntimeError('No validation-selected checkpoint available')
            status('test',group=group,training_complete=dict(zip(GROUPS,completed)))
            worker = launch('visuotactile.train_long',['--group',group,'--stage','test'],group+'_test')
            if wait([worker],session['deadline'])[0] or not (root/(group+'_full')/'test.json').exists():
                raise RuntimeError('Held-out test did not complete; no automatic retry')
        status('closed_loop',training_complete=dict(zip(GROUPS,completed)))
        worker = launch('visuotactile.evaluate_long',[],'closed_loop')
        codes = wait([worker],session['deadline'])
        report_path = root/'closed_loop/report.json'
        if codes[0] or not report_path.exists() or not json.loads(report_path.read_text()).get('complete'):
            raise RuntimeError('Closed-loop report incomplete, regardless of simulator exit code')
        videos = decode_videos(root/'closed_loop')
        status('complete',training_complete=dict(zip(GROUPS,completed)),videos=videos,
               scope='pipeline complete; task outcomes are in closed_loop/report.json')
    except BaseException as error:
        status('interrupted',error=repr(error))
        raise
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        wait(children,time.time())
        for handle in handles:
            handle.close()
        if not active_pilots(root):
            (root/'STOP_MONITOR').touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--resume-training',action='store_true')
    parser.add_argument('--loss',choices=['A','B'],default=DEFAULT_DP_OBJECTIVE,
                        help='Formal default B; A only for explicit historical reproduction')
    args = parser.parse_args()
    if args.status:
        print((args.root/'pipeline.json').read_text())
        return
    with (args.root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'pipeline.json').exists():
            previous = json.loads((args.root/'pipeline.json').read_text())
            if not args.resume_training or previous['state']!='training':
                raise ValueError('Preserve prior pipeline evidence; only explicit training continuation is allowed')
            if active_trainers(args.root,'full'):
                raise ValueError('Previous trainers must finish saving and exit before continuation')
            atomic_json(args.root/f'pipeline_before_resume_{time.time_ns()}.json',previous)
        elif args.resume_training:
            raise ValueError('No existing pipeline to continue')
        run(args.root,args.resume_training,args.loss)


if __name__ == '__main__':
    main()
