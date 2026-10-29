"""Run two independent trainers, then their paired evaluation. Own children only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .training_resources import atomic_json


def read_status(root):
    """Read live status without importing CUDA or acquiring the runner lock."""
    root=Path(root)
    def read(path):
        return json.loads(path.read_text()) if path.exists() else None
    result=dict(root=str(root.resolve()),runner=read(root/'status.json'),variants={})
    for variant in ('b0','b1'):
        status=read(root/variant/'status.json')
        child=(result['runner'] or {}).get('children',{}).get(variant,{})
        pid=child.get('pid')
        alive=False
        if pid:
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
                alive=b'visuotactile.train_dp' in cmd and str(root).encode() in cmd
            except FileNotFoundError:
                pass
        result['variants'][variant]=dict(pid=pid,alive=alive,status=status,
            resources=read(root/variant/'resources.json'),
            checkpoints={name:dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime)
                for name in ('last.pt','best.pt') if (p:=root/variant/name).exists()})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path('outputs/visuotactile/data/close_v4/manifest.json'))
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--mode',choices=('parallel','sequential'),default='parallel')
    p.add_argument('--skip-evaluation',action='store_true')
    p.add_argument('--status',action='store_true')
    args=p.parse_args()
    if args.status:
        print(json.dumps(read_status(args.root),indent=2));return
    args.root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.root/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[False]
    def stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    children={};streams=[];pending=['b0','b1'];returns={};start=time.time();stop_sent=set()
    try:
        while pending or any(c.poll() is None for c in children.values()):
            if (args.root/'STOP').exists():stopped[0]=True
            if stopped[0]:
                for variant,child in children.items():
                    if child.poll() is None and variant not in stop_sent:
                        child.send_signal(signal.SIGTERM);stop_sent.add(variant)
                pending=[]
            elif pending and (args.mode=='parallel' or all(c.poll() is not None for c in children.values())):
                variant=pending.pop(0);output=args.root/variant;output.mkdir(exist_ok=True)
                if (output/'report.json').exists():
                    returns[variant]=0;continue
                env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                     'MALLOC_ARENA_MAX':'2','PYTHONUNBUFFERED':'1'}
                cpus='8-11' if variant=='b0' else '12-15'
                cmd=['nice','-n','10','taskset','-c',cpus,sys.executable,'-m','visuotactile.train_dp',
                    '--manifest',str(args.manifest),'--variant',variant,'--output',str(output),
                    '--epochs',str(args.epochs),'--chunk-size','32','--batch-size','2','--amp','bf16',
                    '--vision-checkpoint-images','12','--prefetch','1','--cuda-memory-fraction','.36',
                    '--min-gpu-free-mib','4096','--min-ram-available-gib','10','--resource-wait-timeout-s','3600']
                if (output/'last.pt').exists():cmd.extend(['--resume',str(output/'last.pt')])
                stream=(output/'stdout.log').open('a',buffering=1);streams.append(stream)
                child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                children[variant]=child
                (output/'pid').write_text(str(child.pid)+'\n')
            for variant,child in children.items():
                if child.poll() is not None:returns[variant]=child.returncode
            atomic_json(args.root/'status.json',dict(state='stopping' if stopped[0] else 'training',
                pid=os.getpid(),started_utc=start,updated_utc=time.time(),mode=args.mode,
                children={v:dict(pid=c.pid,returncode=c.poll()) for v,c in children.items()},pending=pending))
            time.sleep(2)
        returns.update({v:c.returncode for v,c in children.items()})
        if stopped[0]:
            atomic_json(args.root/'status.json',dict(state='stopped',returncodes=returns));return
        if any(rc!=0 for rc in returns.values()) or not all((args.root/v/'report.json').exists() for v in ('b0','b1')):
            raise RuntimeError(f'Training incomplete: {returns}; inspect child logs/checkpoints before resume')
        if not args.skip_evaluation:
            output=args.root/'evaluation'
            if not (output/'report.json').exists():
                # Never silently overwrite a partial evaluation attempt.
                if output.exists():raise RuntimeError('Evaluation partial exists: inspect it before a new attempt')
                cmd=[sys.executable,'-m','visuotactile.evaluate_baselines',
                     '--b0',str(args.root/'b0/best.pt'),'--b1',str(args.root/'b1/best.pt'),
                     '--output',str(output),'--pairs','50','--seed-start','981000','--max-steps','2400']
                log=(args.root/'evaluation.log').open('a');streams.append(log)
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                while child.poll() is None:
                    if stopped[0] or (args.root/'STOP').exists():
                        child.terminate();child.wait(timeout=30);raise RuntimeError('Evaluation stopped; partial retained')
                    atomic_json(args.root/'status.json',dict(state='evaluating',pid=os.getpid(),evaluation_pid=child.pid,
                        started_utc=start,updated_utc=time.time()))
                    time.sleep(5)
                if child.returncode or not (output/'report.json').exists():
                    raise RuntimeError('Evaluation failed; inspect evaluation.log')
        atomic_json(args.root/'status.json',dict(state='complete',started_utc=start,finished_utc=time.time()))
    except BaseException as error:
        atomic_json(args.root/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error),updated_utc=time.time(),
            children={v:dict(pid=c.pid,returncode=c.poll()) for v,c in children.items()}))
        raise
    finally:
        for stream in streams:stream.close()

if __name__=='__main__':main()
