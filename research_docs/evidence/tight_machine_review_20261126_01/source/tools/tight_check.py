"""Supervise the current development-fixture GPU gate through owned exit."""
import argparse
import fcntl
import subprocess
import time
from pathlib import Path
from tools.runtime import CODE,RUN,read,write,supervised_owner,kill_owned


@supervised_owner
def run(root):
    deadline=read(root/'session.json')['development_deadline']
    while not read(root/'status.json').get('owned_workers_exited'):
        if time.time()>=deadline:raise TimeoutError('Original development deadline')
        time.sleep(2)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=read(root/'status.json');monitor=worker=None
        try:
            if not (root/'diagnostic_data').exists():
                subprocess.run(RUN+['-m','tools.tight_audit','fixture','--root',str(root)],cwd=CODE,check=True)
            (root/'STOP_MONITOR').unlink(missing_ok=True)
            with (root/'monitor.log').open('a') as log:
                monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            stamp=str(time.time_ns());logpath=root/('gpu_audit_'+stamp+'.log')
            with logpath.open('w') as log:
                worker=subprocess.Popen(RUN+['-m','tools.tight_audit','gpu','--root',str(root),'--data',str(root/'diagnostic_data')],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'status.json',dict(state='running',stage='foundation_gpu_gate',log=str(logpath),formal_updates=0,formal_episodes=0,owned_workers_exited=False))
            while worker.poll() is None:
                if monitor.poll() is not None:raise RuntimeError('Continuous monitor stopped')
                if time.time()>=deadline or (root/'STOP').exists():raise TimeoutError('Original deadline/STOP')
                time.sleep(2)
            write(root/('gpu_audit_exit_'+stamp+'.json'),dict(returncode=worker.returncode,log=str(logpath)))
            if worker.returncode:raise RuntimeError('Foundation GPU gate failed; exact log preserved')
            write(root/'status.json',dict(original,foundation_gpu_passed=True,owned_workers_exited=False))
        except BaseException as error:
            write(root/'status.json',dict(state='blocked',stage='foundation_gpu_gate',reason=repr(error),formal_updates=0,formal_episodes=0))
            raise
        finally:
            if worker is not None:kill_owned(worker)
            (root/'STOP_MONITOR').touch()
            if monitor is not None:
                try:monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:kill_owned(monitor)
            status=read(root/'status.json');status['owned_workers_exited']=True;write(root/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
