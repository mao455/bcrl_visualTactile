"""One bounded paired implementation checkpoint, independent of publication."""
import argparse,json,os,subprocess,time
from pathlib import Path
from tools.runtime import RUN,kill_owned,supervised_owner,write,sha
from tools.tight_execution_learning import run_child

@supervised_owner
def run(root):
    reg=json.loads((root/'registration.json').read_text())
    if (root/'probe_stage_launch.json').exists():raise ValueError('No implicit stage retry')
    files=subprocess.check_output(['git','ls-files','*.py','visuotactile/run_cpu.sh'],text=True).splitlines()
    write(root/'source_freeze.json',{str(Path(p).resolve()):sha(p) for p in files})
    write(root/'probe_stage_launch.json',dict(pid=os.getpid(),utc=time.time(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),deadline=reg['absolute_deadline']))
    monitor=None
    try:
        with (root/'monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.time()
        while not (root/'shared_resources.json').exists():
            if monitor.poll() is not None or time.time()-start>30:raise RuntimeError('Monitor not ready')
            time.sleep(.2)
        run_child(root,'initial_paired_probe',['-m','tools.tight_execution_learning','probe','--root',str(root)],min(reg['compute_deadline']-10800,time.time()+1800))
        write(root/'probe_stage_complete.json',dict(complete=True,utc=time.time(),next='adaptive diagnosis and exposure; failed first gates are not terminal'))
    finally:
        (root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        write(root/'probe_stage_exit.json',dict(utc=time.time(),monitor_exited=monitor is None or monitor.poll() is not None))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
