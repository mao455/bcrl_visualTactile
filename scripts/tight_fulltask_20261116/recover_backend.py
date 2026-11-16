"""Finish the missing B chain without repeating A; preserve coordinator failure.

The initial coordinator used a shared status.json.tmp from two threads. Stage
receipts remain distinct. This recovery writes B status separately and joins A's
actual finished/offline receipts before any full-policy evaluation.
"""
import os,sys,time,subprocess,threading,traceback
from pathlib import Path
from scripts.tight_fulltask_20261116 import run as driver
from scripts.tight_fulltask_20261116.run import ROOT,DEADLINE,read,atomic_json,monitor_start
from scripts.tight_fulltask_20261116.train import OUT,eval_prepare
from tools.tight_fresh_cycle import backend_chain
from tools.tight_fresh_rollouts import completed_pair
original_atomic=driver.atomic_json

def local_atomic(path,value):
    if Path(path)==ROOT/'status.json':path=ROOT/'status_B_recovery.json'
    original_atomic(path,value)
driver.atomic_json=local_atomic
stop=threading.Event();restarted=[]
def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
monitor_pid=read(ROOT/'shared_resources.json')['pid']
def sustain_monitor():
    while not stop.is_set() and time.time()<DEADLINE:
        if not alive(monitor_pid) and not restarted:
            stamp=str(time.time_ns())
            for name in ['STOP_MONITOR','monitor.log']:
                if (ROOT/name).exists():(ROOT/name).rename(ROOT/(name+'.before_recovery_'+stamp))
            p=monitor_start();restarted.append(p)
            atomic_json(ROOT/'monitor_recovery.json',dict(old_pid=monitor_pid,new_pid=p.pid,utc=time.time(),reason='Completed/failed original coordinator stops its monitor; independent finite B chain still requires telemetry'))
        stop.wait(.5)

def main():
    assert not (OUT/'B/backend').exists(),'Never duplicate an existing B backend'
    assert (ROOT/'backend_cache_A_worker.json').exists()
    atomic_json(ROOT/'backend_recovery_registration.json',dict(utc=time.time(),deadline=DEADLINE,A='Existing independent chain retained, no optimizer/cache repeat',
        B='Missing B stage; own per-role status filename avoids common temporary-file race',epochs=2))
    thread=threading.Thread(target=sustain_monitor,daemon=True);thread.start()
    def execute(name,args,completion,**kwargs):
        driver.execute(name,args);assert completion.exists()
    try:
        backend_chain(execute,OUT,'B',2,continue_rear=True,offline_anchor_batch=32,
            frozen_visual_io=read(OUT/'frozen_visual_io.json'),backend_phase_weighting=read(OUT/'backend_phase_weighting.json'))
        while not (OUT/'A/backend/A/offline/result.json').exists():
            if time.time()>DEADLINE-180:raise TimeoutError('No reserve for paired full evaluation')
            time.sleep(1)
        # Wait for original coordinator to expose its preserved B-stage failure,
        # avoiding a second evaluation if it unexpectedly proceeded successfully.
        while not (ROOT/'training_failure.json').exists():
            if (ROOT/'updated_evaluation/run_started.json').exists():raise RuntimeError('Original evaluation already started; no duplicate')
            if time.time()>DEADLINE-180:raise TimeoutError('Coordinator handoff missed evaluation reserve')
            time.sleep(.5)
        eval_prepare(ROOT/'updated_evaluation',{r:completed_pair(OUT,r) for r in 'AB'})
        driver.execute('updated_evaluation',['scripts.tight_fulltask_20261116.video_eval','--output',ROOT/'updated_evaluation'])
        atomic_json(ROOT/'complete.json',dict(complete=True,utc=time.time(),deadline=DEADLINE,recovered_missing_B_stage=True))
    finally:
        stop.set();thread.join(timeout=2)
        (ROOT/'STOP_MONITOR').touch()
        for p in restarted:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(ROOT/'backend_recovery_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
