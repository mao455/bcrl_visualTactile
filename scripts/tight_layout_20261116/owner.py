"""One durable owner/monitor, immutable stages, no silent simulation retries."""
import sys,os,time,fcntl,subprocess,traceback,json
from pathlib import Path
from scripts.tight_layout_20261116.collect import ROOT,TREE,read
from scripts.tight_layout_20261116.pilot import stop
from visuotactile.training_resources import atomic_json
from tools.runtime import supervised_owner

@supervised_owner
def run():
 with (ROOT/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (ROOT/'collection_owner_started.json').exists()
  atomic_json(ROOT/'collection_owner_started.json',dict(pid=os.getpid(),utc=time.time(),source_tree=str(TREE)))
  monitor=child=None
  try:
   with (ROOT/'monitor.log').open('x') as log:
    monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   until=time.time()+25
   while not (ROOT/'shared_resources.json').exists():
    assert monitor.poll() is None and time.time()<until;time.sleep(.25)
   for name,extra,receipt in [('capture_gate',['--gate'],'collection_gate_complete.json'),('capture',[],'collection_complete.json'),('data',[],'prepared.json')]:
    command=[sys.executable,'-m','scripts.tight_layout_20261116.collect','data' if name=='data' else 'worker','--root',str(ROOT),*extra]
    start=time.time()
    with (ROOT/(name+'.log')).open('x') as log:
     child=subprocess.Popen(command,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    atomic_json(ROOT/(name+'_started.json'),dict(command=command,pid=child.pid,utc=start))
    while child.poll() is None:
     assert monitor.poll() is None,'Monitor exited during active worker'
     if time.time()>read(ROOT/'session.json')['collection_deadline']-30 or (ROOT/'STOP').exists():raise TimeoutError('Collection deadline or STOP')
     progress=[ROOT/'status.json',ROOT/'collection_results.json',*ROOT.glob('episodes/*/progress.json'),*ROOT.glob('episodes/*/result.json')]
     latest=max([start]+[p.stat().st_mtime for p in progress if p.exists()])
     if time.time()-latest>600:raise TimeoutError('No progress for600s')
     atomic_json(ROOT/'owner_status.json',dict(stage=name,worker_pid=child.pid,monitor_pid=monitor.pid,utc=time.time()));time.sleep(2)
    assert child.returncode==0 and read(ROOT/receipt).get('complete',name=='data'),'Stage failed; preserve original attempt'
    assert not Path('/proc',str(child.pid)).exists(),'Reaped process still present'
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    assert str(child.pid) not in [p.strip() for p in gpu],'Reaped worker retains GPU context'
    atomic_json(ROOT/('collection_gate_exit.json' if name=='capture_gate' else name+'_exit.json'),dict(exit_code=child.returncode,receipt=receipt,utc=time.time(),monitor_alive=monitor.poll() is None))
    child=None
   atomic_json(ROOT/'owner_status.json',dict(stage='data_ready_pending_training',utc=time.time()))
  except BaseException:
   atomic_json(ROOT/'owner_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
  finally:
   stop(child);(ROOT/'STOP_MONITOR').touch()
   if monitor is not None:
    try:monitor.wait(timeout=15)
    except subprocess.TimeoutExpired:stop(monitor)
   atomic_json(ROOT/'collection_owner_exit.json',dict(utc=time.time(),children_reaped=True,monitor_returncode=None if monitor is None else monitor.poll()))
if __name__=='__main__':run()
