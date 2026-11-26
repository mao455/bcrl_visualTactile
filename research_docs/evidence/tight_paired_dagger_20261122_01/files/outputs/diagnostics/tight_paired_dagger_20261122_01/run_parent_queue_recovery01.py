import os,signal,sys,time,traceback
from tools.tight_paired_dagger import ROOT,CONFIG,read,sources
from visuotactile.training_resources import atomic_json
from driver_support import external_test_pids,run_owned

def interrupted(signum,frame):raise KeyboardInterrupt('Owned parent queue interrupted '+str(signum))
for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
status=ROOT/'parent_queue.json';end=read(ROOT/'session.json')['deadline'];idle=0
command=[sys.executable,'-m','tools.tight_paired_learning','project','train','--job',str(CONFIG/'parent_train_recovery01.json')]
try:
 while time.time()<end:
  if (ROOT/'STOP_PARENT_QUEUE').exists():raise InterruptedError('Owned queue stop')
  external=external_test_pids()
  parent=ROOT/'common180_parents';cache=ROOT/'owners/shared_demo_cache'
  ready=(ROOT/'source_ready.json').exists() and read(ROOT/'source_ready.json').get('passed')
  ready=ready and read(parent/'preflight.json').get('passed') and read(parent/'complete_exposure_preflight.json').get('passed')
  ready=ready and (cache/'project_complete.json').exists() and not (cache/'project_failure.json').exists() and read(cache/'project_exit.json').get('children_reaped')
  ready=ready and all(read(ROOT/'shared_demo_preparation'/r/'cache_index.json')['complete'] for r in 'AB')
  ready=ready and read(ROOT/'experts_shared/result.json')['software_complete'] and not external
  watcher=ROOT.parents[2]/'diagnostics/tight_paired_dagger_20261122_01/resource_watch_v3_status.json'
  ready=ready and watcher.exists() and time.time()-read(watcher)['utc']<8
  atomic_json(status,dict(pid=os.getpid(),utc=time.time(),external_test_pids=external,ready=bool(ready),policy='Wait external test naturally; one own GPU stage; no external signals'))
  idle=idle+1 if ready else 0
  if idle>=2:break
  time.sleep(30)
 else:raise TimeoutError('Original finite parent session expired')
 assert sources()==read(parent/'training_source_identity.json')['sources']
 atomic_json(parent/'recovery01_source_identity.json',dict(utc=time.time(),sources=sources()))
 with (ROOT/'parent_queue_training_recovery01.log').open('x') as stream:
  rc=run_owned(command,stream,end,ROOT/'STOP_PARENT_QUEUE')
 atomic_json(ROOT/'parent_queue_result.json',dict(returncode=rc,utc=time.time(),command=command))
except BaseException as error:
 atomic_json(ROOT/'parent_queue_result.json',dict(returncode=1,utc=time.time(),error=repr(error),traceback=traceback.format_exc(),command=command))
 raise
raise SystemExit(rc)
