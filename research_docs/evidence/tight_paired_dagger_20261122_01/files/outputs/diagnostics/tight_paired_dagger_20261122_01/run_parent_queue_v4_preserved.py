import json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
from tools.tight_paired_dagger import ROOT,CONFIG,read
from visuotactile.training_resources import atomic_json
def interrupted(signum,frame):raise KeyboardInterrupt('Owned parent queue interrupted '+str(signum))
for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
status=ROOT/'parent_queue.json'
from tools.tight_paired_dagger import sources
idle_checks=0
end=read(ROOT/'session.json')['deadline']
while time.time()<end:
 expert=ROOT/'experts/result.json';preflight=ROOT/'common180_parents/preflight.json'
 external=[]
 for p in psutil.process_iter(['pid','cmdline']):
  try:
   args=' '.join(p.info['cmdline'] or [])
   if p.info['pid'] != os.getpid() and 'python' in (p.info['cmdline'] or [''])[0].lower():
    cwd=p.cwd()
    if 'FingerEyeLab' in args or 'FingerEyeLab' in cwd:external.append(p.info['pid'])
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 cache=ROOT/'owners/shared_demo_cache/project_exit.json'
 source_ready=ROOT/'source_ready.json'
 complete_preflight=ROOT/'common180_parents/complete_exposure_preflight.json'
 ready=source_ready.exists() and read(source_ready).get('passed') and complete_preflight.exists() and read(complete_preflight).get('passed') and cache.exists() and read(cache).get('children_reaped') and expert.exists() and read(expert).get('software_complete') and preflight.exists() and read(preflight).get('passed') and not external
 atomic_json(status,dict(pid=os.getpid(),utc=time.time(),external_test_pids=external,expert_complete=expert.exists(),preflight=preflight.exists(),ready=ready,
  policy='One own GPU stage at a time; wait for other test entire external batch and GPU worker to leave before sustained training; never signal external processes'))
 idle_checks=idle_checks+1 if ready else 0
 if idle_checks>=2:break
 time.sleep(30)
else:raise TimeoutError('Original finite session expired while waiting')
atomic_json(ROOT/'common180_parents/training_source_identity.json',dict(utc=time.time(),sources=sources()))
command=[sys.executable,'-m','tools.tight_paired_learning','project','train','--job',str(CONFIG/'parent_train.json')]
with (ROOT/'parent_queue_training.log').open('x') as stream:
 child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while child.poll() is None:
   if time.time()>=end or (ROOT/'STOP_PARENT_QUEUE').exists():raise TimeoutError('Finite parent deadline or owned stop')
   time.sleep(5)
 finally:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=45)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
 result=child
atomic_json(ROOT/'parent_queue_result.json',dict(returncode=result.returncode,utc=time.time(),command=command))
raise SystemExit(result.returncode)
