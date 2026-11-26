import json,os,subprocess,sys,time
from pathlib import Path
import psutil
from tools.tight_paired_dagger import ROOT,CONFIG,read
from visuotactile.training_resources import atomic_json
status=ROOT/'parent_queue.json'
end=read(ROOT/'session.json')['deadline']
while time.time()<end:
 expert=ROOT/'experts/result.json';preflight=ROOT/'common180_parents/preflight.json'
 external=[]
 for p in psutil.process_iter(['pid','cmdline']):
  try:
   args=' '.join(p.info['cmdline'] or [])
   if 'FingerEyeLab' in args and 'stage9_backend.py' in args:external.append(p.info['pid'])
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 ready=expert.exists() and read(expert).get('software_complete') and preflight.exists() and read(preflight).get('passed') and not external
 atomic_json(status,dict(pid=os.getpid(),utc=time.time(),external_test_pids=external,expert_complete=expert.exists(),preflight=preflight.exists(),ready=ready,
  policy='One own GPU stage at a time; wait for other test GPU worker to leave before sustained training; never signal external processes'))
 if ready:break
 time.sleep(30)
else:raise TimeoutError('Original finite session expired while waiting')
command=[sys.executable,'-m','tools.tight_paired_learning','project','train','--job',str(CONFIG/'parent_train.json')]
with (ROOT/'parent_queue_training.log').open('x') as stream:
 result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
atomic_json(ROOT/'parent_queue_result.json',dict(returncode=result.returncode,utc=time.time(),command=command))
raise SystemExit(result.returncode)
