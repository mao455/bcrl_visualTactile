"""Output-only resource admission; unchanged canonical controller/math/RNG."""
from pathlib import Path
import faulthandler,json,os,sys,threading,time
from outputs.diagnostics.tight_full_curve_20261127_01.train_recovery import bounded_admission
from visuotactile.training_resources import atomic_json

def resource_watchdog(output):
 stream=(output/'watchdog_tracebacks.log').open('x');began=time.time();resumed=began
 deadline=json.loads((output/'session.json').read_text())['deadline']
 wait=Path('outputs/diagnostics/tight_full_curve_20261127_01')/f'resource_wait_{os.getpid()}.jsonl'
 atomic_json(output/'watchdog_policy14.json',dict(startup_seconds=600,control_idle_seconds=300,first_control_marker='batch_progress.json',explicit_capacity_wait_excluded=True,deadline=deadline,physics_timeout_changed=False))
 def watch():
  nonlocal resumed
  while True:
   time.sleep(5);now=time.time()
   waiting=wait.exists() and now-wait.stat().st_mtime<8
   if waiting:resumed=now
   files=[*output.glob('batch_progress.json'),*output.glob('*/progress.json'),*output.glob('*/result.json')]
   newest=max([began,resumed]+[p.stat().st_mtime for p in files if p.exists()])
   control=(output/'batch_progress.json').exists();bound=300 if control else 600
   if now>=deadline or not waiting and now-newest>bound:
    atomic_json(output/'watchdog_failure.json',dict(utc=now,pid=os.getpid(),control_started=control,last_progress_utc=newest,limit_s=bound,reason='Finite deadline' if now>=deadline else 'Worker no progress excluding explicit capacity waits'))
    faulthandler.dump_traceback(file=stream,all_threads=True);stream.flush();os._exit(124)
 threading.Thread(target=watch,daemon=True).start()

if len(sys.argv)>1 and sys.argv[1]=='project':
 from tools import tight_project
 tight_project.ENTRIES['curve_eval']=('evaluate','outputs.diagnostics.tight_full_curve_20261127_01.eval_recovery',[])
 sys.argv.pop(1);tight_project.main()
else:
 from tools import tight_machine_review
 tight_machine_review.watchdog=resource_watchdog
 tight_machine_review.main()
