"""Finite post-training diagnosis, independently managed; no training/collection."""
import json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_budget_20261116.run import ROOT,execute,read
from visuotactile.training_resources import atomic_json

def main():
 deadline=read(ROOT/'session.json')['deadline']
 assert not (ROOT/'finish_started.json').exists(),'One registered finishing owner only'
 atomic_json(ROOT/'finish_started.json',dict(utc=time.time(),deadline=deadline,
  scope='wait for existing training, read-only endpoint fit, bounded short autonomy and reports; no DAgger'))
 try:
  while not (ROOT/'serial_recovery_complete.json').exists():
   if (ROOT/'serial_recovery_failed.json').exists():raise RuntimeError('Existing serial recovery failed; no automatic retraining')
   if time.time()>deadline-180 or (ROOT/'STOP').exists():raise RuntimeError('Fixed deadline/stop before training completed')
   atomic_json(ROOT/'finish_state.json',dict(stage='waiting_existing_training',heartbeat=time.time()))
   time.sleep(10)
  assert read(ROOT/'training_complete.json')['complete']
  for stage,receipt in [('endpoints','endpoint_comparison.json'),('evaluate','evaluation_complete.json')]:
   atomic_json(ROOT/'finish_state.json',dict(stage=stage,heartbeat=time.time()))
   execute('finish_'+stage,['scripts.tight_budget_20261116.'+stage],ROOT/receipt)
  # CPU reports are also finite managed subprocesses, with their completion
  # receipt supplied by this driver after actual successful exit.
  import subprocess
  from scripts.tight_budget_20261116.run import TREE
  for stage in ['report','behavior_report']:
   if time.time()>deadline-60:raise RuntimeError('Fixed deadline before reports')
   atomic_json(ROOT/'finish_state.json',dict(stage=stage,heartbeat=time.time()))
   with (ROOT/'logs'/('finish_'+stage+'.log')).open('x') as log:
    subprocess.run([sys.executable,'-m','scripts.tight_budget_20261116.'+stage],cwd=TREE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()-30))
  atomic_json(ROOT/'finish_complete.json',dict(complete=True,utc=time.time(),reports=True,adoption=False))
  atomic_json(ROOT/'finish_state.json',dict(stage='complete_awaiting_scientific_review',heartbeat=time.time()))
 except BaseException as e:
  atomic_json(ROOT/'finish_failed.json',dict(utc=time.time(),error=repr(e),traceback=traceback.format_exc()))
  raise

if __name__=='__main__':main()
