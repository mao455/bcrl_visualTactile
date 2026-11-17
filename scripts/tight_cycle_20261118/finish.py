"""Finite dependent tail: wait existing owner, run one offline review and two evaluations."""
import os,sys,subprocess,time,traceback
from scripts.tight_cycle_20261118.cycle import ROOT,read,write
from scripts.tight_layout_20261116.pilot import stop

def main():
 assert not (ROOT/'tail_started.json').exists();write(ROOT/'tail_started.json',dict(pid=os.getpid(),utc=time.time()));child=None
 def execute(name,args):
  nonlocal child
  log=ROOT/(name+'.log');cmd=[sys.executable,'-m',*map(str,args)];began=time.time()
  with log.open('x') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/(name+'_started.json'),dict(pid=child.pid,utc=began,command=cmd))
  while child.poll() is None:
   if time.time()>read(ROOT/'session.json')['deadline']-30 or (ROOT/'STOP').exists():raise TimeoutError('Fixed cycle deadline/STOP')
   write(ROOT/'tail_status.json',dict(stage=name,pid=child.pid,elapsed_s=time.time()-began,utc=time.time()));time.sleep(3)
  assert child.returncode==0,name
  write(ROOT/(name+'_complete.json'),dict(seconds=time.time()-began,utc=time.time()));child=None
 try:
  while not (ROOT/'learning/owner_exit.json').exists():
   if time.time()>read(ROOT/'session.json')['training_deadline']:raise TimeoutError('Learning deadline')
   write(ROOT/'tail_status.json',dict(stage='waiting_existing_learning_owner',utc=time.time()));time.sleep(10)
  assert read(ROOT/'learning/complete.json')['complete'],'No candidate if learning incomplete'
  execute('offline',['scripts.tight_cycle_20261118.offline'])
  for part in range(2):
   execute(f'eval_prepare{part}',['scripts.tight_cycle_20261118.evaluate',part])
   execute(f'eval{part}',['scripts.tight_cycle_20261118.cycle','run_wave',ROOT/f'comparison{part:02d}'])
  execute('review_learning',['scripts.tight_cycle_20261118.review','learning'])
  execute('review_evaluation',['scripts.tight_cycle_20261118.review','evaluation'])
  write(ROOT/'tail_complete.json',dict(utc=time.time(),complete=True,scientific_review_required=True))
 except BaseException:write(ROOT/'tail_failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  stop(child);write(ROOT/'tail_exit.json',dict(utc=time.time(),children_reaped=True))
if __name__=='__main__':main()
