"""Finite dependent closure after registered collection, no implicit repeated cycles."""
import os,sys,subprocess,time,traceback
from scripts.tight_recovery_20261118.run import ROOT,OUT,COLLECTION,read,write
from scripts.tight_layout_20261116.pilot import stop

def main():
 assert not(ROOT/'tail_recovery02_started.json').exists();write(ROOT/'tail_recovery02_started.json',dict(pid=os.getpid(),utc=time.time()));child=None
 def execute(name,args):
  nonlocal child
  cmd=[sys.executable,'-m',*map(str,args)];start=time.time()
  with (ROOT/(name+'.log')).open('x') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(ROOT/(name+'_started.json'),dict(pid=child.pid,utc=start,command=cmd))
  while child.poll() is None:
   if time.time()>read(ROOT/'session.json')['deadline']-30 or (ROOT/'STOP').exists():raise TimeoutError('Registered deadline or explicit STOP')
   write(ROOT/'tail_recovery02_status.json',dict(stage=name,pid=child.pid,elapsed_s=time.time()-start,utc=time.time()));time.sleep(3)
  assert child.returncode==0,name
  write(ROOT/(name+'_complete.json'),dict(utc=time.time(),seconds=time.time()-start));child=None
 try:
  while not(ROOT/COLLECTION/'owner_exit.json').exists():
   if time.time()>read(ROOT/'session.json')['collection_deadline']:raise TimeoutError('Collection budget')
   write(ROOT/'tail_recovery02_status.json',dict(stage='waiting_registered_collection',utc=time.time()));time.sleep(5)
  assert read(ROOT/COLLECTION/'result.json')['software_complete']
  execute('collection_coverage',['scripts.tight_recovery_20261118.coverage',COLLECTION])
  execute('collection_support',['scripts.tight_recovery_20261118.support',COLLECTION])
  execute('collection_review',['scripts.tight_recovery_20261118.review','collect'])
  rows=read(OUT/(COLLECTION+'_coverage.json'))['rows'];rescued=[r for r in rows if r['accepted']]
  assert len(rescued)>=4 and all(r['transition_gap']==2 for r in rescued),'Insufficient new collection, scientific review required'
  write(OUT/'collection_decision.json',dict(continue_learning=True,accepted=len(rescued),legal_ticks=sum(r['legal_ticks'] for r in rescued),rule='At least4 additional legal actual recovery states beyond passed pilot; no repeat collection or grid. Registered original30 replay3 / all corrections1 in both components. Full initial-parent72 pairing.',utc=time.time()))
  execute('learning_prepare',['scripts.tight_recovery_20261118.learning'])
  execute('learning',['scripts.tight_layout_20261116.learn','run','--root',ROOT/'learning'])
  execute('learning_review',['scripts.tight_recovery_20261118.review','learning'])
  execute('offline',['scripts.tight_recovery_20261118.review','offline'])
  for part in range(4):
   execute(f'eval_prepare{part}',['scripts.tight_recovery_20261118.review','evaluation_prepare',part])
   execute(f'eval{part}',['scripts.tight_recovery_20261118.run','wave',f'comparison{part:02d}'])
  execute('evaluation_review',['scripts.tight_recovery_20261118.review','evaluation'])
  write(ROOT/'tail_recovery02_complete.json',dict(complete=True,utc=time.time(),scientific_review_required=True,no_automatic_promotion=True))
 except BaseException:write(ROOT/'tail_recovery02_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
 finally:stop(child);write(ROOT/'tail_recovery02_exit.json',dict(utc=time.time(),children_reaped=True))
if __name__=='__main__':main()
