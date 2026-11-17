"""Reuse existing phase-error computation after full paired training; labels unchanged."""
import os,sys,subprocess,time,traceback
from scripts.tight_cycle_20261118.cycle import ROOT,OUT,read,write
from scripts.tight_layout_20261116.pilot import stop

def run():
 from scripts.tight_balance_20261117 import offline_phases as front,backend_phases as rear
 root=ROOT/'learning';out=OUT/'offline';out.mkdir(exist_ok=True)
 front.ROOT=root;front.OUT=out;rear.OUT=out
 review=root/'phase_review';review.mkdir(exist_ok=False);s=read(root/'session.json');s.update(resource_root=str(review),owned_worker_roots=[str(review)]);write(review/'session.json',s)
 monitor=None;began=time.time()
 try:
  with (review/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(review)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (review/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  front.calculate(review)
  # Reused calculation historically calls its candidate76/new4. Correct output
  # identity after computation; filenames and identity fields both updated.
  for p in list(out.iterdir()):
   if p.suffix not in ['.json','.csv']:continue
   text=p.read_text().replace('candidate76','candidate105').replace('new4','new_to_parent33').replace('legal76','legal105');name=p.name.replace('candidate76','candidate105')
   (out/name).write_text(text)
   if name!=p.name:p.unlink()
  write(out/'identity.json',dict(candidate_root=str(root),candidate_trajectories=105,new_to_parent=33,reference='Selected72 full A/B',reuse='Unchanged existing fixed-seed7091 complete causal-history inference; teacher forcing is not autonomous ability; each backend uses its own refreshed DP cache.',seconds=time.time()-began))
 except BaseException:
  write(review/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  (review/'STOP_MONITOR').touch()
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  write(review/'owner_exit.json',dict(children_reaped=True,utc=time.time(),seconds=time.time()-began))
if __name__=='__main__':run()
