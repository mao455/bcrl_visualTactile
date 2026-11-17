"""Bounded ordinary post-training orchestration; no automatic research iteration."""
import json,sys,subprocess,time,os,traceback
from pathlib import Path
from scripts.tight_balance_20261117.offline_phases import ROOT,BASE,OUT
from scripts.tight_layout_20261116.publication import write
TREE=Path.cwd()
def read(p):return json.loads(p.read_text())
def main():
 follow=ROOT/'followup';follow.mkdir(exist_ok=False);reg=read(ROOT/'registration.json');child=None
 def execute(name,args,deadline):
  nonlocal child
  start=time.time();path=follow/(name+'.log')
  with path.open('x') as f:child=subprocess.Popen([sys.executable,'-m',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(follow/(name+'_started.json'),dict(pid=child.pid,args=args,utc=start))
  while child.poll() is None:
   if time.time()>deadline or (ROOT/'STOP_FOLLOWUP').exists():
    from scripts.tight_layout_20261116.pilot import stop
    stop(child);raise TimeoutError('Fixed cycle deadline/stop; no automatic repeat')
   time.sleep(3)
  write(follow/(name+'_exit.json'),dict(code=child.returncode,seconds=time.time()-start,utc=time.time()))
  if child.returncode:raise RuntimeError('Stage failed: '+name)
  child=None
 try:
  while not (ROOT/'owner_exit.json').exists():
   if time.time()>reg['training_deadline']:raise TimeoutError('Training did not close by fixed deadline')
   time.sleep(5)
  assert not (ROOT/'failure.json').exists() and read(ROOT/'complete.json')['complete']
  execute('offline_phases',['scripts.tight_balance_20261117.offline_phases'],reg['deadline']-3600)
  for part in range(4):
   wave=ROOT/f'comparison{part:02d}'
   execute(f'prepare{part}',['scripts.tight_layout_20261116.learn','prepare_development','--root',str(wave),'--source',str(ROOT),'--registration',str(ROOT/'registration.json'),'--baseline',str(BASE/'round05/learning'),'--seed',str(28824000+2*part)],reg['deadline']-1800)
   execute(f'run{part}',['scripts.tight_layout_20261116.learn','run','--root',str(wave)],reg['deadline']-600)
   execute(f'report{part}',['scripts.tight_layout_20261116.phase_report','--root',str(wave/'development'),'--output',str(OUT/f'comparison{part:02d}.json')],reg['deadline'])
  write(follow/'complete.json',dict(complete=True,utc=time.time(),next='Scientific review required; no automatic DAgger or promotion'))
 except BaseException:
  write(follow/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  if child is not None:
   from scripts.tight_layout_20261116.pilot import stop
   stop(child)
  write(follow/'owner_exit.json',dict(children_reaped=True,utc=time.time()))
if __name__=='__main__':main()
