"""Finite read-only science follow-up; never publishes or starts extra training."""
from pathlib import Path
import json,os,subprocess,sys,time
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';read=lambda p:json.loads(p.read_text())
try:
 while not (R/'curve_complete.json').exists():
  if (D/'continuation13_failure.json').exists():raise RuntimeError('Main scheduler failed; preserve actual attempt and inspect before any recovery')
  if time.time()>read(R/'session.json')['deadline']:raise TimeoutError('Original finite round deadline reached')
  status={}
  for c in ['old_only','plus_corrected24']:
   for role in 'AB':
    p=R/c/role/'training/progress.json'
    if p.exists():
     v=read(p);status[c+'_'+role]={k:v[k] for k in ['epoch','updates']}
  atomic_json(D/'postprocess_wait.json',dict(utc=time.time(),status=status));print(json.dumps(status),flush=True);time.sleep(45)
 for name,args in [('summarize',['--plot']),('audit_training',[]),('audit_fitting',[]),('audit_evaluation',['--cuda','--final']),('audit_runtime',[])]:
  with (D/('postprocess_'+name+'.log')).open('x') as f:
   result=subprocess.run([sys.executable,str(D/(name+'.py')),*args],cwd=P,env=dict(os.environ,PYTHONPATH=str(P)),stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(name+' failed; inspect preserved log before publishing')
 atomic_json(D/'audits_complete.json',dict(utc=time.time(),publication_done=False,optimizer_updates=0,physics_steps=0))
except BaseException as e:
 atomic_json(D/'postprocess13_failure.json',dict(utc=time.time(),error=repr(e)));raise
