"""Finite stage scheduler; scientific consumers remain canonical."""
from pathlib import Path
import copy,json,os,subprocess,sys,time
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';C=P/'configs/experiments/full_curve_20261127';py=sys.executable
read=lambda p:json.loads(p.read_text())
def run(name,args,cpus='8-15'):
 env=dict(os.environ,PYTHONPATH=str(P),RESEARCH_CPUS=cpus,OMP_NUM_THREADS='2')
 with (D/(name+'.log')).open('x') as stream:
  process=subprocess.Popen([py,*args],cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 atomic_json(D/(name+'_pid.json'),dict(pid=process.pid,args=args,utc=time.time()))
 return process
def join(p,name):
 while p.poll() is None:
  if time.time()>read(R/'session.json')['deadline']:raise TimeoutError('Finite round deadline expired')
  time.sleep(2)
 if p.returncode:raise RuntimeError(name+' failed; preserve and inspect')
def wait_for(owners):
 while not all((o/'project_complete.json').exists() for o in owners):
  for o in owners:
   if (o/'project_failure.json').exists():raise RuntimeError(str(o)+' failed')
  if time.time()>read(R/'session.json')['deadline']:raise TimeoutError('Finite collection wait expired')
  time.sleep(2)
def job(root,name,operation,entry,args):
 owner=root/('owner_'+name);owner.mkdir();s=copy.deepcopy(read(R/'session.json'));s['resource_root']=str(owner);atomic_json(owner/'session.json',s)
 link=root/'shared_resources.json';link.unlink(missing_ok=True);link.symlink_to(owner/'shared_resources.json')
 path=C/(root.name+'_'+name+'.json');atomic_json(path,dict(version='tight_project_job_v1',operation=operation,resource_root=str(owner),stages=[dict(entry=entry,args=args)]));return path
try:
 wait_for([R/'experts_00',R/'experts_01'])
 p=run('experts_02',['-m','outputs.diagnostics.tight_full_curve_20261127_01.dispatch','project','prepare','--job',str(C/'experts_02.json')],'16-23')
 e=run('evaluation_e00_start',['-m','tools.tight_machine_review','project','evaluate','--job',str(C/'evaluation_e00_start.json')])
 join(p,'experts02')
 prep=run('prepare_learning',[str(D/'prepare_learning.py')],'16-23');join(prep,'prepare_learning')
 # Cache preparation is a single process, sharing a bounded SHA audit and real DINO features.
 cache_root=R/'plus_corrected24';cj=job(cache_root,'cache','prepare','curve_cache',['--root',str(R)])
 cache=run('cache',['-m','outputs.diagnostics.tight_full_curve_20261127_01.cache_worker','project','prepare','--job',str(cj)],'16-23');join(cache,'cache')
 join(e,'evaluation0')
 fronts=[]
 for c,cpus in [('old_only','8-15'),('plus_corrected24','16-23')]:
  root=R/c;j=job(root,'frontend','train','curve_train',['frontend','--root',str(root)])
  fronts.append((run(c+'_frontend',['-m','outputs.diagnostics.tight_full_curve_20261127_01.train_worker','project','train','--job',str(j)],cpus),c))
 for p,n in fronts:join(p,n+' frontend')
 atomic_json(R/'all_frontends_complete.json',dict(utc=time.time()))
 for epochs in (30,60):
  rears=[]
  for c,cpus in [('old_only','8-15'),('plus_corrected24','16-23')]:
   root=R/c;j=job(root,'rear'+str(epochs),'train','curve_train',['rear','--root',str(root),'--epochs',str(epochs)])
   rears.append((run(c+'_rear'+str(epochs),['-m','outputs.diagnostics.tight_full_curve_20261127_01.train_worker','project','train','--job',str(j)],cpus),c))
  for p,n in rears:join(p,n+' rear'+str(epochs))
  prep=run('prepare_evaluation'+str(epochs),[str(D/'prepare_evaluation.py'),'--epoch',str(epochs)]);join(prep,'prepare eval')
  # One2lane queue at a time: numerical results and sampler/layout receipts retained.
  for role in 'AB':
   p=run(f'evaluation_e{epochs:02d}_{role}',['-m','tools.tight_machine_review','project','evaluate','--job',str(C/f'evaluation_e{epochs:02d}_{role}.json')]);join(p,'evaluation')
 finalprep=run('prepare_final16',[str(D/'prepare_evaluation.py'),'--epoch','60','--holdout']);join(finalprep,'prepare final16')
 for role in 'AB':
  p=run(f'final16_{role}',['-m','tools.tight_machine_review','project','evaluate','--job',str(C/f'final16_{role}.json')]);join(p,'independent final16')
 atomic_json(R/'curve_complete.json',dict(utc=time.time(),milestones=[0,30,60],architecture_changed=False))
except BaseException as e:
 atomic_json(D/'pipeline_failure.json',dict(utc=time.time(),error=repr(e)));raise
