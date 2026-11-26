"""Finite owned startup-only CPU placement aid; no science reads or signals."""
from pathlib import Path
import os,json,time,math,psutil
B=Path('/home/CNS2025331827/docs/bc+rl_RSS')
A=B/'outputs/diagnostics/tight_paired_dagger_20261122_01'
R=B/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
O=R/'owners/test_continuation15';D=A/'startup_affinity16_actual01'
MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.run_test_continuation15'
PYTHON='/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python'
OLD=set(range(8,24));WIDE=set(range(1,24));DEADLINE=1795684571.2261844

def require(v,msg):
 if not v:raise ValueError(msg)
def read(p):return json.loads(p.read_bytes())
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def same(p,b):
 try:return p.is_running() and abs(p.create_time()-b)<.001
 except psutil.NoSuchProcess:return False
def owned_thread(tid,p,b):
 if not same(p,b):return False
 try:
  fields=dict(x.split(':',1) for x in Path('/proc/'+str(tid)+'/status').read_text().splitlines() if ':' in x)
  return int(fields['Tgid'])==p.pid
 except (FileNotFoundError,ProcessLookupError):return False
def masks(p,b):
 out=[]
 for t in p.threads():
  if owned_thread(t.id,p,b):
   try:out.append((t.id,set(os.sched_getaffinity(t.id))))
   except ProcessLookupError:pass
 return out
def identity(s,p,b,w):
 require(type(p.pid) is int and p.pid>0 and type(b) in (int,float) and math.isfinite(b) and b>0,'PID/birth')
 require(same(p,b) and p.cmdline()==[PYTHON,'-m',MODULE,'worker','--output',str(w)] and p.cwd()==str(B) and p.nice()==10,'Exact own worker required')
 require(s['command']==p.cmdline() and 16<=s['stage']<=46 and w==R/'evaluation/test'/('wave%02d'%(s['stage']+17)),'Original remaining wave only')
 require(w==w.resolve() and not w.is_symlink(),'Canonical wave')
def change(p,b,cpus):
 tids=[]
 for tid,_ in masks(p,b):
  if owned_thread(tid,p,b):
   try:os.sched_setaffinity(tid,cpus);tids.append(tid)
   except ProcessLookupError:pass
 return tids
def ready(w):
 return (w/'initialization.json').is_file() or read(w/'environment_initialization.json').get('current_stage')=='environment_ready'
def probe(s):
 w=Path(s['command'][-1]);p=psutil.Process(s['pid']);b=p.create_time();identity(s,p,b,w)
 if time.time()-s['utc']<60 or ready(w):return None
 if read(w/'environment_initialization.json').get('current_stage')!='environment_constructor_started':return None
 observed=masks(p,b);require(observed and all(cpus==OLD for _,cpus in observed),'Original all-thread masks required')
 row=dict(started=time.time(),pid=p.pid,created=b,wave=str(w),stage=s['stage'],old=sorted(OLD),temporary=sorted(WIDE),nice=10,cap_seconds=40,signals_sent=0,external_mutations=0)
 # Durable intent before the only owned mutation; an attempted wave is never retried.
 write(D/(w.name+'_intent.json'),row)
 try:
  identity(s,p,b,w)
  if ready(w):row['readiness_race_no_change']=True;return row
  row['changed_threads']=change(p,b,WIDE);until=time.monotonic()+40
  while same(p,b) and time.monotonic()<until and not ready(w) and not (D/'STOP').exists():time.sleep(.5)
  row['native_initialized']=(w/'initialization.json').is_file();row['environment_stage']=read(w/'environment_initialization.json').get('current_stage')
 finally:
  if same(p,b):
   row['restored_threads']=change(p,b,OLD);current=masks(p,b);row['restored_masks']=sorted({tuple(sorted(c)) for _,c in current});row['restored']=bool(current) and all(c==OLD for _,c in current);require(row['restored'],'Failed original thread restoration')
  else:row['worker_exited']=True;row['restored']=None
  row['ended']=time.time();write(D/(w.name+'_exit.json'),row)
 return row

def main():
 require(Path.cwd().resolve()==B,'Canonical cwd');D.mkdir(exist_ok=False);write(D/'started.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),utc=time.time(),deadline=DEADLINE,indices=list(range(33,64)),external_mutations=0,signals_sent=0))
 attempted=set();error=None
 try:
  while time.time()<DEADLINE and not (D/'STOP').exists():
   if (A/'driver_test_continuation15/driver_exit.json').exists():break
   s=read(O/'project_status.json');wave=Path(s['command'][-1]).name
   if s['stage']>=16 and wave not in attempted:
    try:
     row=probe(s)
     if row is not None:attempted.add(wave)
    except (psutil.NoSuchProcess,FileNotFoundError):pass
   time.sleep(2)
 except BaseException as e:error=repr(e);raise
 finally:write(D/'exit.json',dict(utc=time.time(),pid=os.getpid(),attempted=sorted(attempted),error=error,no_science_or_configuration_changes=True,signals_sent=0,external_mutations=0))
if __name__=='__main__':main()
