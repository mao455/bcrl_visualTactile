"""Bounded read-only OS page warming; never loads a model or changes training."""
from pathlib import Path
import collections,json,os,subprocess,time
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01').resolve();D=Path('outputs/diagnostics/tight_full_curve_20261127_01');read=lambda p:json.loads(p.read_text())
os.nice(15);os.sched_setaffinity(0,{16,17});subprocess.run(['ionice','-c','2','-n','7','-p',str(os.getpid())],check=True)
def available():
 return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def current():
 values=[]
 for c in ['old_only','plus_corrected24']:
  for role in 'AB':
   child=R/c/role;p=child/'training/progress.json'
   if p.exists() and not (child/'training/finished.json').exists():
    v=read(p);values.append((v['utc'],child,v))
 return max(values,key=lambda v:v[0]) if values else None
started=time.time();log=D/'prefetch_pilot.jsonl';assert not log.exists();previous=None;warmed=set();summaries=[]
for phase,on in [('off_before',False),('on_first',True),('off_reverse',False),('on_reverse',True)]:
 begin=time.time();first=current();nbytes=0;files=0;max_batch=0
 while time.time()-begin<45:
  item=current()
  if item is None or time.time()-item[0]>90:time.sleep(.5);continue
  _,child,v=item
  with log.open('a') as f:f.write(json.dumps(dict(utc=time.time(),phase=phase,child=str(child),epoch=v['epoch'],updates=v['updates'],seconds=v['seconds'],available_bytes=available()))+'\n')
  if on and available()>18*2**30:
   order=read(child/'training'/('deck_%03d.json'%v['epoch']))['order'];offset=v['batch_index']*2;ids=order[offset:offset+2]
   paths=[child/'state_data/perception'/(i+'.pt') for i in ids];key=(str(child),v['epoch'],v['batch_index'])
   size=sum(p.stat().st_size for p in paths)
   if key not in warmed and size<=3*2**30:
    warmed.add(key);max_batch=max(max_batch,size)
    for p in paths:
     before=p.stat();fd=os.open(p,os.O_RDONLY)
     try:
      os.posix_fadvise(fd,0,0,os.POSIX_FADV_SEQUENTIAL)
      while True:
       block=os.read(fd,8*2**20)
       if not block:break
       nbytes+=len(block)
     finally:os.close(fd)
     after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);files+=1
  time.sleep(.2 if on else 1)
 last=current();elapsed=time.time()-begin
 summaries.append(dict(phase=phase,on=on,seconds=elapsed,start_update=first[2]['updates'] if first else None,end_update=last[2]['updates'] if last else None,warmed_read_bytes=nbytes,files_read=files,maximum_next_batch_bytes=max_batch));print(json.dumps(summaries[-1]),flush=True)
atomic_json(D/'prefetch_pilot_summary.json',dict(started=started,ended=time.time(),pid=os.getpid(),phases=summaries,model_updates=0,rng_draws=0,physics_steps=0,input_or_model_payloads_changed=False,scope='Live sequential on/off page-warming pilot; episode mixtures differ and timing does not establish a causal long-run speedup'))
