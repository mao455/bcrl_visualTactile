"""Finite fixed-pool training experiment, with epoch recovery and no collection."""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.bounded_extension import profile,split_audit
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_fresh_20261116_04/round_19';ROOT=BASE/'tight_budget_20261116_01'
read=lambda p:json.loads(Path(p).read_text())

def prepare():
 import torch
 from visuotactile.trajectory_training import budget
 from visuotactile.frozen_visual_io import mapped_profile
 assert not ROOT.exists(),'Preserve existing experiment'
 assert shutil.disk_usage(BASE).free>28*2**30
 ROOT.mkdir();(ROOT/'logs').mkdir()
 session=read(OLD/'session.json');now=time.time()
 session.update(started_utc=now,deadline=now+6*3600,cache_deadline=now+6*3600,
  training_deadline=now+6*3600,offline_deadline=now+6*3600,resource_root=str(ROOT),
  owned_worker_roots=[str(ROOT)],reserve_bytes=12*2**30,
  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip(),
  authorization='Fixed pool and unchanged sampling; bounded 8-to24 training diagnosis; no DAgger',
  extension_from=str(OLD))
 session['resources']['min_disk_gib']=12
 atomic_json(ROOT/'session.json',session)
 for name in ['metadata.json','gate.json','training_sampling.json','cuda_allocator_policy.json',
              'backend_episode_sampling.json','backend_phase_weighting.json']:
  shutil.copy2(OLD/name,ROOT/name)
 atomic_json(ROOT/'frozen_visual_io.json',mapped_profile())
 reg=read(OLD/'full_trajectory_batch.json');extension=dict(policy=profile(),roles={})
 for role in 'AB':
  source=OLD/role;dest=ROOT/role;dest.mkdir();(dest/'training').mkdir()
  for name in ['state_data','cache_index.json']:(dest/name).symlink_to(source/name)
  shutil.copy2(source/'continuation.json',dest/'continuation.json')
  data=read(source/'state_data/data.json');audit=split_audit(data)
  atomic_json(dest/'split_audit.json',audit)
  s=torch.load(source/'training/last.pt',map_location='cpu',mmap=True,weights_only=False)
  assert s['epoch']==8 and s['updates']==s['ema_optimization_step']==1568
  assert len(s['optimizer']['state']) and s['optimizer']['param_groups'][0]['lr']==3e-5
  assert s['train_ids']==data['splits']['train'] and s['validation_ids']==data['splits']['validation']
  for name in ['last.pt','best.pt']:os.link(source/'training'/name,dest/'training'/name)
  for p in (source/'training').iterdir():
   if p.suffix in ('.json','.jsonl') and p.name not in ('finished.json','configuration.json'):
    shutil.copy2(p,dest/'training'/p.name)
  extension['roles'][role]=dict(last_sha256=sha256(source/'training/last.pt'),
   baseline_best_sha256=sha256(source/'training/best.pt'),source=str(source),
   original_source=s['configuration']['continuation'],epoch=8,updates=1568)
  reg['roles'][role].update(budget(392,24))
 atomic_json(ROOT/'full_trajectory_batch.json',reg);atomic_json(ROOT/'bounded_extension.json',extension)
 atomic_json(ROOT/'prepared.json',dict(utc=now,complete=True,original=str(OLD),roles=extension['roles'],
  deadline=session['deadline'],collection=False,policy=profile(),
  backend='New selected DP inputs, same original A11/B15 rear initialization and seed; fresh AdamW. First8 retained as same-input control; rear has no EMA.'))

def execute(name,args,receipt):
 if receipt.exists():return
 path=ROOT/'logs'/name;path.mkdir(exist_ok=True)
 attempts=list(path.glob('attempt*.json'))
 if len(attempts)>=2:raise RuntimeError('Bounded attempt limit: '+name)
 attempt=len(attempts)+1;start=time.time();command=[sys.executable,'-m',*map(str,args)]
 env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 with (path/f'attempt{attempt}.log').open('x') as log:
  child=subprocess.Popen(command,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  state=dict(name=name,command=command,pid=child.pid,started=start,attempt=attempt)
  atomic_json(path/f'attempt{attempt}.json',state)
  previous=None;changed=time.time()
  while child.poll() is None:
   candidates=[p for p in ROOT.rglob('*.jsonl') if 'resources' not in p.parts and 'allocator_' not in p.name]
   progress=max((p.stat().st_mtime for p in candidates),default=start)
   if progress!=previous:changed=time.time();previous=progress
   state.update(heartbeat=time.time(),last_progress=progress,elapsed=time.time()-start)
   atomic_json(path/'state.json',state)
   deadline=read(ROOT/'session.json')['deadline']
   if time.time()>deadline-180 or time.time()-changed>900 or (ROOT/'STOP').exists():
    os.killpg(child.pid,15)
    try:child.wait(timeout=90)
    except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
    raise RuntimeError('Bounded deadline/stop/stagnation: '+name)
   time.sleep(3)
  state.update(exit_code=child.returncode,ended=time.time());atomic_json(path/'state.json',state)
 if child.returncode or not receipt.exists():raise RuntimeError('Stage failed: '+name+' (preserve checkpoint/log; explicit recovery required)')

def dp():
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(execute,'dp_'+role,['tools.tight_fresh_train','train','--root',ROOT,'--role',role,
   '--epochs','24','--offline-anchor-batch','32','--full-trajectory-batch'],ROOT/role/'training/finished.json') for role in 'AB']
  for job in jobs:job.result()
 atomic_json(ROOT/'dp_complete.json',dict(complete=True,utc=time.time()))

def backend_prepare():
 from tools.tight_fresh_backend import prepare as prep
 assert read(ROOT/'dp_complete.json')['complete']
 for role in 'AB':
  rear=ROOT/role/'backend'
  if not rear.exists():
   prep(ROOT,role,24,continue_rear=True,fresh_recovery_weighting=read(ROOT/'backend_phase_weighting.json'),
    offline_anchor_batch=32,frozen_visual_io=read(ROOT/'frozen_visual_io.json'))
   for p in [rear/'session.json',rear/'configuration.json',rear/role/'session.json',rear/role/'configuration.json']:
    value=read(p);value['residual_pair'].update(bounded_extension=profile(),minimum_control_tick_epochs=16,
      comparison_control_tick_epochs=[8]);atomic_json(p,value)
   atomic_json(rear/'extension_prepared.json',dict(complete=True,utc=time.time(),policy=profile()))
  assert (rear/'extension_prepared.json').exists()

def rear():
 from concurrent.futures import ThreadPoolExecutor
 backend_prepare()
 # Cache+training stages are joined separately: a cache does not compete with a trainer.
 for stage in ['cache','train','offline']:
  with ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[]
   for role in 'AB':
    root=ROOT/role/'backend'
    if stage=='cache':args=['tools.tight_finish','residual_cache','--root',root];receipt=root/'cache_complete.json'
    else:
     args=['visuotactile.anchor_continue','residual-'+stage,'--root',root/role,'--job','image','--output',root/role/('training' if stage=='train' else 'offline')]
     receipt=root/role/('training/finished.json' if stage=='train' else 'offline/result.json')
    jobs.append(pool.submit(execute,'rear_'+stage+'_'+role,args,receipt))
   for job in jobs:job.result()
 atomic_json(ROOT/'training_complete.json',dict(complete=True,utc=time.time(),autonomy_complete=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dp','rear','train']);a=p.parse_args()
 if a.stage=='prepare':prepare();return
 try:
  if a.stage in ('dp','train'):dp()
  if a.stage in ('rear','train'):rear()
 except BaseException as e:
  atomic_json(ROOT/'failure.json',dict(stage=a.stage,error=repr(e),traceback=traceback.format_exc(),utc=time.time()))
  raise

if __name__=='__main__':main()
