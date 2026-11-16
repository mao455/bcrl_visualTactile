"""Fresh common-layout training and paired native development; bounded stages."""
import copy,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from tools.runtime import supervised_owner
from scripts.tight_layout_20261116.pilot import TREE,read,reset_configuration,stop
RUNS=TREE/'outputs/visuotactile/run'
SOURCE=RUNS/'tight_layout10_20261116_01_recovery02'
ROOT=RUNS/'tight_layout10_20261116_learning01'
DEV_SEEDS=list(range(28820000,28820004))

def prepare():
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.frozen_visual_io import mapped_profile
 from visuotactile.bounded_extension import fresh_layout_profile
 assert read(SOURCE/'prepared.json')['source_hashes_verified']
 assert read(SOURCE/'collection_owner_exit.json')['children_reaped']
 ROOT.mkdir(exist_ok=False);(ROOT/'logs').mkdir()
 s=read(SOURCE/'session.json');s.update(resource_root=str(ROOT.resolve()),owned_worker_roots=[str(ROOT.resolve())],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip())
 atomic_json(ROOT/'session.json',s)
 for name in ['metadata.json','source_data.json','selection.json']:
  shutil.copyfile(SOURCE/name,ROOT/name)
 old=RUNS/'tight_budget_20261116_01'
 for name in ['training_sampling.json','cuda_allocator_policy.json']:shutil.copyfile(old/name,ROOT/name)
 atomic_json(ROOT/'frozen_visual_io.json',mapped_profile())
 reg=dict(profile=profile(),roles={})
 for role in 'AB':
  dest=ROOT/role;out=dest/'state_data';out.mkdir(parents=True)
  for name in ['data.json','statistics.pt','numeric.pt']:
   os.link(SOURCE/role/'state_data'/name,out/name)
  (dest/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
  atomic_json(dest/'session.json',s)
  d=read(out/'data.json');ids=d['splits']['train']
  reg['roles'][role]=dict(**budget(len(ids),80),data_sha256=sha256(out/'data.json'),episode_multiplicities=dict.fromkeys(ids,1))
 assert read(ROOT/'A/state_data/data.json')['splits']==read(ROOT/'B/state_data/data.json')['splits']
 atomic_json(ROOT/'full_trajectory_batch.json',reg)
 atomic_json(ROOT/'bounded_extension.json',dict(policy=fresh_layout_profile(),roles={}))
 atomic_json(ROOT/'prepared.json',dict(complete=True,source=str(SOURCE.resolve()),random_student_initialization=True,
  old_student_loaded=False,development_seeds=DEV_SEEDS,final_test_seeds=read(SOURCE/'collection_plan.json')['test_seeds'],
  unchanged_sampling_sha256=sha256(ROOT/'training_sampling.json'),deadline=s['deadline'],source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},utc=time.time()))

def cache():
 from visuotactile.precision import state_fp32
 from visuotactile.native_data import cache_visual_pair
 state_fp32();cache_visual_pair(ROOT/'A');cache_visual_pair(ROOT/'B',reuse_index=ROOT/'A/cache_index.json')
 atomic_json(ROOT/'cache_complete.json',dict(complete=True,utc=time.time()))

def backend_prepare(role):
 from tools.tight_fresh_backend import prepare as prep
 from visuotactile.bounded_extension import fresh_layout_profile
 assert read(ROOT/role/'training/finished.json')['complete']
 prep(ROOT,role,40,offline_anchor_batch=32,frozen_visual_io=read(ROOT/'frozen_visual_io.json'))
 rear=ROOT/role/'backend'
 for p in [rear/'session.json',rear/'configuration.json',rear/role/'session.json',rear/role/'configuration.json']:
  v=read(p);v['residual_pair'].update(bounded_extension=fresh_layout_profile('backend'),minimum_control_tick_epochs=20,comparison_control_tick_epochs=[]);atomic_json(p,v)
 atomic_json(rear/'extension_prepared.json',dict(complete=True,utc=time.time()))

def evaluation_prepare():
 from tools.tight_fresh_rollouts import completed_pair
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 directory=ROOT/'development'
 batch_prepare(ROOT,directory,lanes=8,count=4,seed=DEV_SEEDS[0])
 plan=read(directory/'plan.json')
 # No reuse across waves; native reset stream and actual state checks match the accepted evaluator.
 for c in plan['cases']:
  c['spec']['narrow_reset']=reset_configuration(read(ROOT/'session.json')['domain'],c['spec']['seed'],.10)
  c['spec']['layout']=read(ROOT/'session.json')['layout']
  c['video']=c['spec']['seed']==DEV_SEEDS[0]
 plan['evaluation_scope']=dict(protocol='Full native task, no grasp censor; paired private native reset and pre-action verification',development_seeds=DEV_SEEDS,test_used=False)
 plan['source_sha256']={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116','scripts/tight_fulltask_20261116'] for p in (TREE/folder).rglob('*.py')}
 atomic_json(directory/'plan.json',plan)
 atomic_json(ROOT/'evaluation_prepared.json',dict(complete=True,plan_sha256=sha256(directory/'plan.json'),utc=time.time()))

def evaluation_worker():
 from scripts.tight_fulltask_20261116.paired_eval import install
 install(ROOT/'development')
 from scripts.tight_fulltask_20261116.video_eval import main
 sys.argv=[sys.argv[0],'--output',str(ROOT/'development')];main()

@supervised_owner
def run():
 with (ROOT/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (ROOT/'owner_started.json').exists()
  for name,digest in read(ROOT/'prepared.json')['source_sha256'].items():assert sha256(TREE/name)==digest,name
  atomic_json(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time(),source_tree=str(TREE)))
  monitor=child=None
  def execute(name,args,receipt):
   nonlocal child
   assert not receipt.exists(),'Duplicate stage '+name
   start=time.time();command=[sys.executable,'-m',*map(str,args)]
   with (ROOT/'logs'/(name+'.log')).open('x') as log:
    child=subprocess.Popen(command,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
   atomic_json(ROOT/'logs'/(name+'_started.json'),dict(command=command,pid=child.pid,utc=start))
   while child.poll() is None:
    progress=[p.stat().st_mtime for p in ROOT.rglob('*.jsonl') if 'resources' not in p.parts and not p.name.startswith('allocator_')]
    progress += [p.stat().st_mtime for p in ROOT.rglob('*progress*.json')]
    if time.time()-max([start]+progress)>1200:raise TimeoutError('No recorded learning/cache/control progress for1200s: '+name)
    assert monitor.poll() is None,'Monitor exited while stage active'
    s=read(ROOT/'session.json');deadline=s['training_deadline'] if name.startswith(('dp_','rear_','cache')) else s['deadline']
    if time.time()>deadline-30 or (ROOT/'STOP').exists():raise TimeoutError('Original deadline or STOP')
    atomic_json(ROOT/'status.json',dict(stage=name,pid=child.pid,utc=time.time(),elapsed=time.time()-start))
    time.sleep(3)
   assert child.returncode==0 and receipt.exists(),'Incomplete stage '+name
   atomic_json(ROOT/'logs'/(name+'_complete.json'),dict(exit_code=child.returncode,receipt_sha256=sha256(receipt),utc=time.time(),seconds=time.time()-start,monitor_alive=monitor.poll() is None))
   child=None
  try:
   with (ROOT/'monitor.log').open('x') as log:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   until=time.time()+30
   while not (ROOT/'shared_resources.json').exists():
    assert monitor.poll() is None and time.time()<until;time.sleep(.25)
   execute('cache',[__spec__.name,'cache','--root',ROOT],ROOT/'cache_complete.json')
   execute('gate',['tools.tight_fresh_train','gate','--root',ROOT],ROOT/'gate.json')
   for role in 'AB':
    execute('dp_'+role,['tools.tight_fresh_train','train','--root',ROOT,'--role',role,'--epochs',80,'--offline-anchor-batch',32,'--full-trajectory-batch'],ROOT/role/'training/finished.json')
   for role in 'AB':
    rear=ROOT/role/'backend'
    execute('rear_prepare_'+role,[__spec__.name,'backend_prepare',role,'--root',ROOT],rear/'extension_prepared.json')
    execute('rear_cache_'+role,['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json')
    execute('rear_train_'+role,['visuotactile.anchor_continue','residual-train','--root',rear/role,'--job','image','--output',rear/role/'training'],rear/role/'training/finished.json')
    execute('rear_offline_'+role,['visuotactile.anchor_continue','residual-offline','--root',rear/role,'--job','image','--output',rear/role/'offline'],rear/role/'offline/result.json')
   execute('evaluation_prepare',[__spec__.name,'evaluation_prepare','--root',ROOT],ROOT/'evaluation_prepared.json')
   execute('evaluation',[__spec__.name,'evaluation_worker','--root',ROOT],ROOT/'development/result.json')
   assert read(ROOT/'development/paired_initial_check.json')['passed']
   atomic_json(ROOT/'complete.json',dict(complete=True,utc=time.time(),dagger_complete=False))
   atomic_json(ROOT/'status.json',dict(stage='complete_pending_scientific_review',utc=time.time()))
  except BaseException:
   atomic_json(ROOT/'failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
  finally:
   stop(child);(ROOT/'STOP_MONITOR').touch()
   if monitor is not None:
    try:monitor.wait(timeout=15)
    except subprocess.TimeoutExpired:stop(monitor)
   atomic_json(ROOT/'owner_exit.json',dict(utc=time.time(),children_reaped=True,monitor_returncode=None if monitor is None else monitor.poll()))

def wait_and_run():
 """Wait for the current capture owner to finish; never compete with its GPU."""
 waiting=RUNS/'tight_layout10_20261116_learning_wait01'
 waiting.mkdir(exist_ok=False)
 atomic_json(waiting/'started.json',dict(pid=os.getpid(),source=str(SOURCE.resolve()),utc=time.time()))
 try:
  while not (SOURCE/'collection_owner_exit.json').exists():
   assert time.time()<read(SOURCE/'session.json')['collection_deadline'],'Original collection deadline'
   assert not (SOURCE/'STOP').exists(),'Capture STOP retained'
   atomic_json(waiting/'status.json',dict(stage='waiting_capture_owner_exit',utc=time.time()))
   time.sleep(3)
  assert (SOURCE/'prepared.json').exists(),'Capture/data did not complete; preserve failure'
  prepare()
  atomic_json(waiting/'status.json',dict(stage='starting_learning_owner',utc=time.time()))
  run()
  atomic_json(waiting/'complete.json',dict(complete=True,utc=time.time()))
 except BaseException:
  atomic_json(waiting/'failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise

if __name__=='__main__':
 if sys.argv[1]=='backend_prepare':backend_prepare(sys.argv[2])
 else:globals()[sys.argv[1]]()
