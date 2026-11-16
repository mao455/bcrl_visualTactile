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

def prepare_fixed():
 """Fork immutable fixed-data evidence; resume full DP state in a new root."""
 import torch
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.bounded_extension import long_fixed_profile
 source=Path(sys.argv[sys.argv.index('--source')+1]).resolve()
 registration=Path(sys.argv[sys.argv.index('--registration')+1]).resolve()
 epochs=int(sys.argv[sys.argv.index('--epochs')+1]);policy=long_fixed_profile(maximum_epochs=epochs)
 assert not ROOT.exists() and read(source/'owner_exit.json')['children_reaped']
 run=read(registration);assert time.time()<run['training_deadline']
 ROOT.mkdir();(ROOT/'logs').mkdir()
 session=copy.deepcopy(read(source/'session.json'))
 session.update(version='fixed56_long_session_v1',fixed56_long=True,
  started_utc=run['created_utc'],deadline=run['deadline'],training_deadline=run['training_deadline'],
  collection_deadline=run['training_deadline'],cache_deadline=run['training_deadline'],offline_deadline=run['deadline'],
  resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],
  authorization='Fixed56 substantial full-state DP continuation, regenerated backend, autonomous development')
 atomic_json(ROOT/'session.json',session)
 for name in ['metadata.json','source_data.json','selection.json','training_sampling.json','cuda_allocator_policy.json','frozen_visual_io.json']:
  if (source/name).exists():shutil.copyfile(source/name,ROOT/name)
 reg=dict(profile=profile(),roles={});extension=dict(policy=policy,roles={});cache_hashes={}
 for role in 'AB':
  src=source/role;dest=ROOT/role;dest.mkdir()
  # These inputs remain immutable. New learned history and rear plans are rebuilt.
  (dest/'state_data').symlink_to(src/'state_data',target_is_directory=True)
  for name in ['cache_index.json','continuation.json']:shutil.copyfile(src/name,dest/name)
  (dest/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
  atomic_json(dest/'session.json',session)
  def copy_state(a,b):
   return os.link(a,b) if str(a).endswith('.pt') else shutil.copy2(a,b)
  shutil.copytree(src/'training',dest/'training',copy_function=copy_state)
  (dest/'training/finished.json').unlink()
  old=torch.load(src/'training/last.pt',map_location='cpu',weights_only=False,mmap=True)
  assert old['updates']==read(src/'training/finished.json')['updates']
  data=read(dest/'state_data/data.json');ids=data['splits']['train']
  assert len(ids)==56 and len(data['splits']['validation'])==6
  b=budget(len(ids),epochs)
  reg['roles'][role]=dict(**b,data_sha256=sha256(dest/'state_data/data.json'),episode_multiplicities=dict.fromkeys(ids,1))
  extension['roles'][role]=dict(version='fixed56_full_state_extension_v1',
   last_sha256=sha256(src/'training/last.pt'),old_policy=old['configuration']['bounded_extension'],
   new_policy=policy,new_budget=b,restart_learning_rate=3e-5 if epochs==192 else 1.5e-5,
   original_epoch=old['epoch'],original_updates=old['updates'],additional_updates=b['planned_optimizer_updates']-old['updates'])
  front=src/'training/best.pt';rear=src/'backend'/role/'training/best.pt'
  atomic_json(dest/'backend_continuation.json',dict(path=str(front),sha256=sha256(front),
   previous_backend=str(rear),previous_backend_sha256=sha256(rear)))
  cache_hashes[role]=sha256(dest/'cache_index.json')
 atomic_json(ROOT/'full_trajectory_batch.json',reg);atomic_json(ROOT/'bounded_extension.json',extension)
 atomic_json(ROOT/'gate.json',dict(passed=True,inherited_input_gate=str(source/'gate.json'),
  inherited_input_gate_sha256=sha256(source/'gate.json'),meaning='Identical fixed56 numeric/visual/mask/stat inputs'))
 atomic_json(ROOT/'prepared.json',dict(complete=True,continue_student_pair=True,reuse_frozen_cache=True,
  cache_index_sha256=cache_hashes,source=str(source),registration=str(registration),registration_sha256=sha256(registration),
  source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},utc=time.time()))


def cache():
 from visuotactile.precision import state_fp32
 from visuotactile.native_data import cache_visual_pair
 state_fp32()
 if (ROOT/'A/continuation.json').exists():
  from tools.tight_fresh_dagger import cache as aggregate_cache
  aggregate_cache(ROOT)
 else:
  cache_visual_pair(ROOT/'A');cache_visual_pair(ROOT/'B',reuse_index=ROOT/'A/cache_index.json')
 atomic_json(ROOT/'cache_complete.json',dict(complete=True,utc=time.time()))

def backend_prepare(role):
 from tools.tight_fresh_backend import prepare as prep
 from visuotactile.bounded_extension import fresh_layout_profile,phase_dagger_profile,long_fixed_profile,matched_update_profile
 policy=long_fixed_profile('backend',64) if read(ROOT/'session.json').get('fixed56_long') else phase_dagger_profile('backend') if read(ROOT/'session.json').get('phase_dagger') else fresh_layout_profile('backend')
 if read(ROOT/'session.json').get('matched_update_diagnostic'):
  policy=matched_update_profile('backend',64)
 assert read(ROOT/role/'training/finished.json')['complete']
 prep(ROOT,role,policy['maximum_epochs'],continue_rear=(ROOT/role/'continuation.json').exists(),offline_anchor_batch=32,frozen_visual_io=read(ROOT/'frozen_visual_io.json'))
 rear=ROOT/role/'backend'
 for p in [rear/'session.json',rear/'configuration.json',rear/role/'session.json',rear/role/'configuration.json']:
  v=read(p);v['residual_pair'].update(bounded_extension=policy,minimum_control_tick_epochs=policy['minimum_epochs'],comparison_control_tick_epochs=[])
  if read(ROOT/'session.json').get('matched_update_diagnostic'):
   v['residual_pair']['matched_update_stop']=read(ROOT/'matched_updates.json')['rear'][role]
  atomic_json(p,v)
 atomic_json(rear/'extension_prepared.json',dict(complete=True,utc=time.time()))

def evaluation_prepare():
 from tools.tight_fresh_rollouts import completed_pair
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 directory=ROOT/'development'
 prepared=read(ROOT/'prepared.json')
 seeds=prepared.get('development_seeds',DEV_SEEDS)
 comparison=prepared.get('comparison_baseline')
 assert len(seeds)==(2 if comparison else 4) and seeds==list(range(seeds[0],seeds[0]+len(seeds)))
 accepted=read(RUNS/'tight_reliable_eval_20261116_01/28500100/plan.json')
 batch_prepare(ROOT,directory,lanes=8,count=4,seed=seeds[0],
  execution_backend=accepted['execution_backend'],queued_gpu=True,vision_microbatch=4)
 plan=read(directory/'plan.json')
 if comparison:
  baseline=prepared['comparison_baseline_pairs']
  for role in 'AB':assert completed_pair(Path(comparison),role)==baseline[role]
  plan['pairs'].update({r+'_baseline':p for r,p in baseline.items()})
  templates={c['spec']['seed']:c for c in plan['cases']};cases=[]
  for seed in seeds:
   for role in ('A','A_baseline','B','B_baseline'):
    c=copy.deepcopy(templates[seed]);c.update(role=role,name=role+'_'+c['spec']['identity'])
    cases.append(c)
  plan['cases']=cases
  plan['comparison']=dict(baseline=comparison,scope='Four complete frozen policies in the same actual paired reset wave; development only')
 # No reuse across waves; native reset stream and actual state checks match the accepted evaluator.
 for c in plan['cases']:
  c['spec']['narrow_reset']=reset_configuration(read(ROOT/'session.json')['domain'],c['spec']['seed'],.10)
  c['spec']['layout']=read(ROOT/'session.json')['layout']
  c['video']=c['spec']['seed']==seeds[0]
 plan['evaluation_scope']=dict(protocol='Full native task, no grasp censor; paired private native reset and pre-action verification',development_seeds=seeds,test_used=False)
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
   if not (read(ROOT/'prepared.json').get('evaluation_only_recovery') or read(ROOT/'prepared.json').get('evaluation_only_snapshot')):
    if read(ROOT/'prepared.json').get('reuse_frozen_cache'):
     for role,digest in read(ROOT/'prepared.json')['cache_index_sha256'].items():
      assert sha256(ROOT/role/'cache_index.json')==digest
     atomic_json(ROOT/'cache_complete.json',dict(complete=True,reused_unchanged_inputs=True,utc=time.time()))
    else:execute('cache',[__spec__.name,'cache','--root',ROOT],ROOT/'cache_complete.json')
    if read(ROOT/'prepared.json').get('continue_student_pair'):
     g=read(ROOT/'gate.json');assert g['passed'] and sha256(g['inherited_input_gate'])==g['inherited_input_gate_sha256']
    else:execute('gate',['tools.tight_fresh_train','gate','--root',ROOT],ROOT/'gate.json')
    for role in 'AB':
     execute('dp_'+role,['tools.tight_fresh_train','train','--root',ROOT,'--role',role,'--epochs',read(ROOT/'bounded_extension.json')['policy']['maximum_epochs'],'--offline-anchor-batch',32,'--full-trajectory-batch'],ROOT/role/'training/finished.json')
    for role in 'AB':
     rear=ROOT/role/'backend'
     execute('rear_prepare_'+role,[__spec__.name,'backend_prepare',role,'--root',ROOT],rear/'extension_prepared.json')
     execute('rear_cache_'+role,['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json')
     execute('rear_train_'+role,['visuotactile.anchor_continue','residual-train','--root',rear/role,'--job','image','--output',rear/role/'training'],rear/role/'training/finished.json')
     execute('rear_offline_'+role,['visuotactile.anchor_continue','residual-offline','--root',rear/role,'--job','image','--output',rear/role/'offline'],rear/role/'offline/result.json')
   else:
    from tools.tight_fresh_rollouts import completed_pair
    registered=read(ROOT/'prepared.json')['pairs']
    for role in 'AB':assert completed_pair(Path(read(ROOT/'prepared.json')['source']),role)==registered[role]
   execute('evaluation_prepare',[__spec__.name,'evaluation_prepare','--root',ROOT],ROOT/'evaluation_prepared.json')
   execute('evaluation',[__spec__.name,'evaluation_worker','--root',ROOT],ROOT/'development/result.json')
   assert read(ROOT/'development/paired_initial_check.json')['passed']
   atomic_json(ROOT/'complete.json',dict(complete=True,utc=time.time(),dagger_complete=False,dagger_round_trained=read(ROOT/'prepared.json').get('continue_student_pair',False)))
   atomic_json(ROOT/'status.json',dict(stage='complete_pending_scientific_review',utc=time.time()))
  except BaseException:
   atomic_json(ROOT/'failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
  finally:
   stop(child);(ROOT/'STOP_MONITOR').touch()
   if monitor is not None:
    try:monitor.wait(timeout=15)
    except subprocess.TimeoutExpired:stop(monitor)
   atomic_json(ROOT/'owner_exit.json',dict(utc=time.time(),children_reaped=True,monitor_returncode=None if monitor is None else monitor.poll()))

def prepare_development():
 """Evaluate a frozen complete pair on a registered additional four-family wave."""
 from tools.tight_fresh_rollouts import completed_pair
 source=Path(sys.argv[sys.argv.index('--source')+1]).resolve()
 registration=Path(sys.argv[sys.argv.index('--registration')+1]).resolve()
 baseline=Path(sys.argv[sys.argv.index('--baseline')+1]).resolve() if '--baseline' in sys.argv else None
 start=int(sys.argv[sys.argv.index('--seed')+1]);seeds=list(range(start,start+(2 if baseline else 4)))
 reg=read(registration);assert set(seeds)<=set(reg['development_seeds'])
 assert not set(seeds)&set(reg['heldout_seeds'])
 assert not ROOT.exists() and read(source/'owner_exit.json')['children_reaped']
 pairs={role:completed_pair(source,role) for role in 'AB'}
 session=copy.deepcopy(read(source/'session.json'))
 session.update(resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],deadline=reg['deadline'],training_deadline=reg['training_deadline'])
 assert time.time()<session['deadline']-1800
 ROOT.mkdir();(ROOT/'logs').mkdir()
 for role in 'AB':(ROOT/role).symlink_to(source/role,target_is_directory=True)
 atomic_json(ROOT/'session.json',session)
 atomic_json(ROOT/'prepared.json',dict(complete=True,evaluation_only_snapshot=True,
  source=str(source),pairs=pairs,development_seeds=seeds,registration=str(registration),
  **(dict(comparison_baseline=str(baseline),comparison_baseline_pairs={r:completed_pair(baseline,r) for r in 'AB'}) if baseline else {}),
  registration_sha256=sha256(registration),optimizer_updates_added=0,
  source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},utc=time.time()))


def prepare_evaluation_recovery():
 """Resume an unstarted evaluation without changing or retraining either pair."""
 from tools.tight_fresh_rollouts import completed_pair
 source=Path(sys.argv[sys.argv.index('--source')+1]).resolve()
 assert not ROOT.exists() and read(source/'owner_exit.json')['children_reaped']
 assert 'Incomplete stage evaluation_prepare' in read(source/'failure.json')['error']
 assert not (source/'development').exists() and not (source/'complete.json').exists()
 assert not read(source/'prepared.json').get('evaluation_only_recovery')
 pairs={role:completed_pair(source,role) for role in 'AB'}
 session=read(source/'session.json');assert time.time()<session['deadline']-1800
 ROOT.mkdir();(ROOT/'logs').mkdir()
 for role in 'AB':(ROOT/role).symlink_to(source/role,target_is_directory=True)
 session.update(resource_root=str(ROOT),owned_worker_roots=[str(ROOT)])
 atomic_json(ROOT/'session.json',session)
 atomic_json(ROOT/'prepared.json',dict(complete=True,evaluation_only_recovery=True,continue_student_pair=True,
  source=str(source),pairs=pairs,physical_attempts_repeated=0,optimizer_updates_added=0,
  source_failure_sha256=sha256(source/'failure.json'),original_deadline=session['deadline'],
  source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},utc=time.time()))

def wait_and_run():
 """Wait for the current capture owner to finish; never compete with its GPU."""
 waiting=RUNS/'tight_layout10_20261116_learning_wait02'
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
 if '--root' in sys.argv:ROOT=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 if sys.argv[1]=='backend_prepare':backend_prepare(sys.argv[2])
 else:globals()[sys.argv[1]]()
