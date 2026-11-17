"""One A-only ordinary-sampling continuation; reuse existing matched endpoint."""
import copy,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
TREE=Path(__file__).resolve().parents[2]
RUNS=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ON=RUNS/'tight_balance76_20261117_01_recovery01'
ON_EVAL=RUNS/'tight_balance76_20261117_01_recovery02'
HIST=RUNS/'tight_long_20261117_01/round06/learning'
PARENT=RUNS/'tight_long_20261117_01/round05/learning'
ROOT=RUNS/'tight_sampling76_A_20261117_01'
OUT=TREE/'research_docs/evidence/tight_long_20261117/ordinary76_A'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):atomic_json(Path(p),v)

def prepare():
 import torch
 from visuotactile.bounded_extension import matched_update_profile
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.anchor_sampling import grasp_sampling_profile
 assert not ROOT.exists();ROOT.mkdir();(ROOT/'logs').mkdir();OUT.mkdir(parents=True,exist_ok=True)
 now=time.time();session=read(ON/'session.json')
 session.update(version='ordinary76_A_matched_endpoint_v1',started_utc=now,deadline=now+3*3600,training_deadline=now+2*3600,cache_deadline=now+2*3600,offline_deadline=now+3*3600,collection_deadline=now+3*3600,resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],reserve_bytes=8*1024**3,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),authorization='One A-only ordinary-start matched endpoint experiment, no new loss/DAgger; full24 development executions',phase_dagger=False,matched_update_diagnostic=True)
 session['resources']['min_disk_gib']=8
 assert shutil.disk_usage(ROOT).free>12*1024**3
 write(ROOT/'session.json',session)
 for name in ['metadata.json','selection.json','cuda_allocator_policy.json','frozen_visual_io.json','frozen_visual_hotset.json']:
  if (ON/name).exists():shutil.copyfile(ON/name,ROOT/name)
 sampling=read(ON/'training_sampling.json');sampling['anchors']=grasp_sampling_profile();assert sampling==read(HIST/'training_sampling.json');write(ROOT/'training_sampling.json',sampling)
 assert read(ROOT/'frozen_visual_hotset.json')['max_bytes']==4*1024**3
 protected={};models={}
 for r in ['A','B']:
  d=ROOT/r;d.mkdir();(d/'state_data').symlink_to(ON/r/'state_data',target_is_directory=True)
  shutil.copyfile(ON/r/'cache_index.json',d/'cache_index.json')
  if r=='A':
   shutil.copyfile(ON/r/'continuation.json',d/'continuation.json');write(d/'session.json',session);(d/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
  for p in [ON/r/'state_data/data.json',ON/r/'state_data/statistics.pt',PARENT/r/'training/best.pt',PARENT/r/'backend'/r/'training/best.pt']:
   protected[str(p)]=sha256(p)
 for label,source in [('onset',ON),('historical_ordinary',HIST)]:
  dp=torch.load(source/'A/training/best.pt',map_location='cpu',weights_only=False,mmap=True);rear=torch.load(source/'A/backend/A/training/best.pt',map_location='cpu',weights_only=False,mmap=True)
  models[label]=dict(dp_sha256=sha256(source/'A/training/best.pt'),rear_sha256=sha256(source/'A/backend/A/training/best.pt'),dp_epoch=dp['epoch'],dp_updates=dp['updates'],rear_epoch=rear['epoch'],rear_updates=rear['updates'],dp_continuation=dp['configuration']['continuation'],rear_continuation=rear['continuation_from'],data_sha256=dp['configuration']['data_sha256'],statistics_sha256=dp['configuration']['statistics_sha256'],dp_learning_rates=[r['next_learning_rates'] for r in dp['plateau_state']['events']],rear_learning_rates=[r['next_learning_rates'] for r in rear['plateau_state']['events']])
 assert models['onset']['dp_updates']==494 and models['onset']['rear_updates']==369
 assert models['onset']['dp_continuation']==models['historical_ordinary']['dp_continuation']
 assert models['onset']['data_sha256']==models['historical_ordinary']['data_sha256']
 assert all(x==[3e-5] for x in models['onset']['dp_learning_rates']) and all(x==[1e-4] for x in models['onset']['rear_learning_rates'])
 ids=read(ROOT/'A/state_data/data.json')['splits']['train'];assert len(ids)==76
 write(ROOT/'full_trajectory_batch.json',dict(profile=profile(),roles=dict(A=dict(**budget(76,13),data_sha256=sha256(ROOT/'A/state_data/data.json'),episode_multiplicities=dict.fromkeys(ids,1)))))
 write(ROOT/'bounded_extension.json',dict(policy=matched_update_profile('frontend',13),roles={}))
 write(ROOT/'gate.json',dict(passed=True,inherited_input_gate=str(ON/'gate.json'),inherited_input_gate_sha256=sha256(ON/'gate.json')))
 reg=dict(root=str(ROOT),created_utc=now,deadline=session['deadline'],training_deadline=session['training_deadline'],models=models,protected=protected,no_existing_strict_control=True,reason='Historical ordinary selected DP722/rear1207; no retained DP494/rear369 full pair',factor='Only correction start allocation v3 onset24 half -> previous v2 ordinary; original grasp allocation unchanged',dp_updates=494,dp_epochs=13,rear_updates=369,rear_epochs=3,selection='Both arms frozen endpoint DP494/rear369 for THIS comparison; ordinary final endpoint only, no alternate validation-best deployed',selection_limitation='Reused onset endpoint was historically validation-selected across longer run. Conditional matched-budget contrast, NOT prospectively randomized identical historical model selection; horizon informed by onset, cannot erase this bias.',learning_rate='DP3e-5/rear1e-4 constant through compared horizons; verified onset logs no reduction',rng=dict(model_seed=2026111401,dp_deck=2026111402,dp_noise=2026111403,rear=2026111404,anchor=2026111501,meaning='Same initial streams and episode order; selected starts can differ. Inference DDIM same per-lane seed7093.'),supervision='Original expert-only masks, A q_reached plus geometry, backend actual expert commands. No old-output loss.',development_seeds=list(range(28824000,28824008)),evaluation='Four6lane waves,2 families x3 complete A models; actual pre-action state check. Previously opened development; no final20 rerun. New6lane context differs historical8lane, rerun all3 in same wave.',budget=dict(hours=3,new_output_cap_gib=4,disk_reserve_gib=8,starting_free_bytes=shutil.disk_usage(ROOT).free,reason='Bounded new local budget; preserve all existing artifacts, no deletion'),reuse_diffusion_control='fork_diagnosis/same_input.json: complete causal histories, common generator states, self-replay command<1e-6rad; no repeat timing bug audit')
 write(ROOT/'registration.json',reg);write(OUT/'registration.json',reg)
 freeze=dict(source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_sampling_20261117','scripts/tight_fulltask_20261116'] for p in (TREE/folder).rglob('*.py')})
 write(ROOT/'prepared.json',freeze);print(ROOT)

def backend_prepare():
 from tools.tight_fresh_backend import prepare as prep
 from visuotactile.bounded_extension import matched_update_profile,rear_group_boundaries
 import torch
 assert read(ROOT/'A/training/finished.json')['updates']==494
 prep(ROOT,'A',3,continue_rear=True,offline_anchor_batch=32,frozen_visual_io=read(ROOT/'frozen_visual_io.json'))
 rear=ROOT/'A/backend';data=read(ROOT/'A/state_data/data.json');records=torch.load(ROOT/'A/state_data/numeric.pt',weights_only=False,map_location='cpu')['records']
 masks={i:records[i].get('supervision_mask',torch.ones(len(records[i]['time']),dtype=torch.bool)) for i in data['splits']['train']}
 boundary=rear_group_boundaries(masks,maximum_epochs=3)[369];assert boundary==dict(epoch=3,next_group=76)
 for p in [rear/'session.json',rear/'configuration.json',rear/'A/session.json',rear/'A/configuration.json']:
  v=read(p);v['residual_pair'].update(bounded_extension=matched_update_profile('backend',3),minimum_control_tick_epochs=3,comparison_control_tick_epochs=[],matched_update_stop=dict(updates=369,boundary=boundary))
  write(p,v)
 write(rear/'extension_prepared.json',dict(complete=True,boundary=boundary,utc=time.time()))

def evaluation_prepare(part):
 from tools.tight_fresh_rollouts import completed_pair
 part=int(part);directory=ROOT/f'comparison{part:02d}';directory.mkdir()
 template=read(ON_EVAL/f'comparison{part:02d}/development/plan.json')
 on=completed_pair(ON,'A');base=completed_pair(PARENT,'A');ordinary=completed_pair(ROOT,'A')
 plan=copy.deepcopy(template);plan.update(pairs=dict(A_ordinary=ordinary,A_baseline=base,A_onset=on),cases=[],resource_root=str(ROOT),source_commit=read(ROOT/'session.json')['source_commit'],source_sha256=read(ROOT/'prepared.json')['source_sha256'])
 plan['batch'].update(lanes=6);seeds=read(ROOT/'registration.json')['development_seeds'][part*2:part*2+2]
 for seed in seeds:
  c=next(c for c in template['cases'] if c['spec']['seed']==seed)
  for role in plan['pairs']:
   x=copy.deepcopy(c);x.update(role=role,name=role+'_'+x['spec']['identity'],video=seed in [28824002,28824006]);plan['cases'].append(x)
 plan['evaluation_scope']=dict(protocol='Full native natural termination, no censor. Three complete A policies, paired actual observable start. Development only.',seeds=seeds,test_used=False)
 plan['comparison']=dict(baseline=str(PARENT),onset=str(ON),ordinary=str(ROOT),sampling_only_conditional_budget=True,historical_selection_limitation=read(ROOT/'registration.json')['selection_limitation'])
 write(directory/'session.json',read(ROOT/'session.json'));write(directory/'plan.json',plan);print(directory)

def evaluation_worker(part):
 directory=ROOT/f'comparison{int(part):02d}'
 from scripts.tight_fulltask_20261116.paired_eval import install
 install(directory)
 from scripts.tight_fulltask_20261116.video_eval import main
 sys.argv=[sys.argv[0],'--output',str(directory)];main()

def run():
 from scripts.tight_layout_20261116.pilot import stop
 assert not (ROOT/'owner_started.json').exists()
 for p,h in read(ROOT/'prepared.json')['source_sha256'].items():assert sha256(TREE/p)==h,p
 write(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time()));monitor=child=None
 def execute(name,args,receipt):
  nonlocal child
  assert not receipt.exists();start=time.time();cmd=[sys.executable,'-m',*map(str,args)]
  with (ROOT/'logs'/(name+'.log')).open('x') as log:child=subprocess.Popen(cmd,cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
  write(ROOT/'logs'/(name+'_started.json'),dict(command=cmd,pid=child.pid,utc=start))
  while child.poll() is None:
   assert monitor.poll() is None
   deadline=read(ROOT/'session.json')['deadline' if name.startswith('eval') else 'training_deadline']
   if time.time()>deadline-30 or (ROOT/'STOP').exists():raise TimeoutError('Finite study deadline/stop')
   write(ROOT/'status.json',dict(stage=name,pid=child.pid,utc=time.time(),elapsed_s=time.time()-start));time.sleep(3)
  assert child.returncode==0 and receipt.exists(),name
  write(ROOT/'logs'/(name+'_complete.json'),dict(seconds=time.time()-start,utc=time.time(),receipt_sha256=sha256(receipt)));child=None
 try:
  with (ROOT/'monitor.log').open('x') as log:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (ROOT/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  execute('dp_A',['tools.tight_fresh_train','train','--root',ROOT,'--role','A','--epochs',13,'--offline-anchor-batch',32,'--full-trajectory-batch'],ROOT/'A/training/finished.json')
  rear=ROOT/'A/backend'
  execute('rear_prepare',[__spec__.name,'backend_prepare'],rear/'extension_prepared.json')
  execute('rear_cache',['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json')
  execute('rear_train',['visuotactile.anchor_continue','residual-train','--root',rear/'A','--job','image','--output',rear/'A/training'],rear/'A/training/finished.json')
  execute('rear_offline',['visuotactile.anchor_continue','residual-offline','--root',rear/'A','--job','image','--output',rear/'A/offline'],rear/'A/offline/result.json')
  for part in range(4):
   execute(f'eval_prepare{part}',[__spec__.name,'evaluation_prepare',part],ROOT/f'comparison{part:02d}/plan.json')
   execute(f'eval{part}',[__spec__.name,'evaluation_worker',part],ROOT/f'comparison{part:02d}/result.json')
  write(ROOT/'complete.json',dict(complete=True,utc=time.time(),no_dagger=True,no_promotion=True))
 except BaseException:
  write(ROOT/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  stop(child);(ROOT/'STOP_MONITOR').touch()
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  write(ROOT/'owner_exit.json',dict(children_reaped=True,utc=time.time()))
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
