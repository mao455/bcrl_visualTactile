"""New layout-only successful experts and train-only fresh A/B statistics."""
import json,copy,time,sys,os,subprocess
from pathlib import Path
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.pilot import TREE,ROOT as PILOT,read,reset_configuration,geometry
ROOT=TREE/'outputs/visuotactile/run/tight_layout10_20261116_01'

def prepare():
 assert read(PILOT/'owner_exit.json')['children_reaped']
 rows=read(PILOT/'pilot_complete.json')['rows'];selected=[r for r in rows if r['layout']['distance_m']==.10]
 assert len(selected)==3 and all(r['legal'] and r['success'] for r in selected)
 ROOT.mkdir(exist_ok=False);s=read(PILOT/'session.json');now=time.time()
 s.update(version='peg_spacing10_fresh30_v1',started_utc=now,deadline=now+24*3600,collection_deadline=now+3*3600,cache_deadline=now+6*3600,training_deadline=now+22*3600,offline_deadline=now+24*3600,
  resource_root=str(ROOT.resolve()),owned_worker_roots=[str(ROOT.resolve())],reserve_bytes=24*2**30,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip(),
  layout=dict(version='peg_spacing_reset_v1',distance_m=.10,source_pilot=str(PILOT.resolve())),
  visual_stride=1,control_hz=60,planning_hz=5,trainable_initialization='random DP and Transformer, no old student weights',authorization='New validated10cm layout; fresh common30/6 data, A/B training and full-policy shared-pool DAgger; independent24h bound')
 atomic_json(ROOT/'session.json',s);jobs=[]
 reservation=read(PILOT/'registration.json')['heldout_seed_reservation']
 for split,key in [('validation','validation'),('train','training')]:
  for seed in reservation[key]:
   name=f'layout10_{split}_{seed}'
   jobs.append(dict(identity=name,family=name,split=split,cohort='fresh',seed=seed,event_seed=seed+10000,collision_gate='live_collision_mesh_bounds_v1',narrow_reset=reset_configuration(s['domain'],seed,.10),layout=s['layout']))
 atomic_json(ROOT/'collection_plan.json',dict(jobs=jobs,target_train_successes=30,maximum_train_attempts=40,validation_attempts=6,test_seeds=reservation['test'],
  maximum_consecutive_task_failures=4,minimum_train_success_rate_after10=.7,source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},
  failed_starts_counted=True,no_wider_distribution=True,no_old_layout_training=True,utc=now))

def worker():
 import torch
 from tools.tight_state import environment
 from visuotactile import state_benchmark
 from visuotactile.position_control import ExpertPositionBridge
 from visuotactile.camera_validity import inspect_capture
 torch.set_num_threads(2);plan=read(ROOT/'collection_plan.json');s=read(ROOT/'session.json')
 for name,digest in plan['source_sha256'].items():assert sha256(TREE/name)==digest,'Source drift '+name
 original=state_benchmark.prepare
 def observed(env,isolated,spec,source,output):
  value=original(env,isolated,spec,source,output);geometry(env,output);return value
 state_benchmark.prepare=observed
 rows=read(ROOT/'collection_results.json') if (ROOT/'collection_results.json').exists() else []
 counts={k:sum(r['split']==k and r['success'] for r in rows) for k in ['train','validation']};attempts=0;consecutive=0;app=env=None
 completed={r['identity'] for r in rows};gate='--gate' in sys.argv
 if not gate:assert read(ROOT/'collection_gate_exit.json')['exit_code']==0
 try:
  app,env,player,lease=environment(ROOT,startup_diagnostics=True)
  atomic_json(ROOT/'initialization.json',dict(utc=time.time(),pid=os.getpid()))
  for spec in plan['jobs']:
   if spec['identity'] in completed:continue
   if spec['split']=='train' and counts['train']>=30:break
   lease.check(s['collection_deadline']);out=ROOT/'episodes'/spec['identity'];env.speed_config=dict(ExpertPositionBridge.speed_scheme)
   atomic_json(ROOT/'status.json',dict(stage='expert_collection',identity=spec['identity'],completed=len(rows),successes=counts,utc=time.time()))
   result=state_benchmark.trial(env,lease,out,spec,{},player,s['expert_checkpoint'],expert_speed=dict(ExpertPositionBridge.speed_scheme),capture_dataset=True,capture_compression='lzf',video=False)
   if result['legal']:
    visual=inspect_capture(Path(result['dataset_capture']['path']),preview=out/'three_views.jpg');result['visual_acceptance']=visual
    if not visual['passed']:raise RuntimeError('New sensor quality failure; no admission')
   rows.append(result);counts[spec['split']]+=int(result['success']);atomic_json(ROOT/'collection_results.json',rows)
   if spec['split']=='train':
    attempts+=1;consecutive=0 if result['success'] else consecutive+1
    if consecutive>=4 or (attempts>=10 and counts['train']/attempts<.7):raise RuntimeError('Expert reliability failed; bounded collection stopped')
   if gate and len(rows)==2:
    assert all(r['legal'] and r['success'] for r in rows),'Capture gate needs two complete legal successful records'
    atomic_json(ROOT/'collection_gate_complete.json',dict(complete=True,cases=2,utc=time.time()));return
  if counts!={'train':30,'validation':6}:raise RuntimeError('Insufficient preregistered independent successes: '+str(counts))
  atomic_json(ROOT/'collection_complete.json',dict(complete=True,successes=counts,attempts=len(rows),rows=rows,utc=time.time()))
 finally:
  # All trial HDF5/video writers finish synchronously before receipts. Use the
  # established Kit process-exit boundary; env.close/app.close can spin after completion.
  atomic_json(ROOT/'worker_finalization.json',dict(utc=time.time(),completed=len(rows),exception_active=sys.exc_info()[0] is not None,mode='closed_writer_process_exit'))

def data():
 import torch,h5py
 from visuotactile.narrow_data import record,actual_configuration_labels
 from visuotactile.native_data import limits_affine
 from visuotactile.position_control import contract,continuous_image_contract
 from visuotactile.checkpointing import save
 torch.set_num_threads(2);assert not (ROOT/'prepared.json').exists()
 rows=read(ROOT/'collection_complete.json')['rows'];s=read(ROOT/'session.json');entries={};records={};labels={};splits={'train':[],'validation':[],'test':[]};meta=None
 for row in rows:
  if not row['success']:continue
  i=row['identity'];p=Path(row['dataset_capture']['path']);digest=sha256(p)
  entry=dict(path=str(p.resolve()),episode=i,source_sha256=digest,family=row['family'],length=row['steps'],split=row['split'],layout=s['layout'])
  _,meta,r=record(p,privileged=False);q,labels[i]=actual_configuration_labels(entry);r['q_reached']=q
  entries[i]=entry;records[i]=r;splits[row['split']].append(i)
 assert len(splits['train'])==30 and len(splits['validation'])==6
 fingerprints=[]
 for i,e in entries.items():
  initial=read(Path(e['path']).parents[1]/'initial_observable.json')
  fingerprints.append(json.dumps({k:initial[k] for k in ['joint_position_rad','object_pose_w','hole_pose_w']},sort_keys=True))
 assert len(set(fingerprints))==36
 atomic_json(ROOT/'source_data.json',dict(entries=entries,splits=splits,layout=s['layout']))
 for role,supervision in [('A','q_reached'),('B','u_expert')]:
  dest=ROOT/role;out=dest/'state_data';out.mkdir(parents=True)
  definition=dict(visual_pair=True,visual_stride=1,supervision=supervision,contract=contract(24,True),normalization_identity=continuous_image_contract(supervision)['normalization_identity'],
   splits=splits,entries=entries,source_data_sha256=sha256(ROOT/'source_data.json'),normalization_scope='selected train30 only',normalization_train_ids=splits['train'],
   policy_inputs=['robot178','DINO_RGB_every_control_tick','metric_depth_every_control_tick'],privileged_inputs=False,layout=s['layout'])
  atomic_json(out/'data.json',definition);save(out/'numeric.pt',dict(contract=contract(24,True),data_sha256=sha256(out/'data.json'),records=records))
  x=torch.cat([records[i]['robot'] for i in splits['train']]);y=torch.cat([records[i]['q_reached' if role=='A' else 'command'] for i in splits['train']])
  stats=dict(robot_mean=x.mean(0),robot_std=x.std(0).clamp_min(1e-4));stats['action_scale'],stats['action_offset']=limits_affine(y);save(out/'statistics.pt',stats)
  atomic_json(dest/'session.json',dict(s,resource_root=str(ROOT.resolve())))
 atomic_json(ROOT/'metadata.json',meta);atomic_json(ROOT/'selection.json',dict(splits=splits,labels=labels,selection='First30 successful distinct preregistered train families; all6 fixed validation; no test read',layout=s['layout']))
 atomic_json(ROOT/'prepared.json',dict(train=30,validation=6,test_read=False,old_student_checkpoint_loaded=False,source_hashes_verified=True,all_q_next_alignment_passed=True,utc=time.time()))

if __name__=='__main__':
 if sys.argv[1]!='worker':{'prepare':prepare,'data':data}[sys.argv[1]]()
 else:
  import traceback
  code=0
  try:worker()
  except BaseException:
   code=1;atomic_json(ROOT/'collection_failure.json',dict(utc=time.time(),error=traceback.format_exc()));traceback.print_exc()
  if code==0:
   for path in ROOT.rglob('*'):
    if path.is_file() and not path.is_symlink():
     with path.open('rb') as stream:os.fsync(stream.fileno())
  sys.stdout.flush();sys.stderr.flush();os._exit(code)
