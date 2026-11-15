"""Finish the already trained 26-correction DP candidate under a new resource lease."""
import argparse,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE));BUNDLE=Path(__file__).resolve().parent
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from tools.tight_fresh_collect_cycle_owner import run_stage
from tools.tight_fresh_cycle import backend_chain,paired_backends,wait_parallel_admission
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
OLD=BASE/'tight_fresh_20261115_13';ROOT=BASE/'tight_fresh_20261116_00';SOURCE=OLD/'round_15';OUT=ROOT/'round_15'
DEADLINE=datetime(2026,11,16,9,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
read=lambda p:json.loads(p.read_text())
FRONT_TREE=Path('/home/CNS2025331827/docs/bc+rl_RSS_fresh_allocator_20261116')

def bind_frontend(dest, role):
 """A completed frontend is immutable; incomplete training resumes its durable epoch.

 Only checkpoint files may be hardlinked: the trainer atomically replaces them.
 Append-only learning/validation logs are copied, never linked into old evidence.
 """
 src=SOURCE/role/'training';finished=src/'finished.json'
 if finished.exists():
  done=read(finished)
  assert done['complete'] and done['epochs']==12 and done['updates']==2460
  assert sha256(src/'best.pt')==done['checkpoint_sha256']
  (dest/'training').symlink_to(src,target_is_directory=True)
  return dict(complete=True,finished_sha256=sha256(finished),selected_sha256=done['checkpoint_sha256'],updates=2460)
 import torch
 last=src/'last.pt';saved=torch.load(last,map_location='cpu',mmap=True,weights_only=False)
 assert 0<saved['epoch']<12 and saved['updates']==saved['epoch']*205
 for relative,digest in saved['configuration']['source_sha256'].items():
  assert sha256(FRONT_TREE/relative)==digest
 (dest/'training').mkdir()
 for path in src.iterdir():
  if path.is_file() and (path.name in ('last.pt','best.pt') or path.suffix in ('.json','.jsonl')):
   if path.suffix=='.pt':os.link(path,dest/'training'/path.name)
   else:shutil.copy2(path,dest/'training'/path.name)
 return dict(complete=False,resume_epoch=saved['epoch'],durable_updates=saved['updates'],
  checkpoint_sha256=sha256(last),remaining_updates=2460-saved['updates'],
  resume_tree=str(FRONT_TREE),resume_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=FRONT_TREE,text=True).strip(),
  semantics='Resume exact model, optimizer, EMA, RNG and deck at last complete epoch; old partial log remains preserved and is reconciled only in this new copy')

def launch(expected):
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()==expected
 assert not subprocess.check_output(['git','status','--porcelain','--','tools','visuotactile','scripts'],cwd=TREE,text=True).strip()
 assert not ROOT.exists() and (OLD/'monitor_exit.json').exists() and (OLD/'continuation/owner_finished.json').exists()
 normal_boundary=(OLD/'night_boundary_request.json').exists() and (SOURCE.with_name(SOURCE.name+'_stages')/'STOP').exists()
 # The old temporary wall deadline is preserved even if a slow disk prevents
 # finishing DP. No active worker is moved into a rewritten resource session.
 expired=time.time()>=read(OLD/'session.json')['deadline'] and not read(OLD/'continuation/owner_finished.json')['software_complete']
 assert normal_boundary or expired
 assert not (SOURCE/'A/backend').exists() and not (SOURCE/'B/backend').exists()
 assert DEADLINE-time.time()>7200
 ROOT.mkdir();OUT.mkdir();deadline=DEADLINE
 old=read(OLD/'session.json');session=dict(old,started_utc=time.time(),deadline=deadline,cache_deadline=deadline,
  training_deadline=deadline,offline_deadline=deadline,resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],source_commit=expected,
  session_kind='independent_completed_frontend_handover_v1',authorization='Latest user explicitly permits overnight continuation; old temporary deadline preserved')
 session['handover']=dict(source_root=str(SOURCE),old_deadline=old['deadline'],no_restart_from_initialization=True,
  reason='Closed old lease before rear preparation; complete frontends reused, any incomplete frontend resumes its exact durable checkpoint in the original frozen trainer')
 for dest in (ROOT,OUT):
  atomic_json(dest/'session.json',session)
  for name in ('metadata.json','gate.json'):shutil.copy2(SOURCE/name,dest/name)
 (OUT/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
 fronts={}
 for role in 'AB':
  dest=OUT/role;dest.mkdir();fronts[role]=bind_frontend(dest,role)
  for name in ('state_data','cache_index.json'):(dest/name).symlink_to(SOURCE/role/name,target_is_directory=name!='cache_index.json')
  shutil.copy2(SOURCE/role/'continuation.json',dest/'continuation.json')
  atomic_json(dest/'session.json',session);(dest/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
 for name in ('training_sampling.json','frozen_visual_io.json','backend_episode_sampling.json','cuda_allocator_policy.json','full_trajectory_batch.json'):
  shutil.copy2(SOURCE/name,OUT/name)
 atomic_json(OUT/'prepared.json',dict(complete=True,source=str(SOURCE),frontends=fronts,
  planned_remaining_DP_updates={r:fronts[r].get('remaining_updates',0) for r in 'AB'},
  data_sha256={r:sha256(OUT/r/'state_data/data.json') for r in 'AB'}))
 review=read(TREE/'scripts/tight_fresh_20261115_10/grasp_review.json')
 review.update(hard_families=['expanded_session20261115_03_prefetch13_s29000067','expanded_session20261115_03_prefetch13_s29000078'],
  development_count_per_role=2,lanes=4,vision_microbatch=4)
 atomic_json(ROOT/'grasp_review.json',review)
 python=str(TREE/'.venv_isaacsim/bin/python');base=['systemd-run','--user','--service-type=exec',
  '--property=WorkingDirectory='+str(TREE),'--property=Restart=no','--property=KillMode=control-group',
  '--property=RuntimeMaxSec='+str(math.floor(deadline-time.time()))]
 subprocess.run(base+['--unit=tight-fresh-finish26-monitor-20261116',python,'-m','visuotactile.state_resources','--root',str(ROOT)],check=True)
 end=time.time()+40
 while not (ROOT/'shared_resources.json').exists():
  if time.time()>end:raise RuntimeError('Monitor missing')
  time.sleep(.5)
 subprocess.run(base+['--unit=tight-fresh-finish26-20261116','--property=MemoryHigh=28G','--property=MemoryMax=34G',
  '--property=ExecStopPost=/usr/bin/touch '+str(ROOT/'STOP_MONITOR'),python,str(Path(__file__)),
  'run','--root',str(ROOT)],check=True)
 atomic_json(ROOT/'launch.json',dict(utc=time.time(),source_commit=expected,frontends=fronts,deadline=deadline,startup_only=True))

def run():
 ledger=OUT.with_name(OUT.name+'_stages');ledger.mkdir();deadline=read(ROOT/'session.json')['deadline']-120
 python=str(TREE/'.venv_isaacsim/bin/python');stops=(ROOT/'STOP',OUT/'STOP',ledger/'STOP');rows=[]
 atomic_json(ledger/'registration.json',dict(pid=os.getpid(),utc=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip(),
  source_root=str(SOURCE),output=str(OUT),deadline=deadline,frontends_reused=read(OUT/'prepared.json')))
 def execute(name,args,completion,**kwargs):
  worker_tree=kwargs.pop('worker_tree',TREE)
  folder=ledger/name;folder.mkdir();context=dict(ledger=folder,paths=dict(worker_tree=worker_tree,collection=BASE/'tight_fresh_20261115_09/new_family_15',output=OUT))
  cmd=[python,'-m',*map(str,args)];r=run_stage(name,cmd,context,deadline,stops);rows.append(r)
  if not r['complete'] or not completion.exists():raise RuntimeError('Tail stage failed: '+name)
  atomic_json(ledger/(name+'.json'),dict(r,completion=str(completion),completion_sha256=sha256(completion)))
  proof=read(completion)
  if any(proof.get(k) is False for k in ('complete','software_complete','target_reached')):raise RuntimeError('Incomplete tail stage: '+name)
 try:
  prepared=read(OUT/'prepared.json')
  for role in 'AB':
   if not prepared['frontends'][role]['complete']:
    atomic_json(ledger/'status.json',dict(stage='resume_frontend_'+role,utc=time.time()))
    execute('resume_frontend_'+role,['tools.tight_fresh_train','train','--root',OUT,'--role',role,
     '--epochs',12,'--offline-anchor-batch',32,'--full-trajectory-batch'],OUT/role/'training/finished.json',worker_tree=FRONT_TREE)
  for role in 'AB':
   done=read(OUT/role/'training/finished.json');assert done['complete'] and done['updates']==2460
  atomic_json(ledger/'status.json',dict(stage='backend_admission',utc=time.time()))
  admission=wait_parallel_admission(ROOT/'shared_resources.json',ledger/'admission.json',deadline=deadline,stop_paths=stops)
  atomic_json(ledger/'status.json',dict(stage='backend_chains',parallel=admission['admitted'],utc=time.time()))
  opts=dict(continue_rear=True,stop_paths=stops,offline_anchor_batch=32,frozen_visual_io=read(OUT/'frozen_visual_io.json'))
  if admission['admitted']:
   result=paired_backends(execute,OUT,12,**opts);atomic_json(ledger/'paired_backends.json',result)
   if not result['complete']:raise RuntimeError(str(result['errors']))
  else:
   for role in 'AB':backend_chain(execute,OUT,role,12,**opts)
  dev=OUT/'development';atomic_json(ledger/'status.json',dict(stage='development',utc=time.time()))
  execute('development_prepare',['tools.tight_fresh_batch_run','prepare','--root',OUT,'--output',dev,'--count',8,'--seed',28500000,
   '--lanes',16,'--vision-microbatch',8,'--queued-gpu','--grasp-evaluation-ticks',300,'--backend-json',BUNDLE/'backend.json'],dev/'plan.json')
  execute('development',['tools.tight_fresh_batch_run','run','--output',dev],dev/'result.json')
  atomic_json(ledger/'finished.json',dict(software_complete=True,stages=rows,
   new_DP_updates=prepared['planned_remaining_DP_updates'],total_DP_updates_per_role=2460,
   meaning='Same26 update completed across an explicit resource handover; abandoned old partial epoch excluded',utc=time.time()))
  execute('frontend_hold_baseline',['tools.tight_fresh_frontend_baseline','--root',OUT,'--output',OUT/'frontend_hold_baseline.json'],OUT/'frontend_hold_baseline.json')
  for role in 'AB':execute('pair_fit_'+role,['tools.tight_fresh_pair_fit','--root',OUT,'--role',role,'--output',OUT/('pair_fit_'+role+'.json')],OUT/('pair_fit_'+role+'.json'))
  probe=OUT/'original_training_probe'
  execute('original_probe_prepare',['tools.tight_fresh_original_probe','--root',OUT,'--output',probe,'--backend',BUNDLE/'backend.json'],probe/'plan.json')
  execute('original_probe',['tools.tight_fresh_batch_run','run','--output',probe],probe/'result.json')
  execute('grasp_review',['tools.tight_fresh_grasp_review','--root',OUT,'--collection',BASE/'tight_fresh_20261115_09/new_family_15',
   '--output',OUT/'grasp_review','--configuration',ROOT/'grasp_review.json'],OUT/'grasp_review/result.json')
  atomic_json(ledger/'status.json',dict(stage='complete',utc=time.time()))
  atomic_json(ROOT/'owner_finished.json',dict(software_complete=True,utc=time.time(),stages=rows))
 except BaseException as e:
  import traceback
  atomic_json(ROOT/'owner_finished.json',dict(software_complete=False,error=repr(e),traceback=traceback.format_exc(),utc=time.time(),stages=rows))
  raise
 finally:(ROOT/'STOP_MONITOR').touch(exist_ok=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','run']);p.add_argument('--expected-commit');p.add_argument('--root',type=Path)
 a=p.parse_args()
 if a.action=='launch':launch(a.expected_commit)
 else:
  assert a.root.resolve()==ROOT;run()
