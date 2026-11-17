"""Finite capability and collection cycle using existing complete-policy execution."""
import copy,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from visuotactile.training_resources import atomic_json as write
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.pilot import reset_configuration,stop
TREE=Path(__file__).resolve().parents[2];RUNS=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=RUNS/'tight_cycle_20261118_01';PARENT=RUNS/'tight_long_20261117_01/round05/learning';DATA=RUNS/'tight_long_20261117_01/round06/learning'
OUT=TREE/'research_docs/evidence/tight_cycle_20261118'
def read(p):return json.loads(Path(p).read_text())
def hashes():return {str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_fulltask_20261116'] for p in (TREE/folder).rglob('*.py')}
def prepare():
 from tools.tight_fresh_rollouts import completed_pair
 ROOT.mkdir(exist_ok=False);OUT.mkdir(parents=True,exist_ok=True)
 now=time.time();s=read(PARENT/'session.json');s.update(version='capability_cycle_v1',started_utc=now,deadline=now+6*3600,training_deadline=now+5*3600,cache_deadline=now+5*3600,offline_deadline=now+6*3600,collection_deadline=now+3*3600,reserve_bytes=12*1024**3,authorization='Training-family versus new-development capability then one evidence-supported full learning cycle; latest user attachment',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());s['resources']['min_disk_gib']=12
 write(ROOT/'session.json',s)
 pairs={r:completed_pair(PARENT,r) for r in 'AB'}
 training=[28817000+i for i in [0,4,8,12,16,20,24,28]];dev=list(range(28826000,28826008))
 reg=dict(created_utc=now,deadline=s['deadline'],parent_pairs=pairs,data76=str(DATA),training_seeds=training,new_development_seeds=dev,reserved_new_training_seeds=list(range(28827000,28827016)),capability_opportunities=32,initial_capability_concurrency=32,selection='Spread deterministic original train IDs; disjoint new same-domain development IDs chosen before outcomes; no final tests',decision=dict(support='Both76 updates lost development success; no full training-family capability evidence yet',against='Small samples and batch-dependent native trajectories prevent sampler-only attribution',action='32 complete baseline72 A/B executions, train8 versus freshdev8',expected='Separate in-family failure from initial generalization and actual rollout deviation',falsification='Strong training performance and weak new-family performance favors expansion; similarly weak favors recovery or learning, judged with state distributions'),limits='6h total,5h training,3h collection; no deadline reset; physics/network unchanged')
 write(ROOT/'registration.json',reg);write(OUT/'registration.json',reg)
 d=ROOT/'capability';d.mkdir();template=read(RUNS/'tight_sampling76_A_20261117_01/evaluation_recovery01/comparison00/plan.json');p=copy.deepcopy(template);p.update(pairs=pairs,cases=[],resource_root=str(d),source_sha256=hashes(),source_commit=s['source_commit']);p.pop('comparison',None);p['batch']['lanes']=32
 data=read(DATA/'A/state_data/data.json')
 for cohort,seeds in [('training',training),('new_development',dev)]:
  for seed in seeds:
   family=f'layout10_train_{seed}' if cohort=='training' else f'cycle_dev_{seed}'
   spec=dict(identity='capability_'+family,family=family,split='train' if cohort=='training' else 'development',seed=seed,event_seed=seed+100000,cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=reset_configuration(s['domain'],seed,.1),layout=s['layout'])
   if cohort=='training':
    e=data['entries'][family];meta=read(Path(e['path']).parents[1]/'metadata.json');old=meta.get('spec',meta)
    # Actual recorded spec is the reset authority, not seed equality alone.
    if 'narrow_reset' in old:assert spec['narrow_reset']==old['narrow_reset']
   for role in 'AB':p['cases'].append(dict(name=role+'_'+spec['identity'],role=role,spec=copy.deepcopy(spec),video=seed==seeds[0],diagnostic_cohort=cohort))
 p['evaluation_scope']=dict(protocol='Full natural termination; actual A/B pre-action pairing; train diagnostic versus new development, no blind-test claim',test_used=False)
 s=copy.deepcopy(s);s.update(resource_root=str(d),owned_worker_roots=[str(d)])
 write(d/'session.json',s);write(d/'plan.json',p)
 print(ROOT)
def worker(directory):
 from scripts.tight_fulltask_20261116.paired_eval import install
 directory=Path(directory);install(directory)
 if read(directory/'plan.json')['kind']=='autonomous':
  from scripts.tight_fulltask_20261116.video_eval import main
  sys.argv=[sys.argv[0],'--output',str(directory)];main()
 else:
  from tools.tight_fresh_batch_run import run
  run(directory)
def run_wave(directory):
 d=Path(directory);assert not (d/'owner_started.json').exists();write(d/'owner_started.json',dict(utc=time.time(),pid=os.getpid()));monitor=child=None;start=time.time()
 try:
  with (d/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(d)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (d/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  cmd=[sys.executable,'-m',__spec__.name,'worker',str(d),'--root',str(d)]
  with (d/'worker.log').open('x') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(d/'worker_started.json',dict(command=cmd,pid=child.pid,utc=time.time()))
  while child.poll() is None:
   assert monitor.poll() is None
   if time.time()>min(start+3600,read(d/'session.json')['deadline']-30) or (d/'STOP').exists():raise TimeoutError('Registered wave/session deadline')
   log=(d/'worker.log').read_text(errors='replace')
   if time.time()-start>30 and 'Traceback (most recent call last)' in log:raise RuntimeError('Worker traceback; preserve and reap owned Kit')
   write(d/'owner_status.json',dict(elapsed_s=time.time()-start,pid=child.pid,utc=time.time()));time.sleep(3)
  assert child.returncode==0 and read(d/'result.json')['software_complete']
  write(d/'owner_complete.json',dict(seconds=time.time()-start,utc=time.time()))
 except BaseException:
  write(d/'owner_failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  stop(child);(d/'STOP_MONITOR').touch()
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  write(d/'owner_exit.json',dict(children_reaped=True,utc=time.time(),seconds=time.time()-start))
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:3])
