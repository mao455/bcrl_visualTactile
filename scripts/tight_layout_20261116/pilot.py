"""Registered peg-only reset displacement; original full expert task and control."""
import copy,json,os,sys,time,traceback,subprocess,signal,fcntl
from pathlib import Path
import numpy as np
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
TREE=Path(__file__).resolve().parents[2]
ROOT=TREE/'outputs/visuotactile/run/tight_layout_20261116_01'
SEEDS=[28816001,28816002,28816003]
DISTANCES=[0.,.05,.10]
def read(p):return json.loads(Path(p).read_text())
def displacement(domain,distance):
 v=-np.asarray(domain['center']['relative_xyz'][:2],dtype=float)
 return v/np.linalg.norm(v)*distance

def reset_configuration(domain,seed,distance):
 from visuotactile.narrow_data import direct_configuration
 if not np.isfinite(distance) or not 0<=distance<=.12:raise ValueError('Unregistered displacement')
 cfg=direct_configuration(domain,seed);delta=displacement(domain,distance)
 cfg['fixed_start_pose'][:2]=(np.asarray(cfg['fixed_start_pose'][:2])+delta).tolist()
 return cfg

def box_distance(a,b):
 """Exact Euclidean separation of two solid OBBs, via bounded convex LS."""
 from scipy.optimize import lsq_linear
 ca,ra,ha=a;cb,rb,hb=b
 fit=lsq_linear(np.column_stack((ra,-rb)),cb-ca,bounds=(-np.r_[ha,hb],np.r_[ha,hb]),tol=1e-12,max_iter=300)
 if not fit.success:raise RuntimeError('OBB separation solver failed')
 pa=ca+ra@fit.x[:3];pb=cb+rb@fit.x[3:]
 return dict(distance_m=float(np.linalg.norm(pa-pb)),object_point_w=pa.tolist(),other_point_w=pb.tolist())

def geometry(env,output):
 from visuotactile.initial_geometry import boxes
 d={k:v[0].cpu().numpy() for k,v in env._diagnostics().items()}
 assets=env.cfg.assets.to_dict();obj=boxes(assets['object_urdf'],d['object_pose_w'])
 report={}
 for name,key in [('base','receptive_urdf'),('support','table_urdf')]:
  other=boxes(assets[key],d['hole_pose_w' if name=='base' else 'table_pose_w'])
  rows=[dict(object_collider=i,other_collider=j,**box_distance(a,b)) for i,a in enumerate(obj) for j,b in enumerate(other)]
  report[name]=dict(minimum=min(rows,key=lambda x:x['distance_m']),pairs=rows)
 report.update(object_pose_w=d['object_pose_w'].tolist(),hole_pose_w=d['hole_pose_w'].tolist(),
  table_pose_w=d['table_pose_w'].tolist(),assets=assets,asset_sha256={k:sha256(assets[k]) for k in ['object_urdf','receptive_urdf','table_urdf']},
  scope='Euclidean distance between original URDF box collision solids at actual pre-action body poses. Contact/rest margins excluded; live conservative robot clearance is recorded separately.')
 atomic_json(output/'layout_geometry.json',report)

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 assert not (ROOT/'session.json').exists()
 source=read(TREE/'outputs/visuotactile/run/tight_reliable_eval_20261116_01/session.json')
 now=time.time();session={k:copy.deepcopy(source[k]) for k in ['resource_policy','resources','expert_checkpoint','expert_sha256','dino_checkpoint','dino_sha256','precision_policy','camera_content_validation','domain']}
 session.update(version='peg_spacing_pilot_v1',started_utc=now,deadline=now+7200,collection_deadline=now+7200,training_deadline=now+7200,reserve_bytes=24*2**30,
  resource_root=str(ROOT.resolve()),owned_worker_roots=[str(ROOT.resolve())],authorization='New peg-only layout hypothesis, nine preregistered expert full-task opportunities; gate before new data/training')
 session['resources']['min_disk_gib']=24
 atomic_json(ROOT/'session.json',session)
 jobs=[]
 for i,seed in enumerate(SEEDS):
  for distance in DISTANCES:
   name=f'layout{round(distance*100):02d}_seed{seed}'
   jobs.append(dict(identity=name,family=f'layout_development_{seed}',seed=seed,event_seed=seed+10000,split='layout_development',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=reset_configuration(session['domain'],seed,distance),
    layout=dict(version='peg_spacing_reset_v1',distance_m=distance,translation_xy_m=displacement(session['domain'],distance).tolist(),hole_unchanged=True),video=i==0))
 atomic_json(ROOT/'registration.json',dict(jobs=jobs,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=TREE).strip(),
  source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116'] for p in (TREE/folder).rglob('*.py')},
  selection_rule='Keep all nine outcomes; only consider a displaced layout with >=2/3 native full-task successes, all starts legal, visible supported starts, and video/geometry review. At most one evidence-based minor adjustment cohort, independently registered. No student rejection gate.',
  heldout_seed_reservation=dict(training=list(range(28817000,28817040)),validation=list(range(28818000,28818006)),test=list(range(28819000,28819008)))))
 print(ROOT)

def worker():
 import torch
 from tools.tight_state import environment
 from visuotactile.position_control import ExpertPositionBridge
 from visuotactile import state_benchmark
 torch.set_num_threads(2);registration=read(ROOT/'registration.json')
 for name,digest in registration['source_sha256'].items():assert sha256(TREE/name)==digest,'Source drift: '+name
 original=state_benchmark.prepare
 def observed(env,isolated,spec,source,output):
  result=original(env,isolated,spec,source,output);geometry(env,output);return result
 state_benchmark.prepare=observed
 app=env=None;rows=[]
 try:
  app,env,player,lease=environment(ROOT,startup_diagnostics=True)
  atomic_json(ROOT/'initialization.json',dict(utc=time.time(),pid=os.getpid()))
  for spec in registration['jobs']:
   output=ROOT/'pilot'/spec['identity'];env.speed_config=dict(ExpertPositionBridge.speed_scheme)
   atomic_json(ROOT/'status.json',dict(stage='expert_layout_pilot',identity=spec['identity'],completed=len(rows),utc=time.time()))
   result=state_benchmark.trial(env,lease,output,spec,{},player,read(ROOT/'session.json')['expert_checkpoint'],expert_speed=dict(ExpertPositionBridge.speed_scheme),video=spec['video'],capture_dataset=False,video_title=f"Expert | peg displacement {spec['layout']['distance_m']*100:.0f}cm")
   rows.append(result);atomic_json(ROOT/'pilot_results.json',rows)
  atomic_json(ROOT/'pilot_complete.json',dict(complete=True,cases=len(rows),rows=rows,utc=time.time()))
 finally:
  if env is not None:env.close()
  if app is not None:app.close()

def stop(child):
 if child is None or child.poll() is not None:return
 os.killpg(child.pid,signal.SIGTERM)
 try:child.wait(timeout=15)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)

def run():
 with (ROOT/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (ROOT/'owner_started.json').exists(),'No silent retry'
  atomic_json(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time()))
  monitor=child=None
  try:
   with (ROOT/'monitor.log').open('x') as log:
    monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   until=time.time()+25
   while not (ROOT/'shared_resources.json').exists():
    assert monitor.poll() is None and time.time()<until
    time.sleep(.25)
   with (ROOT/'worker.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker','--root',str(ROOT)],cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
     env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
   start=time.time();deadline=read(ROOT/'session.json')['deadline']
   while child.poll() is None:
    if monitor.poll() is not None:raise RuntimeError('Monitor exited during worker')
    if time.time()>=deadline-30 or (ROOT/'STOP').exists():raise TimeoutError('Pilot finite deadline/STOP')
    progress=[ROOT/'status.json',ROOT/'initialization.json',*ROOT.glob('pilot/*/progress.json'),*ROOT.glob('pilot/*/result.json')]
    latest=max([start]+[p.stat().st_mtime for p in progress if p.exists()])
    if time.time()-latest>420:raise TimeoutError('No startup or control progress for 420 seconds')
    atomic_json(ROOT/'owner_status.json',dict(pid=os.getpid(),worker_pid=child.pid,monitor_pid=monitor.pid,utc=time.time(),stage='pilot'))
    time.sleep(2)
   assert child.returncode==0 and read(ROOT/'pilot_complete.json')['complete']
   atomic_json(ROOT/'status.json',dict(stage='pilot_complete_pending_scientific_review',utc=time.time()))
  except BaseException:
   atomic_json(ROOT/'failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
  finally:
   stop(child);(ROOT/'STOP_MONITOR').touch()
   if monitor is not None:
    try:monitor.wait(timeout=15)
    except subprocess.TimeoutExpired:stop(monitor)
   atomic_json(ROOT/'owner_exit.json',dict(utc=time.time(),worker_returncode=None if child is None else child.poll(),monitor_returncode=None if monitor is None else monitor.poll(),children_reaped=True))

if __name__=='__main__':{'prepare':prepare,'run':run,'worker':worker}[sys.argv[1]]()
