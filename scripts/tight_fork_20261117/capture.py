"""One bounded paired diagnostic capture, preserving exact frozen visual inputs."""
import copy,json,os,sys,time,subprocess,traceback
from .common import *
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256

def prepare():
 assert not ROOT.exists();ROOT.mkdir();OUT.mkdir(parents=True,exist_ok=True)
 old=OLD/'comparison01';session=read(old/'session.json');now=time.time()
 session.update(version='first_fork_diagnostic_v1',started_utc=now,deadline=now+7200,training_deadline=now+7200,offline_deadline=now+7200,resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],authorization='No training/DAgger. One8lane121tick paired observation capture for same-input complete-policy inference',reserve_bytes=12*1024**3)
 atomic_json(ROOT/'session.json',session)
 plan=read(old/'development/plan.json');other=read(OLD/'comparison03/development/plan.json')
 plan['cases']=plan['cases'][:4]+other['cases'][:4]
 for c in plan['cases']:c.update(video=False,grasp_evaluation_ticks=121,diagnostic_cohort='first_fork_input_capture_only')
 plan.update(resource_root=str(ROOT),evaluation_scope=dict(protocol='121 observed actions only, censored not native task failures',full_task_evaluation=False,test_used=False),source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_fulltask_20261116','scripts/tight_fork_20261117'] for p in (TREE/folder).rglob('*.py')})
 plan['batch']['seed']=28824002
 out=ROOT/'capture';out.mkdir();atomic_json(out/'session.json',session);atomic_json(out/'plan.json',plan)
 reg=dict(root=str(ROOT),start=now,deadline=now+7200,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cases=[dict(role=c['role'],seed=c['spec']['seed']) for c in plan['cases']],control_ticks=121,physical_starts=8,source_plan_sha256=sha256(old/'development/plan.json'),pairs=plan['pairs'],no_new_training=True,no_dagger=True,input_budget_gib=4,reason='Original numeric/plans/videos omit real depth and per-tick visual features, so cannot reconstruct complete sensory history. Capture new streams; never call them exact old trajectory replays.',same_input='Replay from tick0 through120 with each model own computed latent memory and own plan; actual recorded issued/applied commands and targets forced as historical inputs. Common DDIM seeds/states, unchanged planning phases; self-replay required before cross-model interpretation.')
 atomic_json(ROOT/'registration.json',reg);atomic_json(OUT/'registration.json',reg)
 print(ROOT)

def worker():
 import torch
 from visuotactile.gpu_batch_controller import MultiPolicyController
 from visuotactile.models.native_vision import NativeDinoSpatial
 from scripts.tight_fulltask_20261116.paired_eval import install
 torch.set_num_threads(2)
 out=ROOT/'capture';snap=ROOT/'inputs';snap.mkdir()
 oldinit=MultiPolicyController.__init__;oldact=MultiPolicyController.act;oldvision=NativeDinoSpatial.forward_lanes
 holder={};count=[0];payload=[0]
 def cpu(x):
  if torch.is_tensor(x):return x.detach().cpu().clone()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  return x
 def init(self,saved,metadata,dino,n,**kwargs):
  oldinit(self,saved,metadata,dino,n,**kwargs);atomic_json(ROOT/'replay_metadata.json',metadata)
  torch.save({r:[c.base.generator.get_state() for c in self.models[r].lanes] for r in saved},ROOT/'initial_generators.pt')
 def features(self,rgb,**kwargs):
  value=oldvision(self,rgb,**kwargs);holder['features']=cpu(value);return value
 def act(self,packet,diagnostics,active):
  tick=int(packet['step_id'][0]);assert tick==count[0] and tick<121 and all(active)
  before=dict(packet=cpu({k:v for k,v in packet.items() if torch.is_tensor(v) and k!='image'}),diagnostics=cpu(diagnostics),active=list(active))
  result=oldact(self,packet,diagnostics,active)
  before.update(features=holder.pop('features'),command=cpu(result),controllers=[])
  for c in self.lanes:
   before['controllers'].append(dict(nominal=cpu(c.nominal),delta=cpu(c.delta),plan=cpu(c.plan),diagnostics=cpu(c.residual_diagnostics),robot=cpu(c.base.last_robot),native=cpu(c.base.memory.native),visual_memory=cpu(c.base.memory.visual),visual_read=cpu(c.visual),next_tick=c.base.memory.next_tick,robot_length=len(c.base.memory.robot),frame_length=len(c.base.memory.frames),rear_state=cpu(c.state),generator=cpu(c.base.generator.get_state())))
  p=snap/f'{tick:03d}.pt';torch.save(before,p);payload[0]+=p.stat().st_size;assert payload[0]<4*1024**3
  count[0]+=1;atomic_json(ROOT/'capture_progress.json',dict(ticks=count[0],bytes=payload[0],utc=time.time()))
  return result
 MultiPolicyController.__init__=init;MultiPolicyController.act=act;NativeDinoSpatial.forward_lanes=features
 install(out)
 from tools.tight_fresh_batch_run import run
 run(out)

def run():
 from scripts.tight_layout_20261116.pilot import stop
 monitor=child=None
 assert not (ROOT/'owner_started.json').exists();atomic_json(ROOT/'owner_started.json',dict(pid=os.getpid(),utc=time.time()))
 try:
  with (ROOT/'monitor.log').open('x') as f:monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (ROOT/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  with (ROOT/'capture.log').open('x') as f:child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  start=time.time()
  while child.poll() is None:
   assert monitor.poll() is None
   if time.time()>min(start+1200,read(ROOT/'session.json')['deadline']):raise TimeoutError('Bounded capture deadline')
   time.sleep(2)
  assert child.returncode==0 and read(ROOT/'capture/result.json')['software_complete']
  assert read(ROOT/'capture_progress.json')['ticks']==121
  atomic_json(ROOT/'capture_complete.json',dict(complete=True,seconds=time.time()-start,returncode=child.returncode,utc=time.time()))
 except BaseException:
  atomic_json(ROOT/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
 finally:
  stop(child);(ROOT/'STOP_MONITOR').touch()
  if monitor:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  atomic_json(ROOT/'owner_exit.json',dict(children_reaped=True,utc=time.time()))
if __name__=='__main__':globals()[sys.argv[1]]()
