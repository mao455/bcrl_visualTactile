"""One reviewed layout-specific DAgger round; common pool and full task evaluation."""
import argparse,copy,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from scripts.tight_layout_20261116.pilot import TREE,read,stop
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.runtime import supervised_owner


def source_hashes():
 return {str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116','scripts/tight_fulltask_20261116'] for p in (TREE/folder).rglob('*.py')}


def prepare(source,output,round_index,offset,trigger):
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 from tools.tight_fresh_stage_takeover import EARLY_CONTRACT,validate_plan
 from visuotactile.grasp_curriculum import APPROACH,CONTACT_ONSET
 assert read(source/'complete.json')['complete'] and read(source/'owner_exit.json')['children_reaped']
 assert read(source/'development/result.json')['software_complete'] and read(source/'development/paired_initial_check.json')['passed']
 assert not output.exists()
 data=read(source/'A/state_data/data.json');other=read(source/'B/state_data/data.json')
 assert data['splits']==other['splits']
 identities=data['normalization_train_ids'][offset:offset+4]
 assert len(identities)==4 and all(data['entries'][i]['layout']['distance_m']==.1 for i in identities)
 session=copy.deepcopy(read(source/'session.json'))
 assert session['deadline']-time.time()>7200,'Insufficient original window for collection and complete learning'
 output.mkdir();(output/'logs').mkdir()
 session.update(resource_root=str(output),owned_worker_roots=[str(output)],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 atomic_json(output/'session.json',session)
 plan=batch_prepare(source,output/'collection',lanes=8,identities=identities,delays=[0]*4,round_index=round_index,
  seed=28830000+100*round_index,session_root=output,
  execution_backend=read(source/'development/plan.json')['execution_backend'],queued_gpu=True,vision_microbatch=4,
  writer_bytes=512<<20,verification_workers=2,verification_max_pending=4,capture_audit_memo=True)
 if trigger!='contact':
  plan['stage_takeover_contract']=EARLY_CONTRACT
  for c in plan['cases']:c['dagger']['trigger']=copy.deepcopy(APPROACH if trigger=='approach' else CONTACT_ONSET)
 for c in plan['cases']:
  c['spec']['layout']=copy.deepcopy(session['layout']);c['video']=False
 validate_plan(plan)
 plan.update(source_sha256=source_hashes(),layout_dagger=dict(source=str(source),offset=offset,trigger=trigger,
  pairing='Private native reset streams and pre-action verification, no hidden-state exactness claim',
  scope='Four original training families, both full students; all successful actual expert suffixes shared, no test access'))
 atomic_json(output/'collection/plan.json',plan)
 atomic_json(output/'prepared.json',dict(source=str(source),source_sha256=source_hashes(),round=round_index,
  original_deadline=session['deadline'],training_identities=identities,trigger=trigger,utc=time.time()))


def collection_worker(output):
 from scripts.tight_fulltask_20261116.paired_eval import install
 from tools.tight_fresh_batch_run import run
 install(output/'collection');run(output/'collection')


def prepare_learning(output):
 from tools.tight_fresh_dagger import prepare_aggregate
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.bounded_extension import fresh_layout_profile
 source=Path(read(output/'prepared.json')['source']);collection=output/'collection';target=output/'learning'
 assert read(collection/'result.json')['software_complete'] and read(collection/'paired_initial_check.json')['passed']
 assert read(output/'collection_owner_exit.json')['children_reaped']
 entries=read(collection/'correction_entries.json')
 assert entries,'No successful recovery: preserve evidence and review, never train an empty pool'
 prepare_aggregate(source,collection,target)
 (target/'logs').mkdir()
 session=copy.deepcopy(read(output/'session.json'));session.update(resource_root=str(target),owned_worker_roots=[str(target)])
 atomic_json(target/'session.json',session)
 (target/'shared_resources.json').unlink()
 registration=dict(profile=profile(),roles={})
 for role in 'AB':
  child=target/role;(child/'shared_resources.json').unlink();(child/'shared_resources.json').symlink_to(target/'shared_resources.json')
  atomic_json(child/'session.json',session)
  data=read(child/'state_data/data.json');ids=data['splits']['train']
  registration['roles'][role]=dict(**budget(len(ids),80),data_sha256=sha256(child/'state_data/data.json'),episode_multiplicities=dict.fromkeys(ids,1))
  assert data['normalization_train_ids']==read(source/role/'state_data/data.json')['normalization_train_ids']
 assert read(target/'A/state_data/data.json')['splits']==read(target/'B/state_data/data.json')['splits']
 for name in ['training_sampling.json','cuda_allocator_policy.json','frozen_visual_io.json']:shutil.copyfile(source/name,target/name)
 atomic_json(target/'full_trajectory_batch.json',registration)
 atomic_json(target/'bounded_extension.json',dict(policy=fresh_layout_profile(),roles={}))
 prepared=read(target/'prepared.json');prepared.update(continue_student_pair=True,source_sha256=source_hashes(),
  note='Common frozen aggregate; current pair EMA/Transformer continuation with fresh optimizers; refresh rear inputs after new DP; no new weighting')
 atomic_json(target/'prepared.json',prepared)
 return target


@supervised_owner
def run(output):
 import fcntl
 with (output/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (output/'owner_started.json').exists()
  for name,digest in read(output/'prepared.json')['source_sha256'].items():assert sha256(TREE/name)==digest,name
  atomic_json(output/'owner_started.json',dict(pid=os.getpid(),utc=time.time()))
  monitor=child=None
  try:
   with (output/'logs/monitor.log').open('x') as log:
    monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(output)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   limit=time.time()+30
   while not (output/'shared_resources.json').exists():
    assert monitor.poll() is None and time.time()<limit;time.sleep(.25)
   started=time.time()
   with (output/'logs/collection.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   atomic_json(output/'collection_worker_started.json',dict(pid=child.pid,utc=time.time()))
   while child.poll() is None:
    progress=[p.stat().st_mtime for p in (output/'collection').rglob('*progress*.json')]
    progress += [p.stat().st_mtime for p in (output/'collection').rglob('result.json')]
    if time.time()-max([started]+progress)>1200:raise TimeoutError('No capture/control progress for1200s')
    assert monitor.poll() is None,'Monitor exited during collection'
    assert not (output/'STOP').exists() and time.time()<min(started+3600,read(output/'session.json')['deadline']-120)
    atomic_json(output/'status.json',dict(stage='collection',pid=child.pid,utc=time.time()))
    time.sleep(3)
   assert child.returncode==0 and read(output/'collection/result.json')['software_complete']
  except BaseException:
   atomic_json(output/'failure.json',dict(error=traceback.format_exc(),utc=time.time()));raise
  finally:
   stop(child);(output/'STOP_MONITOR').touch()
   if monitor is not None:
    try:monitor.wait(timeout=15)
    except subprocess.TimeoutExpired:stop(monitor)
   atomic_json(output/'collection_owner_exit.json',dict(children_reaped=True,monitor_returncode=None if monitor is None else monitor.poll(),utc=time.time()))
  target=prepare_learning(output)
  # A separate owner/monitor is started only after the capture process exits.
  atomic_json(output/'status.json',dict(stage='learning',root=str(target),utc=time.time()))
  subprocess.run([sys.executable,'-m','scripts.tight_layout_20261116.learn','run','--root',str(target)],check=True)
  atomic_json(output/'complete.json',dict(complete=True,utc=time.time(),requires_scientific_review=True))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','run']);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--source',type=Path);p.add_argument('--round',type=int,default=1);p.add_argument('--offset',type=int,default=0)
 p.add_argument('--trigger',choices=['approach','touch','contact'],default='approach');a=p.parse_args();a.output=a.output.resolve()
 if a.mode=='prepare':prepare(a.source.resolve(),a.output,a.round,a.offset,a.trigger)
 elif a.mode=='worker':collection_worker(a.output)
 else:
  try:run(a.output)
  except BaseException:
   atomic_json(a.output/'round_failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
