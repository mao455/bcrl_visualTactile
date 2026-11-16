"""One untouched eight-family final test, same paired full-task execution."""
import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
from scripts.tight_layout_20261116 import learn
from scripts.tight_layout_20261116.dagger import source_hashes
from scripts.tight_layout_20261116.pilot import stop
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.runtime import supervised_owner


def validate(plan):
 assert plan['kind']=='autonomous' and plan['test_used'] is True and plan['holdout_contract']=='layout_final_test_v1'
 registration=learn.read(Path(plan['final_test_registration']))
 assert sha256(plan['final_test_registration'])==plan['final_test_registration_sha256']
 seeds=[c['spec']['seed'] for c in plan['cases']]
 assert len(seeds)==8 and len(set(seeds))==4 and set(seeds)<=set(registration['seeds'])
 for seed in set(seeds):
  cases=[c for c in plan['cases'] if c['spec']['seed']==seed]
  assert {c['role'] for c in cases}=={'A','B'} and len(cases)==2
  assert cases[0]['spec']['narrow_reset']==cases[1]['spec']['narrow_reset']
  assert all(c['spec']['split']=='test' and 'grasp_evaluation_ticks' not in c and 'dagger' not in c for c in cases)
 assert plan['pairs']==registration['pairs']


def prepare(source,output,holdout_registration=None):
 from tools.tight_fresh_rollouts import completed_pair
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 assert not output.exists() and learn.read(source/'complete.json')['complete'] and learn.read(source/'owner_exit.json')['children_reaped']
 session=copy.deepcopy(learn.read(source/'session.json'));assert session['deadline']-time.time()>1800
 if holdout_registration is None:
  seeds=learn.read(learn.SOURCE/'collection_plan.json')['test_seeds']
 else:
  holdout=learn.read(holdout_registration);seeds=holdout['heldout_seeds']
  assert holdout['created_utc']<learn.read(source/'complete.json')['utc']
  assert not set(seeds)&set(holdout['old_test_regression_only']+holdout['development_seeds'])
 assert len(seeds)==8 and len(set(seeds))==8
 assert not set(seeds)&set(learn.DEV_SEEDS)
 data=learn.read(source/'A/state_data/data.json')
 for i in data['splits']['train']+data['splits']['validation']:
  entry=data['entries'][i]
  # Original episode identities end in their registered seed; corrections also carry seed explicitly.
  seed=entry.get('seed',int(entry['family'].rsplit('_',1)[-1]));assert seed not in seeds
 output.mkdir();(output/'logs').mkdir()
 session.update(resource_root=str(output),owned_worker_roots=[str(output)])
 atomic_json(output/'session.json',session)
 pairs={r:completed_pair(source,r) for r in 'AB'}
 registration=dict(source=str(source),seeds=seeds,pairs=pairs,utc=time.time(),test_used=True,
  selection='All eight originally registered final test families, once each per role; no training/model selection after these results',
  source_sha256=source_hashes(),original_deadline=session['deadline'])
 if holdout_registration is not None:
  registration.update(holdout_registration=str(holdout_registration),holdout_registration_sha256=sha256(holdout_registration),
   resource_roots=[str(source),str(output)],selection='All eight separately preregistered untouched phase-session test families, once each per role; no selection afterward')
 atomic_json(output/'registration.json',registration)
 for index in range(2):
  part=output/f'part{index}';chosen=seeds[index*4:index*4+4]
  assert chosen==list(range(chosen[0],chosen[0]+4))
  batch_prepare(source,part,lanes=8,count=4,seed=chosen[0],queued_gpu=True,vision_microbatch=4,
   execution_backend=learn.read(source/'development/plan.json')['execution_backend'])
  plan=learn.read(part/'plan.json')
  for c in plan['cases']:
   c['spec']['narrow_reset']=learn.reset_configuration(session['domain'],c['spec']['seed'],.1)
   c['spec'].update(split='test',layout=session['layout'])
   c['video']=c['spec']['seed']==seeds[0]
  plan.update(resource_root=str(output),test_used=True,holdout_contract='layout_final_test_v1',
   source_sha256=source_hashes(),final_test_registration=str(output/'registration.json'),
   final_test_registration_sha256=sha256(output/'registration.json'))
  validate(plan);atomic_json(part/'plan.json',plan);atomic_json(part/'session.json',session)


def worker(output):
 from scripts.tight_fulltask_20261116.paired_eval import install
 from scripts.tight_fulltask_20261116.video_eval import main
 validate(learn.read(output/'plan.json'));install(output)
 sys.argv=[sys.argv[0],'--output',str(output)];main(allow_heldout=True)


@supervised_owner
def run(output):
 assert not (output/'owner_started.json').exists()
 for name,digest in learn.read(output/'registration.json')['source_sha256'].items():assert sha256(learn.TREE/name)==digest
 atomic_json(output/'owner_started.json',dict(pid=os.getpid(),utc=time.time()))
 monitor=child=None
 try:
  with (output/'logs/monitor.log').open('x') as log:
   monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(output)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  until=time.time()+30
  while not (output/'shared_resources.json').exists():
   assert monitor.poll() is None and time.time()<until;time.sleep(.25)
  for index in range(2):
   part=output/f'part{index}';started=time.time()
   with (output/'logs'/f'part{index}.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-m',__spec__.name,'worker','--output',str(part)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    assert monitor.poll() is None and not (output/'STOP').exists()
    assert time.time()<min(started+1200,learn.read(output/'session.json')['deadline']-30)
    atomic_json(output/'status.json',dict(stage=f'part{index}',pid=child.pid,utc=time.time()));time.sleep(3)
   assert child.returncode==0 and learn.read(part/'result.json')['software_complete'] and learn.read(part/'paired_initial_check.json')['passed']
   child=None
  atomic_json(output/'complete.json',dict(complete=True,utc=time.time(),test_used=True))
 except BaseException:
  atomic_json(output/'failure.json',dict(utc=time.time(),error=traceback.format_exc()));raise
 finally:
  stop(child);(output/'STOP_MONITOR').touch()
  if monitor is not None:
   try:monitor.wait(timeout=15)
   except subprocess.TimeoutExpired:stop(monitor)
  atomic_json(output/'owner_exit.json',dict(children_reaped=True,utc=time.time(),monitor_returncode=None if monitor is None else monitor.poll()))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','run']);p.add_argument('--source',type=Path);p.add_argument('--holdout-registration',type=Path);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();output=a.output.resolve()
 if a.mode=='prepare':prepare(a.source.resolve(),output,None if a.holdout_registration is None else a.holdout_registration.resolve())
 elif a.mode=='worker':worker(output)
 else:run(output)
