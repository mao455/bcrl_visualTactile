from pathlib import Path
import argparse,json,sys,time,os,traceback
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
read=lambda p:json.loads(Path(p).read_text())
def experts(root):
 from tools.tight_state import environment
 from visuotactile.state_benchmark import trial
 from visuotactile.camera_validity import inspect_capture
 from visuotactile.narrow_data import record
 from visuotactile.position_control import ExpertPositionBridge
 from visuotactile.native_evaluation import exit_completed_kit_worker
 plan=read(root/'plan.json');s=read(root/'session.json')
 for p,h in plan['source_sha256'].items():assert sha256(p)==h,p
 atomic_json(root/'run_started.json',dict(pid=os.getpid(),utc=time.time()))
 app,env,player,lease=environment(root,camera_content_validation=True)
 env.speed_config=dict(ExpertPositionBridge.speed_scheme);entries={};rows=[]
 previous=[]
 for d in plan.get('complete_previous_shards',[]):
  assert read(Path(d)/'result.json')['software_complete']
  previous.extend(read(Path(d)/'expert_entries.json'))
 target=plan.get('target_total_accepted',len(plan['cases']))-len(previous)
 for c in plan['cases']:
  if len(entries)>=target:break
  for attempt in range(plan['attempts']):
   name=f'corrected_full_s{c["seed"]}_attempt{attempt:02d}'
   spec=dict(identity=name,family=c['family'],seed=c['seed'],split='train',cohort='fresh',event_seed=c['seed']+100000+attempt,collision_gate='live_collision_mesh_bounds_v1',narrow_reset=c['narrow_reset'],reset_admission=c['reset_admission'],layout=s['layout'])
   row=trial(env,lease,root/name,spec,{},player,s['expert_checkpoint'],expert_speed=ExpertPositionBridge.speed_scheme,capture_dataset=True,capture_compression='lzf',video=False)
   accepted=False
   if row.get('legal') and row.get('software_complete') and row.get('success'):
    path=Path(row['dataset_capture']['path']);v=inspect_capture(path);record(path,episode=name,privileged=False);assert v['passed']
    assert read(root/name/'actual_layout_admission.json')['passed']
    entries[name]=dict(path=str(path.resolve()),episode=name,length=row['steps'],family=c['family'],split='train',source_sha256=sha256(path),seed=c['seed'],trajectory_kind='full_expert_demonstration_v1',expert_speed=ExpertPositionBridge.speed_scheme,layout_verified=True)
    accepted=True
   rows.append(dict(family=c['family'],attempt=attempt,case=name,accepted=accepted,result=row));atomic_json(root/'results.json',rows);atomic_json(root/'expert_entries.json',entries)
   print(json.dumps(dict(case=name,accepted=accepted,steps=row.get('steps'),accepted_families=len(entries))),flush=True)
   if accepted:break
 atomic_json(root/'result.json',dict(software_complete=True,accepted=len(entries),families=len(plan['cases']),attempts=len(rows),utc=time.time()))
 exit_completed_kit_worker(root)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['curve_experts']=('prepare','outputs.diagnostics.tight_full_curve_20261127_01.dispatch',[])
  sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('mode',choices=['experts']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
  try:experts(a.root)
  except BaseException as e:
   atomic_json(a.root/'worker_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=time.time()));traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
