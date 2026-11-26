from pathlib import Path
import copy,json,time
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.position_control import ExpertPositionBridge
from visuotactile.adaptive_takeover import profile
from scripts.tight_layout_20261116.pilot import reset_configuration,displacement
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';O=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01';C=P/'configs/experiments/correction_20261127';read=lambda p:json.loads(p.read_text())
summary=read(P/'outputs/diagnostics/tight_correction_20261127_01/development_before_summary.json')
assert all(x['complete']==8 for x in summary.values()),'Finish corrected32 before phase selection'
owner=R/'current_takeovers';s=session_at(owner,duration=12*3600);s['resources']['min_ram_gib']=6
s['authorization']='Current complete A/B own-student targeted captures on two existing train families: one contact-stall and one never-entered alignment opportunity each,8total; expert ownhistory tick0, two flush ticks masked; no terminal64 access'
atomic_json(owner/'session.json',s);pairs={r:read(R/'registration.json')['pairs'][r+'_new'] for r in 'AB'};stages=[];plans={}
for opportunity in ['contact','alignment']:
 for seed in (28910000,28910001):
  name=f'{opportunity}_{seed}';out=owner/name;out.mkdir();template=read(O/'collection/plan.json');cases=[]
  old=next(x for x in read(O/'cohorts.json')['train'] if x['seed']==seed)['narrow_reset'];corrected=reset_configuration(s['domain'],seed,.1)
  for role in ('AB' if seed%2==0 else 'BA'):
   data=read(O/f'learning/{role}/state_data/data.json');source=next(i for i in data['paired_pool']['shared_full_ids'] if data['entries'][i]['seed']==seed);family=data['entries'][source]['family'];identity=f'{role}_{name}'
   spec=dict(identity=identity,family=family,source_identity=source,seed=seed,event_seed=seed+100000,split='train',cohort='dagger_train',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=corrected,layout=s['layout'],takeover_tick=10**9,
    reset_admission=dict(object_xy_m=corrected['fixed_start_pose'][:2],hole_xy_m=[old['hole_x_range'][0],old['hole_y_range'][0]],unshifted_object_xy_m=old['fixed_start_pose'][:2],displacement_xy_m=displacement(s['domain'],.1).tolist(),tolerance_m=1e-5,baseline_source=str(O/'cohorts.json')))
   cases.append(dict(name=identity,role=role,opportunity=opportunity,spec=spec,video=False,source_data=str(O/f'learning/{role}/state_data/data.json'),dagger=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme,trigger=profile(opportunity,never_entered_only=opportunity=='alignment'))))
  for k in ['training_expansion_registry','parent_pairs_path','parent_pairs_sha256']:template.pop(k,None)
  template.update(kind='takeover',stage_takeover_contract='current_targeted_correction_v1',resource_root=str(owner),source_root=str(O/'learning'),pairs=pairs,cases=cases,source_sha256=sources(),source_session_sha256=sha256(owner/'session.json'),source_data_sha256={r:sha256(O/f'learning/{r}/state_data/data.json') for r in 'AB'},round_index=2,test_used=False,first_plan_forensics=False)
  template['batch'].update(lanes=2,seed=seed,writer_bytes=64<<20,verification_workers=0)
  template['execution_backend'].update(kit_tasking_threads=4,physics_threads=2)
  atomic_json(out/'session.json',s);atomic_json(out/'plan.json',template);plans[name]=sha256(out/'plan.json')
  stages.append(dict(entry='machine_review',args=['worker','--output',str(out),'--mode','takeover']))
atomic_json(C/'current_takeovers.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=stages))
atomic_json(owner/'registration.json',dict(utc=time.time(),opportunities_per_role=4,families=[28910000,28910001],profiles={p:profile(p,never_entered_only=p=='alignment') for p in ['contact','alignment']},plans=plans,pairs=pairs,terminal64_used=False,successful_student_interception='Contact screen can intercept healthy late grasps; rescues never counted as student gains',main_data_preserved=True))
print(owner)
