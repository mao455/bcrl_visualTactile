from pathlib import Path
import argparse,copy,json,time
import torch
from tools.tight_machine_review import sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.state_benchmark import dagger_policy_identity
from visuotactile.narrow_data import direct_configuration
from scripts.tight_layout_20261116.pilot import reset_configuration,displacement
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';C=P/'configs/experiments/full_curve_20261127';PREV=P/'outputs/visuotactile/run/tight_correction_20261127_01';read=lambda p:json.loads(Path(p).read_text())
def prepare(epoch,holdout=False):
 reg=read(R/'registration.json');s0=read(R/'session.json');jobs=[]
 for which in (['start'] if epoch==0 else ['A','B']):
  owner=R/(f'final16_{which}' if holdout else f'evaluation_e{epoch:02d}_{which}');owner.mkdir();s=copy.deepcopy(s0);s['resource_root']=str(owner);atomic_json(owner/'session.json',s)
  out=owner/'rollouts';out.mkdir();pairs={};cases=[]
  if epoch==0:pairs={r:reg['starting_pairs'][r] for r in 'AB'}
  else:
   for condition in reg['conditions']:
    root=R/condition/f'rear{epoch}';rear=root/which/'backend'/which/'training/best.pt';saved=torch.load(rear,map_location='cpu',weights_only=False,mmap=True);pairs[which+'_'+condition]=dagger_policy_identity(saved,str(rear))
  seeds=reg['sealed_new_final_families'] if holdout else reg['training_families']+reg['development_families']
  for j,seed in enumerate(seeds):
   reset=reset_configuration(s['domain'],seed,.1);old=direct_configuration(s['domain'],seed);cohort='independent_final16' if holdout else 'training_curve' if seed in reg['training_families'] else 'development_curve'
   for name in (list(pairs) if j%2==0 else list(reversed(pairs))):
    identity=f'{name}_{seed}';spec=dict(identity=identity,family=f'physical_seed_{seed}',seed=seed,event_seed=seed+100000,split='development',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=reset,layout=s['layout'],reset_admission=dict(object_xy_m=reset['fixed_start_pose'][:2],hole_xy_m=[reset['hole_x_range'][0],reset['hole_y_range'][0]],unshifted_object_xy_m=old['fixed_start_pose'][:2],displacement_xy_m=displacement(s['domain'],.1).tolist(),tolerance_m=1e-5,baseline_source='Historical direct family reset'))
    cases.append(dict(name=identity,role=name,spec=spec,video=False,diagnostic_cohort=cohort,opportunity='fixed_curve'))
  template=read(PREV/'development_before/wave00/plan.json');template.update(resource_root=str(owner),pairs=pairs,cases=cases,source_sha256=sources(),paired_queue=True,observe_only=True,sampler_episode_seed=7091,expert_permission=False,record_sampler_rng=True,source_commit=reg.get('source_commit','f069a5769286fbe80a6f53613028877e5115afec'))
  template['batch'].update(lanes=2,seed=seeds[0],writer_bytes=64<<20,verification_workers=0)
  template['execution_backend'].update(kit_tasking_threads=4,physics_threads=2)
  template['evaluation_scope']=dict(epoch=epoch,training_families=reg['training_families'],development_families=reg['development_families'],terminal64_used=False,selection='Descriptive fixed0/30/60 curve, no development-best deployment',complete_policy=True)
  atomic_json(out/'session.json',s);atomic_json(out/'plan.json',template)
  job=C/(f'final16_{which}.json' if holdout else f'evaluation_e{epoch:02d}_{which}.json');atomic_json(job,dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=[dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous'])]));jobs.append(str(job))
 return jobs
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True,choices=[0,30,60]);p.add_argument('--holdout',action='store_true');a=p.parse_args();assert not a.holdout or a.epoch==60;print(prepare(a.epoch,a.holdout))
