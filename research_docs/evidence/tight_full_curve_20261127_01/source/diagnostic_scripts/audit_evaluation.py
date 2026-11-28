"""Audit completed real outcomes; partial runs never become new denominators."""
from pathlib import Path
import argparse,collections,csv,json,math,time
import numpy as np
from tools.tight_paired_evaluate import audit_rng
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';read=lambda p:json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');p.add_argument('--final',action='store_true');args=p.parse_args()
rows=list(csv.DictReader((D/'per_initial_state.csv').open()));assert len(rows)==len({(r['epoch'],r['role'],r['condition'],r['seed'],r['cohort']) for r in rows})
if args.final:assert (R/'curve_complete.json').exists() and len(rows)==224
proof=[];initial=[];rejections=[];verified_policy_hashes={}
def digest(path):
 if path not in verified_policy_hashes:verified_policy_hashes[path]=sha256(path)
 return verified_policy_hashes[path]
for row in rows:
 directory=P/Path(row['source']).parent;res=read(directory/'result.json');assert res['software_complete']
 assert not res['success'] or res['legal']
 admission=read(directory/'actual_layout_admission.json');assert admission['passed']
 for a in admission['rows']:assert a['passed'] and max(a['errors'].values())<=1e-5
 owner=R/row['owner'];plan=read(owner/'rollouts/plan.json')
 assert plan['observe_only'] and plan['expert_permission'] is False and plan['sampler_episode_seed']==7091
 if res.get('not_started'):
  assert not res['success'] and not res['legal'];rejections.append(dict(row,reason=res['reason']));continue
 case=next(c for c in plan['cases'] if c['name']==directory.name);pair=plan['pairs'][case['role']]
 assert res['complete_student_policy']==pair
 assert res['checkpoint_sha256']==pair['backend_sha256']
 assert read(directory/'metadata.json')['policy_checkpoint_sha256']==pair['backend_sha256']
 for component in ['frontend','backend']:assert digest(pair[component])==pair[component+'_sha256']
 steps=res['steps'];assert steps>0 and res['legal']
 with np.load(directory/'plans.npz') as saved:plans=len(saved['plans'])
 assert plans==math.ceil(steps/12)
 reset=read(directory/'sampler_episode_reset.json');assert reset['seed']==7091 and reset['extra_random_draws']==0
 rng=audit_rng(directory,plans,reconstruct_cuda=args.cuda)
 proof.append(dict(epoch=int(row['epoch']),role=row['role'],condition=row['condition'],seed=int(row['seed']),cohort=row['cohort'],steps=steps,planning_calls=plans,rng=rng,source=str(directory),actual_layout_passed=True))
 v=read(directory/'initial_observable.json');base=np.asarray(v['base_pose_w']).reshape(-1,7)[0]
 observed={}
 for key in ['joint_position_rad','joint_velocity_rad_s','actuator_target_rad','object_velocity_w_m_s_rad_s','goal_velocity_w_m_s_rad_s']:
  if key in v:observed[key]=np.asarray(v[key]).reshape(-1).tolist()
 for key in ['object_pose_w','hole_pose_w','goal_pose_w','palm_body_pose_w','table_pose_w']:
  if key in v:
   value=np.asarray(v[key]).reshape(-1,7)[0].copy();value[:3]-=base[:3];observed[key]=value.tolist()
 initial.append(dict(seed=int(row['seed']),case=directory.name,epoch=int(row['epoch']),source=str(directory),observed=observed))
# Compare observed reset variables across policies, checkpoints and recovery owners.
# Relative base positions remove lane origins; hidden PhysX state is not covered.
paired=[]
for seed in sorted({r['seed'] for r in initial}):
 values=[r for r in initial if r['seed']==seed];ref=values[0]
 for other in values[1:]:
  errors={key:float(np.max(np.abs(np.asarray(value)-other['observed'][key]))) for key,value in ref['observed'].items()}
  paired.append(dict(seed=seed,reference=ref['source'],case=other['source'],errors=errors,maximum_error=max(errors.values()),passed=max(errors.values())<=1e-5))
assert all(x['passed'] for x in paired),'Preserve outcomes and inspect actual initial-state mismatch'
freeze=None
if args.final:
 reg=read(R/'registration.json');final_owners=[R/'final16_A',R/'final16_B']
 final_start=min(read(o/'project_job_started.json')['utc'] for o in final_owners)
 training_end=max(read(R/c/'rear60_complete.json')['utc'] for c in reg['conditions'])
 assert training_end<final_start
 assert not set(reg['sealed_new_final_families'])&set(reg['training_families']+reg['development_families']+reg['sealed_historical64'])
 freeze=dict(last_final_pair_training_complete_utc=training_end,first_independent16_owner_start_utc=final_start,all_four_final_pairs_frozen_before_independent_evaluation=True)
summary=dict(frozen_final_proof=freeze,actual_policy_hashes=verified_policy_hashes,utc=time.time(),final=args.final,complete_registered_outcomes=len(rows),physically_executed=len(proof),pre_action_collision_rejections=rejections,noise_receipts=proof,native_cuda_noise_reconstruction=args.cuda,paired_initial_checks=paired,all_observed_pairs_passed=True,maximum_initial_error=max((x['maximum_error'] for x in paired),default=0),old64_read=0,expert_permission=False,scope='Observable initial variables and diffusion draws only; no hidden-state identity or bitwise DINO determinism claim')
atomic_json(D/'evaluation_contract_audit.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['noise_receipts','paired_initial_checks','pre_action_collision_rejections']}),flush=True)
