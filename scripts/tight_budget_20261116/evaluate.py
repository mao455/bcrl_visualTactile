"""Finite same-initial-state short full-policy comparisons; never collection."""
import copy,json,sys,time,subprocess
from pathlib import Path
TREE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TREE))
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.narrow_data import direct_configuration
from scripts.tight_budget_20261116.run import ROOT,OLD,execute,read

def main():
 import torch
 from tools.tight_fresh_rollouts import completed_pair
 from visuotactile.state_benchmark import dagger_policy_identity
 assert read(ROOT/'training_complete.json')['complete']
 baseline={r:completed_pair(OLD,r) for r in 'AB'}
 extended={r:completed_pair(ROOT,r) for r in 'AB'}
 rear8={}
 for r in 'AB':
  p=ROOT/r/'backend'/r/'training/best_through_epoch8.pt'
  s=torch.load(p,map_location='cpu',weights_only=False)
  assert s['epoch']<=8 and s['updates']>0 and s['configuration']['bases'][r]['sha256']==extended[r]['frontend_sha256']
  rear8[r]=dagger_policy_identity(s,str(p))
 candidates=[('baseline8',baseline),('extended_dp_rear8',rear8)]
 if any(rear8[r]['backend_sha256']!=extended[r]['backend_sha256'] for r in 'AB'):candidates.append(('extended_best',extended))
 else:atomic_json(ROOT/'evaluation_duplicate_omitted.json',dict(reason='Both selected rear models already selected within first8',equivalent='extended_dp_rear8'))
 session=read(ROOT/'session.json')
 for label,pairs in candidates:
  output=ROOT/('evaluation_'+label)
  if not output.exists():
   output.mkdir();plan=copy.deepcopy(read(OLD/'development/plan.json'))
   plan.update(pairs=pairs,resource_root=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip(),
    source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools'] for p in sorted(Path(folder).rglob('*.py'))},utc=time.time())
   for c in plan['cases']:c.update(video=False,grasp_evaluation_ticks=300)
   for seed in [28500004,28500005]:
    for role in 'AB':
     spec=dict(identity=f'budget_development_{seed}',family=f'fresh_development_{seed}',split='development',seed=seed,event_seed=seed+100000,
      cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(session['domain'],seed))
     plan['cases'].append(dict(name=role+'_'+spec['identity'],role=role,spec=spec,video=False,grasp_evaluation_ticks=300,diagnostic_cohort='additional_development'))
   assert len(plan['cases'])==16
   plan['batch']['lanes']=16;plan['evaluation_scope']=dict(grasp_horizon=300,full_task_per_role=0,
    scope='8 fixed development +4 previously corrected train families +4 additional development; same registered cases per pair; no full-task insertion denominator')
   atomic_json(output/'session.json',session);atomic_json(output/'plan.json',plan)
  execute('evaluation_'+label,['tools.tight_fresh_batch_run','run','--output',output],output/'result.json')
 atomic_json(ROOT/'evaluation_complete.json',dict(complete=True,utc=time.time(),groups=[x[0] for x in candidates],attempts_per_group=16))
if __name__=='__main__':main()
