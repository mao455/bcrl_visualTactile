"""CPU-only B rear6/rear22 diagnosis on all four fixed development cases."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py,numpy as np,torch
from visuotactile.compact import sha256
from tools import tight_fresh_expert_counterfactual as expert
from scripts.tight_budget_20261116.run import ROOT,read

TREE=Path(__file__).resolve().parents[2]
O=TREE/'research_docs/evidence/tight_diagnosis_20261116/budget_extension/closed_command_gap.json'
assert not O.exists(),'Preserve completed diagnostic'
assert read(ROOT/'evaluation_complete.json')['complete']
previous=read(TREE/'research_docs/evidence/tight_diagnosis_20261116/round19_command_chain.json')
assert previous['complete'] and previous['source_sha256']==sha256(Path(expert.__file__))
validation=previous['validation']
assert validation['recorded_raw_mean_max_abs_difference']<=1e-4
assert validation['recorded_mapped_target_max_abs_difference']<=2e-6
session=read(ROOT/'session.json');checkpoint=Path(session['expert_checkpoint'])
assert sha256(checkpoint)==previous['expert_sha256']==session['expert_sha256']
torch.set_num_threads(2);started=time.time()
groups=['extended_dp_rear8','extended_best'];cases=[];rows=[];comparisons=[]
for group in groups:
 plan=read(ROOT/('evaluation_'+group)/'plan.json')
 selected=[c for c in plan['cases'] if c['role']=='B' and c.get('diagnostic_cohort','fixed_development')=='fixed_development']
 assert len(selected)==4
 cases.append([c['name'] for c in selected])
assert cases[0]==cases[1]
sample=ROOT/('evaluation_'+groups[0])/cases[0][0]/'numeric.h5'
with h5py.File(sample) as f:size=f['episode/transition/expert_actor_input_diagnostic'].shape[-1]
model=expert.load_model(checkpoint,size,TREE)
behavior=read(O.parent/'behavior_cases.json')
for group in groups:
 for case in cases[0]:
  directory=ROOT/('evaluation_'+group)/case
  assert read(directory/'result.json')['legal']
  row=expert.query(model,directory/'numeric.h5',TREE,ticks=150,decompose_student=True)
  assert row['expert_sha256']==session['expert_sha256']
  row.update(group=group,case=case,role='B')
  physical=next(r for r in behavior if r['group']==group and r['case']==case)
  row.update(first_contact=physical['first_contact'],first_two=physical['first_two'],hold60=physical['hold60'])
  with h5py.File(directory/'numeric.h5') as f:
   t=f['episode/transition'];target=t['target_after_rad'][:150];applied=t['actuator_substep_target_rad'][:150]
   row['actuator_target_max_abs_error_rad']=float(np.abs(applied-target[:,None]).max())
   assert row['actuator_target_max_abs_error_rad']<=2e-6
  rows.append(row)
for case in cases[0]:
 directories=[ROOT/('evaluation_'+g)/case for g in groups]
 with np.load(directories[0]/'plans.npz') as a,np.load(directories[1]/'plans.npz') as b:
  plan_gap=float(np.abs(a['plans'][0]-b['plans'][0]).max())
 with np.load(directories[0]/'trace.npz') as a,np.load(directories[1]/'trace.npz') as b:
  target_gap=np.sqrt(((a['target'][:150,7:]-b['target'][:150,7:])**2).mean(-1))
  first=next((i for i in range(148) if np.all(target_gap[i:i+3]>.001)),None)
 comparisons.append(dict(case=case,first_dp_plan_max_abs_gap_rad=plan_gap,
  first_sustained_hand_target_difference_action_index=first,descriptive_difference_threshold_rad=.001,
  minimum_consecutive_controls=3,hand_target_difference_rad=target_gap.tolist(),
  note='Between-policy difference, NOT expert error. Same DP weights; subsequent inputs/history evolve with each policy.'))
summary=[]
for group in groups:
 chosen=[r for r in rows if r['group']==group]
 for key in ['controls_0_48','controls_48_100','controls_100_150']:
  stages=chosen[0]['command_decomposition'][key]
  summary.append(dict(group=group,controls=key,**{name:float(np.sqrt(np.mean([r['command_decomposition'][key][name]['hand_rmse_rad']**2 for r in chosen])))
   for name in ['nominal','projected_nominal_diagnostic','rear_before_projection','final']}))
O.write_text(json.dumps(dict(complete=True,seconds=time.time()-started,rows=rows,comparisons=comparisons,summary=summary,
 reused_validation=dict(path=str(TREE/'research_docs/evidence/tight_diagnosis_20261116/round19_command_chain.json'),
  source_sha256=previous['source_sha256'],raw_mean_max_difference=validation['recorded_raw_mean_max_abs_difference'],
  mapped_max_difference=validation['recorded_mapped_target_max_abs_difference']),
 expert_sha256=session['expert_sha256'],source_sha256=sha256(Path(__file__)),simulator_starts=0,optimizer_updates=0,
 scope='All four preregistered B fixed-development cases, same DP with rear6 versus rear22. Actual recorded states and complete recurrent expert input history; CPU diagnostic suggestions only, never executed, admitted or used for tuning. No exact hidden-state or expert recoverability claim.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
