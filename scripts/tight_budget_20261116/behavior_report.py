"""Stage-specific short autonomy denominators and observable initial-state checks."""
import json,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visuotactile.grasp_curriculum import holding_metrics
from tools.tight_fresh_closed_gap import object_position_in_base
from scripts.tight_budget_20261116.run import ROOT,read
O=Path('research_docs/evidence/tight_diagnosis_20261116/budget_extension');O.mkdir(parents=True,exist_ok=True)
assert read(ROOT/'evaluation_complete.json')['complete']
keys=['contact','two_finger','native_lift','contact_height','hold12','hold60','final_hold60','alignment','subgoal','observed_insertion']
rows=[];counts=[];initial=[]
for label in read(ROOT/'evaluation_complete.json')['groups']:
 d=ROOT/('evaluation_'+label);plan=read(d/'plan.json')
 for case in plan['cases']:
  r=read(d/case['name']/'result.json');v=dict(group=label,role=case['role'],case=case['name'],family=case['spec']['family'],cohort=case.get('diagnostic_cohort','fixed_development'),legal=r['legal'],directory=str(d/case['name']))
  if r['legal']:
   with h5py.File(d/case['name']/'numeric.h5') as f:
    t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:300];force=t['object_contact_force_w'][:300]
    fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1);settled=np.arange(len(pose))>=48;height=pose[:,2]>=np.median(pose[24:48,2])+.02
    def first(mask):a=np.flatnonzero(mask);return int(a[0]+1) if len(a) else None
    h12=holding_metrics(pose,force,window=12);h60=holding_metrics(pose,force)
    v.update(ticks=len(pose),contact=bool(((fingers>0)&settled).any()),two_finger=bool(((fingers>=2)&settled).any()),
     native_lift=bool(t['diagnostics_after/lifted_object'][:300].any()),contact_height=bool(((fingers>=2)&height&settled).any()),
     hold12=h12['sustained'],hold60=h60['sustained'],final_hold60=h60['held_at_final_window'],alignment=bool(t['native_alignment_condition'][:300].any()),
     subgoal=bool(t['diagnostics_after/successes'][:300].max()>0),first_contact=first((fingers>0)&settled),first_two=first((fingers>=2)&settled),hold_details=h60,
     insertion=None,observed_insertion=bool(r.get('success')),full_task_eligible=False,censored=r.get('grasp_stage_censored'))
   baseline=ROOT/'evaluation_baseline8'/case['name']
   if label!='baseline8' and read(baseline/'result.json')['legal']:
    with h5py.File(baseline/'numeric.h5') as f,h5py.File(d/case['name']/'numeric.h5') as g:
     x=f['episode/transition/diagnostics_before'];y=g['episode/transition/diagnostics_before']
     q=float(np.abs(x['joint_position_rad'][0]-y['joint_position_rad'][0]).max())
     ob=object_position_in_base(x['object_pose_w'][:1],x['base_pose_w'][:1])-object_position_in_base(y['object_pose_w'][:1],y['base_pose_w'][:1])
     initial.append(dict(group=label,case=case['name'],joint_max_abs_rad=q,object_base_position_gap_m=float(np.linalg.norm(ob)),hidden_state_exact=False))
  rows.append(v)
 for role in 'AB':
  for cohort in sorted({x['cohort'] for x in rows if x['group']==label and x['role']==role}):
   selected=[x for x in rows if x['group']==label and x['role']==role and x['cohort']==cohort];legal=[x for x in selected if x['legal']]
   counts.append(dict(group=label,role=role,cohort=cohort,attempts=len(selected),legal=len(legal),illegal=len(selected)-len(legal),**{k:sum(bool(x[k]) for x in legal) for k in keys}))
(O/'behavior_cases.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'initial_comparison.json').write_text(json.dumps(initial,indent=2)+'\n')
with (O/'behavior_counts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(counts[0]),lineterminator='\n');w.writeheader();w.writerows(counts)
fig,axes=plt.subplots(3,2,figsize=(16,12))
for col,role in enumerate('AB'):
 for line,cohort in enumerate(['fixed_development','training_family_recheck','additional_development']):
  ax=axes[line,col];group=[x for x in counts if x['role']==role and x['cohort']==cohort]
  a=np.array([[x[k]/x['legal'] if x['legal'] else 0 for k in keys] for x in group]);ax.imshow(a,aspect='auto',vmin=0,vmax=1,cmap='YlGn')
  for i,x in enumerate(group):
   for j,k in enumerate(keys):ax.text(j,i,f"{x[k]}/{x['legal']}",ha='center',va='center',fontsize=8)
  ax.set_yticks(range(len(group)),[x['group'] for x in group],fontsize=8);ax.set_xticks(range(len(keys)),keys,rotation=45,ha='right',fontsize=8);ax.set_title(role+' '+cohort)
fig.suptitle('Full DP+Transformer, 16 lanes, first300 control ticks = 5s. Distinct criteria; no full insertion failure denominator.')
fig.tight_layout();fig.savefig(O/'behavior_stages.png',dpi=150);plt.close(fig)
print(json.dumps(counts,indent=2))
