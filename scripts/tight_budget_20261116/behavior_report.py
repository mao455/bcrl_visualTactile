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
def rotation_gap(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 return float(2*np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
for label in read(ROOT/'evaluation_complete.json')['groups']:
 d=ROOT/('evaluation_'+label);plan=read(d/'plan.json')
 for case in plan['cases']:
  r=read(d/case['name']/'result.json');pair=plan['pairs'][case['role']]
  v=dict(group=label,role=case['role'],case=case['name'],family=case['spec']['family'],cohort=case.get('diagnostic_cohort','fixed_development'),legal=r['legal'],directory=str(d/case['name']),
   frontend_sha256=pair['frontend_sha256'],backend_sha256=pair['backend_sha256'])
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
     physical=dict(group=label,case=case['name'],joint_max_abs_rad=q,object_base_position_gap_m=float(np.linalg.norm(ob)),hidden_state_exact=False,
      joint_velocity_max_abs_rad_s=float(np.abs(x['joint_velocity_rad_s'][0]-y['joint_velocity_rad_s'][0]).max()),
      object_velocity_max_abs_components=np.abs(x['object_velocity_w_m_s_rad_s'][0]-y['object_velocity_w_m_s_rad_s'][0]).tolist(),
      world_orientation_gaps_rad={k:rotation_gap(x[k][0,3:7],y[k][0,3:7]) for k in ['base_pose_w','object_pose_w','goal_pose_w','hole_pose_w']},
      discrete_equal={k:bool(np.array_equal(x[k][0],y[k][0])) for k in ['native_stage_step','successes','retract_phase','max_goals']})
     for key in ['goal_pose_w','hole_pose_w']:
      gap=object_position_in_base(x[key][:1],x['base_pose_w'][:1])-object_position_in_base(y[key][:1],y['base_pose_w'][:1])
      physical[key+'_base_position_gap_m']=float(np.linalg.norm(gap))
     initial.append(physical)
  rows.append(v)
 for role in 'AB':
  for cohort in sorted({x['cohort'] for x in rows if x['group']==label and x['role']==role}):
   selected=[x for x in rows if x['group']==label and x['role']==role and x['cohort']==cohort];legal=[x for x in selected if x['legal']]
   counts.append(dict(group=label,role=role,cohort=cohort,attempts=len(selected),legal=len(legal),illegal=len(selected)-len(legal),**{k:sum(bool(x[k]) for x in legal) for k in keys}))
groups=read(ROOT/'evaluation_complete.json')['groups'];paired=[];changes=[]
for role in 'AB':
 for cohort in sorted({r['cohort'] for r in rows if r['role']==role}):
  maps={g:{r['case']:r for r in rows if r['group']==g and r['role']==role and r['cohort']==cohort} for g in groups}
  common=set.intersection(*[{k for k,v in values.items() if v['legal']} for values in maps.values()])
  paired.append(dict(role=role,cohort=cohort,common_legal_cases=sorted(common),
   excluded_registered_cases=sorted(set.union(*[set(v) for v in maps.values()])-common),
   groups={g:dict(n=len(common),**{k:sum(bool(values[i][k]) for i in common) for k in keys}) for g,values in maps.items()}))
  for g in groups[1:]:
   changes.append(dict(group=g,role=role,cohort=cohort,common_legal_n=len(common),
    gained={k:sorted(i for i in common if not maps['baseline8'][i][k] and maps[g][i][k]) for k in ['native_lift','hold12','hold60','final_hold60']},
    lost={k:sorted(i for i in common if maps['baseline8'][i][k] and not maps[g][i][k]) for k in ['native_lift','hold12','hold60','final_hold60']}))
(O/'paired_behavior.json').write_text(json.dumps(dict(counts=paired,changes_from_baseline8=changes,scope='Intersection of legal registered cases across groups; observable initial-state differences reported separately; no hidden-state exact claim.'),indent=2)+'\n')
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
