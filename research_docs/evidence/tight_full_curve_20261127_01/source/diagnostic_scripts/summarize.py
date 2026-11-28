from pathlib import Path
import collections,csv,json,sys,time
import numpy as np
import h5py
from tools.tight_paired_evaluate import wilson,contrast,geometry_metrics
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01');D=Path('outputs/diagnostics/tight_full_curve_20261127_01');read=lambda p:json.loads(p.read_text())
reg=read(R/'registration.json');rows=[];attempts=[]
def csvout(path,records):
 fields=list(dict.fromkeys(k for row in records for k in row))
 with path.open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(records)
for owner in sorted(list(R.glob('evaluation_*'))+list(R.glob('final16_*'))):
 p=owner/'rollouts/plan.json'
 if not p.exists():continue
 plan=read(p);epoch=plan['evaluation_scope']['epoch']
 for case in plan['cases']:
  directory=p.parent/case['name'];res=read(directory/'result.json') if (directory/'result.json').exists() else {}
  row=dict(owner=owner.name,epoch=epoch,role=case['role'][0],condition='start' if epoch==0 else case['role'][2:],seed=case['spec']['seed'],cohort=case['diagnostic_cohort'],software_complete=bool(res.get('software_complete')),legal=res.get('legal'),success=bool(res.get('success')),steps=res.get('steps'),reason=json.dumps(res.get('reasons',res.get('reason','technical_unfinished')),ensure_ascii=False),source=str(directory/'result.json'))
  row['not_started']=bool(res.get('not_started'));row['physical_executed']=bool(res.get('steps',0));row['case_started']=(directory/'started.json').exists()
  row['attempt_status']='pre_action_collision_rejected' if row['software_complete'] and row['not_started'] else 'completed_physical' if row['software_complete'] else 'technical_interruption_after_admission' if row['case_started'] else 'not_dispatched'
  attempts.append(row.copy())
  if not row['software_complete']:continue
  if (directory/'numeric.h5').exists():
   with h5py.File(directory/'numeric.h5') as f:
    tr=f['episode/transition'];before=tr['diagnostics_before/successes'][:];after=tr['diagnostics_after/successes'][:]
    metric=geometry_metrics(tr['diagnostics_after/object_pose_w'][:],tr['object_contact_force_w'][:],tr['native_keypoints_max_dist'][:],before)
    row.update(metric);row['native_first_subgoal']=bool((after>=1).any())
    obj=tr['diagnostics_after/object_pose_w'][:];goal=tr['diagnostics_before/goal_pose_w'][:];root=np.linalg.norm(obj[:,:3]-goal[:,:3],axis=-1)
    contact=(np.linalg.norm(tr['object_contact_force_w'][:],axis=-1)>.1).sum(-1).min(-1)
    held=np.zeros(len(obj),bool)
    if len(obj)>=48:
     good=(contact>=2)&(obj[:,2]>=np.median(obj[24:48,2])+.02);run=0
     for i,v in enumerate(good):run=run+1 if v else 0;held[i]=run>=12 and i>=47
    row['stable_near_hole']=bool((held&(root<=.09)&(before==0)).any())
  else:row.update(stable_hold=False,first_15mm_gate=False,native_first_subgoal=False,stable_near_hole=False)
  admission=directory/'actual_layout_admission.json';row['actual_layout_passed']=read(admission)['passed'] if admission.exists() else False
  row['category']='strict_success' if row['success'] else 'post_subgoal_failure' if row['native_first_subgoal'] else 'entered_unconfirmed' if row['first_15mm_gate'] else 'held_near_never_gate' if row['stable_near_hole'] else 'held_not_near' if row['stable_hold'] else 'grasp_not_held'
  rows.append(row)
# A retry replaces only a technical unfinished entry, never a completed physical failure.
keys=[(r['epoch'],r['role'],r['condition'],r['seed'],r['cohort']) for r in rows];assert len(keys)==len(set(keys)),'No repeated physical outcomes may be silently selected'
summary=[]
for (ep,role,cond,cohort),items in __import__('itertools').groupby(sorted(rows,key=lambda r:(r['epoch'],r['role'],r['condition'],r['cohort'])),key=lambda r:(r['epoch'],r['role'],r['condition'],r['cohort'])):
 items=list(items);n=16 if cohort=='independent_final16' else 8
 summary.append(dict(epoch=ep,role=role,condition=cond,cohort=cohort,registered=n,completed=len(items),physical_executed=sum(r['physical_executed'] for r in items),pre_action_rejections=sum(r['not_started'] for r in items),legal=sum(bool(r['legal']) for r in items),software_unfinished=n-len(items),strict_success=sum(r['success'] for r in items),wilson95=wilson(sum(r['success'] for r in items),n),stable_hold=sum(r['stable_hold'] for r in items),stable_near_hole=sum(r['stable_near_hole'] for r in items),first_15mm_gate=sum(r['first_15mm_gate'] for r in items),native_first_subgoal=sum(r['native_first_subgoal'] for r in items),categories=dict(collections.Counter(r['category'] for r in items))))
paired=[]
for cohort,seeds in [('training_curve',reg['training_families']),('development_curve',reg['development_families']),('independent_final16',reg['sealed_new_final_families'])]:
 for ep in [30,60]:
  for role in 'AB':
   sets={c:{r['seed']:int(r['success']) for r in rows if r['cohort']==cohort and r['epoch']==ep and r['role']==role and r['condition']==c} for c in reg['conditions']}
   if all(set(v)==set(seeds) for v in sets.values()):paired.append(dict(cohort=cohort,epoch=ep,role=role,comparison='plus_corrected24 minus old_only',**contrast([sets['plus_corrected24'][i] for i in seeds],[sets['old_only'][i] for i in seeds])))
  for cond in reg['conditions']:
   a={r['seed']:int(r['success']) for r in rows if r['cohort']==cohort and r['epoch']==ep and r['role']=='A' and r['condition']==cond};b={r['seed']:int(r['success']) for r in rows if r['cohort']==cohort and r['epoch']==ep and r['role']=='B' and r['condition']==cond}
   if set(a)==set(b)==set(seeds):paired.append(dict(cohort=cohort,epoch=ep,condition=cond,comparison='A minus B',**contrast([a[i] for i in seeds],[b[i] for i in seeds])))
for ep in [30,60]:
 for cohort,seeds in [('training_curve',reg['training_families']),('development_curve',reg['development_families'])]:
  for role in 'AB':
   before={r['seed']:int(r['success']) for r in rows if r['cohort']==cohort and r['epoch']==0 and r['role']==role}
   for cond in reg['conditions']:
    after={r['seed']:int(r['success']) for r in rows if r['cohort']==cohort and r['epoch']==ep and r['role']==role and r['condition']==cond}
    if set(before)==set(after)==set(seeds):paired.append(dict(cohort=cohort,epoch=ep,role=role,condition=cond,comparison='milestone minus frozen_start',**contrast([after[i] for i in seeds],[before[i] for i in seeds])))
if rows:csvout(D/'per_initial_state.csv',rows)
if attempts:csvout(D/'all_attempts.csv',attempts)
if summary:csvout(D/'learning_curve.csv',summary)
atomic_json(D/'primary_summary.json',dict(utc=time.time(),rows=summary,paired_contrasts=paired,complete_opportunities=len(rows),actual_layout_all_passed=all(r['actual_layout_passed'] for r in rows),old64_used=False))
print(json.dumps(summary,ensure_ascii=False))
if '--plot' in sys.argv and summary:
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,cohort in zip(axes,['training_curve','development_curve']):
  for role,color in [('A','#356ac3'),('B','#d87719')]:
   start=next((r for r in summary if r['epoch']==0 and r['cohort']==cohort and r['role']==role),None)
   for cond,style in [('old_only','--'),('plus_corrected24','-')]:
    values=([start] if start else [])+[r for r in summary if r['epoch']>0 and r['cohort']==cohort and r['role']==role and r['condition']==cond]
    values=sorted(values,key=lambda r:r['epoch']);ax.plot([r['epoch'] for r in values],[r['strict_success']/r['registered']*100 for r in values],style+'o',color=color,label=role+' '+('old pool' if cond=='old_only' else '+corrected demos'))
    for r in values:
     ax.annotate(f"{r['strict_success']}/{r['registered']}",(r['epoch'],r['strict_success']/r['registered']*100),xytext=(0,5 if cond=='plus_corrected24' else -12),textcoords='offset points',fontsize=8)
  ax.set(title='Fixed training families' if cohort=='training_curve' else 'Fixed development families',xlabel='DP epoch (paired backend at same epoch)',xticks=[0,30,60],ylim=(-5,105));ax.grid(alpha=.25)
 axes[0].set_ylabel('Strict insertion success (%)');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(D/'autonomous_learning_curve.png',dpi=180)
