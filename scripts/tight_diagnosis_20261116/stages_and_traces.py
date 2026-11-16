"""Physical stage counts and failure locations from saved numeric trajectories."""
import sys,json,csv
from pathlib import Path
import h5py,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visuotactile.grasp_curriculum import holding_metrics
B=Path('outputs/visuotactile/run');O=Path('research_docs/evidence/tight_diagnosis_20261116');read=lambda p:json.loads(p.read_text())
R={n:B/p/f'round_{n}' for n,p in [(16,'tight_fresh_20261116_01'),(17,'tight_fresh_20261116_02'),(18,'tight_fresh_20261116_02'),(19,'tight_fresh_20261116_04')]}
def csvout(n,rows):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
def save(fig,n):fig.tight_layout();fig.savefig(O/n,dpi=160);plt.close(fig)
def measured(directory,case,horizon=None):
 result=read(directory/case['name']/'result.json');r=dict(case=case['name'],role=case['role'],family=case['spec']['family'],event_seed=case['spec'].get('event_seed'),legal=result['legal'],censored=result.get('grasp_stage_censored',False),directory=str(directory),horizon=horizon)
 if not result['legal']:return r
 with h5py.File(directory/case['name']/'numeric.h5') as f:
  t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:horizon];force=t['object_contact_force_w'][:horizon];n=len(pose)
  fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1);settled=np.arange(n)>=48;height=pose[:,2]>=np.median(pose[24:48,2])+.02
  def first(x):a=np.flatnonzero(x);return int(a[0]+1) if len(a) else None
  h12=holding_metrics(pose,force,window=12);h60=holding_metrics(pose,force)
  r.update(ticks=n,contact=bool(((fingers>0)&settled).any()),two_finger=bool(((fingers>=2)&settled).any()),native_lift=bool(t['diagnostics_after/lifted_object'][:horizon].any()),contact_height=bool(((fingers>=2)&height&settled).any()),hold12=h12['sustained'],hold60=h60['sustained'],final_hold60=h60['held_at_final_window'],alignment=bool(t['native_alignment_condition'][:horizon].any()),subgoal=bool(t['diagnostics_after/successes'][:horizon].max()>0),insertion=bool(result['success']) if horizon is None else None,first_contact=first((fingers>0)&settled),first_two=first((fingers>=2)&settled),first_contact_height=first((fingers>=2)&height&settled),hold60_details=h60)
 return r
keys=['contact','two_finger','native_lift','contact_height','hold12','hold60','final_hold60','alignment','subgoal'];cases=[];counts=[]
for label,role,directory in [('A11','A',B/'tight_fresh_20261116_01/source_backend_baseline'),('B15','B',B/'tight_fresh_20261116_00/round_15/development')]+[(str(n),role,root/('development_camera_repaired_01' if n==16 else 'development')) for n,root in R.items() for role in 'AB']:
 plan=read(directory/'plan.json');selected=[c for c in plan['cases'] if c['role']==role]
 if label in ('A11','B15'):selected=[c for c in selected if c['spec']['family'] in {f'fresh_development_{28500000+i}' for i in range(4)}]
 for c in selected:
  row=measured(directory,c,300);row.update(round=label,cohort=c.get('diagnostic_cohort','fixed_development'));cases.append(row)
 for cohort in sorted({r['cohort'] for r in cases if r['round']==label and r['role']==role}):
  group=[r for r in cases if r['round']==label and r['role']==role and r['cohort']==cohort];legal=[r for r in group if r['legal']]
  counts.append(dict(round=label,role=role,cohort=cohort,horizon=300,legal=len(legal),illegal=len(group)-len(legal),**{k:sum(bool(r[k]) for r in legal) for k in keys}))
csvout('matched_grasp300.csv',counts);(O/'matched_grasp300_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(15,5))
for ax,role in zip(axs,'AB'):
 group=[r for r in counts if r['role']==role and r['cohort']=='fixed_development'];matrix=[[r[k]/r['legal'] for k in keys] for r in group];ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
 for i,r in enumerate(group):
  for j,k in enumerate(keys):ax.text(j,i,f"{r[k]}/{r['legal']}",ha='center',va='center',fontsize=8)
 ax.set_xticks(range(len(keys)),keys,rotation=55,ha='right');ax.set_yticks(range(len(group)),[r['round'] for r in group]);ax.set_title(role+' matched registered families, first300controls')
fig.suptitle('Independent criteria; contact-height is not native lift. First300 = 5 simulation seconds.');save(fig,'matched_stages.png')
# Historical completed development cohorts kept separate; no pooling CPU and GPU.
historical=[];unavailable=[]
for root in sorted(B.glob('tight_fresh_*/round_*')):
 if not root.is_dir() or root.name not in [f'round_{n:02}'+v for n in range(1,20) for v in ('','_v2','_v3')]:continue
 if root.name=='round_02' and root.with_name('round_02_v2').exists():continue
 d=root/('development_camera_repaired_01' if root==R[16] else 'development_recovery_01' if (root/'development_recovery_01/plan.json').exists() else 'development')
 if root.name=='round_14':d=B/'tight_fresh_20261115_07/throughput_preflight/round14_recovery'
 if not (d/'plan.json').exists():continue
 backend=read(d/'execution_backend_observed.json') if (d/'execution_backend_observed.json').exists() else read(d/'plan.json').get('execution_backend',{})
 complete_cases=[c for c in read(d/'plan.json')['cases'] if (d/c['name']/'result.json').exists()]
 missing_cases=[c['name'] for c in read(d/'plan.json')['cases'] if c not in complete_cases]
 if missing_cases:unavailable.append(dict(directory=str(d),software_unresolved_cases=missing_cases))
 try:allrows=[measured(d,c) for c in complete_cases]
 except (KeyError,FileNotFoundError) as e:unavailable.append(dict(directory=str(d),reason=str(e)));continue
 if root.name=='round_14':
  original=root/'development'
  for c in read(original/'plan.json')['cases']:
   if (original/c['name']/'result.json').exists() and c['name'] not in {r['case'] for r in allrows}:allrows.append(measured(original,c))
 for role in 'AB':
  group=[r for r in allrows if r['role']==role];legal=[r for r in group if r['legal']];full=[r for r in legal if not r['censored']]
  historical.append(dict(root=str(root),role=role,backend=json.dumps(backend,sort_keys=True),legal=len(legal),illegal=len(group)-len(legal),**{k:sum(bool(r[k]) for r in legal) for k in keys},full_task_legal=len(full),insertion=sum(bool(r['insertion']) for r in full),cohort='entire registered cohort; differs by round; not a matched causal comparison'))
csvout('historical_stages.csv',historical);(O/'historical_missing.json').write_text(json.dumps(unavailable,indent=2)+'\n')
# New expert rescues are a separate population, never student successes.
rescues=[]
for n,root in R.items():
 d=root.parent/f'new_family_{n}';plan=read(d/'plan.json');rows=[measured(d,c) for c in plan['cases']]
 for role in 'AB':
  group=[r for r in rows if r['role']==role];legal=[r for r in group if r['legal']];result=[read(d/r['case']/'result.json') for r in legal]
  rescues.append(dict(round=n,role=role,attempts=len(group),legal=len(legal),takeovers=sum(x['dagger']['takeover_triggered'] for x in result),expert_success=sum(x['success'] for x in result),admitted=sum(x['success'] for x in result),**{k:sum(bool(r[k]) for r in legal) for k in keys}))
csvout('expert_rescues_separate.csv',rescues)
# Same-state expert command gap + physical event traces, first150 error then real300 physical history.
cf=read(O/'round19_command_chain.json');trace_rows=[];events=[]
chosen=[r for r in cf['rows'] if 'recheck' in r['case']];fig,axs=plt.subplots(3,4,figsize=(16,9))
for column,r in enumerate(chosen):
 d=Path(r['evaluation'])/r['case']
 with h5py.File(d/'numeric.h5') as f:
  t=f['episode/transition'];pose=t['diagnostics_after/object_pose_w'][:300];force=t['object_contact_force_w'][:300];q=t['diagnostics_after/joint_position_rad'][:300];target=t['target_after_rad'][:300];before=t['diagnostics_before/joint_position_rad'][:300]
  executed=t['actuator_substep_target_rad'][:300];assert np.max(np.abs(executed-target[:,None]))<2e-5
 count=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1);height=pose[:,2]-np.median(pose[24:48,2]);held=holding_metrics(pose,force);handtracking=np.sqrt(((q[:,7:]-target[:,7:])**2).mean(-1));error=r['hand_command_error_rad'];ticks=np.arange(1,len(pose)+1)
 axs[0,column].plot(np.arange(len(error))/60,error,label='final vs same-state expert');axs[0,column].axhline(.015,color='gray',ls=':',label='descriptive threshold')
 axs[0,column].set_title(r['role']+' family'+('76' if '76_' in r['case'] else '86'));axs[0,column].set_ylabel('command RMS gap (rad)')
 axs[1,column].plot(ticks/60,height*100,label='object rise');axs[1,column].axhline(2,color='gray',ls=':');axs[1,column].set_ylabel('height above settled median (cm)')
 axs[2,column].step(ticks/60,count,where='post',label='fingers in both substeps');axs[2,column].set_ylabel('contact finger count');axs[2,column].set_ylim(-.2,5.3)
 contact=np.flatnonzero((count>0)&(ticks>48));first=int(contact[0]+1) if len(contact) else None
 for ax in axs[:,column]:
  if first:ax.axvline(first/60,ls='--',color='orange',label='first contact')
  ax.grid(alpha=.2);ax.set_xlabel('simulation seconds');ax.legend(fontsize=6)
 on=r['sustained_hand_error_onset']['zero_based_control_tick'];two_indices=np.flatnonzero((count>=2)&(ticks>48));first_two=int(two_indices[0]+1) if len(two_indices) else None
 loss_two=next((int(k+1) for k in range(first_two or len(count),len(count)) if count[k]<2),None)
 loss_window=next((k for k in range((held['first_completed_tick'] or len(pose))+1,len(pose)+1) if not holding_metrics(pose[:k],force[:k])['held_at_final_window']),None)
 if loss_window:
  for ax in axs[1:,column]:ax.axvline(loss_window/60,color='red',ls=':',label='hold60 window lost');ax.legend(fontsize=6)
 events.append(dict(case=r['case'],role=r['role'],first_contact_completed_tick=first,error_onset_action_index=on,error_before_contact=on is not None and first is not None and on<first,hold=held,first_two_finger_tick=first_two,first_lost_two_finger_tick=loss_two,first_lost_hold60_window_tick=loss_window,actuator_targets_exact=True))
 for j in range(len(pose)):trace_rows.append(dict(case=r['case'],action_index=j,completed_tick=j+1,sim_seconds_after=(j+1)/60,height_above_settled_m=float(height[j]),two_substep_contact_fingers=int(count[j]),hand_target_tracking_rmse_rad=float(handtracking[j]),same_state_expert_hand_command_gap_rad=error[j] if j<len(error) else None))
save(fig,'round19_failures.png');csvout('failure_trace_ticks.csv',trace_rows);(O/'failure_events.json').write_text(json.dumps(events,indent=2)+'\n')
# Existing true two-original fitting experiment; no invented intermediate epochs.
fig,axs=plt.subplots(1,2,figsize=(11,4));fixed=[]
for ax,role in zip(axs,'AB'):
 hist=read(B/f'tight_fresh_20261116_02/fixed_fit/{role}/metrics.json')
 for identity in hist[0]['episodes']:
  ax.plot([r['updates'] for r in hist],[r['episodes'][identity]['before150']['hand_rmse_rad'] for r in hist],'-o',ms=3,label=identity)
  for r in hist:fixed.append(dict(role=role,identity=identity,updates=r['updates'],passes=r['passes'],**r['episodes'][identity]['before150']))
 ax.set_title(role+' frozen DP + fitted rear, original2');ax.set_xlabel('actual optimizer updates');ax.set_ylabel('first150 hand command RMSE (rad)');ax.legend(fontsize=8);ax.grid(alpha=.2)
save(fig,'fixed_train2_fit.png');csvout('fixed_train2_fit.csv',fixed)
print(json.dumps(dict(matched=counts,historical_cohorts=len(historical),historical_missing=unavailable,events=events),indent=2))
# Same-family source/candidate observable reset check; no hidden-state replay claim.
from tools.tight_fresh_closed_gap import object_position_in_base
pairs=[]
source_dirs={'A':B/'tight_fresh_20261116_01/source_backend_baseline','B':B/'tight_fresh_20261116_00/round_15/development'}
for row in cases:
 if row['round'] in ('A11','B15') or row['cohort']!='fixed_development' or not row['legal']:continue
 a=source_dirs[row['role']]/row['case'];d=Path(row['directory'])/row['case']
 if not read(a/'result.json')['legal']:continue
 with h5py.File(a/'numeric.h5') as f,h5py.File(d/'numeric.h5') as g:
  x=f['episode/transition/diagnostics_before'];y=g['episode/transition/diagnostics_before'];q=x['joint_position_rad'][0]-y['joint_position_rad'][0]
  obj=object_position_in_base(x['object_pose_w'][:1],x['base_pose_w'][:1])-object_position_in_base(y['object_pose_w'][:1],y['base_pose_w'][:1])
  pairs.append(dict(round=row['round'],role=row['role'],case=row['case'],initial_joint_max_abs_rad=float(np.abs(q).max()),initial_object_base_gap_m=float(np.linalg.norm(obj)),source=str(a),candidate=str(d),hidden_state_exact=False))
(O/'development_initial_comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')
