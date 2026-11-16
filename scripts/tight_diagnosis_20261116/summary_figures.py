"""Small pooled plots and quantitative conclusions; all points bind saved metrics."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,h5py,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.tight_fresh_closed_gap import object_position_in_base
O=Path('research_docs/evidence/tight_diagnosis_20261116');B=Path('outputs/visuotactile/run');read=lambda p:json.loads(p.read_text())
def rows(name):return list(csv.DictReader((O/name).open()))
def save(fig,name):fig.tight_layout();fig.savefig(O/name,dpi=160);plt.close(fig)
def pool(a,key):
 n=sum(int(r['ticks']) for r in a)
 return (sum(int(r['ticks'])*float(r[key])**2 for r in a)/n)**.5 if n else None
end=rows('dp_endpoints.csv');pooled=[];phases=['pre_first_contact','first_contact12','contact_height','hold60_window_end','later_ge150']
fig,axs=plt.subplots(2,2,figsize=(13,8))
for i,role in enumerate('AB'):
 for j,cohort in enumerate(('original2','latest4')):
  ax=axs[i,j]
  for offset,ckpt in enumerate(('source','selected','last')):
   vals=[]
   for phase in phases+['all_legal','before150']:
    a=[r for r in end if r['role']==role and r['cohort']==cohort and r['checkpoint']==ckpt and r['phase']==phase];v=pool(a,'hand_rmse_rad');pooled.append(dict(role=role,cohort=cohort,checkpoint=ckpt,phase=phase,ticks=sum(int(r['ticks']) for r in a),hand_rmse_rad=v))
    if phase in phases:vals.append(np.nan if v is None else v)
   ax.bar(np.arange(len(phases))+.23*(offset-1),vals,width=.23,label=ckpt)
  ax.set_xticks(range(len(phases)),phases,rotation=35,ha='right');ax.set_title(role+' '+cohort+' (same recorded inputs)');ax.set_ylabel('DP hand RMSE (rad)');ax.legend();ax.grid(axis='y',alpha=.2)
save(fig,'dp_endpoint_phases.png');(O/'dp_endpoint_summary.json').write_text(json.dumps(pooled,indent=2)+'\n')
phase=rows('offline_phase_metrics.csv');fig,axs=plt.subplots(1,2,figsize=(12,4))
physical=['approach_no_object_contact','contact_low_object','transport_above_3cm','near_goal','alignment','insertion']
for ax,role in zip(axs,'AB'):
 chosen=[r for r in phase if r['role']==role and r['source']=='stage_errors.csv' and r['split']=='validation' and r['group']=='hand' and r['motion']=='all' and r['phase']!='all'];names=list(dict.fromkeys(r['phase'] for r in chosen))
 for n in range(16,20):
  a=[r for r in chosen if int(r['round'])==n];mapping={r['phase']:r for r in a};ax.plot(range(len(names)),[float(mapping[k]['rmse_rad']) if mapping[k]['rmse_rad'] else np.nan for k in names],'-o',label=f'r{n}')
 ax.set_xticks(range(len(names)),names,rotation=30,ha='right');ax.set_title(role+' rear, fixed validation6');ax.set_ylabel('final expert-command hand RMSE (rad)');ax.legend();ax.grid(alpha=.2)
save(fig,'rear_physical_phases.png')
s=read(O/'training_summary.json');fig,axs=plt.subplots(1,2,figsize=(12,4))
for ax,role in zip(axs,'AB'):
 for n in range(16,20):
  h=s[f'{n}{role}']['rear_history'];ax.plot([r['epoch'] for r in h],[r['validation']['all']['normalized_equal_mse'] for r in h],'-o',label=f'r{n} val6')
  ax.plot([r['epoch'] for r in h],[r['train']['all']['normalized_equal_mse'] for r in h],'--',label=f'r{n} train2')
 ax.set_title(role+' rear normalized equal Arm/Hand');ax.set_xlabel('control-tick epoch');ax.set_ylabel('normalized MSE');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
save(fig,'rear_normalized_selection.png')
# Preserve genuine cumulative IO counters and separate warm/cold controlled probes.
io=[]
for role in 'AB':
 p=B/f'tight_fresh_20261116_04/round_19/{role}/training/validation.jsonl'
 for line in p.open():
  x=json.loads(line);m=x['training_epoch_io'];io.append(dict(role=role,epoch=x['epoch'],wall_s=m['wall_seconds'],read_gib=m['process_io']['read_bytes']/2**30,major_faults=m['process_io']['major_faults'],minor_faults=m['process_io']['minor_faults'],consumer_wait_s=m['batch_prefetch']['consumer_wait_s'],prefetch_fallbacks=m['batch_prefetch']['admission_fallbacks']))
fig,axs=plt.subplots(1,2,figsize=(11,4))
for role in 'AB':
 a=[r for r in io if r['role']==role];axs[0].plot([r['epoch'] for r in a],[r['wall_s'] for r in a],'-o',label=role+' epoch wall');axs[0].plot([r['epoch'] for r in a],[r['consumer_wait_s'] for r in a],'--o',label=role+' wait for features')
axs[0].set_xlabel('round19 epoch');axs[0].set_ylabel('seconds');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
probe=read(O/'io_probe_cold.json')['rows'];axs[1].bar(range(4),[r['load_s'] for r in probe],label='open/load');axs[1].bar(range(4),[r['update_s'] for r in probe],bottom=[r['load_s'] for r in probe],label='real B2 update');axs[1].set_xticks(range(4),[r['mode'] for r in probe]);axs[1].set_ylabel('seconds');axs[1].set_title('Same1.56GB input, cold file pages, ABBA');axs[1].legend();save(fig,'feature_io.png')
(O/'io_history.json').write_text(json.dumps(io,indent=2)+'\n')
# Source-vs-fitted reset matches do NOT prove exact demonstration initialization.
init=[];data=read(B/'tight_fresh_20261116_04/round_19/A/state_data/data.json')
for kind in ('fixed_fit_source_evaluation','fixed_fit_evaluation'):
 d=B/'tight_fresh_20261116_03'/kind
 for c in read(d/'plan.json')['cases']:
  identity=c['spec']['source_identity'];entry=data['entries'][identity]
  with h5py.File(entry['path']) as f,h5py.File(d/c['name']/'numeric.h5') as g:
   x=f['episodes'][entry['episode']]['transition/diagnostics_before'];y=g['episode/transition/diagnostics_before'];q=x['joint_position_rad'][0]-y['joint_position_rad'][0]
   op=object_position_in_base(x['object_pose_w'][:1],x['base_pose_w'][:1])-object_position_in_base(y['object_pose_w'][:1],y['base_pose_w'][:1]);init.append(dict(kind=kind,case=c['name'],demo_initial_joint_max_abs_rad=float(np.abs(q).max()),demo_initial_object_gap_m=float(np.linalg.norm(op))))
(O/'original_initial_comparison.json').write_text(json.dumps(init,indent=2)+'\n')
# Already computed same-state counterfactual, offline improvement and its physical effect.
cf=read(B/'tight_fresh_20261116_03/expert_counterfactual_fitted.json');fig,axs=plt.subplots(2,2,figsize=(12,7))
for ax,(role,family) in zip(axs.flat,[(r,f) for r in 'AB' for f in ('formal_050','formal_029')]):
 for row in cf['rows']:
  if row['role']==role and row['family']==family:
   label='source' if 'source_evaluation' in row['evaluation'] else 'fitted rear';ax.plot(np.arange(row['ticks'])/60,row['hand_command_error_rad'],label=label)
 ax.set_title(role+' '+family+' actual autonomous states');ax.set_xlabel('action time (simulation seconds)');ax.set_ylabel('same-state expert command gap (rad)');ax.legend();ax.grid(alpha=.2)
save(fig,'fixed_train2_closed_gap.png')
print(json.dumps(dict(io_total_read_tib=sum(r['read_gib'] for r in io)/1024,initials=init),indent=2))
