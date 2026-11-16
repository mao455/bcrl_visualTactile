"""Render only measured epoch/cohort points from the bounded experiment."""
import csv,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('outputs/visuotactile/run/tight_budget_20261116_01').resolve()
O=Path('research_docs/evidence/tight_diagnosis_20261116/budget_extension');O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):(O/p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def plot_end(fig,name):fig.tight_layout();fig.savefig(O/name,dpi=150);plt.close(fig)
def all_fig(role,kind,rows):
 fig,ax=plt.subplots(2,3,figsize=(16,8))
 epochs=[x['epoch'] for x in rows]
 if kind=='dp':
  ax[0,0].plot(epochs,[x['loss'] for x in rows],'-o',ms=3,label='actual stochastic train objective')
  for name in ['original','historical','latest','validation']:
   d=[x for x in rows if 'cohort_diagnostics' in x]
   ax[0,1].plot([x['epoch'] for x in d],[x['cohort_diagnostics'][name]['fixed_noise_objective'] for x in d],'-o',ms=3,label=name+' fixed-noise')
  ax[0,1].set_title('Fixed-noise objective (geometry fixed2 diagnostic)')
 else:
  loss=[json.loads(x) for x in (R/role/'backend'/role/'training/loss.jsonl').read_text().splitlines()]
  ax[0,0].plot(epochs,[np.mean([v['loss'] for v in loss if v['epoch']==e]) for e in epochs],'-o',ms=3,label='train FY / fixed128 window')
  ax[0,1].plot(epochs,[x['validation']['all']['normalized_equal_mse'] for x in rows],'-o',ms=3,label='validation executed-command normalized MSE')
  ax[0,1].set_title('Validation command loss (not FY training objective)')
 for name in ['original','historical','latest','validation']:
  d=[x for x in rows if 'cohort_diagnostics' in x];e=[x['epoch'] for x in d]
  for axis,phase,metric in [(ax[0,2],'all','arm_rmse_rad'),(ax[1,0],'all','hand_rmse_rad'),(ax[1,1],'pre_contact','hand_rmse_rad'),(ax[1,2],'grasp_contact','hand_rmse_rad')]:
   axis.plot(e,[x['cohort_diagnostics'][name]['phases'][phase][metric] for x in d],'-o',ms=3,label=name)
 for a,title in zip(ax.flat,['Actual training objective','', 'Arm RMSE all legal','Hand RMSE all legal','Hand RMSE before first touch','Hand RMSE during two-finger contact']):
  if title:a.set_title(title)
  a.set_xlabel('completed epoch');a.axvline(8,color='gray',ls='--',label='original budget8');a.grid(alpha=.2);a.legend(fontsize=7)
 for row in rows:
  if row.get('stopping',{}).get('event')=='reduce_lr':
   for a in ax.flat:a.axvline(row['epoch'],color='red',ls=':',label='LR reduced after epoch')
 fig.suptitle(role+' '+kind+': fixed data/sampling; train cohorts are fitting diagnostics, not held-out validation')
 plot_end(fig,f'{kind}_{role}.png')
 # The selected score is on the same scale on both sides of the old boundary.
 fig,axes=plt.subplots(1,2,figsize=(13,4))
 train=[x['train'] if kind=='dp' else x['train']['all']['normalized_equal_mse'] for x in rows]
 val=[x['validation'] if kind=='dp' else x['validation']['all']['normalized_equal_mse'] for x in rows]
 best=int(np.argmin(val))
 for ax,xs,xlabel in [(axes[0],epochs,'epoch'),(axes[1],[x.get('updates',x.get('update')) for x in rows],'optimizer update')]:
  ax.plot(xs,train,'o-',ms=3,label='fixed train2 fit diagnostic')
  ax.plot(xs,val,'o-',ms=3,label='independent validation6')
  ax.scatter([xs[best]],[val[best]],marker='*',s=130,color='red',label=f'selected epoch {epochs[best]}')
  ax.set_xlabel(xlabel);ax.set_ylabel('equal Arm/Hand normalized MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle(role+' '+kind+': training examples are NOT independent validation')
 plot_end(fig,f'{kind}_{role}_selection.png')
 if kind=='dp':
  d=[x for x in rows if 'cohort_diagnostics' in x]
  names=list(d[-1]['cohort_diagnostics']['validation']['loss_components'])
  fig,axes=plt.subplots(1,len(names),figsize=(max(7,len(names)*3.4),4),squeeze=False)
  for ax,name in zip(axes[0],names):
   for cohort in ['original','historical','latest','validation']:
    ax.plot([x['epoch'] for x in d],[x['cohort_diagnostics'][cohort]['loss_components'][name] for x in d],'o-',ms=3,label=cohort)
   ax.set_title(name);ax.set_xlabel('epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
  fig.suptitle(role+' DP fixed-noise loss components; geometry diagnostic uses fixed2, actual training rotates half')
  plot_end(fig,f'dp_{role}_components.png')
summary={}
phase_rows=[]
def phase_table(role,kind,rows):
 for row in rows:
  for cohort,metrics in row.get('cohort_diagnostics',{}).items():
   for phase,values in metrics['phases'].items():
    phase_rows.append(dict(role=role,network=kind,epoch=row['epoch'],updates=row.get('updates',row.get('update')),cohort=cohort,phase=phase,**values))
for role in 'AB':
 dprows=[json.loads(x) for x in (R/role/'training/validation.jsonl').read_text().splitlines()]
 initial=read(R/role/'baseline_epoch8_diagnostics.json')
 for row in dprows:
  if row['epoch']==8:row['cohort_diagnostics']=initial['diagnostics']
 dump(f'dp_{role}_epochs.json',dprows);all_fig(role,'dp',dprows);phase_table(role,'dp',dprows)
 rearpath=R/role/'backend'/role/'training'
 if (rearpath/'finished.json').exists():
  rearrows=read(rearpath/'validation.json');dump(f'rear_{role}_epochs.json',rearrows);all_fig(role,'rear',rearrows);phase_table(role,'rear',rearrows)
  summary[role]=dict(dp=read(R/role/'training/finished.json'),rear=read(rearpath/'finished.json'),rear_selected=read(rearpath/'selected.json'))
 for kind,path,name in [('dp',R/role/'training/learning.jsonl','learning'),('rear',rearpath/'loss.jsonl','loss')]:
  if path.exists():
   with gzip.open(O/f'{kind}_{role}_{name}.jsonl.gz','wt') as f:
    for line in path.open():
     value=json.loads(line)
     if kind=='dp':
      value['visits']=[{k:v for k,v in visit.items() if k in ('episode','anchors','target_ticks','visual_frames','parts','loss','loss_weight','episode_index','episode_visit')} for visit in value.get('visits',[])]
     f.write(json.dumps(value,separators=(',',':'))+'\n')
dump('summary.json',summary)
dump('registration.json',read(R/'prepared.json'))
dump('split_audit.json',read(R/'A/split_audit.json'))
dump('split_identity_evidence.json',read(R/'split_identity_evidence.json'))
with (O/'phase_errors.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(phase_rows[0]),lineterminator='\n');w.writeheader();w.writerows(phase_rows)
print('Wrote measured curves',str(O))
if (R/'endpoint_comparison.json').exists():
 endpoint=read(R/'endpoint_comparison.json');dump('endpoint_comparison.json',endpoint)
 for phase in ['all','before150','pre_contact','grasp_contact']:
  fig,axes=plt.subplots(2,2,figsize=(14,8))
  for j,kind in enumerate(['dp','rear']):
   for k,role in enumerate('AB'):
    selected=[x for x in endpoint['rows'] if x['network']==kind and x['role']==role]
    names=['original','historical','latest','validation'];ax=axes[j,k];width=.8/len(selected)
    for n,row in enumerate(selected):
     vals=[row['diagnostics'][name]['phases'][phase]['hand_rmse_rad'] for name in names]
     ax.bar(np.arange(4)-.4+(n+.5)*width,vals,width,label=row['label']+f" ep{row['epoch']}")
    ax.set_xticks(range(4),names);ax.set_title(role+' '+kind+' '+phase);ax.set_ylabel('hand RMSE (rad)');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
  fig.suptitle('Identical recorded histories; A DP = reached q[t+1], B DP/rear = expert commands; each rear uses its own DP cache')
  plot_end(fig,f'endpoint_{phase}.png')
# Every point below is one actual optimizer transaction. No smoothing/interpolation.
fig,axes=plt.subplots(2,2,figsize=(14,8))
for j,kind in enumerate(['dp','rear']):
 for k,role in enumerate('AB'):
  name='learning' if kind=='dp' else 'loss';p=O/f'{kind}_{role}_{name}.jsonl.gz'
  if not p.exists():continue
  with gzip.open(p,'rt') as f:vals=[json.loads(line) for line in f]
  xs=[v.get('updates',v.get('update')) for v in vals];ys=[v['loss'] for v in vals]
  axes[j,k].plot(xs,ys,'.',ms=1,alpha=.35);axes[j,k].set_title(role+' '+kind+' actual optimizer updates');axes[j,k].set_xlabel('optimizer update');axes[j,k].set_ylabel('actual training objective');axes[j,k].grid(alpha=.2)
plot_end(fig,'optimizer_losses.png')
