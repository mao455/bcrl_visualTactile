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
summary={}
for role in 'AB':
 dprows=[json.loads(x) for x in (R/role/'training/validation.jsonl').read_text().splitlines()]
 initial=read(R/role/'baseline_epoch8_diagnostics.json')
 for row in dprows:
  if row['epoch']==8:row['cohort_diagnostics']=initial['diagnostics']
 dump(f'dp_{role}_epochs.json',dprows);all_fig(role,'dp',dprows)
 rearpath=R/role/'backend'/role/'training'
 if (rearpath/'finished.json').exists():
  rearrows=read(rearpath/'validation.json');dump(f'rear_{role}_epochs.json',rearrows);all_fig(role,'rear',rearrows)
  summary[role]=dict(dp=read(R/role/'training/finished.json'),rear=read(rearpath/'finished.json'),rear_selected=read(rearpath/'selected.json'))
 for kind,path,name in [('dp',R/role/'training/learning.jsonl','learning'),('rear',rearpath/'loss.jsonl','loss')]:
  if path.exists():
   with gzip.open(O/f'{kind}_{role}_{name}.jsonl.gz','wt') as f:
    for line in path.read_text().splitlines():
     value=json.loads(line)
     if kind=='dp':
      for visit in value.get('visits',[]):visit.pop('frozen_visual_io',None)
     f.write(json.dumps(value,separators=(',',':'))+'\n')
dump('summary.json',summary)
dump('registration.json',read(R/'prepared.json'))
dump('split_audit.json',read(R/'A/split_audit.json'))
print('Wrote measured curves',str(O))
