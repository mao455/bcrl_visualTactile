"""Static research figures from measured receipts; no simulator or model imports."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('research_docs/evidence/tight_long_20261117/balanced76')
def read(p):return json.loads(p.read_text())
def main():
 coverage=read(OUT/'coverage.json')['groups'];planned=read(OUT/'planned_exposure.json')['counts']
 fig,ax=plt.subplots(1,2,figsize=(12,4.7))
 sources=['original','A','B'];left=np.zeros(3)
 names=['Pre-contact','Contact, not held','Held, far','Held, rotating','Held, near goal','After first subgoal','Contact lost']
 for index,(phase,label) in enumerate(zip(coverage['original']['phases'],names)):
  values=np.array([coverage[s]['phases'][phase] for s in sources]);ax[0].barh(sources,values,left=left,label=label,color=plt.cm.tab10(index));left+=values
 ax[0].set(xlabel='Unique legal expert ticks',title='76 trajectories: geometry phase proxies')
 ax[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.15),ncol=2)
 x=np.arange(2);width=.35
 for offset,(version,label) in enumerate([('historical_v2','Old sampler'),('candidate_v3','Onset sampler')]):
  row=planned[version];v=[row['original_early_slots'],row['correction_onset_slots']]
  ax[1].bar(x+(offset-.5)*width,v,width,label=label)
 ax[1].set(xticks=x,xticklabels=['Original before tick150','Correction first24'],ylabel='Repeated supervised target slots',title='40-epoch CPU sampling preview; no training')
 ax[1].legend();fig.subplots_adjust(bottom=.3,wspace=.3);fig.savefig(OUT/'coverage_sampling.png',dpi=150);plt.close(fig)
 if not (OUT/'completed.json').exists():return
 d=read(OUT/'completed.json');fig,ax=plt.subplots(1,3,figsize=(12,4))
 for k,(metric,title) in enumerate([('success','Full insertion'),('hold120','Sustained holding120 (proxy)'),('subgoal','First native subgoal')]):
  for j,version in enumerate(['baseline','candidate']):
   vals=[d['counts'][r+('_baseline' if version=='baseline' else '')][metric] for r in 'AB']
   bars=ax[k].bar(x+(j-.5)*width,vals,width,label='Selected72' if j==0 else 'Fixed76 onset')
   ax[k].bar_label(bars,labels=[f'{v}/8' for v in vals])
  ax[k].set(title=title,xticks=x,xticklabels=list('AB'),ylim=(0,9),ylabel='Development initial states')
 ax[0].legend();fig.suptitle('Same-wave observable initial-state pairs; descriptive, not significance')
 fig.tight_layout();fig.savefig(OUT/'paired_outcomes.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(10,6))
 for i,role in enumerate('AB'):
  for j,field in enumerate(['arm_rmse_rad','hand_rmse_rad']):
   for version in ['parent72','candidate76']:
    p=read(OUT/f'offline_{role}_{version}.json')['phases'];keys=['original_early','correction_onset24','near_alignment','after_subgoal']
    ax[i,j].plot(range(len(keys)),[p[k][field] for k in keys],marker='o',label=version)
   ax[i,j].set(xticks=range(4),xticklabels=['Original early','Onset24','Near','After subgoal'],title=f'{role}: '+field,ylabel='Radians');ax[i,j].grid(alpha=.25)
 ax[0,0].legend();fig.suptitle('Teacher-forced legal76, DP commit12; separate A/B label spaces')
 fig.tight_layout();fig.savefig(OUT/'offline_phase_errors.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
