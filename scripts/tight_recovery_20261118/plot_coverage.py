"""Static actual label timing and support figures, not a causal classifier plot."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tight_recovery_20261118.run import OUT,read

def main():
 old=[r for r in read(OUT/'old_regression_audit.json')['rows'] if r['wave']=='collection' and r['first_legal_tick'] is not None]
 new=[];support=[]
 for name in ['pilot_admission_recovery02','collection_completed']:
  new += [r for r in read(OUT/(name+'_coverage.json'))['rows'] if r['accepted']]
  support += read(OUT/(name+'_support.json'))['rows']
 fig,ax=plt.subplots(1,3,figsize=(13,3.5));bins=np.arange(36,325,24)
 ax[0].hist([r['first_legal_tick'] for r in old],bins=bins,alpha=.6,label='Previous29');ax[0].hist([r['first_legal_tick'] for r in new],bins=bins,alpha=.6,label='New25');ax[0].set_xlabel('First lawful expert action (control tick)');ax[0].set_ylabel('Actual corrections');ax[0].legend()
 names=['Palm position','Relative rotation','Velocity','Issued history'];keys=['palm_local_object_m','palm_local_rotation_rad','object_speed_m_s','issued_history12_rms_rad']
 values=[]
 for k in keys:
  rr=[r for r in support if r['metric']==k and r['region']=='first24_legal'];values.append(100*sum(r['far_ticks'] for r in rr)/sum(r['ticks'] for r in rr))
 ax[1].barh(names,values);ax[1].set_xlabel('First24 ticks beyond old105 phase p95 (%)');ax[1].set_xlim(0,100)
 summary=read(OUT/'new_data_summary.json');ph=summary['first24_phase_ticks'];ax[2].barh(['Contact lost','Contact, not held','Held near','Other'],[ph['contact_lost'],ph['contact_not_held'],ph['held_near_goal60'],600-ph['contact_lost']-ph['contact_not_held']-ph['held_near_goal60']]);ax[2].set_xlabel('Lawful ticks in first24 windows (total600)')
 fig.suptitle('New recovery supervision: later error states; support distance is not causal value',fontsize=11);fig.tight_layout();fig.savefig(OUT/'coverage.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
