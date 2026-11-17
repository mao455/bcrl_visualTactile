"""Static physical-unit diagnostic plots from completed evidence."""
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tight_cycle_20261118.cycle import OUT,read
b=OUT/'capability';rows=read(b/'support.json');stages=['grasp_without_contact','contact_not_held','held_far','held_rotating','held_near_goal60','after_first_native_subgoal','contact_lost'];keys=['object_palm_m','object_rotation_rad','joint_rms_rad','object_speed_m_s','object_spin_rad_s','issued_history12_rms_rad'];m=[]
for k in keys:
 values=[]
 for stage in stages:
  rr=[r for r in rows if r['metric']==k and r['phase']==stage];values.append(sum(r['ticks']*r['far_fraction'] for r in rr)/sum(r['ticks'] for r in rr))
 m.append(values)
fig,ax=plt.subplots(figsize=(11,4.4));im=ax.imshow(m,vmin=0,vmax=1,cmap='YlOrRd');ax.set_xticks(range(7),['precontact','contact','held far','held rotating','held near','after subgoal1','lost contact'],rotation=20,ha='right');ax.set_yticks(range(6),['object/palm position (m)','object orientation (rad)','joint RMS (rad)','object linear velocity (m/s)','object angular velocity (rad/s)','past12 mean command RMS (rad)'])
for i in range(6):
 for j in range(7):ax.text(j,i,f'{m[i][j]:.0%}',ha='center',va='center',fontsize=9)
fig.colorbar(im,ax=ax,label='Fraction above phase-specific training leave-family-out P95');ax.set_title('Baseline72 autonomous support distance on32 capability executions\nIndependent physical metrics; phase labels are not a strict funnel');fig.tight_layout();fig.savefig(b/'phase_support.png',dpi=150);plt.close(fig)
ini=read(b/'initial_preaction.json');ks=['object_base_m','object_palm_m','object_rotation_rad','joint_rms_rad','joint_speed_rms_rad_s'];fig,axes=plt.subplots(1,5,figsize=(13,3.5))
for ax,k in zip(axes,ks):
 rr=[r for r in ini if r['metric']==k];scale=1000 if k.endswith('_m') else 1
 vals=[[r['nearest']*scale for r in rr if r['cohort']==co] for co in ['training','new_development']]
 ax.boxplot(vals,tick_labels=['train','new dev']);ax.axhline(rr[0]['train_leave_family_p95']*scale,color='r',ls='--',label='train LFO P95');ax.set_title(k.replace('_','\n'),fontsize=9);ax.set_ylabel('mm' if scale==1000 else 'rad/s' if k.endswith('_s') else 'rad');ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Actual pre-action nearest original-expert family; metrics kept in physical units');fig.tight_layout();fig.savefig(b/'initial_distances.png',dpi=140);plt.close(fig)
