"""CPU-only same-state expert suggestions around the observed A contact losses."""
import json,time
import h5py,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tight_stability_20261116.replay import expert_model,diagnostic_expert
from scripts.tight_stability_20261116.analyze_existing import OLD,OUT,read

def main():
    torch.set_num_threads(2);model=expert_model();result=[];start=time.time()
    for row in read(OUT/'a_existing.json')['rows']:
        d=OLD/'evaluation_extended_best'/row['case'];mapped,q=diagnostic_expert(model,d/'numeric.h5');target=mapped.numpy()
        with np.load(d/'trace.npz') as f:a={k:f[k] for k in f.files}
        with h5py.File(d/'numeric.h5') as f:names=json.loads(f.attrs['metadata_json'])['joint_order']
        event=row['events']['all_contact_loss6_after_hold'] or row['events']['two_finger_loss4_after_hold']
        rr=dict(case=row['case'],hand_error_rad=q['hand_command_error_rad'],arm_error_rad=q['arm_command_error_rad'],event=event)
        if event:
            i=event-1;lo=max(48,i-24);sl=slice(lo,i+1);error=a['target'][sl]-target[sl]
            rr['prior24_joint_bias_rad']={names[j]:float(error[:,j].mean()) for j in range(29)}
            rr['prior24_hand_rmse_rad']=float(np.sqrt(np.mean(error[:,7:]**2)))
            rr['prior24_tracking_hand_rmse_rad']=float(np.sqrt(np.mean((a['target'][sl,7:]-a['joint'][sl,7:])**2)))
            rr['prior24_projection_hand_rmse_rad']=float(np.sqrt(np.mean(a['execution_projection_rad'][sl,7:]**2)))
            js=np.argsort(np.mean(error[:,7:]**2,axis=0))[-3:]+7
            fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True);left=max(0,i-60);right=min(len(target),i+25);ss=np.arange(left+1,right+1)/60
            axes[0].plot(ss,a['object'][left:right,2]-np.median(a['object'][24:48,2]),label='Height above settled / m')
            axes[0].step(ss,(np.linalg.norm(a['object_contact'][left:right],axis=-1)>.1).sum(-1).min(-1)/10,label='Finger count /10')
            for ax,j in zip(axes[1:],js):
                for key,label in [('nominal','DP reached-q prediction'),('execution_preprojection_target_rad','Rear pre-projection'),('target','Actually executed')]:ax.plot(ss,a[key][left:right,j],label=label)
                ax.plot(ss,target[left:right,j],label='Same-state expert suggestion',ls='--');ax.set_ylabel(names[j]+' (rad)',fontsize=8)
            for ax in axes:ax.axvline(event/60,color='red',ls='--');ax.legend(fontsize=7);ax.grid(alpha=.2)
            axes[-1].set_xlabel('Simulation seconds; red = sustained contact loss onset (not necessarily drop)');fig.suptitle(row['case'],fontsize=9);fig.tight_layout();fig.savefig(OUT/(row['case']+'_command_zoom.png'),dpi=120);plt.close(fig)
        result.append(rr)
    (OUT/'a_expert_suggestions.json').write_text(json.dumps(dict(rows=result,seconds=time.time()-start,scope='CPU queries of unchanged privileged expert on real student states; suggestions are not executed expert labels. A nominal predicts reached q, not command. No causal claim from joint error ranking.'),indent=2)+'\n')

if __name__=='__main__':main()
