"""Separate deterministic recentering from the learned rear displacement."""
import gzip,json
import h5py,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from visuotactile.execution_learning import ideal_execution_bounds
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION,project_processed_position_targets
from scripts.tight_stability_20261116.analyze_existing import OLD,OUT,read,rms

def main():
    rows=[];points=[]
    for case in read(OUT/'a_existing.json')['rows']:
        d=OLD/'evaluation_extended_best'/case['case']
        with np.load(d/'trace.npz') as f:a={k:f[k] for k in f.files}
        with np.load(d/'plans.npz') as f:plans=f['plans']
        with h5py.File(d/'numeric.h5') as f:metadata=json.loads(f.attrs['metadata_json'])
        limits=torch.tensor(metadata['joint_limits_rad']).reshape(29,2)
        previous=torch.from_numpy(a['execution_previous_target_rad']);base=torch.from_numpy(a['nominal'])
        bounds=ideal_execution_bounds(previous,limits,REACHABLE_POSITION_EXECUTION)
        anchor=base.clamp(min=bounds[...,0],max=bounds[...,1])
        pre=base+torch.from_numpy(a['residual_proposed_rad'])
        learned=pre-anchor;recenter=anchor-base
        reproduced=project_processed_position_targets(pre.clamp(min=limits[:,0],max=limits[:,1]),previous,limits,REACHABLE_POSITION_EXECUTION)
        error=float((reproduced-torch.from_numpy(a['target'])).abs().max());assert error<2e-6
        qobj=Rotation.from_quat(a['object'][:,[4,5,6,3]])
        qpalm=Rotation.from_quat(a['palm'][:,[4,5,6,3]])
        relative=qpalm.inv()*qobj;reference=(case['events']['hold12'] or 49)-1
        rel_angle=(relative[reference].inv()*relative).magnitude()
        perstep=np.r_[0,(relative[:-1].inv()*relative[1:]).magnitude()]
        replan=[]
        for k in range(1,len(plans)):
            i=k*12;shock=plans[k,0]-plans[k-1,12]
            replan.append(dict(completed_tick=i+1,hand_new_plan_vs_old_next_rmse_rad=float(rms(shock[7:])),
                hand_actual_step_rmse_rad=float(rms(np.diff(a['target'][i-1:i+1,7:],axis=0)[0])),
                learned_hand_step_rmse_rad=float(rms((learned[i]-learned[i-1]).numpy()[7:]))))
        events={}
        for label,tick in case['events'].items():
            if tick is None:continue
            i=tick-1;lo=max(0,i-24);sl=slice(lo,i+1)
            events[label]=dict(completed_tick=tick,hand_learned_correction_rmse_rad=float(learned[sl,7:].square().mean().sqrt()),
                hand_deterministic_recenter_rmse_rad=float(recenter[sl,7:].square().mean().sqrt()),
                hand_final_increment_rmse_rad=float(torch.from_numpy(a['target'][sl,7:]-a['execution_previous_target_rad'][sl,7:]).square().mean().sqrt()),
                relative_rotation_last12_deg=float((relative[max(0,i-12)].inv()*relative[i]).magnitude()*180/np.pi),
                nearest_preceding_replan=next((r for r in reversed(replan) if r['completed_tick']<=tick),None))
        rows.append(dict(case=case['case'],reconstruction_max_abs_rad=error,events=events,replans=replan))
        for i in range(len(base)):
            points.append(dict(case=case['case'],tick=i+1,palm_relative_rotation_since_firsthold12_rad=float(rel_angle[i]),
                palm_relative_rotation_step_rad=float(perstep[i]),learned_hand_rms_rad=float(learned[i,7:].square().mean().sqrt()),
                deterministic_recenter_hand_rms_rad=float(recenter[i,7:].square().mean().sqrt())))
        if case['events']['hold60']:
            fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True);t=np.arange(1,len(base)+1)/60
            ax[0].plot(t,rel_angle*180/np.pi,label='Relative rotation since first12-tick hold');ax[0].set_ylabel('Object in palm (deg)')
            ax[1].plot(t,rms(recenter.numpy()[:,7:]),label='Deterministic recentering');ax[1].plot(t,rms(learned.numpy()[:,7:]),label='Learned rear displacement');ax[1].set_ylabel('Hand RMS (rad)')
            ax[2].plot([r['completed_tick']/60 for r in replan],[r['hand_new_plan_vs_old_next_rmse_rad'] for r in replan],label='New DP head vs previous plan next',marker='.')
            ax[2].plot([r['completed_tick']/60 for r in replan],[r['hand_actual_step_rmse_rad'] for r in replan],label='Executed increment at replan',marker='.');ax[2].set_ylabel('Hand RMS (rad)')
            for axy in ax:axy.grid(alpha=.2);axy.legend(fontsize=8)
            ax[-1].set_xlabel('Simulation seconds');fig.suptitle(case['case']+'\nStored total residual includes deterministic recentering; it is not just the network output',fontsize=9);fig.tight_layout();fig.savefig(OUT/(case['case']+'_decomposition.png'),dpi=120);plt.close(fig)
    (OUT/'a_command_decomposition.json').write_text(json.dumps(dict(rows=rows,scope='Current nominal-centered linear contract: anchor=clip(nominal,reachable_box); learned=unclipped_proposal-anchor. No control modification; replay reconstructs recorded targets. Quaternion wxyz converted explicitly.'),indent=2)+'\n')
    with gzip.open(OUT/'a_command_decomposition_ticks.jsonl.gz','wt') as f:
        for r in points:f.write(json.dumps(r)+'\n')
if __name__=='__main__':main()
