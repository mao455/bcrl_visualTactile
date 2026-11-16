"""Read-only grasp/command forensics on all seven retained A executions."""
import csv, gzip, json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.tight_fresh_closed_gap import object_position_in_base
from visuotactile.grasp_curriculum import holding_metrics

TREE = Path(__file__).resolve().parents[2]
OLD = TREE / 'outputs/visuotactile/run/tight_budget_20261116_01'
OUT = TREE / 'research_docs/evidence/tight_diagnosis_20261116/stability'

def read(p): return json.loads(Path(p).read_text())
def first(mask, start=0, length=1):
    run = 0
    for i in range(start, len(mask)):
        run = run+1 if mask[i] else 0
        if run >= length: return i-length+2  # completed controls, first in run
    return None

def rms(a): return np.sqrt(np.mean(a*a, axis=-1))
def angle(a,b):
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return 2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root=OLD/'evaluation_extended_best';rows=[];points=[]
    for c in read(root/'plan.json')['cases']:
        p=root/c['name']
        if c['role']!='A' or not read(p/'result.json')['legal']:continue
        with np.load(p/'trace.npz') as f: a={k:f[k] for k in f.files}
        with h5py.File(p/'numeric.h5') as f:
            t=f['episode/transition'];goal=t['diagnostics_after/goal_pose_w'][:]
            names=json.loads(f.attrs['metadata_json'])['joint_order']
        obj=a['object'];n=len(obj);ticks=np.arange(1,n+1);secs=ticks/60
        z=obj[:,2];rise=z-np.median(z[24:48]);force=np.linalg.norm(a['object_contact'],axis=-1)
        fingers=(force>.1).sum(-1).min(-1);total_force=force.sum(-1).min(-1)
        local=object_position_in_base(obj,a['palm']);relstep=np.r_[0,np.linalg.norm(np.diff(local,axis=0),axis=-1)]
        hold={str(w):holding_metrics(obj,a['object_contact'],window=w) for w in (12,60,120,180)}
        h=hold['60']['first_completed_tick'];h12=hold['12']['first_completed_tick']
        after=h or h12 or n
        events=dict(first_contact=first((fingers>0)&(ticks>48)),first_two_finger=first((fingers>=2)&(ticks>48)),
            first_elevated_contact=first((fingers>=2)&(rise>=.02)&(ticks>48)),
            native_lift=first(a['lifted'].astype(bool)),hold12=h12,hold60=h,
            two_finger_loss4_after_hold=first(fingers<2,after,4),
            all_contact_loss6_after_hold=first(fingers==0,after,6),
            below_height_floor_after_hold=first(rise<.02,after,3),
            subgoal=first(a['subgoals']>0),retract=first(a['retract'].astype(bool)))
        jumps={k:np.vstack((np.zeros((1,29)),np.diff(a[v],axis=0))) for k,v in (
            ('nominal','nominal'),('rear_proposal','residual_proposed_rad'),('final','target'))}
        projection=a['execution_projection_rad'];replan=(ticks-1)%12==0
        selected_event=events['all_contact_loss6_after_hold'] or events['two_finger_loss4_after_hold']
        event_detail={}
        for label,tick in events.items():
            if tick is None:continue
            i=tick-1;lo=max(0,i-12);hi=min(n,i+4)
            event_detail[label]=dict(completed_tick=tick,simulation_seconds=tick/60,action_phase=i%12,
                fingers=int(fingers[i]),height_above_reference_m=float(rise[i]),
                minimum_substep_total_force_N=float(total_force[i]),object_in_palm_m=local[i].tolist(),
                goal_orientation_error_rad=float(angle(obj[i:i+1,3:],goal[i:i+1,3:])[0]),
                hand_command_change_last12_rad=(a['target'][i,7:]-a['target'][lo,7:]).tolist(),
                largest_command_joints=[dict(joint=names[j],change_rad=float(a['target'][i,j]-a['target'][lo,j])) for j in np.argsort(np.abs(a['target'][i]-a['target'][lo]))[-5:][::-1]],
                hand_projection_rmse_rad=float(rms(projection[lo:hi,7:]).mean()),
                arm_projection_rmse_rad=float(rms(projection[lo:hi,:7]).mean()),
                nominal_jump_hand_max_rad=float(rms(jumps['nominal'][lo:hi,7:]).max()),
                final_jump_hand_max_rad=float(rms(jumps['final'][lo:hi,7:]).max()),
                local_translation_change_last12_m=float(np.linalg.norm(local[i]-local[lo])))
        row=dict(case=c['name'],cohort=c.get('diagnostic_cohort','fixed_development'),events=events,
            hold=hold,event_details=event_detail,end_height_m=float(rise[-1]),end_fingers=int(fingers[-1]),
            no_task_release_observed=not bool(a['retract'].any()),
            command_mapping_max_abs_error_rad=float(np.abs(a['actuator_targets']-a['target'][:,None]).max()),
            boundary_statistics={k:{part:dict(boundary_mean_rad=float(rms(v[replan,sl]).mean()),
                between_mean_rad=float(rms(v[~replan,sl]).mean())) for part,sl in [('arm',slice(0,7)),('hand',slice(7,None))]} for k,v in jumps.items()},
            final_contact_free_ticks=int(np.argmax((fingers[::-1]>0))) if np.any(fingers>0) else n)
        rows.append(row)
        fig,ax=plt.subplots(6,1,figsize=(13,14),sharex=True)
        ax[0].plot(secs,rise*100,label='Object height above settled reference');ax[0].axhline(2,ls='--',color='k',lw=.7)
        ax[0].set_ylabel('Height (cm)')
        ax[1].step(secs,fingers,label='Fingers >0.1 N in both physics samples');ax[1].plot(secs,total_force,label='Minimum total force / N',alpha=.5);ax[1].set_ylim(0,max(5,min(15,np.quantile(total_force,.98))));ax[1].set_ylabel('Contact / force')
        for j,axis in enumerate('xyz'):ax[2].plot(secs,local[:,j]*100,label='Object in palm '+axis)
        ax[2].set_ylabel('Position (cm)')
        ax[3].plot(secs,angle(obj[:,3:],goal[:,3:])*180/np.pi,label='Object–current goal orientation')
        ax[3].plot(secs,angle(obj[:,3:],np.broadcast_to(obj[48:49,3:],(n,4)))*180/np.pi,label='Object rotation since tick49');ax[3].set_ylabel('Angle (deg)')
        for k,v in jumps.items():ax[4].plot(secs,rms(v[:,7:]),label=k+' hand change')
        ax[4].plot(secs,rms(projection[:,7:]),label='Hand execution projection',alpha=.6);ax[4].set_ylabel('Command RMS (rad)')
        for j in np.argsort(np.ptp(a['target'][:,7:],axis=0))[-4:]:
            ax[5].plot(secs,a['target'][:,7+j],label=names[7+j])
        ax[5].set_ylabel('Finger targets (rad)');ax[5].set_xlabel('Simulation seconds (completed controls / 60); replans every 12 controls')
        for axes in ax:
            for s in secs[replan]:axes.axvline(s,color='grey',alpha=.1,lw=.5)
            for key,color in [('hold60','green'),('all_contact_loss6_after_hold','red'),('two_finger_loss4_after_hold','orange')]:
                if events[key]:axes.axvline(events[key]/60,color=color,ls='--',alpha=.7,label=key if axes is ax[0] else None)
            axes.grid(alpha=.2);axes.legend(fontsize=7,loc='best')
        fig.suptitle(c['name']+'\nNo expert control; hold-window failure alone is not a dropped object',fontsize=10);fig.tight_layout()
        fig.savefig(OUT/(c['name']+'.png'),dpi=110);plt.close(fig)
        for i in range(n):
            points.append(dict(case=c['name'],completed_tick=i+1,simulation_s=(i+1)/60,action_phase=i%12,
                height_m=float(rise[i]),fingers=int(fingers[i]),total_force_N=float(total_force[i]),
                palm_x_m=float(local[i,0]),palm_y_m=float(local[i,1]),palm_z_m=float(local[i,2]),
                relative_translation_step_m=float(relstep[i]),goal_orientation_error_rad=float(angle(obj[i:i+1,3:],goal[i:i+1,3:])[0]),
                hand_nominal_jump_rad=float(rms(jumps['nominal'][i,7:])),hand_rear_jump_rad=float(rms(jumps['rear_proposal'][i,7:])),
                hand_final_jump_rad=float(rms(jumps['final'][i,7:])),hand_projection_rad=float(rms(projection[i,7:])),
                arm_projection_rad=float(rms(projection[i,:7])),native_lift=bool(a['lifted'][i]),subgoals=int(a['subgoals'][i])))
    (OUT/'a_existing.json').write_text(json.dumps(dict(rows=rows,scope='All seven legal A extended-best executions; descriptive event thresholds, no new success criteria; completed ticks start at1, action phase uses tick-1.'),indent=2)+'\n')
    with gzip.open(OUT/'a_existing_ticks.csv.gz','wt') as f:
        w=csv.DictWriter(f,list(points[0]));w.writeheader();w.writerows(points)
    print(json.dumps([dict(case=r['case'],events=r['events'],end_height_m=r['end_height_m'],end_fingers=r['end_fingers']) for r in rows],indent=2))

if __name__=='__main__': main()
