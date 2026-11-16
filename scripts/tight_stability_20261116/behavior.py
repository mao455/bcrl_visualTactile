"""New preregistered development families, common480 window and separate full tasks."""
import json,csv
from pathlib import Path
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visuotactile.grasp_curriculum import holding_metrics
from scripts.tight_stability_20261116.analyze_existing import OUT,read,first
from scripts.tight_stability_20261116.run import ROOT
from tools.tight_fresh_closed_gap import object_position_in_base

def main():
    assert read(ROOT/'complete.json')['complete'];rows=[];summary=[];initial=[]
    keys=['contact','two_finger','native_lift','contact_height','hold60','hold120','hold180','final60','alignment','subgoal']
    for group in ['candidates','fallbacks']:
        plan=read(ROOT/group/'plan.json')
        for c in plan['cases']:
            d=ROOT/group/c['name'];r=read(d/'result.json');row=dict(group=group,case=c['name'],role=c['role'],family=c['spec']['family'],legal=r['legal'],training_family=False,full=c['grasp_evaluation_ticks'] is None)
            if r['legal']:
                with np.load(d/'trace.npz') as f:a={k:f[k] for k in f.files}
                n=min(480,len(a['object']));obj=a['object'][:n];force=a['object_contact'][:n];fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
                h={str(w):holding_metrics(obj,force,window=w) for w in [60,120,180]};lift=bool(a['lifted'][:n].any());idx=np.arange(n)
                row.update(actual_total_controls=len(a['object']),common_window_controls=n,
                    contact=bool(((fingers>0)&(idx>=48)).any()),two_finger=bool(((fingers>=2)&(idx>=48)).any()),native_lift=lift,
                    contact_height=bool(((fingers>=2)&(idx>=48)&(obj[:,2]>=np.median(obj[24:48,2])+.02)).any()),
                    hold60=h['60']['sustained'],hold120=h['120']['sustained'],hold180=h['180']['sustained'],final60=h['60']['held_at_final_window'],
                    alignment=bool(a['alignment'][:n].any()),subgoal=bool(a['subgoals'][:n].max()>0),hold_details=h,
                    first_contact=first((fingers>0)&(idx>=48)),first_two=first((fingers>=2)&(idx>=48)),
                    full_insertion_success=bool(r['success']) if row['full'] else None,
                    observed_insertion_success=bool(r['success']),
                    terminal_before480=len(a['object'])<480,at480_observed=len(a['object'])>=480,
                    native_termination_reasons=r.get('reasons'),grasp_censored=r.get('grasp_stage_censored'))
                if group=='fallbacks' and read(ROOT/'candidates'/c['name']/'result.json')['legal']:
                    with h5py.File(d/'numeric.h5') as f,h5py.File(ROOT/'candidates'/c['name']/'numeric.h5') as g:
                        x=f['episode/transition/diagnostics_before'];y=g['episode/transition/diagnostics_before']
                        gap=object_position_in_base(x['object_pose_w'][:1],x['base_pose_w'][:1])-object_position_in_base(y['object_pose_w'][:1],y['base_pose_w'][:1])
                        initial.append(dict(case=c['name'],joint_max_abs_rad=float(np.abs(x['joint_position_rad'][0]-y['joint_position_rad'][0]).max()),object_base_gap_m=float(np.linalg.norm(gap)),hidden_state_exact=False))
            rows.append(row)
        for role in 'AB':
            allrows=[r for r in rows if r['group']==group and r['role']==role];legal=[r for r in allrows if r['legal']];full=[r for r in legal if r['full']]
            at480=[r for r in legal if r['at480_observed']]
            summary.append(dict(group=group,role=role,attempts=len(allrows),legal=len(legal),**{k:sum(r[k] for r in legal) for k in keys},
                at480_n=len(at480),at480_final60=sum(r['final60'] for r in at480),
                success_before480=sum(r['observed_insertion_success'] and r['terminal_before480'] for r in legal),
                full_task_n=len(full),full_success=sum(r['full_insertion_success'] for r in full)))
    paired=[]
    for role in 'AB':
        maps={g:{r['case']:r for r in rows if r['group']==g and r['role']==role and r['legal']} for g in ['candidates','fallbacks']}
        common=set(maps['candidates'])&set(maps['fallbacks'])
        paired.append(dict(role=role,common_legal_n=len(common),cases=sorted(common),gained={k:sorted(i for i in common if maps['candidates'][i][k] and not maps['fallbacks'][i][k]) for k in keys},lost={k:sorted(i for i in common if maps['fallbacks'][i][k] and not maps['candidates'][i][k]) for k in keys}))
    result=dict(complete=True,rows=rows,summary=summary,paired=paired,initial_comparison=initial,
        scope='All preregistered new development families, not train or final test. Up to first480 controls=8sim seconds; native terminal episodes end sooner and are explicit. final60 in the raw table means last recorded window, not uniform tick480; at480_* separately excludes early native terminations. Successful insertion before480 is NOT a grasp failure. Each group/role seed28500100 continues to native termination; only those form full-task denominator. No replacement of illegal starts. Native lift distinct from contact-height. No exact hidden-state replay claim.')
    (OUT/'new_behavior.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(14,5));matrix=np.array([[r[k]/r['legal'] if r['legal'] else 0 for k in keys] for r in summary]);ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
    for i,r in enumerate(summary):
        for j,k in enumerate(keys):ax.text(j,i,f"{r[k]}/{r['legal']}",ha='center',va='center')
    ax.set_yticks(range(4),[r['group']+' '+r['role'] for r in summary]);ax.set_xticks(range(len(keys)),keys,rotation=30,ha='right');ax.set_title('New independent development families: full DP+Transformer, first480 controls /8s');fig.tight_layout();fig.savefig(OUT/'new_behavior.png',dpi=150)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
