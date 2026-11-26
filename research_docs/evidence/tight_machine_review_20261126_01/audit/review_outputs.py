"""Read-only diagnostic tables. Never selects from terminal64."""
from pathlib import Path
import json,csv,collections,time
import numpy as np
import h5py
from audit_existing import replay

R=Path.cwd(); A=R/'outputs/diagnostics/tight_machine_review_20261126_01'
ROOT=R/'outputs/visuotactile/run/tight_machine_review_20261126_01'
def read(p): return json.loads(Path(p).read_text())
def write(name,value): (A/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (A/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def historical():
    census=read(A/'train_development_stage_census.json')['rows']
    exact=[]
    for r in census:
        if r['cohort']!='historical_train' or r['student_prefix_ticks'] is None:continue
        with h5py.File(r['path']) as f:
            g=f['episodes'][r['identity']] if 'episodes' in f else f['episode']
            if 'dagger/control_source' not in g['transition']:continue
            tr=g['transition'];source=tr['dagger/control_source'][:]
            if not (source!=0).any():continue
            handoff=int(np.flatnonzero(source!=0)[0])
            if handoff not in r['eligible_completed_ticks']:continue
            valid=tr['dagger/expert_label_valid'][:].astype(bool)
            first=int(np.flatnonzero(valid)[0]);state=tr['diagnostics_before/object_pose_w'][handoff]
            exact.append(dict(identity=r['identity'],family=r['family'],category=r['category'],
                handoff_tick=handoff,first_legal_tick=first,queue_flush_ticks=first-handoff,
                legal_ticks=int(valid.sum()),success=bool(r['success']),raw_path=r['path'],
                actual_authority_source=int(source[handoff]),mask_prefix_zero=not valid[:first].any(),
                object_pose_at_handoff=state.tolist(),minimum_selected_successful_steps=60))
    write('historical_exact_never_gate_rescues.json',dict(rows=exact,trajectories=len(exact),
        families=len({r['family'] for r in exact}),new_diagnostic_episodes=False,
        qualification='Actual historical expert recoveries whose last completed student state satisfies the causal never15mm held-near stall screen; not fresh efficacy estimates'))
    csvout('historical_exact_never_gate_rescues.csv',exact)

def angle(q1,q2):
    return float(2*np.arccos(np.clip(abs(np.dot(q1,q2))/(np.linalg.norm(q1)*np.linalg.norm(q2)),0,1)))
def compare(g1,g2,t):
    out={}
    paths={'joint_rad':'transition/diagnostics_before/joint_position_rad',
           'joint_velocity_rad_s':'transition/diagnostics_before/joint_velocity_rad_s',
           'object_velocity':'transition/diagnostics_before/object_velocity_w_m_s_rad_s',
           'contact_force_N':'obs/contact_substeps_w_N','actuator_target_rad':'obs/actuator_target_rad',
           'prior_plan_rad':'obs/prior_plan','previous_nominal_rad':'obs/previous_nominal_action',
           'expert_delayed_action':'transition/dagger/expert_delayed_action',
           'expert_queue_length':'transition/dagger/expert_queue_length'}
    for label,p in paths.items():
        if p not in g1 or p not in g2:out[label+'_max_abs']=None;continue
        out[label+'_max_abs']=float(np.max(np.abs(g1[p][t].astype(float)-g2[p][t].astype(float))))
    p='transition/diagnostics_before/object_pose_w';a,b=g1[p][t].copy(),g2[p][t].copy()
    # Vector environments translate entire scenes. Compare in each robot base,
    # as the production paired-initial verifier does, not world lane offsets.
    base='transition/diagnostics_before/base_pose_w'
    a[:3]-=g1[base][t][:3];b[:3]-=g2[base][t][:3]
    out.update(object_translation_m=float(np.linalg.norm(a[:3]-b[:3])),object_angle_rad=angle(a[3:],b[3:]),
               original_commit_phase=t%12,comparison_tick=t)
    out['observed_state_equal_at_1e_5']=all(v is not None and v<=1e-5 for k,v in out.items() if (k.endswith('_max_abs') and not k.startswith('expert_')) or k in ('object_translation_m','object_angle_rad'))
    return out

def pilots():
    rows=[];pairs=[]
    for owner in sorted(ROOT.glob('never_gate_pilot*')):
        roll=owner/'rollouts'
        if not (roll/'plan.json').exists():continue
        plan=read(roll/'plan.json');byseed=collections.defaultdict(dict)
        for case in plan['cases']:
            d=roll/case['name'];rp=d/'result.json';result=read(rp) if rp.exists() else {}
            row=dict(pilot=owner.name,identity=case['name'],seed=case['spec']['seed'],
                arm='continue' if case['dagger']['trigger'].get('observe_only') else 'expert_enabled',
                software_complete=bool(result.get('software_complete')),success=bool(result.get('success')),
                steps=result.get('steps'),raw_path=str(d/'numeric.h5'),
                handoff=False,legal_ticks=0,main_pool_admission=False)
            numeric=d/'numeric.h5'
            partial=d/'demonstration/data.h5.partial'
            if not numeric.exists() and partial.exists():
                # Preserve and inspect interrupted authority, rather than
                # assuming that a missing final result means no handoff.
                retained=d/'numeric_retained.h5'
                if not retained.exists():
                    with h5py.File(partial) as f,h5py.File(retained,'x') as out:
                        g=next(iter(f['episodes'].values()));e=out.create_group('episode')
                        for key in ('obs','transition','terminated','truncated'):
                            if key in g:g.copy(key,e)
                numeric=retained
            if numeric.exists():
                row['raw_path']=str(numeric);row.update(replay(numeric))
                with h5py.File(numeric) as f:
                    tr=f['episode/transition'];src=tr['dagger/control_source'][:]
                    valid=tr['dagger/expert_label_valid'][:].astype(bool)
                    row.update(handoff=bool((src!=0).any()),legal_ticks=int(valid.sum()),
                        handoff_tick=int(np.flatnonzero(src!=0)[0]) if (src!=0).any() else None,
                        first_legal_tick=int(np.flatnonzero(valid)[0]) if valid.any() else None)
            if not row['software_complete']:
                fail=roll/'worker_failure.json';row['technical_error']=read(fail).get('error') if fail.exists() else 'No complete per-case result; see owner log'
            rows.append(row);byseed[row['seed']][row['arm']]=row
        for seed,s in byseed.items():
            if set(s)!={'continue','expert_enabled'}:continue
            x,y=s['continue'],s['expert_enabled']
            if not x['software_complete'] or not y['software_complete']:continue
            with h5py.File(x['raw_path']) as f1,h5py.File(y['raw_path']) as f2:
                a,b=f1['episode'],f2['episode'];n=min(len(a['obs/step_id']),len(b['obs/step_id']))
                tq=y.get('handoff_tick')
                if tq is None:tq=x.get('never_alignment_first') or y.get('never_alignment_first')
                tq=min(tq,n-1) if tq is not None else n-1
                comparisons=[dict(point='reset',**compare(a,b,0)),dict(point='handoff_or_last_common',**compare(a,b,tq))]
                q1=a['transition/diagnostics_before/joint_position_rad'][:n]
                q2=b['transition/diagnostics_before/joint_position_rad'][:n]
                d=np.max(np.abs(q1-q2),axis=-1);idx=np.flatnonzero(d>1e-5)
                pairs.append(dict(pilot=owner.name,seed=seed,continue_success=x['success'],expert_enabled_success=y['success'],
                    real_handoff=y['handoff'],first_joint_divergence_tick=int(idx[0]) if len(idx) else None,
                    comparisons=comparisons,same_state_branch=False,
                    interpretation='Different physical lanes, only observed initial states paired; divergence before handoff prevents exact counterfactual claim'))
    write('diagnostic_branch_stage_census.json',rows);csvout('diagnostic_branch_results.csv',rows)
    write('branch_state_pairing.json',pairs)
    write('diagnostic_branch_summary.json',dict(planned=len(rows),complete=sum(r['software_complete'] for r in rows),
        technical=sum(not r['software_complete'] for r in rows),handoffs=sum(r['handoff'] for r in rows),
        recovered=sum(r['handoff'] and r['success'] for r in rows),legal_ticks=sum(r['legal_ticks'] for r in rows),
        complete_legal_ticks=sum(r['legal_ticks'] for r in rows if r['software_complete']),
        interrupted_handoffs=sum(r['handoff'] and not r['software_complete'] for r in rows),
        terminal64_selection=False,new_pool_trajectories=0,optimizer_updates=0))

def positives():
    rows=[];initial=[]
    for out in sorted(ROOT.glob('positive*')):
        if not (out/'plan.json').exists():continue
        for c in read(out/'plan.json')['cases']:
            d=out/c['name'];rp=d/'result.json';r=read(rp) if rp.exists() else {}
            rows.append(dict(run=out.name,role=c['role'],seed=c['spec']['seed'],success=bool(r.get('success')),
                software_complete=bool(r.get('software_complete')),steps=r.get('steps'),result_path=str(rp)))
    for r in rows:
        if not r['software_complete'] or r['run']=='positive_r0_unshifted':continue
        ref=ROOT/'positive_r0_unshifted'/f"{r['role']}_positive_r0_unshifted"/'numeric.h5'
        p=Path(r['result_path']).parent/'numeric.h5'
        with h5py.File(ref) as f1,h5py.File(p) as f2:
            initial.append(dict(reference=str(ref),candidate=str(p),run=r['run'],role=r['role'],**compare(f1['episode'],f2['episode'],0)))
    write('positive_controls.json',dict(student_rows=rows,expert_rows=read(ROOT/'expert_controls/results.json'),
        initial_comparisons=initial,qualification='Known seed reused only for debugging. The unshifted second repeat had no controls; not a completed matched 2x2 comparison.'))
    csvout('positive_controls.csv',rows)

if __name__=='__main__':historical();pilots();positives();print('Diagnostic tables refreshed')
