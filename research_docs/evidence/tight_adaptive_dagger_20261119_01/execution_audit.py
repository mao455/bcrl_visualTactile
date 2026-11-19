"""Read completed autonomous records; separate policy clipping from actuator feedback."""
import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.tight_adaptive_dagger import ROOT, PUBLIC, read, write
from tools.tight_frozen_comparison import table
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION, reachable_position_bounds
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.geometry_sampling import STAGES


def maxabs(value):
    return float(np.abs(value).max()) if value.size else 0.


def audit(waves, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    rows=[];phase_rows=[]
    for wave in waves:
        plan=read(wave/'plan.json')
        for case in plan['cases']:
            folder=wave/case['name'];result=read(folder/'result.json')
            assert result['legal'] and result['software_complete']
            with h5py.File(folder/'numeric.h5') as f, np.load(folder/'trace.npz') as trace:
                t=f['episode/transition'];o=f['episode/obs']
                previous=t['target_before_rad'][:];actual=t['target_after_rad'][:]
                position=t['diagnostics_after/joint_position_rad'][:];n=len(actual)
                assert n==result['steps'] and np.isfinite(actual).all() and np.isfinite(position).all()
                metadata=json.loads(f.attrs['metadata_json'])
                limits=torch.as_tensor(metadata['joint_limits_rad'],dtype=torch.float32).reshape(-1,29,2)[0]
                bounds=reachable_position_bounds(torch.from_numpy(previous),limits,REACHABLE_POSITION_EXECUTION).numpy()
                violation=np.maximum(bounds[...,0]-actual,actual-bounds[...,1]).clip(min=0)
                continuity=dict(command_max_rad=maxabs(previous[1:]-actual[:-1]),
                    joint_max_rad=maxabs(t['diagnostics_before/joint_position_rad'][1:]-position[:-1]),
                    obs_actuator_max_rad=maxabs(o['actuator_target_rad'][:]-previous),
                    trace_actual_max_rad=maxabs(trace['target']-actual),
                    trace_previous_max_rad=maxabs(trace['execution_previous_target_rad']-previous),
                    clock_max_s=maxabs(t['time_before_s'][1:]-t['time_after_s'][:-1]),
                    observation_clock_max_s=maxabs(o['time_s'][:]-t['time_before_s'][:]))
                assert max(continuity.values())<=1e-6,(case['name'],continuity)
                assert float(violation.max())<=1.1e-6,(case['name'],float(violation.max()))
                clock=o['time_s'][:]
                assert np.all(o['camera_time_s'][:]<=clock[:,None]+1e-6)
                valid=o['contact_valid'][:].reshape(-1).astype(bool)
                assert np.all(o['contact_time_s'][:][valid]<=clock[valid,None]+1e-6)
                assert np.all(np.diff(o['camera_frame'][:],axis=0)>0)
                proposal_clip=trace['residual_clip_mask'].astype(bool)
                projection=trace['execution_projection_rad']
                projected=np.abs(projection)>1e-6
                preprojection=trace['execution_preprojection_target_rad']
                chain_error=maxabs(actual-preprojection-projection)
                assert chain_error<=1e-6,(case['name'],chain_error)
                lower,upper=bounds[...,0],bounds[...,1]
                boundary=np.minimum(np.abs(actual-lower),np.abs(actual-upper))<=1.1e-6
                camera_age=clock[:,None]-o['camera_time_s'][:]
                contact_age=clock[valid,None]-o['contact_time_s'][:][valid]
                tracking=actual-position
                row=dict(wave=str(wave),case=case['name'],role=case['role'],seed=case['spec']['seed'],
                    success=result['success'],ticks=n,proposal_joint_clip_ticks=int(proposal_clip.any(-1).sum()),
                    proposal_joint_clip_components=int(proposal_clip.sum()),
                    reachable_projection_ticks=int(projected.any(-1).sum()),
                    reachable_projection_components=int(projected.sum()),
                    maximum_reachable_projection_rad=maxabs(projection),
                    maximum_actual_reachable_violation_rad=float(violation.max()),
                    actuator_joint_clip_ticks=int(t['position_clip_mask'][:].reshape(n,-1).any(-1).sum()),
                    tracking_arm_squared_sum=float((tracking[:,:7]**2).sum()),
                    tracking_hand_squared_sum=float((tracking[:,7:]**2).sum()),
                    invalid_contact_observations=int((~valid).sum()),continuity=continuity)
                row.update(proposal_joint_clip_component_fraction=float(proposal_clip.mean()),
                    reachable_boundary_component_fraction=float(boundary.mean()),
                    camera_age_max_s=float(camera_age.max()),
                    contact_age_max_s=float(contact_age.max()) if contact_age.size else None,
                    command_chain_max_error_rad=chain_error)
                rows.append(row)
                phase=phases(t)
                for k,label in enumerate(STAGES):
                    mask=phase==k
                    if not mask.any():continue
                    phase_rows.append(dict(role=case['role'],case=case['name'],success=result['success'],
                        stage=label,ticks=int(mask.sum()),
                        proposal_joint_clip_components=int(proposal_clip[mask].sum()),
                        reachable_projection_components=int(projected[mask].sum()),
                        reachable_boundary_components=int(boundary[mask].sum()),
                        tracking_arm_squared_sum=float((tracking[mask,:7]**2).sum()),
                        tracking_hand_squared_sum=float((tracking[mask,7:]**2).sum())))
    total=collections.defaultdict(list)
    for r in rows:total[r['role']].append(r)
    summary=[]
    for role,rr in total.items():
        ticks=sum(r['ticks'] for r in rr)
        summary.append(dict(role=role,cases=len(rr),ticks=ticks,
            proposal_joint_clip_tick_fraction=sum(r['proposal_joint_clip_ticks'] for r in rr)/ticks,
            reachable_projection_tick_fraction=sum(r['reachable_projection_ticks'] for r in rr)/ticks,
            reachable_projection_component_fraction=sum(r['reachable_projection_components'] for r in rr)/(29*ticks),
            maximum_reachable_projection_rad=max(r['maximum_reachable_projection_rad'] for r in rr),
            maximum_actual_reachable_violation_rad=max(r['maximum_actual_reachable_violation_rad'] for r in rr),
            tracking_arm_rmse_rad=(sum(r['tracking_arm_squared_sum'] for r in rr)/ticks/7)**.5,
            tracking_hand_rmse_rad=(sum(r['tracking_hand_squared_sum'] for r in rr)/ticks/22)**.5,
            continuity_max=max(max(r['continuity'].values()) for r in rr)))
    phase_totals=collections.defaultdict(collections.Counter)
    for r in phase_rows:
        phase_totals[r['role'],r['success'],r['stage']].update({k:r[k] for k in (
            'ticks','proposal_joint_clip_components','reachable_projection_components',
            'reachable_boundary_components','tracking_arm_squared_sum','tracking_hand_squared_sum')})
    phase_summary=[]
    for (role,success,stage),v in sorted(phase_totals.items()):
        ticks=v['ticks']
        phase_summary.append(dict(role=role,success=success,stage=stage,ticks=ticks,
            **{k+'_fraction':v[k]/(29*ticks) for k in ('proposal_joint_clip_components',
                'reachable_projection_components','reachable_boundary_components')},
            tracking_arm_rmse_rad=(v['tracking_arm_squared_sum']/ticks/7)**.5,
            tracking_hand_rmse_rad=(v['tracking_hand_squared_sum']/ticks/22)**.5))
    table(output/'per_case.csv',rows);table(output/'summary.csv',summary)
    table(output/'phase_per_case.csv',phase_rows);table(output/'phase_summary.csv',phase_summary)
    write(output/'receipt.json',dict(cases=len(rows),rows=rows,summary=summary,passed=True,
        scope='Actual complete autonomous records only. Policy proposal joint clipping, reachable projection and actuator clipping are separate events. Tracking error is not the supervised command error. Observable feedback continuity and causal timestamps checked; no hidden-state deterministic replay or latent-history reconstruction claim. Success/phase comparisons are observational and exposure-weighted, not matched causal controls; recorded native-reset camera age may be nonzero.',
        optimizer_updates=0,physics_updates=0,cuda_initialized=torch.cuda.is_initialized()))
    print(dict(cases=len(rows),summary=summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int);p.add_argument('--heldout',action='store_true')
    p.add_argument('--waves',nargs='+',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.waves:
        assert a.output;audit(a.waves,a.output)
    else:
        name='final_test' if a.heldout else f'round{a.round:02d}/evaluation'
        registration=read(ROOT/name/'registration.json')
        audit([Path(w) for w in registration['waves']],PUBLIC/name/'execution_audit')
