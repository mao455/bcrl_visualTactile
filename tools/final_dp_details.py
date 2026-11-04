"""Per-episode expert-reference diagnostics from already recorded numeric traces."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch

from tools.state_image_queue import read
from tools.h24_diagnosis_report import describe
from tools.h24_comparison_report import validated_fk
from tools.state_fk_audit import fk_pose


def details(root, record):
    folder=Path(record['output'])/record['trial']['identity']
    summary,series,trace=describe(folder)
    session=read(root/'session.json');data=read(Path(session['source_root'])/'data.json')
    entry=data['entries'][record['trial']['identity']]
    with h5py.File(entry['path'],'r') as f:
        reference=f['episodes'][entry['episode']]['transition/diagnostics_after/joint_position_rad'][:]
    fk=validated_fk(read(folder/'metadata.json'))
    xyz=fk_pose(fk,reference)[0]
    n=min(len(xyz),len(series['actual_xyz']))
    error=np.linalg.norm(series['actual_xyz'][:n]-xyz[:n],axis=-1)*1000
    contact=summary['events']['contact_tick']
    pre=min(n,contact if contact is not None else n)
    summary['recorded_expert_reference']=dict(common_ticks=n,
        alignment='same recorded control tick; common prefix only; no hidden PhysX exact-replay claim or time warping',
        ee_mean_mm=float(error.mean()),ee_p95_mm=float(np.quantile(error,.95)),
        precontact_ee_mean_mm=float(error[:pre].mean()) if pre else None,
        final_common_tick_error_mm=float(error[-1]))
    events=summary['events']
    phase=('success' if record['trial']['success'] else 'after_alignment' if events['alignment_tick'] is not None else
           'after_first_subgoal' if events['subgoal_tick'] is not None else 'after_lift' if events['lift_tick'] is not None else
           'after_contact' if contact is not None else 'before_contact')
    hand=trace['target'][:,7:];delta=np.diff(hand,axis=0)
    reversals=((delta[1:]*delta[:-1])<0).sum(axis=0)
    return dict(model=record['model'],ordinal=record['ordinal'],cohort=record['trial']['cohort'],
        success=record['trial']['success'],reasons=record['trial']['reasons'],failure_phase=phase,metrics=summary,
        hand_motion=dict(per_joint_target_range_rad=np.ptp(hand,axis=0).tolist(),
            per_joint_direction_reversals=reversals.tolist(),
            note='Descriptive counts, no newly invented oscillation acceptance threshold'),
        before_termination=dict(last_ticks=min(30,len(trace['joint'])),
            object_z_m=trace['object'][-30:,2].tolist(),contact_force_N=series['object_contact_N'][-30:].tolist(),
            hand_target_jump_rad=series['hand_jump_rad'][-30:].tolist(),arm_target_jump_rad=series['arm_jump_rad'][-30:].tolist()),
        maximum_hand_jump=dict(tick=int(np.argmax(series['hand_jump_rad'])),rad=float(np.max(series['hand_jump_rad']))))
