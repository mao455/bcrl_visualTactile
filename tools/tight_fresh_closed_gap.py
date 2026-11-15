"""Compare saved original-family autonomy with its successful demonstration.

Time-aligned expert commands cease to be counterfactual correct labels once the
student state diverges. This report locates divergence, not its sole cause.
Reads numeric fields only and never performs another model or simulator run.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tools.tight_fresh_grasp_evidence import behavior
from visuotactile.training_resources import atomic_json


def first_run(mask, length=3):
    run=0
    for index,value in enumerate(mask):
        run=run+1 if value else 0
        if run>=length:return index-length+1
    return None


def report(root, output):
    if output.exists():raise FileExistsError('Preserve prior comparison')
    read=lambda p:json.loads(p.read_text())
    directory=root/'original_training_probe';plan=read(directory/'plan.json')
    data=read(root/'A/state_data/data.json');rows=[]
    for case in plan['cases']:
        result=read(directory/case['name']/'result.json')
        row=dict(case=case['name'],role=case['role'],legal=result['legal'],family=case['spec']['family'])
        if not result['legal']:
            rows.append(row);continue
        identity=case['spec']['source_identity'];entry=data['entries'][identity]
        fit=read(root/('pair_fit_'+case['role']+'.json'))['results'][identity]
        with h5py.File(entry['path']) as f,h5py.File(directory/case['name']/'numeric.h5') as g:
            original=f['episodes'][entry['episode']];actual=g['episode']
            n=min(150,len(original['obs/time_s']),len(actual['obs/time_s']))
            expert=original['transition/target_after_rad'][:n]
            issued=actual['transition/target_after_rad'][:n]
            error=issued-expert
            arm=np.sqrt(np.mean(error[:,:7]**2,axis=1));hand=np.sqrt(np.mean(error[:,7:]**2,axis=1))
            whole=np.sqrt(np.mean(error**2,axis=1))
            # Model input state is prior to each submitted command. Initial
            # observable differences disclose reset/lane effects explicitly.
            before='transition/diagnostics_before/'
            q=actual[before+'joint_position_rad'][:n]-original[before+'joint_position_rad'][:n]
            obj=actual[before+'object_pose_w'][:n,:3]-original[before+'object_pose_w'][:n,:3]
            applied=actual['transition/actuator_substep_target_rad'][:n]
            mapping_error=float(np.max(np.abs(applied-issued[:,None,:])))
            if mapping_error>2e-5:raise ValueError('Submitted/applied target mapping differs')
            row.update(identity=identity,comparison_ticks=n,
                initial_joint_max_abs_difference_rad=float(np.abs(q[0]).max()),
                initial_object_position_difference_m=float(np.linalg.norm(obj[0])),
                command_to_actuator_max_abs_difference_rad=mapping_error,
                first_three_tick_arm_gap_above_002_rad=first_run(arm>.002),
                first_three_tick_hand_gap_above_02_rad=first_run(hand>.02),
                first_three_tick_q_gap_above_01_rad=first_run(np.max(np.abs(q),axis=1)>.01),
                first_three_tick_object_gap_above_002_m=first_run(np.linalg.norm(obj,axis=1)>.002),
                arm_command_rmse_rad=arm.tolist(),hand_command_rmse_rad=hand.tolist(),
                all_joint_command_rmse_rad=whole.tolist(),
                teacher_forced_all_joint_rmse_rad=fit['first150_actual_target_error'][:n],
                teacher_forced_nominal_to_command_gap_rad=fit['first150_nominal_to_command_gap'][:n])
        rows.append(row)
    atomic_json(output,dict(complete=True,rows=rows,autonomous_behavior=behavior(directory,horizon=300),
        units=dict(indices='zero-based control action index',simulation_seconds='index / 60',control_hz=60,planner_hz=5),
        thresholds='Fixed descriptive thresholds and three consecutive ticks; not task success or acceptance thresholds',
        limitations='Registered reset families reused; no exact hidden-state replay claim. After states diverge the old expert trajectory is not an expert label for the student state. Teacher-forced error and autonomous path divergence answer different questions. A nominal-to-command gap is not A q[t+1] prediction error.',
        model_forward_calls=0,simulator_starts=0,test_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root.resolve(),a.output.resolve())
