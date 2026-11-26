"""Compare inference outputs with actual expert authority and execution feedback."""
from pathlib import Path
import json,csv
import numpy as np,h5py
R=Path.cwd();ROOT=R/'outputs/visuotactile/run/tight_machine_review_20261126_01'
A=R/'outputs/diagnostics/tight_machine_review_20261126_01'
def read(p):return json.loads(Path(p).read_text())
def rmse(x,y):
    d=np.asarray(x)-np.asarray(y)
    return dict(arm_rad=float(np.sqrt(np.mean(d[:7]**2))),hand_rad=float(np.sqrt(np.mean(d[7:]**2))))
rows=[];geometry=[];source=[]
base=ROOT/'fresh_rescue_history02/inference01'
for p in (base/'inference',base/'inference_source'):
    for r in read(p/'rows.json'):
        identity=r['identity'];d=ROOT/'never_gate_pilot07_B72_training/rollouts'/identity
        result=read(d/'result.json');receipt=result['dagger']['trigger_receipt']
        with h5py.File(d/'demonstration/data.h5') as f:
            tr=f['episodes'][identity]['transition'];t=r['absolute_tick'];v=r['values'];dag=tr['dagger']
            actual=dag['actual_issued'][t];applied=dag['actual_applied'][t];target=dag['u_expert'][t]
            q=dag['q_reached'][t];nominal=dag['nominal'][t]
            row=dict(identity=identity,role=r['role'],policy=r['policy'],tick=t,first_legal=r['first_legal'],
                native_anchor=r['native_anchor'],slot=r['slot'],
                nominal_arm=r['nominal']['arm_rmse_rad'],nominal_hand=r['nominal']['hand_rmse_rad'],
                residual_arm=r['residual']['arm_rmse_rad'],residual_hand=r['residual']['hand_rmse_rad'],
                preprojection_arm=r['preprojection']['arm_rmse_rad'],preprojection_hand=r['preprojection']['hand_rmse_rad'],
                final_arm=r['final_command']['arm_rmse_rad'],final_hand=r['final_command']['hand_rmse_rad'],
                projection_arm=r['projection_magnitude']['arm_rmse_rad'],projection_hand=r['projection_magnitude']['hand_rmse_rad'],
                expert_issued_to_applied_max=float(np.max(np.abs(actual-applied))),
                applied_to_expert_target_max=float(np.max(np.abs(applied-target))),
                q_reached_to_observed_max=float(np.max(np.abs(q-tr['diagnostics_after/joint_position_rad'][t]))),
                first_difference='model prediction versus expert target; issued/applied expert chain remains exact')
            rows.append(row)
            if r['policy']=='source':
                source.append(dict(identity=identity,tick=t,
                    nominal_replay_to_live_max=float(np.max(np.abs(np.asarray(v['nominal'])-nominal))),
                    student_command_replay_to_live_max=float(np.max(np.abs(np.asarray(v['final_command'])-dag['student_proposed_target_rad'][t])))))
            if r['role']=='B' and r['policy']=='new' and r['first_legal']:
                switch=receipt['actual_scheduled_takeover_tick'];n=len(tr['step_id'])
                for at in sorted({switch-1,switch,t,t+1,t+12,t+36,t+60,n-1}):
                    if at>=n:continue
                    geometry.append(dict(identity=identity,tick=at,seconds_from_handoff=(at-switch)/60,
                        source=int(dag['control_source'][at]),legal=bool(dag['expert_label_valid'][at]),
                        expert_source_tick=int(dag['expert_source_step'][at]),queue_length=int(dag['expert_queue_length'][at]),
                        keypoint_mm=float(tr['native_keypoints_max_dist'][at]*1000),
                        subgoal_after=float(tr['diagnostics_after/successes'][at]),
                        object_z_m=float(tr['diagnostics_after/object_pose_w'][at][2]),
                        min_substep_contact_fingers=int((np.linalg.norm(tr['object_contact_force_w'][at],axis=-1)>.1).sum(-1).min()),
                        **{'actual_issued_tracking_'+k:v for k,v in rmse(dag['actual_issued'][at],tr['diagnostics_after/joint_position_rad'][at]).items()},
                        expert_label_tracking=rmse(dag['u_expert'][at],tr['diagnostics_after/joint_position_rad'][at]) if bool(dag['expert_label_valid'][at]) else None))
with (A/'fresh_rescue_action_chain.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(A/'fresh_rescue_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
(A/'source_self_replay.json').write_text(json.dumps(source,indent=2)+'\n')
print('rows',len(rows),'source replay max',max(r['student_command_replay_to_live_max'] for r in source))
