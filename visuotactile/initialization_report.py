"""Small measured reset/wrench report from audited episodes, not inferred noise."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .run_rl_trials import validate_completion
from .compact import EpisodeFile
from .trial_utils import write_json


def report(manifest,output):
    manifest=Path(manifest);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    m=json.loads(manifest.read_text());rows=[];initials=[]
    for entry in m['episodes']:
        p=manifest.parent/entry['directory'];trial=validate_completion(p)
        a,s=trial['arrays']['initial'],trial['summary'];initials.append(a['object_pose_w'])
        with EpisodeFile(p/'data.h5') as f:
            g=f['episodes'][entry['episode']];lag=g['transition/control_trace/effective_lag_steps'][:]
        row=dict(id=entry['id'],seed=entry['seed'],outcome=s['outcome'],steps=s['control_steps'],
            object_xyz_w_m=a['object_pose_w'][:3].tolist(),object_quat_wxyz=a['object_pose_w'][3:].tolist(),
            hole_xyz_w_m=a['hole_pose_w'][:3].tolist(),q_arm_rad=a['joint_position_rad'][:7].tolist(),
            qd_norm_rad_s=float(np.linalg.norm(a['joint_velocity_rad_s'])),
            force_probability=float(a['external_force_probability']),torque_probability=float(a['external_torque_probability']),
            force_physics_samples=s['applied_force']['nonzero_steps'],force_duration_s=s['applied_force']['duration_s'],
            peak_force_N=s['applied_force']['peak'],torque_physics_samples=s['applied_torque']['nonzero_steps'],
            peak_torque_Nm=s['applied_torque']['peak'],force_steps_with_contact_before=s['force_steps_with_finger_contact_before'],
            force_steps_with_contact_after=s['force_steps_with_finger_contact_after'],
            effective_delay_counts=np.bincount(lag.astype(int),minlength=3).tolist())
        rows.append(row)
    with (output/'initial_states.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(rows)
    xyz=np.array(initials)[:,:3] if initials else np.empty((0,3))
    outcomes={k:sum(a['outcome']==k for a in m['attempts']) for k in sorted({a['outcome'] for a in m['attempts']})}
    result=dict(source_manifest=str(manifest),episodes=rows,attempt_outcomes=outcomes,
        success_xyz_quantiles_m=np.quantile(xyz,[0,.25,.5,.75,1],axis=0).tolist() if len(xyz) else None,
        success_exact_initial_duplicates=len(initials)-len({tuple(x) for x in initials}),
        zero_force_episode_fraction=sum(r['force_physics_samples']==0 for r in rows)/len(rows) if rows else None,
        limitation='Success-conditioned sample, not native reset distribution; software startup without metadata has no initial state. No empirical noise distribution recorded.')
    write_json(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=report(a.manifest,a.output);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
